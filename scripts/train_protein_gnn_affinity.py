"""Run an isolated Davis affinity experiment with MGNN and a 2D ProteinGNN."""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from protein_gnn import PROTEIN_EMBEDDING_DIM, ProteinGNN
from src.dataset import DavisDataset
from src.metrics import concordance_index, mse, r_m2, rmse
from src.models.mgnn import MGNN
from src.preprocessing import extract_graph_structure, smiles_to_mol
from src.train import resolve_device, set_random_seed, split_train_validation

RAW_PROCESSED_DIR = PROJECT_ROOT / "data/processed/davis"
GRAPH_DIR = PROJECT_ROOT / "results/davis/protein_graphs_2d"
OUTPUT_DIR = PROJECT_ROOT / "results/davis/protein_gnn_experiment"
SEED = 42
VALIDATION_FRACTION = 0.1
BATCH_SIZE = 32
EPOCHS = 30
STEPS_PER_EPOCH = 50
LEARNING_RATE = 5e-4
PATIENCE = 10
DRUG_EMBEDDING_DIM = 96
FUSION_INPUT_DIM = DRUG_EMBEDDING_DIM + PROTEIN_EMBEDDING_DIM


class InteractionDataset(Dataset):
    """Pair each Davis interaction with its drug and sequence-matched protein graph."""

    def __init__(
        self,
        frame: Any,
        protein_graphs: dict[str, dict[str, Any]],
        drug_graph_cache: dict[str, dict[str, Tensor]],
    ) -> None:
        self.rows = list(
            frame[["drug_id", "compound_iso_smiles", "target_id", "target_sequence", "affinity"]]
            .itertuples(index=False, name=None)
        )
        self.protein_graphs = protein_graphs
        self.drug_graph_cache = drug_graph_cache

        for drug_id, smiles, target_id, raw_sequence, _affinity in self.rows:
            sequence = str(raw_sequence).strip().upper()
            if sequence not in self.protein_graphs:
                raise KeyError(
                    f"No residue graph for target {target_id} "
                    f"(drug {drug_id}, sequence length {len(sequence)})"
                )
            if smiles not in self.drug_graph_cache:
                molecule = smiles_to_mol(smiles)
                self.drug_graph_cache[smiles] = extract_graph_structure(molecule)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        drug_id, smiles, target_id, raw_sequence, affinity = self.rows[index]
        sequence = str(raw_sequence).strip().upper()
        protein_graph = self.protein_graphs[sequence]
        return {
            "drug_id": str(drug_id),
            "target_id": str(target_id),
            "drug_x": self.drug_graph_cache[smiles]["x"],
            "drug_edge_index": self.drug_graph_cache[smiles]["edge_index"],
            "protein_x": protein_graph["x"],
            "protein_edge_index": protein_graph["edge_index"],
            "affinity": torch.tensor(float(affinity), dtype=torch.float32),
        }


def load_protein_graphs() -> dict[str, dict[str, Any]]:
    graph_files = sorted(GRAPH_DIR.glob("protein_*.pt"))
    if not graph_files:
        raise FileNotFoundError(f"No protein graph files found in {GRAPH_DIR}")

    sequence_to_graph = {}
    graph_ids = set()
    for graph_path in graph_files:
        graph = torch.load(graph_path, map_location="cpu", weights_only=True)
        sequence = str(graph.get("sequence", "")).strip().upper()
        graph_id = str(graph.get("graph_id", graph_path.stem))
        if not sequence:
            raise ValueError(f"Empty sequence in protein graph {graph_path}")
        if sequence in sequence_to_graph:
            raise ValueError(
                f"Duplicate sequence stored in {graph_path} and "
                f"{sequence_to_graph[sequence]['source_path']}"
            )
        features = graph.get("x")
        edge_index = graph.get("edge_index")
        if not isinstance(features, Tensor) or tuple(features.shape) != (len(sequence), 20):
            raise ValueError(f"Invalid residue features in {graph_path}")
        if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index in {graph_path}")
        graph["source_path"] = str(graph_path)
        graph["graph_id"] = graph_id
        sequence_to_graph[sequence] = graph
        graph_ids.add(graph_id)

    if len(graph_ids) != len(graph_files):
        raise ValueError("Protein graph IDs are not unique")
    return sequence_to_graph


def verify_split_sequence_coverage(protein_graphs: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Check only sequence-to-graph mapping; do not read or score test labels."""
    counts = {}
    failures = []
    for split in ("train", "test"):
        split_path = RAW_PROCESSED_DIR / f"davis_{split}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Required existing Davis split file is missing: {split_path}")
        count = 0
        with split_path.open(newline="", encoding="utf-8") as split_file:
            reader = csv.DictReader(split_file)
            required_fields = {"target_id", "target_sequence"}
            if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
                raise ValueError(f"{split_path} lacks target_id/target_sequence columns")
            for row_number, row in enumerate(reader, start=2):
                count += 1
                sequence = str(row["target_sequence"]).strip().upper()
                if sequence not in protein_graphs:
                    failures.append(
                        f"{split_path.name} row {row_number}, target {row['target_id']}, "
                        f"sequence length {len(sequence)}"
                    )
        counts[split] = count

    if failures:
        preview = "\n".join(f"  - {failure}" for failure in failures[:20])
        remainder = len(failures) - min(len(failures), 20)
        if remainder:
            preview += f"\n  - ... and {remainder} more unmapped interactions"
        raise RuntimeError(
            f"{len(failures)} Davis interactions cannot map to protein graphs:\n{preview}"
        )
    return counts


def collate_interactions(
    interactions: list[dict[str, Any]],
) -> tuple[Batch, Batch, Tensor]:
    drug_batch = Batch.from_data_list(
        [Data(x=item["drug_x"], edge_index=item["drug_edge_index"]) for item in interactions]
    )
    protein_batch = Batch.from_data_list(
        [Data(x=item["protein_x"], edge_index=item["protein_edge_index"]) for item in interactions]
    )
    targets = torch.stack([item["affinity"] for item in interactions]).reshape(-1)
    return drug_batch, protein_batch, targets


class ProteinGNNAffinityModel(nn.Module):
    """Experimental MGNN/ProteinGNN fusion model, not part of MGraphDTA."""

    def __init__(self) -> None:
        super().__init__()
        self.drug_encoder = MGNN(drug_embed_dim=DRUG_EMBEDDING_DIM)
        self.protein_encoder = ProteinGNN()
        self.classifier = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, drug_batch: Batch, protein_batch: Batch) -> Tensor:
        drug_embedding = self.drug_encoder(drug_batch)
        protein_embedding = self.protein_encoder(
            protein_batch.x,
            protein_batch.edge_index,
            protein_batch.batch,
        )
        if drug_embedding.shape[0] != protein_embedding.shape[0]:
            raise RuntimeError(
                "Drug/protein graph batch counts differ: "
                f"{drug_embedding.shape[0]} != {protein_embedding.shape[0]}"
            )
        fused = torch.cat((drug_embedding, protein_embedding), dim=1)
        return self.classifier(fused).reshape(-1)


def move_batches(
    drug_batch: Batch,
    protein_batch: Batch,
    targets: Tensor,
    device: torch.device,
) -> tuple[Batch, Batch, Tensor]:
    return drug_batch.to(device), protein_batch.to(device), targets.to(device)


def run_training_epoch(
    model: ProteinGNNAffinityModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    iterator = iter(loader)
    squared_error_sum = 0.0
    sample_count = 0
    criterion = nn.MSELoss(reduction="sum")

    for _step in range(STEPS_PER_EPOCH):
        try:
            drug_batch, protein_batch, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            drug_batch, protein_batch, targets = next(iterator)
        drug_batch, protein_batch, targets = move_batches(
            drug_batch, protein_batch, targets, device
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = model(drug_batch, protein_batch)
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(predictions.shape)} does not match "
                f"target shape {tuple(targets.shape)}"
            )
        loss_sum = criterion(predictions, targets)
        (loss_sum / targets.numel()).backward()
        optimizer.step()
        squared_error_sum += float(loss_sum.detach().item())
        sample_count += targets.numel()

    return squared_error_sum / sample_count


def evaluate_mse(
    model: ProteinGNNAffinityModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    sample_count = 0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for drug_batch, protein_batch, targets in loader:
            drug_batch, protein_batch, targets = move_batches(
                drug_batch, protein_batch, targets, device
            )
            predictions = model(drug_batch, protein_batch)
            if predictions.shape != targets.shape:
                raise RuntimeError(
                    f"Prediction shape {tuple(predictions.shape)} does not match "
                    f"target shape {tuple(targets.shape)}"
                )
            squared_error_sum += float(criterion(predictions, targets).item())
            sample_count += targets.numel()
    if sample_count == 0:
        raise ValueError("Evaluation loader is empty")
    return squared_error_sum / sample_count


def collect_predictions(
    model: ProteinGNNAffinityModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_batches = []
    target_batches = []
    with torch.no_grad():
        for drug_batch, protein_batch, targets in loader:
            drug_batch, protein_batch, targets = move_batches(
                drug_batch, protein_batch, targets, device
            )
            predictions = model(drug_batch, protein_batch)
            prediction_batches.append(predictions.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())
    if not prediction_batches:
        raise ValueError("Test loader is empty")
    return np.concatenate(target_batches), np.concatenate(prediction_batches)


def make_loader(dataset: Dataset, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_interactions,
        pin_memory=torch.cuda.is_available(),
    )


def save_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        [record["epoch"] for record in history],
        [record["training_mse"] for record in history],
        label="Training MSE",
    )
    axis.plot(
        [record["epoch"] for record in history],
        [record["validation_mse"] for record in history],
        label="Validation MSE",
    )
    axis.set(title="Experimental Davis Protein GNN Affinity Training", xlabel="Epoch", ylabel="MSE")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.epochs != EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError(
            f"This initial experiment is configured for {EPOCHS} epochs and "
            f"batch size {BATCH_SIZE}; use the defaults."
        )
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment outputs: {OUTPUT_DIR}"
        )
    for split in ("train", "test"):
        path = RAW_PROCESSED_DIR / f"davis_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required existing Davis split file is missing: {path}")

    set_random_seed(SEED)
    device = resolve_device()
    protein_graphs = load_protein_graphs()
    split_counts = verify_split_sequence_coverage(protein_graphs)
    print(
        f"Protein graph mapping preflight: PASSED "
        f"({len(protein_graphs)} unique sequences; "
        f"{split_counts['train']} train and {split_counts['test']} test interactions mapped)"
    )

    train_base = DavisDataset(split="train")
    drug_graph_cache: dict[str, dict[str, Tensor]] = {}
    train_dataset = InteractionDataset(train_base.df, protein_graphs, drug_graph_cache)
    training_subset, validation_subset = split_train_validation(
        train_dataset,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
    train_loader = make_loader(training_subset, shuffle=True)
    validation_loader = make_loader(validation_subset, shuffle=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    history_path = OUTPUT_DIR / "training_history.csv"
    metrics_path = OUTPUT_DIR / "final_metrics.json"
    curve_path = OUTPUT_DIR / "training_curve.png"

    model = ProteinGNNAffinityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_validation_mse = math.inf
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    print("===== Experimental Davis Protein GNN Affinity Training =====")
    print(f"Device: {device}")
    print(f"Training interactions: {len(training_subset)}")
    print(f"Validation interactions: {len(validation_subset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}; steps per epoch: {STEPS_PER_EPOCH}")
    print(f"Learning rate: {LEARNING_RATE}; early-stopping patience: {PATIENCE}")
    print("Protocol: experimental prototype; not a paper reproduction")

    training_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        training_mse = run_training_epoch(model, train_loader, optimizer, device)
        validation_mse = evaluate_mse(model, validation_loader, device)
        improved = validation_mse < best_validation_mse
        if improved:
            best_validation_mse = validation_mse
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                    "best_validation_mse": best_validation_mse,
                    "seed": SEED,
                    "model_type": "experimental_mgnn_protein_gnn_affinity",
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "training_mse": training_mse,
                "validation_mse": validation_mse,
                "best_validation_mse": best_validation_mse,
                "is_best": float(improved),
                "patience_counter": float(patience_counter),
            }
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS}: training_mse={training_mse:.6f} "
            f"validation_mse={validation_mse:.6f} "
            f"best_validation_mse={best_validation_mse:.6f} "
            f"patience={patience_counter}/{PATIENCE}"
        )
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (patience {PATIENCE})")
            break

    training_time_seconds = time.perf_counter() - training_started
    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a best validation checkpoint")

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    print(f"Restored best validation checkpoint from epoch {best_epoch}")

    # The independent test dataset is constructed only after best-checkpoint restoration.
    test_base = DavisDataset(split="test")
    test_dataset = InteractionDataset(test_base.df, protein_graphs, drug_graph_cache)
    test_loader = make_loader(test_dataset, shuffle=False)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    test_metrics = {
        "test_mse": float(mse(y_true, y_pred)),
        "test_rmse": float(rmse(y_true, y_pred)),
        "test_ci": float(concordance_index(y_true, y_pred)),
        "test_r_m2": float(r_m2(y_true, y_pred)),
    }

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=[
                "epoch",
                "training_mse",
                "validation_mse",
                "best_validation_mse",
                "is_best",
                "patience_counter",
            ],
        )
        writer.writeheader()
        writer.writerows(history)
    save_training_curve(history, curve_path)

    final_metrics = {
        "experiment": "standalone_davis_mgnn_2d_protein_gnn_affinity",
        "interpretation": "experimental result; not a paper reproduction",
        "best_validation_mse": best_validation_mse,
        "best_epoch": best_epoch,
        "training_time_seconds": training_time_seconds,
        "training_interactions": len(training_subset),
        "validation_interactions": len(validation_subset),
        "test_interactions": len(test_dataset),
        "test_metrics": test_metrics,
        "seed": SEED,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": PATIENCE,
        "device": str(device),
        "protein_graph_count": len(protein_graphs),
    }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print("===== Final Experimental Test Metrics =====")
    print(f"Test MSE: {test_metrics['test_mse']:.6f}")
    print(f"Test RMSE: {test_metrics['test_rmse']:.6f}")
    print(f"Test CI: {test_metrics['test_ci']:.6f}")
    print(f"Test r_m^2: {test_metrics['test_r_m2']:.6f}")
    print(f"Best validation MSE: {best_validation_mse:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Training time: {training_time_seconds:.2f} seconds")
    print(f"Training interactions: {len(training_subset)}")
    print(f"Validation interactions: {len(validation_subset)}")
    print(f"Test interactions: {len(test_dataset)}")
    print(f"Outputs: {OUTPUT_DIR}")
    print("This is an experimental result, not a paper reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
