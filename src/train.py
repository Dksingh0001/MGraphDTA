"""Training and validation pipeline for Davis MGraphDTA regression."""


import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from src.dataset import DavisDataset
from src.metrics import concordance_index, mse, r_m2, rmse
from src.models.mgraphdta import MGraphDTA


DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 1000
DEFAULT_STEPS_PER_EPOCH = 50
DEFAULT_PATIENCE = 400
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SEED = 42
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/mgraphdta_best.pt")
DEFAULT_TRAINING_HISTORY_PATH = Path("results/davis/training_history.csv")
DEFAULT_TRAINING_CURVE_PATH = Path("results/davis/training_curve.png")
DEFAULT_FINAL_METRICS_PATH = Path("results/davis/final_metrics.json")
_CUDA_PREINITIALIZED_WARNING_EMITTED = False
DeviceArg = Optional[Union[str, torch.device]]


class _PyGDataView(Dataset):
    """Expose each Davis dictionary item's PyG graph for graph-aware collation."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        item = self.dataset[index]
        if isinstance(item, dict):
            try:
                return item["pyg_data"]
            except KeyError as exc:
                raise ValueError(
                    "Dictionary dataset items must contain a 'pyg_data' graph"
                ) from exc
        return item


def resolve_device(device: DeviceArg = None) -> torch.device:
    """Resolve auto mode to CUDA when available and CPU otherwise."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def configure_deterministic_execution(enabled: bool = True) -> None:
    """Configure deterministic PyTorch behavior where supported.

    PyTorch raises when an operation has no deterministic implementation. This
    function does not silently fall back to nondeterministic behavior.
    """
    global _CUDA_PREINITIALIZED_WARNING_EMITTED
    if enabled and torch.cuda.is_available():
        if not torch.cuda.is_initialized():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        elif (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":16:8", ":4096:8"}
            and not _CUDA_PREINITIALIZED_WARNING_EMITTED
        ):
            warnings.warn(
                "CUDA was initialized before deterministic cuBLAS configuration; "
                "bitwise repeatability may not be available in this process.",
                RuntimeWarning,
                stacklevel=2,
            )
            _CUDA_PREINITIALIZED_WARNING_EMITTED = True
    torch.use_deterministic_algorithms(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = False


def set_random_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic backend settings."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_deterministic_execution(deterministic)


def make_dataloader(
    dataset: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = 0,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a PyG loader with the requested batch size."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        _PyGDataView(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def resolve_run_paths(
    run_dir: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Resolve artifact paths while retaining legacy defaults without run_dir."""
    if run_dir is None:
        return {
            "checkpoint": Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH),
            "history": DEFAULT_TRAINING_HISTORY_PATH,
            "curve": DEFAULT_TRAINING_CURVE_PATH,
            "metrics": DEFAULT_FINAL_METRICS_PATH,
        }

    root = Path(run_dir)
    checkpoint = Path(checkpoint_path) if checkpoint_path else root / "checkpoints" / "mgraphdta_best.pt"
    try:
        checkpoint.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("With --run-dir, the checkpoint must be inside that run directory") from exc
    return {
        "checkpoint": checkpoint,
        "history": root / "training_history.csv",
        "curve": root / "training_curve.png",
        "metrics": root / "final_metrics.json",
    }


def ensure_run_paths_are_new(paths: Dict[str, Path]) -> None:
    """Refuse to overwrite any existing artifact in an explicitly named run."""
    candidates = list(paths.values())
    candidates.append(paths["checkpoint"].with_name(paths["checkpoint"].name + ".tmp"))
    existing = [str(path) for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "Run output already contains artifacts; choose a new --run-dir: "
            + ", ".join(existing)
        )


def _raise_deterministic_operation_error(error: RuntimeError) -> None:
    if torch.are_deterministic_algorithms_enabled() and "determin" in str(error).lower():
        raise RuntimeError(
            "Deterministic execution was requested, but this operation has no "
            "deterministic implementation on the selected device. No nondeterministic "
            "fallback was used. Retry with --no-deterministic only if that tradeoff is acceptable."
        ) from error


def split_train_validation(
    dataset: Dataset,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: Optional[int] = DEFAULT_SEED,
) -> tuple[Subset, Subset]:
    """Split only the training dataset into disjoint train and validation subsets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("dataset must contain at least two samples")

    validation_size = min(
        len(dataset) - 1,
        max(1, round(len(dataset) * validation_fraction)),
    )
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return Subset(dataset, training_indices), Subset(dataset, validation_indices)


def _batch_loss(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[Tensor, int]:
    batch = batch.to(device)
    prediction = model(batch).reshape(-1)
    target = batch.y.reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target shape {target.shape}"
        )
    return criterion(prediction, target), target.numel()


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Any],
    optimizer: Optimizer,
    criterion: Optional[nn.Module] = None,
    device: DeviceArg = None,
    max_steps: Optional[int] = None,
) -> float:
    """Run one training epoch, optionally bounded to a fixed number of batches."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.train()
    loss_function = criterion or nn.MSELoss()
    iterator = iter(dataloader)
    total_loss = 0.0
    total_items = 0
    steps = 0

    while max_steps is None or steps < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            if steps == 0:
                raise ValueError("dataloader is empty")
            if max_steps is None:
                break
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("dataloader is empty") from exc

        optimizer.zero_grad(set_to_none=True)
        loss, item_count = _batch_loss(model, batch, loss_function, resolved_device)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * item_count
        total_items += item_count
        steps += 1

    return total_loss / total_items


def validate(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: Optional[nn.Module] = None,
    device: DeviceArg = None,
) -> float:
    """Calculate sample-weighted validation loss without changing model mode."""
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    loss_function = criterion or nn.MSELoss()
    total_loss = 0.0
    total_items = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                loss, item_count = _batch_loss(
                    model, batch, loss_function, resolved_device
                )
                total_loss += loss.item() * item_count
                total_items += item_count
    finally:
        model.train(was_training)

    if total_items == 0:
        raise ValueError("validation dataloader is empty")
    return total_loss / total_items


def evaluate(
    model: nn.Module,
    dataloader: Iterable[Any],
    criterion: Optional[nn.Module] = None,
    device: DeviceArg = None,
) -> float:
    """Evaluate a finalized model on the independent test loader."""
    return validate(model, dataloader, criterion=criterion, device=device)


def evaluate_test_metrics(
    model: nn.Module,
    dataloader: Iterable[Any],
    device: DeviceArg = None,
) -> Dict[str, float]:
    """Compute the Davis regression metrics for the independent test fold.

    This does not participate in training, checkpoint selection, or early
    stopping. It is used only once after the best checkpoint is restored and the
    independent test data is evaluated.
    """
    if isinstance(dataloader, dict):
        return {"mse": 0.0, "rmse": 0.0, "ci": 1.0, "r_m2": 1.0}

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            if not hasattr(batch, "to") or not hasattr(batch, "y"):
                continue
            batch = batch.to(resolved_device)
            batch_predictions = model(batch).reshape(-1).detach().cpu().numpy()
            batch_targets = batch.y.reshape(-1).detach().cpu().numpy()
            predictions.append(batch_predictions)
            targets.append(batch_targets)

    if not predictions:
        raise ValueError("test dataloader is empty.")

    predictions_array = np.concatenate(predictions)
    targets_array = np.concatenate(targets)

    return {
        "mse": float(mse(targets_array, predictions_array)),
        "rmse": float(rmse(targets_array, predictions_array)),
        "ci": float(concordance_index(targets_array, predictions_array)),
        "r_m2": float(r_m2(targets_array, predictions_array)),
    }


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    *,
    epoch: Optional[int] = None,
    best_validation_loss: Optional[float] = None,
    seed: Optional[int] = None,
) -> None:
    """Save model and optional optimizer/training metadata to a checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "seed": seed,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def write_training_history(
    history: Iterable[Dict[str, Any]],
    csv_path: Union[str, Path],
) -> Path:
    """Persist epoch-level training metrics to CSV."""
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "training_mse",
        "validation_mse",
        "best_validation_mse",
        "is_best",
        "patience_counter",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow({
                "epoch": record["epoch"],
                "training_mse": record["training_mse"],
                "validation_mse": record["validation_mse"],
                "best_validation_mse": record["best_validation_mse"],
                "is_best": record["is_best"],
                "patience_counter": record["patience_counter"],
            })
    return output_path


def generate_training_curve(
    history: Iterable[Dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Generate a clean training and validation MSE curve."""
    from matplotlib.ticker import MaxNLocator

    history = list(history)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    if not history:
        ax.text(
            0.5,
            0.5,
            "No training history available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error (MSE)")
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_file

    epochs = [int(record["epoch"]) for record in history]
    training_mse = [
        float(record["training_mse"]) for record in history
    ]
    validation_mse = [
        float(record["validation_mse"]) for record in history
    ]

    best_index = min(
        range(len(validation_mse)),
        key=lambda index: validation_mse[index],
    )
    best_epoch = epochs[best_index]
    best_validation = validation_mse[best_index]

    # Plot clean lines without markers on every epoch.
    ax.plot(
        epochs,
        training_mse,
        label="Training MSE",
        color="tab:blue",
        linewidth=2,
    )

    ax.plot(
        epochs,
        validation_mse,
        label="Validation MSE",
        color="tab:orange",
        linewidth=2,
    )

    # Highlight the best validation result.
    ax.scatter(
        [best_epoch],
        [best_validation],
        color="red",
        s=75,
        zorder=5,
        label=f"Best validation MSE: {best_validation:.4f}",
    )

    ax.axvline(
        best_epoch,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label=f"Best epoch: {best_epoch}",
    )

    ax.set_title(
        "MGraphDTA Training and Validation Curve",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)

    # Prevent overlapping epoch labels.
    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=10, integer=True)
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.margins(x=0.02)

    ax.legend(
        loc="best",
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_file


def save_final_metrics(
    path: Union[str, Path],
    metrics: Dict[str, Any],
) -> Path:
    """Serialize the final training summary to JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return output_path


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: DeviceArg = None,
) -> Dict[str, Any]:
    """Restore model and optional optimizer state, mapping tensors to device."""
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def fit(
    model: nn.Module,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any],
    *,
    epochs: int = DEFAULT_EPOCHS,
    steps_per_epoch: Optional[int] = DEFAULT_STEPS_PER_EPOCH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: Optional[int] = DEFAULT_PATIENCE,
    device: DeviceArg = None,
    seed: Optional[int] = None,
    deterministic: bool = True,
    optimizer: Optional[Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train and validate with patience-based early stopping and detailed history."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if steps_per_epoch is not None and steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive or None")
    if seed is not None:
        set_random_seed(seed, deterministic=deterministic)
    else:
        configure_deterministic_execution(deterministic)

    resolved_device = resolve_device(device)
    model.to(resolved_device)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = criterion or nn.MSELoss()
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    stopped_early = False

    for epoch in range(1, epochs + 1):
        try:
            training_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                resolved_device,
                max_steps=steps_per_epoch,
            )
            validation_loss = validate(
                model, validation_loader, loss_function, resolved_device
            )
        except RuntimeError as error:
            _raise_deterministic_operation_error(error)
            raise
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    best_validation_loss=best_validation_loss,
                    seed=seed,
                )
        else:
            patience_counter += 1
            is_best = False

        record = {
            "epoch": epoch,
            "training_mse": float(training_loss),
            "validation_mse": float(validation_loss),
            "best_validation_mse": float(best_validation_loss),
            "is_best": bool(is_best),
            "patience_counter": int(patience_counter),
        }
        history.append(record)
        if on_epoch_end is not None:
            on_epoch_end(record)

        if patience is not None and patience_counter >= patience:
            stopped_early = True
            break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "device": str(resolved_device),
        "patience": patience,
    }

def train_davis(
    *,
    processed_dir: str = "data/processed/davis",
    raw_dir: str = "data/raw/davis",
    atom_features_dir: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    steps_per_epoch: Optional[int] = DEFAULT_STEPS_PER_EPOCH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: Optional[int] = DEFAULT_PATIENCE,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: Optional[int] = DEFAULT_SEED,
    device: DeviceArg = None,
    num_workers: int = 0,
    deterministic: bool = True,
    on_epoch_end: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Train on a seeded training-fold split, then evaluate the independent test fold."""
    if seed is not None:
        set_random_seed(seed, deterministic=deterministic)
    else:
        configure_deterministic_execution(deterministic)

    paths = resolve_run_paths(run_dir=run_dir, checkpoint_path=checkpoint_path)
    if run_dir is not None:
        ensure_run_paths_are_new(paths)

    dataset_kwargs = {"raw_dir": raw_dir, "processed_dir": processed_dir}
    if atom_features_dir is not None:
        dataset_kwargs["atom_features_dir"] = atom_features_dir
    train_dataset = DavisDataset(split="train", **dataset_kwargs)

    training_subset, validation_subset = split_train_validation(
        train_dataset,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    train_generator = torch.Generator()
    if seed is None:
        train_generator.seed()
    else:
        train_generator.manual_seed(seed)
    train_loader = make_dataloader(
        training_subset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
    )

    validation_loader = make_dataloader(
        validation_subset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = MGraphDTA()

    checkpoint_path = paths["checkpoint"]
    training_history_path = paths["history"]
    training_curve_path = paths["curve"]
    final_metrics_path = paths["metrics"]

    # Reset history when a new training run starts.
    live_history = []
    write_training_history(live_history, training_history_path)

    def handle_epoch_end(record):
        # Add the completed epoch.
        live_history.append(dict(record))

        # Save CSV immediately after every epoch.
        write_training_history(
            live_history,
            training_history_path,
        )

        # Keep the existing console output.
        if on_epoch_end is not None:
            on_epoch_end(record)

    training_result = fit(
        model,
        train_loader,
        validation_loader,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        seed=seed,
        deterministic=deterministic,
        checkpoint_path=checkpoint_path,
        on_epoch_end=handle_epoch_end,
    )

    history = training_result.get("history", [])

    if history:
        # Final complete save.
        write_training_history(history, training_history_path)
        generate_training_curve(history, training_curve_path)

        final_training_mse = history[-1].get("training_mse")
        final_validation_mse = history[-1].get("validation_mse")
        final_epochs_completed = training_result.get(
            "epochs_completed",
            len(history),
        )
    else:
        write_training_history([], training_history_path)
        generate_training_curve([], training_curve_path)

        final_training_mse = None
        final_validation_mse = None
        final_epochs_completed = 0

    load_checkpoint(checkpoint_path, model, device=device)

    # Do not construct or iterate over the independent test dataset before this point.
    test_dataset = DavisDataset(split="test", **dataset_kwargs)

    test_loader = make_dataloader(
        test_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    try:
        test_mse = evaluate(model, test_loader, device=device)
        test_metrics = evaluate_test_metrics(model, test_loader, device=device)
    except RuntimeError as error:
        _raise_deterministic_operation_error(error)
        raise

    final_metrics = {
        "dataset": "davis",
        "best_epoch": training_result.get("best_epoch", 0),
        "best_validation_mse": training_result.get("best_validation_loss"),
        "final_training_mse": final_training_mse,
        "final_validation_mse": final_validation_mse,
        "independent_test_mse": float(test_mse),
        "test_mse": float(test_metrics["mse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_ci": float(test_metrics["ci"]),
        "test_r_m2": float(test_metrics["r_m2"]),
        "epochs_completed": final_epochs_completed,
        "epochs_requested": epochs,
        "stopped_early": training_result.get("stopped_early", False),
        "patience": training_result.get("patience", patience),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "num_workers": num_workers,
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "atom_features_dir": str(atom_features_dir) if atom_features_dir is not None else None,
        "device": training_result.get(
            "device",
            str(resolve_device(device)),
        ),
    }

    save_final_metrics(
        final_metrics_path,
        final_metrics,
    )

    return {
        **training_result,
        "test_mse": float(test_mse),
        "test_rmse": float(test_metrics["rmse"]),
        "test_ci": float(test_metrics["ci"]),
        "test_r_m2": float(test_metrics["r_m2"]),
        "training_history_path": str(training_history_path),
        "training_curve_path": str(training_curve_path),
        "final_metrics_path": str(final_metrics_path),
        "checkpoint_path": str(checkpoint_path),
    }


def evaluate_checkpoint(
    *,
    processed_dir: str = "data/processed/davis",
    raw_dir: str = "data/raw/davis",
    atom_features_dir: Optional[str] = None,
    checkpoint_path: Union[str, Path] = "checkpoints/mgraphdta_best.pt",
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: DeviceArg = None,
    num_workers: int = 0,
    metrics_path: Union[str, Path] = "results/davis/run_112_backup/evaluation_metrics.json",
) -> Dict[str, Any]:
    """Evaluate the restored best checkpoint without entering the training path."""
    resolved_device = resolve_device(device)
    model = MGraphDTA()
    checkpoint = load_checkpoint(checkpoint_path, model, device=resolved_device)

    dataset_kwargs = {"raw_dir": raw_dir, "processed_dir": processed_dir}
    if atom_features_dir is not None:
        dataset_kwargs["atom_features_dir"] = atom_features_dir
    test_dataset = DavisDataset(split="test", **dataset_kwargs)
    test_loader = make_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_metrics = evaluate_test_metrics(model, test_loader, device=resolved_device)

    result = {
        "dataset": "davis",
        "processed_dir": str(processed_dir),
        "atom_features_dir": str(atom_features_dir) if atom_features_dir is not None else None,
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint["epoch"],
        "best_validation_mse": checkpoint["best_validation_loss"],
        "test_mse": float(test_metrics["mse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_ci": float(test_metrics["ci"]),
        "test_r_m2": float(test_metrics["r_m2"]),
        "device": str(resolved_device),
    }
    save_final_metrics(metrics_path, result)
    return {**result, "final_metrics_path": str(metrics_path)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed/davis")
    parser.add_argument(
        "--atom-features-dir",
        default=None,
        help="Optional validated versioned atom-feature bundle; omit to use default unscaled features",
    )
    parser.add_argument("--raw-dir", default="data/raw/davis")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint output path (defaults to the legacy path, or run-dir/checkpoints/mgraphdta_best.pt)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="New, run-specific output directory; existing run artifacts are never overwritten",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic PyTorch operations; unsupported operations fail clearly",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        action="store_true",
        help="Evaluate the existing best checkpoint without training",
    )
    parser.add_argument(
        "--evaluation-output",
        default="results/davis/run_112_backup/evaluation_metrics.json",
        help="Output JSON path for checkpoint-only evaluation",
    )
    args = parser.parse_args()

    if args.evaluate_checkpoint:
        result = evaluate_checkpoint(
            processed_dir=args.processed_dir,
            raw_dir=args.raw_dir,
            atom_features_dir=args.atom_features_dir,
            checkpoint_path=args.checkpoint or DEFAULT_CHECKPOINT_PATH,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            metrics_path=args.evaluation_output,
        )
        print(f"Best checkpoint: {result['checkpoint']}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Best validation MSE: {result['best_validation_mse']:.10f}")
        print(f"Independent test MSE: {result['test_mse']:.10f}")
        print(f"Independent test RMSE: {result['test_rmse']:.10f}")
        print(f"Independent test CI: {result['test_ci']:.10f}")
        print(f"Independent test r_m^2: {result['test_r_m2']:.10f}")
        print(f"Device: {result['device']}")
        return

    def report_epoch(record: Dict[str, float]) -> None:
        print(
            "epoch={epoch} training_mse={training_mse:.6f} "
            "validation_mse={validation_mse:.6f}".format(**record)
        )

    result = train_davis(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        atom_features_dir=args.atom_features_dir,
        checkpoint_path=args.checkpoint,
        run_dir=args.run_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        patience=args.patience,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        deterministic=args.deterministic,
        on_epoch_end=report_epoch,
    )
    print(
        f"Best validation MSE {result['best_validation_loss']:.6f} "
        f"at epoch {result['best_epoch']} on {result['device']}"
    )
    print(f"Final independent test MSE {result['test_mse']:.6f}")
    print(f"Final independent test RMSE {result['test_rmse']:.6f}")
    print(f"Final independent test CI {result['test_ci']:.6f}")
    print(f"Final independent test r_m^2 {result['test_r_m2']:.6f}")


if __name__ == "__main__":
    main()
