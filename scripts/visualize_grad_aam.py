"""Render Grad-AAM atom importance for one Davis test drug-protein pair."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from matplotlib import colormaps
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data

from src.dataset import DavisDataset
from src.explainability.grad_aam import GradAAM
from src.models.mgraphdta import MGraphDTA
from src.train import load_checkpoint


DEFAULT_CHECKPOINT = Path("results/davis/run_112_backup/mgraphdta_best.pt")
DEFAULT_OUTPUT = Path("results/davis/grad_aam")


def visualize_sample(
    sample_index: int,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one Davis test pair and save its atom map and metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DavisDataset(split="test")
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f"sample index {sample_index} is outside the Davis test dataset "
            f"of length {len(dataset)}"
        )

    sample = dataset[sample_index]
    smiles = str(sample["drug_smiles"])
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse Davis SMILES at index {sample_index}")

    data: Data = sample["pyg_data"].to(device)
    num_atoms = molecule.GetNumAtoms()
    if data.x.shape[0] != num_atoms:
        raise ValueError(
            f"Davis graph has {data.x.shape[0]} nodes but SMILES has {num_atoms} atoms"
        )

    model = MGraphDTA().to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # Keep autograd enabled for the Grad-AAM pass and for this prediction pass.
    prediction = model(data).reshape(-1)
    if prediction.numel() != 1:
        raise ValueError("Expected one predicted affinity for the selected Davis pair")
    predicted_affinity = float(prediction.detach().cpu().item())

    attribution = GradAAM(model).attribute(data)
    atom_importance = attribution.atom_importance.detach().cpu().reshape(-1)
    if atom_importance.numel() != num_atoms:
        raise ValueError(
            f"Grad-AAM returned {atom_importance.numel()} scores for {num_atoms} atoms"
        )
    if not torch.isfinite(atom_importance).all():
        raise ValueError("Grad-AAM returned non-finite atom importance values")

    importance_scores = [float(score) for score in atom_importance.tolist()]
    atom_indices = list(range(num_atoms))
    cmap = colormaps["Oranges"]
    atom_colors = {
        atom_index: tuple(float(value) for value in cmap(importance_scores[atom_index])[:3])
        for atom_index in atom_indices
    }
    atom_radii = {atom_index: 0.2 for atom_index in atom_indices}

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"davis_sample_{sample_index}"
    png_path = output_dir / f"{output_stem}.png"
    json_path = output_dir / f"{output_stem}.json"

    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    drawer.drawOptions().useBWAtomPalette()
    prepared_molecule = rdMolDraw2D.PrepareMolForDrawing(molecule)
    drawer.DrawMolecule(
        prepared_molecule,
        highlightAtoms=atom_indices,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    png_path.write_bytes(drawer.GetDrawingText())

    true_affinity = sample.get("affinity")
    if true_affinity is None and getattr(data, "y", None) is not None:
        true_affinity = float(data.y.reshape(-1)[0].detach().cpu().item())
    elif true_affinity is not None:
        true_affinity = float(true_affinity)

    result = {
        "dataset": "davis",
        "sample_index": sample_index,
        "smiles": smiles,
        "protein_sequence": sample.get("protein_seq"),
        "protein_original_length": sample.get("protein_orig_len"),
        "true_affinity": true_affinity,
        "predicted_affinity": predicted_affinity,
        "number_of_atoms": num_atoms,
        "atom_importance": importance_scores,
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "activation_shape": list(attribution.activation_shape),
        "png": str(png_path),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the PNG and JSON outputs",
    )
    args = parser.parse_args()

    result = visualize_sample(args.sample_index, args.checkpoint, args.output)
    scores = result["atom_importance"]
    print(f"True affinity: {result['true_affinity']}")
    print(f"Predicted affinity: {result['predicted_affinity']}")
    print(f"Number of atoms: {result['number_of_atoms']}")
    print(f"Importance min/max: {min(scores):.6f} / {max(scores):.6f}")
    print(f"PNG: {result['png']}")
    print(f"JSON: {args.output / f'davis_sample_{args.sample_index}.json'}")


if __name__ == "__main__":
    main()