"""Generate Grad-AAM images and metadata for a Davis test-sample range."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem

from scripts.visualize_grad_aam import DEFAULT_CHECKPOINT, visualize_sample
from src.dataset import DavisDataset


DEFAULT_OUTPUT = Path("results/davis/grad_aam/examples")
SUMMARY_COLUMNS = [
    "sample_index",
    "true_affinity",
    "predicted_affinity",
    "absolute_prediction_error",
    "atom_count",
    "smiles",
]


def generate_examples(
    num_samples: int,
    start_index: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Generate and verify visualizations for a contiguous Davis test range."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if start_index < 0:
        raise ValueError("start_index cannot be negative")

    dataset = DavisDataset(split="test")
    end_index = start_index + num_samples
    if end_index > len(dataset):
        raise IndexError(
            f"requested samples [{start_index}, {end_index}) exceed Davis test "
            f"dataset length {len(dataset)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for sample_index in range(start_index, end_index):
        # Reuse the single-sample path for checkpoint loading, Grad-AAM, and RDKit rendering.
        result = visualize_sample(sample_index, DEFAULT_CHECKPOINT, output_dir)
        smiles = str(result["smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse sample {sample_index} SMILES")

        atom_count = molecule.GetNumAtoms()
        importance = [float(score) for score in result["atom_importance"]]
        if atom_count != result["number_of_atoms"]:
            raise ValueError(
                f"Sample {sample_index}: RDKit atom count differs from visualizer"
            )
        if len(importance) != atom_count:
            raise ValueError(
                f"Sample {sample_index}: {len(importance)} atom scores for "
                f"{atom_count} RDKit atoms"
            )
        if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in importance):
            raise ValueError(
                f"Sample {sample_index}: atom importance must be finite and in [0, 1]"
            )

        true_affinity = result["true_affinity"]
        predicted_affinity = float(result["predicted_affinity"])
        absolute_error = (
            abs(float(true_affinity) - predicted_affinity)
            if true_affinity is not None
            else None
        )
        result["atom_importance"] = importance
        result["atom_count"] = atom_count
        result["absolute_prediction_error"] = absolute_error

        json_path = output_dir / f"davis_sample_{sample_index}.json"
        png_path = output_dir / f"davis_sample_{sample_index}.png"
        if not png_path.is_file():
            raise FileNotFoundError(f"Visualization was not created: {png_path}")
        json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if not json_path.is_file():
            raise FileNotFoundError(f"Metadata was not created: {json_path}")

        saved_result = json.loads(json_path.read_text(encoding="utf-8"))
        saved_scores = saved_result["atom_importance"]
        if len(saved_scores) != molecule.GetNumAtoms():
            raise ValueError(f"Sample {sample_index}: saved score count does not match atoms")
        if not all(
            math.isfinite(float(score)) and 0.0 <= float(score) <= 1.0
            for score in saved_scores
        ):
            raise ValueError(f"Sample {sample_index}: saved scores are invalid")

        results.append(result)

    summary_path = output_dir / "grad_aam_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "sample_index": result["sample_index"],
                "true_affinity": result["true_affinity"],
                "predicted_affinity": result["predicted_affinity"],
                "absolute_prediction_error": result["absolute_prediction_error"],
                "atom_count": result["atom_count"],
                "smiles": result["smiles"],
            })
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary CSV was not created: {summary_path}")

    print(
        f"{'Index':>5} {'True affinity':>15} {'Predicted':>15} "
        f"{'Abs. error':>13} {'Atoms':>7}"
    )
    for result in results:
        print(
            f"{result['sample_index']:>5} "
            f"{result['true_affinity']:>15.6f} "
            f"{result['predicted_affinity']:>15.6f} "
            f"{result['absolute_prediction_error']:>13.6f} "
            f"{result['atom_count']:>7}"
        )
    print(f"Summary CSV: {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_examples(args.num_samples, args.start_index, args.output)


if __name__ == "__main__":
    main()