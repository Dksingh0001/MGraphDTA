"""Plot the completed Davis training histories for MGraphDTA and Protein-GNN."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_HISTORY = (
    PROJECT_ROOT / "results/davis/run_112_backup/training_history.csv"
)
PROTEIN_GNN_HISTORY = (
    PROJECT_ROOT / "results/davis/protein_gnn_experiment/training_history.csv"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/davis/mgraphdta_vs_2d_protein_gnn_training_comparison.png"
)


def read_history(path: Path) -> list[dict[str, float | int]]:
    """Read and validate the epoch and MSE columns required for plotting."""
    required_columns = {"epoch", "training_mse", "validation_mse"}
    with path.open("r", newline="", encoding="utf-8-sig") as history_file:
        reader = csv.DictReader(history_file)
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required_columns))}"
            )

        history: list[dict[str, float | int]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                epoch = int(row["epoch"])
                training_mse = float(row["training_mse"])
                validation_mse = float(row["validation_mse"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid epoch/MSE value at {path}:{row_number}") from exc
            if epoch < 1 or not math.isfinite(training_mse) or not math.isfinite(validation_mse):
                raise ValueError(f"Non-finite or invalid value at {path}:{row_number}")
            history.append(
                {
                    "epoch": epoch,
                    "training_mse": training_mse,
                    "validation_mse": validation_mse,
                }
            )

    if not history:
        raise ValueError(f"No training epochs found in {path}")
    epochs = [int(row["epoch"]) for row in history]
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epoch numbers found in {path}")
    return history


def best_validation(history: list[dict[str, float | int]]) -> tuple[int, float]:
    row = min(history, key=lambda record: float(record["validation_mse"]))
    return int(row["epoch"]), float(row["validation_mse"])


def main() -> None:
    original_history = read_history(ORIGINAL_HISTORY)
    protein_gnn_history = read_history(PROTEIN_GNN_HISTORY)

    original_best_epoch, original_best_mse = best_validation(original_history)
    protein_best_epoch, protein_best_mse = best_validation(protein_gnn_history)

    expected_points = (
        ("Original MGraphDTA", original_best_epoch, original_best_mse, 91, 0.23236284946015257),
        ("2D Protein-GNN", protein_best_epoch, protein_best_mse, 29, 0.573254),
    )
    for label, epoch, mse, expected_epoch, expected_mse in expected_points:
        if epoch != expected_epoch or not math.isclose(
            mse, expected_mse, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"{label} history minimum is epoch {epoch}, MSE {mse:.12g}; "
                f"expected epoch {expected_epoch}, MSE {expected_mse:.12g}"
            )

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.plot(
        [row["epoch"] for row in original_history],
        [row["training_mse"] for row in original_history],
        color="tab:blue",
        linewidth=2,
        label="Original MGraphDTA — Training MSE",
    )
    axis.plot(
        [row["epoch"] for row in original_history],
        [row["validation_mse"] for row in original_history],
        color="tab:orange",
        linewidth=2,
        label="Original MGraphDTA — Validation MSE",
    )
    axis.plot(
        [row["epoch"] for row in protein_gnn_history],
        [row["training_mse"] for row in protein_gnn_history],
        color="tab:green",
        linewidth=2,
        label="2D Protein-GNN — Training MSE",
    )
    axis.plot(
        [row["epoch"] for row in protein_gnn_history],
        [row["validation_mse"] for row in protein_gnn_history],
        color="tab:red",
        linewidth=2,
        label="2D Protein-GNN — Validation MSE",
    )

    axis.scatter(
        [original_best_epoch], [original_best_mse], color="black", marker="*", s=180,
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    axis.scatter(
        [protein_best_epoch], [protein_best_mse], color="purple", marker="*", s=180,
        edgecolor="white", linewidth=0.8, zorder=5,
    )
    axis.annotate(
        f"Original MGraphDTA:\nBest Val MSE = {original_best_mse:.6f} @ Epoch {original_best_epoch}\n\n"
        f"2D Protein-GNN:\nBest Val MSE = {protein_best_mse:.6f} @ Epoch {protein_best_epoch}",
        xy=(0.985, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.95},
    )

    axis.set_title("Comparison of MGraphDTA and 2D Protein-GNN — Davis Training")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE")
    axis.set_xlim(left=1)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.72), framealpha=0.95)
    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError(f"Output PNG was not created successfully: {OUTPUT_PATH}")

    print(f"Original MGraphDTA input: {ORIGINAL_HISTORY}")
    print(f"Original MGraphDTA epochs found: {len(original_history)}")
    print(
        f"Original MGraphDTA best validation: epoch {original_best_epoch}, "
        f"MSE {original_best_mse:.12g}"
    )
    print(f"2D Protein-GNN input: {PROTEIN_GNN_HISTORY}")
    print(f"2D Protein-GNN epochs found: {len(protein_gnn_history)}")
    print(
        f"2D Protein-GNN best validation: epoch {protein_best_epoch}, "
        f"MSE {protein_best_mse:.12g}"
    )
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Output PNG exists: {OUTPUT_PATH.is_file()} ({OUTPUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
