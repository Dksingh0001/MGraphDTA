"""Analyze measured Davis targets for one drug from the existing train split."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/processed/davis/davis_train.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/davis"
DRUG_ID = 11667893
KD_CUTOFF_NM = 10000
TOP_N = 20
KEEP_COLUMNS = [
    "target_id",
    "target_sequence",
    "compound_iso_smiles",
    "affinity_kd",
    "affinity",
]


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Davis training CSV not found: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required_columns = {"drug_id", *KEEP_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"{INPUT_PATH} is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    drug_rows = data.loc[data["drug_id"].astype(str) == str(DRUG_ID), KEEP_COLUMNS]
    if drug_rows.empty:
        raise ValueError(f"No rows found for drug_id={DRUG_ID} in {INPUT_PATH}")

    # Verify grouping does not discard conflicting target-level information.
    grouped = drug_rows.groupby("target_id", sort=False, dropna=False)
    for column in KEEP_COLUMNS[1:]:
        conflicts = grouped[column].nunique(dropna=False)
        if (conflicts > 1).any():
            target_ids = conflicts[conflicts > 1].index.astype(str).tolist()
            raise ValueError(
                f"Conflicting {column} values found within target IDs: "
                f"{', '.join(target_ids[:10])}"
            )

    targets = (
        grouped[KEEP_COLUMNS[1:]]
        .first()
        .reset_index()
        .loc[:, KEEP_COLUMNS]
        .sort_values(["affinity_kd", "target_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    if targets["target_id"].duplicated().any():
        raise RuntimeError("Target grouping did not produce unique target IDs")

    below_cap = targets.loc[targets["affinity_kd"] < KD_CUTOFF_NM].copy()
    at_cap = targets.loc[targets["affinity_kd"] == KD_CUTOFF_NM].copy()
    if len(below_cap) + len(at_cap) != len(targets):
        raise ValueError(
            "Found Kd values outside the requested <10000 and ==10000 categories"
        )
    top_targets = below_cap.head(TOP_N).copy()
    if len(top_targets) < TOP_N:
        raise ValueError(
            f"Expected at least {TOP_N} targets with Kd below {KD_CUTOFF_NM} nM; "
            f"found {len(top_targets)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complete_path = OUTPUT_DIR / f"drug_{DRUG_ID}_targets.csv"
    top_path = OUTPUT_DIR / f"drug_{DRUG_ID}_top_targets.csv"
    plot_path = OUTPUT_DIR / f"drug_{DRUG_ID}_top_targets.png"

    targets.to_csv(complete_path, index=False, encoding="utf-8")
    top_targets.to_csv(top_path, index=False, encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(
        top_targets["target_id"].astype(str),
        top_targets["affinity"].astype(float),
        color="steelblue",
        edgecolor="white",
        linewidth=0.7,
    )
    axis.set_title("Drug 11667893 — Top Measured Protein Targets")
    axis.set_xlabel("Target")
    axis.set_ylabel("pKd")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    for tick in axis.get_xticklabels():
        tick.set_ha("right")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    output_paths = (complete_path, top_path, plot_path)
    missing_outputs = [path for path in output_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(
            "One or more outputs were not created successfully: "
            + ", ".join(str(path) for path in missing_outputs)
        )

    print(f"Input CSV: {INPUT_PATH}")
    print(f"Drug ID: {DRUG_ID}")
    print(f"Total unique target IDs: {len(targets)}")
    print(f"Targets with Kd < {KD_CUTOFF_NM} nM: {len(below_cap)}")
    print(f"Targets with Kd == {KD_CUTOFF_NM} nM: {len(at_cap)}")
    print(f"\nTop {TOP_N} measured targets (sorted by Kd ascending):")
    print(f"{'Target ID':<45} {'Kd (nM)':>12} {'pKd':>10}")
    for row in top_targets.itertuples(index=False):
        print(f"{str(row.target_id):<45} {float(row.affinity_kd):>12g} {float(row.affinity):>10.6f}")

    print(
        f"\nNote: Kd={KD_CUTOFF_NM} nM is the Davis dataset's capped/non-binding "
        "label; it should not be interpreted as strong binding."
    )
    print("\nOutput files:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
