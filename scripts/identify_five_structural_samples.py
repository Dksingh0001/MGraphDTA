"""Map five existing Grad-AAM samples to exact Davis training interactions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "results/davis/grad_aam/examples/grad_aam_summary.csv"
TRAIN_PATH = PROJECT_ROOT / "data/processed/davis/davis_train.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results/davis/structural_interactions/five_samples_davis_mapping.csv"
)
SAMPLE_INDICES = (0, 1, 2, 3, 4)
OUTPUT_COLUMNS = [
    "sample_index",
    "pair_idx",
    "drug_id",
    "target_id",
    "target_sequence",
    "sequence_length",
    "compound_iso_smiles",
    "affinity_kd",
    "affinity",
]


def main() -> None:
    for input_path in (SUMMARY_PATH, TRAIN_PATH):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {input_path}")

    summary = pd.read_csv(SUMMARY_PATH, dtype={"smiles": "string"})
    training = pd.read_csv(
        TRAIN_PATH,
        dtype={"drug_id": "string", "compound_iso_smiles": "string", "target_id": "string"},
    )
    if not {"sample_index", "smiles"}.issubset(summary.columns):
        raise ValueError(f"{SUMMARY_PATH} must contain sample_index and smiles columns")
    required_training_columns = {
        "pair_idx",
        "drug_id",
        "target_id",
        "target_sequence",
        "compound_iso_smiles",
        "affinity_kd",
        "affinity",
    }
    missing = required_training_columns.difference(training.columns)
    if missing:
        raise ValueError(
            f"{TRAIN_PATH} is missing columns: {', '.join(sorted(missing))}"
        )

    mapping_frames: list[pd.DataFrame] = []
    per_sample: dict[int, pd.DataFrame] = {}
    for sample_index in SAMPLE_INDICES:
        sample_rows = summary.loc[summary["sample_index"].eq(sample_index)]
        if len(sample_rows) != 1:
            raise ValueError(
                f"Expected one summary row for sample_index={sample_index}; "
                f"found {len(sample_rows)}"
            )
        smiles = sample_rows.iloc[0]["smiles"]
        if pd.isna(smiles) or not str(smiles):
            raise ValueError(f"Missing SMILES for sample_index={sample_index}")

        matches = training.loc[
            training["compound_iso_smiles"].eq(str(smiles)),
            [
                "pair_idx",
                "drug_id",
                "target_id",
                "target_sequence",
                "compound_iso_smiles",
                "affinity_kd",
                "affinity",
            ],
        ].copy()
        matches.insert(0, "sample_index", sample_index)
        matches["sequence_length"] = matches["target_sequence"].astype(str).str.len()
        matches = matches.loc[:, OUTPUT_COLUMNS].sort_values(
            ["affinity_kd", "target_id", "pair_idx"], kind="stable"
        )
        per_sample[sample_index] = matches
        if not matches.empty:
            mapping_frames.append(matches)

    if mapping_frames:
        mapping = pd.concat(mapping_frames, ignore_index=True).loc[:, OUTPUT_COLUMNS]
    else:
        mapping = pd.DataFrame(columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Summary file: {SUMMARY_PATH}")
    print(f"Davis training CSV: {TRAIN_PATH}")
    for sample_index in SAMPLE_INDICES:
        matches = per_sample[sample_index]
        sample_smiles = str(
            summary.loc[summary["sample_index"].eq(sample_index), "smiles"].iloc[0]
        )
        drug_ids = sorted(matches["drug_id"].dropna().astype(str).unique())
        target_ids = sorted(matches["target_id"].dropna().astype(str).unique())

        print("\n========================================")
        print(f"SAMPLE {sample_index}")
        print("========================================")
        print(f"SMILES: {sample_smiles}")
        print(f"Drug ID: {', '.join(drug_ids) if drug_ids else 'NO MATCH'}")
        print(f"Number of matching Davis interactions: {len(matches)}")
        print("pair_idx | target_id | sequence_length | Kd_nM | affinity")
        for row in matches.itertuples(index=False):
            print(
                f"{row.pair_idx} | {row.target_id} | {row.sequence_length} | "
                f"{float(row.affinity_kd):g} | {float(row.affinity):.12g}"
            )

        if matches.empty:
            print(f"WARNING: sample {sample_index} has no exact SMILES match.")
        if len(drug_ids) > 1:
            print(
                f"WARNING: sample {sample_index} maps to multiple drug IDs: "
                f"{', '.join(drug_ids)}"
            )
        if len(target_ids) > 1:
            print(
                f"WARNING: sample {sample_index} maps to multiple target IDs "
                f"({len(target_ids)} targets); all are listed above."
            )

    print("\n========================================")
    print("COMPACT MAPPING TABLE")
    print("========================================")
    print("sample | drug_id | target_id | Kd_nM | affinity")
    for sample_index in SAMPLE_INDICES:
        matches = per_sample[sample_index]
        for row in matches.itertuples(index=False):
            print(
                f"{sample_index} | {row.drug_id} | {row.target_id} | "
                f"{float(row.affinity_kd):g} | {float(row.affinity):.12g}"
            )

    print(f"\nSaved mapping CSV: {OUTPUT_PATH}")
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError(f"Mapping CSV was not created successfully: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
