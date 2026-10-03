"""Resolve five Grad-AAM drug/protein pairs across Davis train and test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "results/davis/grad_aam/examples"
TRAIN_PATH = PROJECT_ROOT / "data/processed/davis/davis_train.csv"
TEST_PATH = PROJECT_ROOT / "data/processed/davis/davis_test.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "results/davis/structural_interactions/five_pairs_all_splits.csv"
)
SAMPLE_DRUG_IDS = {
    0: "176870",
    1: "11717001",
    2: "208908",
    3: "42642645",
    4: "16722836",
}
SPLIT_COLUMNS = [
    "pair_idx",
    "drug_id",
    "target_id",
    "target_sequence",
    "affinity_kd",
    "affinity",
]
OUTPUT_COLUMNS = [
    "sample_index",
    "split",
    "pair_idx",
    "drug_id",
    "target_id",
    "target_sequence",
    "target_sequence_length",
    "affinity_kd",
    "affinity",
    "true_affinity",
    "predicted_affinity",
    "status",
    "candidate_count",
    "candidate_target_ids",
    "identical_sequence_variants",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"drug_id": "string", "target_id": "string", "target_sequence": "string"},
    )
    missing = set(SPLIT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return frame


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required Davis split not found: {path}")
    train = load_split(TRAIN_PATH)
    test = load_split(TEST_PATH)
    split_frames = (("train", train), ("test", test))

    all_records: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []

    for sample_index, drug_id in SAMPLE_DRUG_IDS.items():
        json_path = EXAMPLES_DIR / f"davis_sample_{sample_index}.json"
        payload = load_json(json_path)
        if int(payload.get("sample_index", -1)) != sample_index:
            raise ValueError(f"Unexpected sample_index in {json_path}")
        sequence = payload.get("protein_sequence")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"Missing protein_sequence in {json_path}")

        matches_by_split = []
        for split_name, frame in split_frames:
            found = frame.loc[
                frame["drug_id"].eq(drug_id)
                & frame["target_sequence"].eq(sequence),
                SPLIT_COLUMNS,
            ].copy()
            found.insert(0, "split", split_name)
            matches_by_split.append(found)
        matches = pd.concat(matches_by_split, ignore_index=True)
        matches = matches.sort_values(["split", "pair_idx", "target_id"], kind="stable")

        target_ids = sorted(matches["target_id"].dropna().astype(str).unique())
        sequence_count = int(matches["target_sequence"].nunique(dropna=False))
        same_sequence_variants = len(matches) > 1 and sequence_count == 1 and len(target_ids) > 1
        if len(matches) == 1:
            status = "EXACT_PAIR_VERIFIED"
        elif len(matches) > 1 and same_sequence_variants:
            status = "MULTIPLE_IDENTICAL_SEQUENCE_VARIANTS"
        elif len(matches) > 1:
            status = "MULTIPLE_ROWS_MATCHED"
        else:
            status = "PAIR_NOT_FOUND_IN_PROCESSED_DAVIS"

        if matches.empty:
            all_records.append(
                {
                    "sample_index": sample_index,
                    "split": "",
                    "pair_idx": None,
                    "drug_id": drug_id,
                    "target_id": "",
                    "target_sequence": sequence,
                    "target_sequence_length": len(sequence),
                    "affinity_kd": None,
                    "affinity": None,
                    "true_affinity": payload.get("true_affinity"),
                    "predicted_affinity": payload.get("predicted_affinity"),
                    "status": status,
                    "candidate_count": 0,
                    "candidate_target_ids": "",
                    "identical_sequence_variants": False,
                }
            )
        else:
            for row in matches.itertuples(index=False):
                all_records.append(
                    {
                        "sample_index": sample_index,
                        "split": row.split,
                        "pair_idx": row.pair_idx,
                        "drug_id": row.drug_id,
                        "target_id": row.target_id,
                        "target_sequence": row.target_sequence,
                        "target_sequence_length": len(str(row.target_sequence)),
                        "affinity_kd": row.affinity_kd,
                        "affinity": row.affinity,
                        "true_affinity": payload.get("true_affinity"),
                        "predicted_affinity": payload.get("predicted_affinity"),
                        "status": status,
                        "candidate_count": len(matches),
                        "candidate_target_ids": ";".join(target_ids),
                        "identical_sequence_variants": same_sequence_variants,
                    }
                )

        sample_summaries.append(
            {
                "sample_index": sample_index,
                "drug_id": drug_id,
                "payload": payload,
                "matches": matches,
                "status": status,
                "target_ids": target_ids,
                "identical_sequence_variants": same_sequence_variants,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records, columns=OUTPUT_COLUMNS).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8"
    )

    print(f"Train CSV: {TRAIN_PATH}")
    print(f"Test CSV: {TEST_PATH}")
    for summary in sample_summaries:
        sample_index = summary["sample_index"]
        payload = summary["payload"]
        matches = summary["matches"]
        print(f"\nSAMPLE {sample_index}: {summary['status']}")
        if matches.empty:
            print(
                f"No row matched drug_id={summary['drug_id']} and the exact "
                "protein_sequence in either processed split."
            )
        else:
            for row in matches.itertuples(index=False):
                print(
                    f"{row.split} | pair_idx={row.pair_idx} | drug_id={row.drug_id} | "
                    f"target_id={row.target_id} | sequence_length={len(str(row.target_sequence))} | "
                    f"Kd_nM={float(row.affinity_kd):g} | affinity={float(row.affinity):.12g}"
                )
            if len(matches) > 1:
                print(f"Matched rows: {len(matches)}")
                print(f"All target IDs: {', '.join(summary['target_ids'])}")
                if summary["identical_sequence_variants"]:
                    print(
                        "These target IDs are identical-sequence variants: every matched "
                        "Davis target_sequence exactly equals the JSON protein_sequence."
                    )
                print(
                    "The JSON does not contain a pair_idx or target_id that distinguishes "
                    "these rows; no target was selected."
                )
        if summary["status"] != "EXACT_PAIR_VERIFIED":
            print(
                f"Explanation: status={summary['status']}; JSON true_affinity="
                f"{payload.get('true_affinity')}, predicted_affinity="
                f"{payload.get('predicted_affinity')}."
            )

    print("\nsample | split | pair_idx | drug_id | target_id | Kd_nM | affinity | status")
    for record in all_records:
        print(
            f"{record['sample_index']} | {record['split']} | {record['pair_idx'] or ''} | "
            f"{record['drug_id']} | {record['target_id'] or ''} | "
            f"{record['affinity_kd'] if pd.notna(record['affinity_kd']) else ''} | "
            f"{record['affinity'] if pd.notna(record['affinity']) else ''} | {record['status']}"
        )

    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError(f"Mapping CSV was not created successfully: {OUTPUT_PATH}")
    print(f"\nSaved all-splits mapping: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
