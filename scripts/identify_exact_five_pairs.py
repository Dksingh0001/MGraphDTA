"""Resolve Grad-AAM samples to exact Davis train interactions when possible."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "results/davis/grad_aam/examples"
MAPPING_PATH = (
    PROJECT_ROOT
    / "results/davis/structural_interactions/five_samples_davis_mapping.csv"
)
TRAIN_PATH = PROJECT_ROOT / "data/processed/davis/davis_train.csv"
OUTPUT_PATH = (
    PROJECT_ROOT / "results/davis/structural_interactions/exact_five_pairs.csv"
)
SAMPLE_INDICES = (0, 1, 2, 3, 4)
OUTPUT_COLUMNS = [
    "sample_index",
    "status",
    "matching_method",
    "drug_id",
    "pair_idx",
    "target_id",
    "target_sequence_length",
    "affinity_kd",
    "affinity",
    "predicted_affinity",
    "candidate_count",
    "candidate_target_ids",
    "resolution_note",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _same_value(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def main() -> None:
    required_files = [MAPPING_PATH, TRAIN_PATH] + [
        EXAMPLES_DIR / f"davis_sample_{sample_index}.json"
        for sample_index in SAMPLE_INDICES
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    mapping = pd.read_csv(
        MAPPING_PATH,
        dtype={"drug_id": "string", "target_id": "string", "compound_iso_smiles": "string"},
    )
    training = pd.read_csv(
        TRAIN_PATH,
        dtype={"drug_id": "string", "target_id": "string", "compound_iso_smiles": "string"},
    )
    needed_mapping = {
        "sample_index", "pair_idx", "drug_id", "target_id", "target_sequence",
        "compound_iso_smiles", "affinity_kd", "affinity",
    }
    needed_training = {
        "pair_idx", "drug_id", "target_id", "target_sequence",
        "compound_iso_smiles", "affinity_kd", "affinity",
    }
    if not needed_mapping.issubset(mapping.columns):
        raise ValueError(f"Mapping CSV lacks required columns: {sorted(needed_mapping - set(mapping.columns))}")
    if not needed_training.issubset(training.columns):
        raise ValueError(f"Training CSV lacks required columns: {sorted(needed_training - set(training.columns))}")

    resolved_rows: list[dict[str, Any]] = []
    reports: list[tuple[int, dict[str, Any], str, pd.DataFrame, str]] = []

    for sample_index in SAMPLE_INDICES:
        payload = _read_json(EXAMPLES_DIR / f"davis_sample_{sample_index}.json")
        if int(payload.get("sample_index", -1)) != sample_index:
            raise ValueError(
                f"Sample index mismatch in davis_sample_{sample_index}.json: "
                f"{payload.get('sample_index')}"
            )

        smiles = payload.get("smiles")
        sample_mapping = mapping.loc[mapping["sample_index"].eq(sample_index)]
        if smiles is not None:
            sample_mapping = sample_mapping.loc[
                sample_mapping["compound_iso_smiles"].eq(str(smiles))
            ]

        method = "none"
        candidates = pd.DataFrame()
        note = ""
        matched_drug_ids = sorted(
            sample_mapping["drug_id"].dropna().astype(str).unique()
        )

        pair_idx = payload.get("pair_idx")
        if pair_idx is not None:
            method = "pair_idx"
            candidates = sample_mapping.loc[sample_mapping["pair_idx"].eq(pair_idx)]
            if len(candidates) != 1:
                candidates = training.loc[training["pair_idx"].eq(pair_idx)]
            if candidates.empty:
                note = f"JSON pair_idx={pair_idx} was not found in the supplied Davis training data."
        else:
            sequence = payload.get("target_sequence", payload.get("protein_sequence"))
            if sequence:
                method = "protein_sequence"
                candidates = sample_mapping.loc[
                    sample_mapping["target_sequence"].eq(str(sequence))
                ]
                if candidates.empty:
                    sequence_rows = training.loc[
                        training["target_sequence"].eq(str(sequence))
                    ]
                    sequence_target_ids = sorted(
                        sequence_rows["target_id"].dropna().astype(str).unique()
                    )
                    drug_rows = sequence_rows
                    if smiles is not None:
                        drug_rows = drug_rows.loc[
                            drug_rows["compound_iso_smiles"].eq(str(smiles))
                        ]
                    note = (
                        "The exact protein sequence does not match a training interaction "
                        "for this sample's SMILES. "
                    )
                    if sequence_target_ids:
                        note += (
                            "The sequence occurs in training under target ID(s): "
                            + ", ".join(sequence_target_ids)
                            + "; however, the matching drug-target row is absent from "
                            "davis_train.csv."
                        )
                    else:
                        note += "The sequence itself is absent from davis_train.csv."
                elif "true_affinity" in payload:
                    affinity_matches = candidates.loc[
                        candidates["affinity"].map(
                            lambda value: _same_value(value, payload["true_affinity"])
                        )
                    ]
                    # The recorded affinity can narrow candidates only when it actually does so.
                    if len(affinity_matches) == 1:
                        candidates = affinity_matches
                    elif len(affinity_matches) > 1:
                        candidates = affinity_matches
            else:
                note = "JSON contains neither pair_idx nor target_sequence/protein_sequence."

        verified = pd.DataFrame()
        if len(candidates) == 1:
            candidate = candidates.iloc[0]
            verified = training.loc[training["pair_idx"].eq(candidate["pair_idx"])]
            if len(verified) == 1:
                check_fields = (
                    "drug_id", "target_id", "target_sequence", "compound_iso_smiles",
                    "affinity_kd", "affinity",
                )
                same = all(
                    str(verified.iloc[0][field]) == str(candidate[field])
                    for field in check_fields
                )
                if same:
                    note = "One Davis interaction uniquely matched and verified by pair_idx."
                else:
                    verified = pd.DataFrame()
                    note = "Mapping row disagrees with davis_train.csv for the same pair_idx."
            else:
                verified = pd.DataFrame()
                note = "Candidate pair_idx is not unique in davis_train.csv."
        elif len(candidates) > 1:
            ids = sorted(candidates["target_id"].dropna().astype(str).unique())
            note = (
                f"Exact sequence/metadata leaves {len(candidates)} Davis interactions "
                f"across target ID(s): {', '.join(ids)}. No unique target can be selected."
            )

        if len(verified) == 1:
            record = verified.iloc[0]
            resolved_rows.append(
                {
                    "sample_index": sample_index,
                    "status": "identified",
                    "matching_method": method,
                    "drug_id": record["drug_id"],
                    "pair_idx": record["pair_idx"],
                    "target_id": record["target_id"],
                    "target_sequence_length": len(str(record["target_sequence"])),
                    "affinity_kd": record["affinity_kd"],
                    "affinity": record["affinity"],
                    "predicted_affinity": payload.get("predicted_affinity"),
                    "candidate_count": 1,
                    "candidate_target_ids": str(record["target_id"]),
                    "resolution_note": note,
                }
            )
        else:
            candidate_ids = sorted(
                candidates["target_id"].dropna().astype(str).unique()
            ) if not candidates.empty else []
            resolved_rows.append(
                {
                    "sample_index": sample_index,
                    "status": "EXACT TARGET NOT IDENTIFIED",
                    "matching_method": method,
                    "drug_id": matched_drug_ids[0] if len(matched_drug_ids) == 1 else None,
                    "pair_idx": None,
                    "target_id": None,
                    "target_sequence_length": (
                        len(str(payload.get("protein_sequence", payload.get("target_sequence"))))
                        if payload.get("protein_sequence", payload.get("target_sequence"))
                        else None
                    ),
                    "affinity_kd": None,
                    "affinity": None,
                    "predicted_affinity": payload.get("predicted_affinity"),
                    "candidate_count": len(candidates),
                    "candidate_target_ids": ";".join(candidate_ids),
                    "resolution_note": note or "Required identifying metadata is missing or ambiguous.",
                }
            )
        reports.append((sample_index, payload, method, candidates, note))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resolved_rows, columns=OUTPUT_COLUMNS).to_csv(
        OUTPUT_PATH, index=False, encoding="utf-8"
    )

    print(f"Mapping CSV: {MAPPING_PATH}")
    print(f"Verification CSV: {TRAIN_PATH}")
    for sample_index, payload, method, candidates, note in reports:
        record = resolved_rows[sample_index]
        print("\n========================================")
        print(f"SAMPLE {sample_index}")
        print("========================================")
        print(f"JSON metadata fields: {', '.join(payload.keys())}")
        for field in ("target_id", "target_sequence", "pair_idx", "drug_id", "affinity"):
            print(f"JSON contains {field}: {'yes' if field in payload else 'no'}")
        print(f"matching method used: {method}")
        if record["status"] == "identified":
            print(f"drug_id: {record['drug_id']}")
            print(f"pair_idx: {record['pair_idx']}")
            print(f"target_id: {record['target_id']}")
            print(f"target_sequence_length: {record['target_sequence_length']}")
            print(f"affinity_kd: {record['affinity_kd']}")
            print(f"affinity: {record['affinity']}")
            print(f"predicted_affinity: {record['predicted_affinity']}")
        else:
            print("EXACT TARGET NOT IDENTIFIED")
            print(f"drug_id: {record['drug_id'] or 'not resolved from SMILES mapping'}")
            print(f"pair_idx: not present in JSON; no unique pair verified")
            print(f"candidate target IDs: {record['candidate_target_ids'] or 'none'}")
            print(f"Explanation: {record['resolution_note']}")
            print(f"JSON true_affinity: {payload.get('true_affinity', 'not present')}")
            print(f"JSON predicted_affinity: {payload.get('predicted_affinity', 'not present')}")

    print("\nsample | pair_idx | drug_id | target_id | Kd_nM | affinity | predicted_affinity")
    for record in resolved_rows:
        print(
            f"{record['sample_index']} | {record['pair_idx'] or ''} | "
            f"{record['drug_id'] or ''} | {record['target_id'] or 'EXACT TARGET NOT IDENTIFIED'} | "
            f"{record['affinity_kd'] if pd.notna(record['affinity_kd']) else ''} | "
            f"{record['affinity'] if pd.notna(record['affinity']) else ''} | "
            f"{record['predicted_affinity']}"
        )

    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError(f"Output mapping was not created successfully: {OUTPUT_PATH}")
    print(f"\nSaved verified mapping: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
