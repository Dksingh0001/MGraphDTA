"""Tests for the opt-in, versioned scaled Davis input bundle (no training)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts.generate_davis_scaling_comparison import FEATURE_NAMES
from src.dataset import DavisDataset
from src.scaled_features import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FEATURE_COMPARISON_DIR,
    DEFAULT_LIGANDS_FILE,
    DEFAULT_PROCESSED_DIR,
    build_scaled_dataset,
    load_scaled_feature_bundle,
)


@pytest.fixture(scope="module")
def scaled_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("scaled_bundle") / "dataset"
    build_scaled_dataset(destination)
    return destination


def _build(tmp_path: Path, **kwargs) -> Path:
    destination = tmp_path / "dataset"
    build_scaled_dataset(destination, **kwargs)
    return destination


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _corrupt_source_copy(tmp_path: Path, filename: str, transform) -> tuple[Path, Path]:
    comparison = tmp_path / "comparison"
    shutil.copytree(DEFAULT_FEATURE_COMPARISON_DIR, comparison)
    target_names = [filename]
    if filename in ("author_scaled_atom_features.csv", "unscaled_atom_features.csv"):
        target_names = ["unscaled_atom_features.csv", "author_scaled_atom_features.csv"]
    for target_name in target_names:
        table = comparison / target_name
        rows = list(csv.DictReader(table.open(newline="", encoding="utf-8")))
        transform(rows)
        _write_csv_rows(table, rows)
    manifest_path = comparison / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for target_name in target_names:
        manifest["outputs"][target_name]["sha256"] = _file_hash(comparison / target_name)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison, manifest_path


def test_bundle_manifest_hashes_and_saved_scaled_table_parity(scaled_dir: Path) -> None:
    manifest_path = scaled_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors, loaded_manifest = load_scaled_feature_bundle(scaled_dir)
    assert manifest == loaded_manifest
    assert manifest["feature_order"] == FEATURE_NAMES
    assert manifest["feature_dimension"] == 22
    assert manifest["ligand_count"] == 68
    assert manifest["atom_count"] == 2180
    assert set(tensors) == {item["drug_id"] for item in manifest["ligands"]}

    table_path = DEFAULT_FEATURE_COMPARISON_DIR / "author_scaled_atom_features.csv"
    table = list(csv.DictReader(table_path.open(newline="", encoding="utf-8")))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in table:
        grouped.setdefault(row["drug_id"], []).append(row)
    assert set(grouped) == set(tensors)
    for drug_id, records in grouped.items():
        records.sort(key=lambda row: int(row["atom_index"]))
        expected = torch.tensor(
            [[float(row[name]) for name in FEATURE_NAMES] for row in records], dtype=torch.float32
        )
        assert torch.equal(tensors[drug_id], expected)
        assert torch.isfinite(tensors[drug_id]).all()
        assert tensors[drug_id].shape == (len(records), 22)

    for filename, metadata in manifest["files"].items():
        assert _file_hash(scaled_dir / filename) == metadata["sha256"]


def test_dataset_csvs_are_byte_identical_to_active_splits_and_membership(scaled_dir: Path) -> None:
    for filename in ("davis_all.csv", "davis_train.csv", "davis_test.csv"):
        assert (scaled_dir / filename).read_bytes() == (DEFAULT_PROCESSED_DIR / filename).read_bytes()

    for split in ("all", "train", "test"):
        active = DavisDataset(split=split, processed_dir=str(DEFAULT_PROCESSED_DIR))
        scaled = DavisDataset(split=split, processed_dir=str(scaled_dir), atom_features_dir=str(scaled_dir))
        pd.testing.assert_frame_equal(active.df, scaled.df, check_dtype=True)
    train = pd.read_csv(scaled_dir / "davis_train.csv")["pair_idx"]
    test = pd.read_csv(scaled_dir / "davis_test.csv")["pair_idx"]
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(pd.read_csv(scaled_dir / "davis_all.csv"))


@pytest.mark.parametrize("split,indexes", [("all", [0, 441, 442]), ("train", [0, 442]), ("test", [0, 442])])
def test_scaled_adapter_changes_only_x_and_preserves_graph_and_labels(
    scaled_dir: Path, split: str, indexes: list[int]
) -> None:
    active = DavisDataset(split=split, processed_dir=str(DEFAULT_PROCESSED_DIR))
    scaled = DavisDataset(split=split, processed_dir=str(scaled_dir), atom_features_dir=str(scaled_dir))
    for index in indexes:
        unscaled_item = active[index]
        scaled_item = scaled[index]
        assert unscaled_item["drug_id"] == scaled_item["drug_id"]
        assert unscaled_item["sample_index"] == scaled_item["sample_index"] == index
        assert unscaled_item["drug_smiles"] == scaled_item["drug_smiles"]
        assert unscaled_item["affinity"] == scaled_item["affinity"]
        assert unscaled_item["protein_seq"] == scaled_item["protein_seq"]
        assert unscaled_item["protein_orig_len"] == scaled_item["protein_orig_len"]
        assert torch.equal(unscaled_item["protein_tensor"], scaled_item["protein_tensor"])
        assert torch.equal(unscaled_item["edge_index"], scaled_item["edge_index"])
        assert unscaled_item["num_atoms"] == scaled_item["num_atoms"]
        assert unscaled_item["num_bonds"] == scaled_item["num_bonds"]
        assert unscaled_item["pyg_data"].x.shape == scaled_item["pyg_data"].x.shape == (
            scaled_item["num_atoms"], 22
        )
        assert scaled_item["pyg_data"].x.dtype == torch.float32
        assert scaled_item["pyg_data"].edge_index.dtype == torch.long
        assert torch.equal(unscaled_item["pyg_data"].edge_index, scaled_item["pyg_data"].edge_index)
        assert torch.equal(unscaled_item["pyg_data"].target, scaled_item["pyg_data"].target)
        assert torch.equal(unscaled_item["pyg_data"].y, scaled_item["pyg_data"].y)
        assert not torch.equal(unscaled_item["pyg_data"].x, scaled_item["pyg_data"].x)
        assert torch.isfinite(scaled_item["pyg_data"].x).all()


def test_default_dataset_remains_unscaled_and_scaled_is_explicit_opt_in(scaled_dir: Path) -> None:
    default = DavisDataset(split="all")
    explicit_unscaled = DavisDataset(split="all", processed_dir=str(DEFAULT_PROCESSED_DIR))
    scaled = DavisDataset(split="all", processed_dir=str(scaled_dir), atom_features_dir=str(scaled_dir))
    assert default.atom_features_dir is None
    assert torch.equal(default[0]["pyg_data"].x, explicit_unscaled[0]["pyg_data"].x)
    assert scaled.atom_features_dir == str(scaled_dir)
    assert not torch.equal(default[0]["pyg_data"].x, scaled[0]["pyg_data"].x)


def test_hash_mismatch_and_missing_bundle_files_fail_clearly(scaled_dir: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tampered"
    shutil.copytree(scaled_dir, tampered)
    with (tampered / "ligand_atom_features.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash validation failed"):
        load_scaled_feature_bundle(tampered)
    with pytest.raises(ValueError, match="hash validation failed"):
        DavisDataset(split="all", processed_dir=str(tampered), atom_features_dir=str(tampered))


@pytest.mark.parametrize("mutation,expected", [
    ("duplicate", "duplicated atom index"),
    ("missing", "missing or duplicated"),
    ("nan", "Non-finite"),
])
def test_builder_rejects_duplicate_missing_or_invalid_feature_records(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    filename = "author_scaled_atom_features.csv" if mutation != "nan" else "unscaled_atom_features.csv"

    def change(rows: list[dict[str, str]]) -> None:
        if mutation == "duplicate":
            rows[1]["atom_index"] = rows[0]["atom_index"]
        elif mutation == "missing":
            rows.pop(0)
        else:
            rows[0][FEATURE_NAMES[0]] = "nan"

    comparison, _ = _corrupt_source_copy(tmp_path, filename, change)
    with pytest.raises(ValueError, match=expected):
        build_scaled_dataset(
            tmp_path / "bad_dataset",
            comparison_dir=comparison,
        )


def test_builder_refuses_existing_output_and_repeats_hashes(tmp_path: Path) -> None:
    first = _build(tmp_path / "first")
    second = _build(tmp_path / "second")
    for filename in ("davis_all.csv", "davis_train.csv", "davis_test.csv", "ligand_atom_features.pt", "manifest.json"):
        assert _file_hash(first / filename) == _file_hash(second / filename)
    with pytest.raises(FileExistsError, match="Refusing"):
        build_scaled_dataset(first)
    assert DEFAULT_DATASET_DIR.name == "davis_scaled_v1"


def test_scaled_bundle_ids_smiles_and_atom_counts_match_current_ligands(scaled_dir: Path) -> None:
    ligands = json.loads(DEFAULT_LIGANDS_FILE.read_text(encoding="utf-8"))
    tensors, manifest = load_scaled_feature_bundle(scaled_dir)
    assert list(ligands) == [record["drug_id"] for record in manifest["ligands"]]
    for drug_id, smiles in ligands.items():
        record = next(item for item in manifest["ligands"] if item["drug_id"] == drug_id)
        assert record["smiles"] == smiles
        assert tensors[drug_id].shape[0] == record["atom_count"]
