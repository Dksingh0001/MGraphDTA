"""Versioned Davis author-scaled atom feature bundle validation and building."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from src.preprocessing import smiles_to_mol


FEATURE_NAMES = [
    "atom_H", "atom_C", "atom_N", "atom_O", "atom_F", "atom_Cl", "atom_S", "atom_Br", "atom_I",
    "atomic_number", "acceptor", "donor", "aromatic", "hybridization_SP", "hybridization_SP2",
    "hybridization_SP3", "total_hydrogens", "explicit_valence", "formal_charge", "implicit_valence",
    "explicit_hydrogens", "radical_electrons",
]
SCHEMA_VERSION = 1
AUTHOR_COMMIT = "54386680e8c9562f708a0fe5d381038c62111df6"
DEFAULT_DATASET_DIR = Path("data/processed/davis_scaled_v1")
DEFAULT_FEATURE_COMPARISON_DIR = Path("data/processed/davis_scaling_experiment")
DEFAULT_PROCESSED_DIR = Path("data/processed/davis")
DEFAULT_LIGANDS_FILE = Path("data/raw/davis/ligands_can.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_feature_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = ["drug_id", "smiles", "atom_index", *FEATURE_NAMES]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected feature columns in {path}: {reader.fieldnames}")
        return list(reader)


def build_scaled_dataset(
    output_dir: Path = DEFAULT_DATASET_DIR,
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    comparison_dir: Path = DEFAULT_FEATURE_COMPARISON_DIR,
    ligands_file: Path = DEFAULT_LIGANDS_FILE,
) -> dict[str, Any]:
    """Build a new dataset bundle. Existing destination paths are never reused."""
    output_dir = Path(output_dir)
    processed_dir = Path(processed_dir)
    comparison_dir = Path(comparison_dir)
    ligands_file = Path(ligands_file)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite dataset directory: {output_dir}")

    source_manifest_path = comparison_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("author_repository_commit") != AUTHOR_COMMIT:
        raise ValueError("Scaling source manifest does not name the pinned author commit")
    for filename in ("unscaled_atom_features.csv", "author_scaled_atom_features.csv"):
        expected_hash = source_manifest.get("outputs", {}).get(filename, {}).get("sha256")
        source_path = comparison_dir / filename
        if not expected_hash or not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise ValueError(f"Scaling source hash validation failed for {source_path}")

    ligands = json.loads(ligands_file.read_text(encoding="utf-8"))
    expected_ligands = {str(key): value for key, value in ligands.items()}
    rows = _read_feature_rows(comparison_dir / "author_scaled_atom_features.csv")
    unscaled_rows = _read_feature_rows(comparison_dir / "unscaled_atom_features.csv")
    if len(rows) != len(unscaled_rows):
        raise ValueError("Scaled and unscaled feature row counts differ")

    for row_number, (scaled_row, raw_row) in enumerate(zip(rows, unscaled_rows)):
        for field in ("drug_id", "smiles", "atom_index"):
            if scaled_row[field] != raw_row[field]:
                raise ValueError(f"Scaled/unscaled row alignment mismatch at row {row_number}, field {field}")

    per_ligand: dict[str, list[tuple[int, list[float]]]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    for row in rows:
        drug_id = str(row["drug_id"])
        if drug_id not in expected_ligands:
            raise ValueError(f"Unknown drug ID in scaled table: {drug_id}")
        if row["smiles"] != expected_ligands[drug_id]:
            raise ValueError(f"SMILES mismatch for drug ID {drug_id}")
        try:
            atom_index = int(row["atom_index"])
            values = [float(row[name]) for name in FEATURE_NAMES]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid atom feature record for drug ID {drug_id}") from error
        pair = (drug_id, atom_index)
        if atom_index < 0 or pair in seen_pairs:
            raise ValueError(f"Negative or duplicated atom index for drug ID {drug_id}: {atom_index}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite atom feature for drug ID {drug_id}, atom {atom_index}")
        seen_pairs.add(pair)
        per_ligand.setdefault(drug_id, []).append((atom_index, values))

    if set(per_ligand) != set(expected_ligands):
        missing = sorted(set(expected_ligands) - set(per_ligand))
        extra = sorted(set(per_ligand) - set(expected_ligands))
        raise ValueError(f"Ligand feature coverage mismatch; missing={missing}, extra={extra}")

    raw_per_ligand: dict[str, list[tuple[int, list[float]]]] = {}
    for row in unscaled_rows:
        drug_id = str(row["drug_id"])
        if drug_id not in expected_ligands or row["smiles"] != expected_ligands[drug_id]:
            raise ValueError(f"Invalid or unknown unscaled feature record for drug ID {drug_id}")
        try:
            atom_index = int(row["atom_index"])
            values = [float(row[name]) for name in FEATURE_NAMES]
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid unscaled atom feature record for drug ID {drug_id}") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite unscaled atom feature for drug ID {drug_id}")
        raw_per_ligand.setdefault(drug_id, []).append((atom_index, values))

    for drug_id, raw_matrix in raw_per_ligand.items():
        raw_matrix.sort(key=lambda item: item[0])
        if [index for index, _ in raw_matrix] != list(range(len(raw_matrix))):
            raise ValueError(f"Unscaled atom indices for drug ID {drug_id} are missing or duplicated")
        scaled_matrix = per_ligand[drug_id]
        scaled_matrix.sort(key=lambda item: item[0])
        if [index for index, _ in scaled_matrix] != list(range(len(scaled_matrix))):
            raise ValueError(f"Scaled atom indices for drug ID {drug_id} are missing or duplicated")
        if len(raw_matrix) != len(scaled_matrix):
            raise ValueError(f"Scaled/unscaled atom count mismatch for drug ID {drug_id}")
        raw_min = min(min(atom) for _, atom in raw_matrix)
        raw_max = max(max(atom) for _, atom in raw_matrix)
        span = raw_max - raw_min
        if span <= 0:
            raise ValueError(f"Author formula is undefined for zero-range drug ID {drug_id}")
        for atom_index, (_, raw_values) in enumerate(raw_matrix):
            scaled_values = per_ligand[drug_id][atom_index][1]
            for raw_value, scaled_value in zip(raw_values, scaled_values):
                expected_value = (raw_value - raw_min) / span
                if abs(scaled_value - expected_value) > 1e-6:
                    raise ValueError(f"Scaled/unscaled formula mismatch for drug ID {drug_id}")

    feature_tensors: dict[str, torch.Tensor] = {}
    ligand_records = []
    for drug_id, smiles in expected_ligands.items():
        ligand_rows = per_ligand[drug_id]
        ligand_rows.sort(key=lambda item: item[0])
        indices = [index for index, _ in ligand_rows]
        if indices != list(range(len(indices))):
            raise ValueError(f"Atom indices for drug ID {drug_id} are missing or non-contiguous")
        matrix = torch.tensor([values for _, values in ligand_rows], dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES) or not torch.isfinite(matrix).all():
            raise ValueError(f"Invalid feature matrix for drug ID {drug_id}: {tuple(matrix.shape)}")
        molecule = smiles_to_mol(smiles)
        if molecule is None or molecule.GetNumAtoms() != matrix.shape[0]:
            raise ValueError(f"Atom count does not match ligand SMILES for drug ID {drug_id}")
        feature_tensors[drug_id] = matrix.contiguous()
        ligand_records.append({"drug_id": drug_id, "smiles": smiles, "atom_count": int(matrix.shape[0])})

    table_names = ("davis_all.csv", "davis_train.csv", "davis_test.csv")
    table_sources = {name: processed_dir / name for name in table_names}
    missing_tables = [str(path) for path in table_sources.values() if not path.is_file()]
    if missing_tables:
        raise FileNotFoundError(f"Missing processed Davis tables: {missing_tables}")
    import pandas as pd

    all_rows = pd.read_csv(table_sources["davis_all.csv"])
    train_rows = pd.read_csv(table_sources["davis_train.csv"])
    test_rows = pd.read_csv(table_sources["davis_test.csv"])
    required_columns = {"pair_idx", "drug_id", "compound_iso_smiles", "target_id", "target_sequence", "affinity_kd", "affinity"}
    if not required_columns.issubset(all_rows.columns):
        raise ValueError("Processed Davis all-table is missing required interaction columns")
    known_pair_ids = set(all_rows["pair_idx"].tolist())
    train_pair_ids = train_rows["pair_idx"].tolist()
    test_pair_ids = test_rows["pair_idx"].tolist()
    if len(train_pair_ids) != len(set(train_pair_ids)) or len(test_pair_ids) != len(set(test_pair_ids)):
        raise ValueError("Processed Davis train/test split contains duplicate pair indices")
    if set(train_pair_ids) & set(test_pair_ids) or set(train_pair_ids) | set(test_pair_ids) != known_pair_ids:
        raise ValueError("Processed Davis train/test split membership does not partition the all-table")
    for frame in (all_rows, train_rows, test_rows):
        for row in frame[["drug_id", "compound_iso_smiles"]].drop_duplicates().itertuples(index=False):
            drug_id = str(row.drug_id)
            if drug_id not in expected_ligands or row.compound_iso_smiles != expected_ligands[drug_id]:
                raise ValueError(f"Processed interaction table has invalid ligand ID/SMILES: {drug_id}")

    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, source in table_sources.items():
        shutil.copyfile(source, output_dir / filename)
    tensor_path = output_dir / "ligand_atom_features.pt"
    torch.save(feature_tensors, tensor_path)

    output_files = {name: sha256_file(output_dir / name) for name in (*table_names, tensor_path.name)}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "Davis author-style scaled atom features",
        "feature_variant": "author_global_minmax_per_molecule",
        "author_repository_commit": AUTHOR_COMMIT,
        "author_preprocessing_sha256": source_manifest["author_preprocessing_sha256"],
        "feature_order": FEATURE_NAMES,
        "feature_dimension": len(FEATURE_NAMES),
        "scaling_formula": "(x - x.min()) / (x.max() - x.min())",
        "scaling_reduction": "scalar min/max over each complete [atoms, 22] molecule matrix",
        "zero_range_policy": "reject zero-range or non-finite matrices; pinned author code itself has no guard",
        "source_artifacts": {
            "comparison_manifest_sha256": sha256_file(source_manifest_path),
            "scaled_features_sha256": sha256_file(comparison_dir / "author_scaled_atom_features.csv"),
            "ligand_dictionary_sha256": sha256_file(ligands_file),
            "source_processed_tables": {
                name: sha256_file(source) for name, source in table_sources.items()
            },
        },
        "ligand_count": len(ligand_records),
        "atom_count": sum(record["atom_count"] for record in ligand_records),
        "ligands": ligand_records,
        "files": {name: {"sha256": digest} for name, digest in output_files.items()},
        "rdkit_version": source_manifest.get("rdkit_version"),
        "torch_version": torch.__version__,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def load_scaled_feature_bundle(feature_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Validate bundle hashes/schema and load CPU float32 per-ligand matrices."""
    feature_dir = Path(feature_dir)
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scaled feature manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported scaled feature schema: {manifest.get('schema_version')}")
    if manifest.get("author_repository_commit") != AUTHOR_COMMIT:
        raise ValueError("Scaled feature bundle does not identify the pinned author commit")
    if manifest.get("feature_order") != FEATURE_NAMES or manifest.get("feature_dimension") != len(FEATURE_NAMES):
        raise ValueError("Scaled feature bundle feature ordering/dimension is incompatible")
    required_files = {"davis_all.csv", "davis_train.csv", "davis_test.csv", "ligand_atom_features.pt"}
    file_entries = manifest.get("files")
    if not isinstance(file_entries, dict) or set(file_entries) != required_files:
        raise ValueError("Scaled feature manifest file list is incomplete or unexpected")
    for filename, metadata in file_entries.items():
        path = feature_dir / filename
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Scaled feature bundle hash validation failed for {path}")

    tensor_path = feature_dir / "ligand_atom_features.pt"
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Scaled ligand feature tensors not found: {tensor_path}")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict):
        raise ValueError("Scaled feature tensor file must contain an ID-to-tensor mapping")
    records = manifest.get("ligands")
    if not isinstance(records, list):
        raise ValueError("Scaled feature manifest has no ligand record list")
    ids = [str(record.get("drug_id")) for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(tensors):
        raise ValueError("Scaled feature manifest and tensor IDs are missing or duplicated")
    if manifest.get("ligand_count") != len(records):
        raise ValueError("Scaled feature manifest ligand count is inconsistent")
    atom_total = 0
    for record in records:
        drug_id = str(record["drug_id"])
        tensor = tensors[drug_id]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"Feature tensor for drug ID {drug_id} must be float32")
        if tensor.ndim != 2 or tensor.shape != (record.get("atom_count"), len(FEATURE_NAMES)):
            raise ValueError(f"Feature tensor shape mismatch for drug ID {drug_id}: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite feature tensor for drug ID {drug_id}")
        atom_total += int(tensor.shape[0])
    if manifest.get("atom_count") != atom_total:
        raise ValueError("Scaled feature manifest atom count is inconsistent")
    return tensors, manifest
