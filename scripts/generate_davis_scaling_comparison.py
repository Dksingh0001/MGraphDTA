"""Generate isolated raw-vs-author-scaled Davis atom-feature tables.

This utility never reads or writes the active processed CSVs and does not import
or invoke the model/training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, rdBase

from src.preprocessing import ATOM_FEATURE_DIM, extract_atom_features_22dim, smiles_to_mol


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_REPO = ROOT / "references" / "mgraphdta_official" / "author_repo"
AUTHOR_PREPROCESSING = AUTHOR_REPO / "regression" / "preprocessing.py"
LIGANDS_FILE = ROOT / "data" / "raw" / "davis" / "ligands_can.txt"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "davis_scaling_experiment"
PINNED_AUTHOR_COMMIT = "54386680e8c9562f708a0fe5d381038c62111df6"

FEATURE_NAMES = [
    "atom_H", "atom_C", "atom_N", "atom_O", "atom_F",
    "atom_Cl", "atom_S", "atom_Br", "atom_I",
    "atomic_number", "acceptor", "donor", "aromatic",
    "hybridization_SP", "hybridization_SP2", "hybridization_SP3",
    "total_hydrogens", "explicit_valence", "formal_charge",
    "implicit_valence", "explicit_hydrogens", "radical_electrons",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_author_preprocessor() -> Any:
    if not AUTHOR_PREPROCESSING.is_file():
        raise FileNotFoundError(f"Pinned author source missing: {AUTHOR_PREPROCESSING}")
    result = subprocess.run(
        ["git", "-C", str(AUTHOR_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip()
    if actual_commit != PINNED_AUTHOR_COMMIT:
        raise RuntimeError(
            f"Expected author commit {PINNED_AUTHOR_COMMIT}, found {actual_commit}"
        )

    spec = importlib.util.spec_from_file_location(
        "mgraphdta_pinned_author_preprocessing", AUTHOR_PREPROCESSING
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {AUTHOR_PREPROCESSING}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def author_raw_features(smiles: str, author_module: Any | None = None) -> torch.Tensor:
    """Return the pinned author implementation's unscaled atom-feature matrix."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    module = author_module or _load_author_preprocessor()
    dataset = object.__new__(module.GNNDataset)
    graph = dataset.mol2graph(mol)
    if graph is None or graph[0] is None:
        raise ValueError(f"Author preprocessing could not construct a graph for {smiles!r}")
    return graph[0].to(dtype=torch.float32)


def author_global_minmax(raw: torch.Tensor) -> torch.Tensor:
    """Exact author formula: one scalar min/max over the complete molecule matrix."""
    return (raw - raw.min()) / (raw.max() - raw.min())


def guarded_global_minmax(raw: torch.Tensor) -> torch.Tensor:
    """Safer alternative: reject a zero-range matrix instead of returning NaNs."""
    span = raw.max() - raw.min()
    if not torch.isfinite(span) or float(span) == 0.0:
        raise ValueError("Cannot scale a constant or non-finite feature matrix")
    return (raw - raw.min()) / span


def feature_pair(smiles: str, author_module: Any | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local unscaled and pinned-author unscaled matrices for parity checks."""
    mol = smiles_to_mol(smiles)
    local = extract_atom_features_22dim(mol).to(dtype=torch.float32)
    official = author_raw_features(smiles, author_module)
    if local.shape != official.shape:
        raise AssertionError(f"Feature shape mismatch: local={local.shape}, author={official.shape}")
    if local.shape[1] != ATOM_FEATURE_DIM or ATOM_FEATURE_DIM != len(FEATURE_NAMES):
        raise AssertionError("Unexpected atom feature dimension/order declaration")
    if not torch.equal(local, official):
        delta = float((local - official).abs().max().item())
        raise AssertionError(f"Local raw features differ from pinned author features (max diff {delta})")
    return local, official


def _write_feature_csv(path: Path, rows: list[tuple[str, str, int, torch.Tensor]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["drug_id", "smiles", "atom_index", *FEATURE_NAMES])
        for drug_id, smiles, atom_index, feature in rows:
            writer.writerow([
                drug_id,
                smiles,
                atom_index,
                *(format(float(value), ".17g") for value in feature.tolist()),
            ])


def generate_comparison(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Write deterministic side-by-side CSVs and a provenance/hash manifest."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite output directory: {output_dir}")
    if not LIGANDS_FILE.is_file():
        raise FileNotFoundError(f"Missing Davis ligand dictionary: {LIGANDS_FILE}")

    author_module = _load_author_preprocessor()
    ligands = json.loads(LIGANDS_FILE.read_text(encoding="utf-8"))
    if len(ligands) != 68:
        raise ValueError(f"Expected 68 Davis ligands, found {len(ligands)}")

    raw_rows: list[tuple[str, str, int, torch.Tensor]] = []
    scaled_rows: list[tuple[str, str, int, torch.Tensor]] = []
    molecule_ranges: list[dict[str, float | str | int]] = []
    atom_count = 0

    for drug_id, smiles in ligands.items():
        local, author_raw = feature_pair(smiles, author_module)
        if local.ndim != 2 or local.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Unexpected matrix shape for drug {drug_id}: {tuple(local.shape)}")
        if not torch.isfinite(local).all() or not torch.isfinite(author_raw).all():
            raise ValueError(f"Non-finite raw atom features for drug {drug_id}")
        span = float((author_raw.max() - author_raw.min()).item())
        if span <= 0:
            raise ValueError(f"Zero-range Davis molecule features for drug {drug_id}")
        scaled = author_global_minmax(author_raw)
        if not torch.isfinite(scaled).all():
            raise ValueError(f"Non-finite scaled atom features for drug {drug_id}")

        for atom_index in range(local.shape[0]):
            raw_rows.append((str(drug_id), smiles, atom_index, local[atom_index].cpu()))
            scaled_rows.append((str(drug_id), smiles, atom_index, scaled[atom_index].cpu()))
        atom_count += int(local.shape[0])
        molecule_ranges.append({
            "drug_id": str(drug_id),
            "atoms": int(local.shape[0]),
            "minimum": float(author_raw.min().item()),
            "maximum": float(author_raw.max().item()),
            "range": span,
        })

    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "unscaled_atom_features.csv"
    scaled_path = output_dir / "author_scaled_atom_features.csv"
    _write_feature_csv(raw_path, raw_rows)
    _write_feature_csv(scaled_path, scaled_rows)

    manifest: dict[str, Any] = {
        "experiment": "Davis atom feature scaling comparison only; no model training",
        "author_repository_commit": PINNED_AUTHOR_COMMIT,
        "author_preprocessing_sha256": sha256_file(AUTHOR_PREPROCESSING),
        "local_ligand_dictionary_sha256": sha256_file(LIGANDS_FILE),
        "rdkit_version": rdBase.rdkitVersion,
        "torch_version": torch.__version__,
        "ligand_count": len(ligands),
        "atom_count": atom_count,
        "feature_dimension": len(FEATURE_NAMES),
        "feature_order": FEATURE_NAMES,
        "unscaled_definition": "Pinned author mol2graph raw node matrix; parity checked against local feature function.",
        "author_scaling_definition": "(x - x.min()) / (x.max() - x.min()); x is [atoms, 22]; min/max are scalar reductions over the complete matrix for one molecule.",
        "constant_matrix_behavior": "Pinned formula has no zero-range guard; this generator rejects zero-range Davis matrices before writing.",
        "molecule_ranges": molecule_ranges,
        "outputs": {
            raw_path.name: {"rows": atom_count, "sha256": sha256_file(raw_path)},
            scaled_path.name: {"rows": atom_count, "sha256": sha256_file(scaled_path)},
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = generate_comparison(args.output_dir)
    print(f"Generated {manifest['atom_count']} atom rows for {manifest['ligand_count']} ligands")
    print(f"Output: {args.output_dir}")
    for name, metadata in manifest["outputs"].items():
        print(f"{name}: SHA-256 {metadata['sha256']}")


if __name__ == "__main__":
    main()
