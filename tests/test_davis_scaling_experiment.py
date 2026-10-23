"""Tests for the isolated Davis feature-scaling comparison (no model training)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from scripts.generate_davis_scaling_comparison import (
    DEFAULT_OUTPUT_DIR,
    FEATURE_NAMES,
    author_global_minmax,
    author_raw_features,
    feature_pair,
    generate_comparison,
    guarded_global_minmax,
)


def test_all_davis_unscaled_features_match_pinned_author_and_are_finite() -> None:
    manifest = generate_comparison.__globals__["_load_author_preprocessor"]()
    ligand_path = generate_comparison.__globals__["LIGANDS_FILE"]
    ligands = json.loads(ligand_path.read_text(encoding="utf-8"))
    assert len(ligands) == 68
    total_atoms = 0
    for smiles in ligands.values():
        local, author = feature_pair(smiles, manifest)
        assert local.ndim == 2
        assert local.shape[1] == len(FEATURE_NAMES) == 22
        assert torch.isfinite(local).all()
        assert torch.equal(local, author)
        total_atoms += local.shape[0]
    assert total_atoms == 2180


def test_author_formula_is_molecule_global_and_zero_range_columns_are_not_separately_scaled() -> None:
    first = torch.tensor([[1.0, 0.0], [1.0, 4.0]])
    second = torch.tensor([[10.0, 20.0], [10.0, 30.0]])
    first_scaled = author_global_minmax(first)
    second_scaled = author_global_minmax(second)
    assert torch.equal(first_scaled, torch.tensor([[0.25, 0.0], [0.25, 1.0]]))
    assert torch.equal(second_scaled, torch.tensor([[0.0, 0.5], [0.0, 1.0]]))
    # The first column has zero range, but the author's matrix-wide denominator is nonzero.
    assert torch.equal(first_scaled[:, 0], torch.tensor([0.25, 0.25]))


def test_constant_matrix_matches_unguarded_author_behavior_and_safe_alternative() -> None:
    constant = torch.ones((3, 22), dtype=torch.float32)
    # Exact author formula divides 0 by 0; it does not return a guarded all-zero matrix.
    assert torch.isnan(author_global_minmax(constant)).all()
    with pytest.raises(ValueError, match="constant"):
        guarded_global_minmax(constant)


@pytest.mark.parametrize("smiles", ["", "   ", "not-a-smiles-%%%"])
def test_empty_or_invalid_smiles_are_rejected(smiles: str) -> None:
    with pytest.raises(ValueError):
        author_raw_features(smiles)


def test_comparison_outputs_have_expected_shapes_and_finite_values(tmp_path: Path) -> None:
    out = tmp_path / "run"
    manifest = generate_comparison(out)
    raw_path = out / "unscaled_atom_features.csv"
    scaled_path = out / "author_scaled_atom_features.csv"
    raw_rows = list(csv.DictReader(raw_path.open(encoding="utf-8", newline="")))
    scaled_rows = list(csv.DictReader(scaled_path.open(encoding="utf-8", newline="")))
    assert len(raw_rows) == len(scaled_rows) == 2180
    assert len({row["drug_id"] for row in raw_rows}) == 68
    assert list(raw_rows[0]) == ["drug_id", "smiles", "atom_index", *FEATURE_NAMES]
    assert [(r["drug_id"], r["atom_index"]) for r in raw_rows] == [
        (r["drug_id"], r["atom_index"]) for r in scaled_rows
    ]
    for row in raw_rows + scaled_rows:
        assert all(torch.isfinite(torch.tensor(float(row[name]))) for name in FEATURE_NAMES)
    assert manifest["outputs"][raw_path.name]["sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert manifest["outputs"][scaled_path.name]["sha256"] == hashlib.sha256(scaled_path.read_bytes()).hexdigest()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert out.resolve() != DEFAULT_OUTPUT_DIR.resolve()


def test_generated_artifact_hashes_are_reproducible(tmp_path: Path) -> None:
    out1, out2 = tmp_path / "first", tmp_path / "second"
    generate_comparison(out1)
    generate_comparison(out2)
    names = ["unscaled_atom_features.csv", "author_scaled_atom_features.csv", "manifest.json"]
    for name in names:
        a, b = (out1 / name).read_bytes(), (out2 / name).read_bytes()
        assert a == b
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing"):
        generate_comparison(out)
    assert marker.read_text(encoding="utf-8") == "preserve"
