"""
Dedicated test for 22-dimensional atom feature extraction.

Tests:
- feature tensor shape is [num_atoms, 22]
- feature dimension is exactly 22 for all atoms
- all values are finite (no NaN, Inf)
- runs on 5 real Davis compound SMILES
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.preprocessing import (
    smiles_to_mol,
    extract_atom_features_22dim,
    ATOM_FEATURE_DIM,
)

# 5 representative Davis molecules (drawn from ligands_can.txt)
TEST_SMILES = [
    # Compound 1: staurosporine-like (complex ring system)
    "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N",
    # Compound 2: imatinib (Gleevec)
    "CN1CCN(CC1)C2=CC=C(C=C2)C(=O)NC3=CC(=NC=C3)C4=CN=CC=C4",
    # Compound 3: erlotinib
    "COCCOC1=C(OCCOC)C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    # Compound 4: gefitinib
    "COC1=CC2=C(C=C1OCCCN3CCOCC3)NC=NC2=NC4=CC(=C(C=C4)Cl)F",
    # Compound 5: lapatinib
    "CS(=O)(=O)CCC1=CC=C(C=C1)NC(=O)NC2=NC=C(C=C2)C3=CC=NC=C3",
]


def test_atom_features_22dim():
    print("=" * 70)
    print("TEST: 22-Dimensional Atom Feature Extraction")
    print("=" * 70)
    print(f"Expected feature dimension per atom: {ATOM_FEATURE_DIM}")
    print("-" * 70)

    for i, smiles in enumerate(TEST_SMILES):
        mol = smiles_to_mol(smiles)
        num_atoms = mol.GetNumAtoms()

        features = extract_atom_features_22dim(mol)

        # --- Assertions ---
        assert features.ndim == 2, (
            f"SMILES {i+1}: Expected 2D tensor, got {features.ndim}D"
        )
        assert features.shape[0] == num_atoms, (
            f"SMILES {i+1}: Expected {num_atoms} rows, got {features.shape[0]}"
        )
        assert features.shape[1] == 22, (
            f"SMILES {i+1}: Expected 22 feature dims, got {features.shape[1]}"
        )
        assert torch.isfinite(features).all(), (
            f"SMILES {i+1}: Feature tensor contains NaN or Inf values"
        )

        # --- Report ---
        print(f"Sample {i + 1}:")
        print(f"  SMILES: {smiles}")
        print(f"  Number of atoms: {num_atoms}")
        print(f"  Atom feature shape: {list(features.shape)}")
        print(f"  Expected feature dimension: {ATOM_FEATURE_DIM}")
        print(f"  All values finite: {torch.isfinite(features).all().item()}")
        print(f"  Feature test: PASSED")
        print("-" * 70)

    print("\nALL 22-DIM ATOM FEATURE TESTS PASSED!")
    print("=" * 70)


def test_with_real_davis_smiles(num_compounds: int = 5):
    """Load actual Davis ligands and run the same assertions."""
    raw_path = os.path.join("data", "raw", "davis", "ligands_can.txt")
    if not os.path.exists(raw_path):
        print(f"Skipping real Davis test (raw file not found: {raw_path})")
        return

    with open(raw_path, "r") as f:
        ligands = json.load(f)

    davis_smiles = list(ligands.values())[:num_compounds]

    print("=" * 70)
    print(f"TEST: 22-Dim Features on {num_compounds} Real Davis Compounds")
    print("=" * 70)

    for i, smiles in enumerate(davis_smiles):
        mol = smiles_to_mol(smiles)
        num_atoms = mol.GetNumAtoms()
        features = extract_atom_features_22dim(mol)

        assert features.ndim == 2
        assert features.shape[1] == 22
        assert features.shape[0] == num_atoms
        assert torch.isfinite(features).all()

        print(f"Sample {i + 1}:")
        print(f"  SMILES: {smiles}")
        print(f"  Number of atoms: {num_atoms}")
        print(f"  Atom feature shape: {list(features.shape)}")
        print(f"  Expected feature dimension: {ATOM_FEATURE_DIM}")
        print(f"  Feature test: PASSED")
        print("-" * 70)

    print(f"\nALL {num_compounds} REAL DAVIS COMPOUND TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_atom_features_22dim()
    test_with_real_davis_smiles()
