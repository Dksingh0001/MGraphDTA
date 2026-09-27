"""
Verification test for Davis dataset loading and molecular/protein preprocessing.
Loads real Davis samples and verifies:
- SMILES validity with RDKit
- Atom & bond counts
- Edge index shape [2, num_directed_edges]
- Protein tensor shape [1200]
- Numeric affinity value (pKd)
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.data import Batch

from src.dataset import DavisDataset
from src.preprocessing import smiles_to_mol
from src.train import make_dataloader


def test_davis_pipeline(num_samples: int = 5):
    print("=" * 70)
    print("TEST: Davis Dataset Preprocessing & Loader Verification")
    print("=" * 70)

    # 1. Initialize dataset
    dataset = DavisDataset(split="all")
    total_samples = len(dataset)
    print(f"Total interactions in Davis dataset: {total_samples}")
    assert total_samples == 30056, f"Expected 30,056 samples, got {total_samples}"

    print("-" * 70)
    print(f"Inspecting first {num_samples} samples:\n")

    for i in range(num_samples):
        sample = dataset[i]

        smiles = sample["drug_smiles"]
        num_atoms = sample["num_atoms"]
        num_bonds = sample["num_bonds"]
        edge_index = sample["edge_index"]
        prot_orig_len = sample["protein_orig_len"]
        prot_tensor = sample["protein_tensor"]
        affinity = sample["affinity"]

        # Validations
        assert smiles is not None and len(smiles) > 0, f"Sample {i}: Missing SMILES"
        mol = smiles_to_mol(smiles)
        assert mol is not None, f"Sample {i}: RDKit could not parse SMILES {smiles}"
        assert num_atoms > 0, f"Sample {i}: Atom count is 0"
        assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
            f"Sample {i}: Edge index shape invalid: {edge_index.shape}"
        )
        assert edge_index.shape[1] == 2 * num_bonds, (
            f"Sample {i}: Expected {2 * num_bonds} directed edges, got {edge_index.shape[1]}"
        )
        assert prot_tensor.shape == torch.Size([1200]), (
            f"Sample {i}: Expected protein tensor shape [1200], got {prot_tensor.shape}"
        )
        assert isinstance(affinity, (int, float)) and not torch.isnan(torch.tensor(affinity)), (
            f"Sample {i}: Invalid affinity {affinity}"
        )

        # Print formatted output as requested
        print(f"Sample number: {i + 1}")
        print(f"Drug SMILES: {smiles}")
        print(f"Number of atoms: {num_atoms}")
        print(f"Number of bonds: {num_bonds} (directed edges: {edge_index.shape[1]})")
        print(f"Protein length before padding: {prot_orig_len}")
        print(f"Protein tensor shape: {list(prot_tensor.shape)}")
        print(f"Affinity value (pKd): {affinity:.4f}")
        print("-" * 70)

    # 2. Verify all unique compounds and targets in Davis
    unique_drugs = dataset.df["compound_iso_smiles"].unique()
    unique_target_ids = dataset.df["target_id"].unique()
    unique_sequences = dataset.df["target_sequence"].unique()

    print(f"Total Unique Compounds: {len(unique_drugs)}")
    print(f"Total Unique Target IDs: {len(unique_target_ids)}")
    print(f"Total Unique Sequences: {len(unique_sequences)} (some target variants share wild-type sequence)")

    assert len(unique_drugs) == 68, f"Expected 68 compounds, got {len(unique_drugs)}"
    assert len(unique_target_ids) == 442, f"Expected 442 targets, got {len(unique_target_ids)}"
    assert len(unique_sequences) == 379, f"Expected 379 unique sequences, got {len(unique_sequences)}"

    # Check that all 68 compounds parse cleanly in RDKit
    for smi in unique_drugs:
        m = smiles_to_mol(smi)
        assert m is not None, f"Failed to parse compound: {smi}"

    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


def test_davis_pyg_batches_variable_edge_counts():
    dataset = DavisDataset(split="train")
    indices = (0, 442)
    samples = [dataset[index] for index in indices]

    assert samples[0]["edge_index"].shape[1] != samples[1]["edge_index"].shape[1]

    batch = next(iter(make_dataloader(
        torch.utils.data.Subset(dataset, indices), batch_size=2
    )))

    assert isinstance(batch, Batch)
    assert batch.num_graphs == 2
    assert batch.y.shape == (2,)
    assert batch.target.shape == (2, 1200)
    assert batch.edge_index.shape[1] == sum(
        sample["edge_index"].shape[1] for sample in samples
    )
    assert batch.x.shape[0] == sum(sample["num_atoms"] for sample in samples)
    assert batch.batch.shape == (batch.x.shape[0],)
    assert batch.edge_index.max().item() < batch.x.shape[0]


if __name__ == "__main__":
    test_davis_pipeline()
    test_davis_pyg_batches_variable_edge_counts()
