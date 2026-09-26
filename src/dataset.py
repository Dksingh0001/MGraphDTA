"""
Davis Dataset Loader and Processing for MGraphDTA.

Parses raw Davis files:
- ligands_can.txt: JSON dictionary of {compound_id: SMILES}
- proteins.txt: JSON dictionary of {target_id: sequence}
- Y: Pickle 2D array of raw Kd affinities in nM
- folds/train_fold_setting1.txt & folds/test_fold_setting1.txt: Standard benchmark split indices

Transforms affinity:
  pKd = -log10(Kd / 1e9) = 9 - log10(Kd)
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.preprocessing import (
    smiles_to_mol,
    extract_graph_structure,
    preprocess_protein
)


def load_raw_davis_data(raw_dir: str = "data/raw/davis") -> Dict[str, Any]:
    """
    Reads the raw Davis files and returns ligands, proteins, Y matrix, and fold indices.
    """
    ligands_file = os.path.join(raw_dir, "ligands_can.txt")
    proteins_file = os.path.join(raw_dir, "proteins.txt")
    affinity_file = os.path.join(raw_dir, "Y")
    train_fold_file = os.path.join(raw_dir, "folds", "train_fold_setting1.txt")
    test_fold_file = os.path.join(raw_dir, "folds", "test_fold_setting1.txt")

    if not os.path.exists(ligands_file):
        raise FileNotFoundError(f"Missing {ligands_file}")
    if not os.path.exists(proteins_file):
        raise FileNotFoundError(f"Missing {proteins_file}")
    if not os.path.exists(affinity_file):
        raise FileNotFoundError(f"Missing {affinity_file}")

    with open(ligands_file, "r") as f:
        ligands = json.load(f)
    with open(proteins_file, "r") as f:
        proteins = json.load(f)
    with open(affinity_file, "rb") as f:
        Y = pickle.load(f, encoding="latin1")

    train_folds = None
    test_fold = None
    if os.path.exists(train_fold_file):
        with open(train_fold_file, "r") as f:
            train_folds = json.load(f)
    if os.path.exists(test_fold_file):
        with open(test_fold_file, "r") as f:
            test_fold = json.load(f)

    return {
        "ligands": ligands,
        "proteins": proteins,
        "Y": np.array(Y, dtype=np.float64),
        "train_folds": train_folds,
        "test_fold": test_fold
    }


def prepare_davis_tables(
    raw_dir: str = "data/raw/davis",
    processed_dir: str = "data/processed/davis"
) -> Dict[str, pd.DataFrame]:
    """
    Extracts compound-protein-affinity interaction pairs from raw files,
    computes pKd = -log10(Kd / 1e9), and persists processed CSV files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_data = load_raw_davis_data(raw_dir)

    ligands = raw_data["ligands"]
    proteins = raw_data["proteins"]
    Y = raw_data["Y"]

    ligand_ids = list(ligands.keys())
    protein_ids = list(proteins.keys())

    records = []
    # Grid of (drug_idx, protein_idx)
    num_drugs = len(ligand_ids)
    num_proteins = len(protein_ids)

    for i in range(num_drugs):
        c_id = ligand_ids[i]
        smi = ligands[c_id]
        for j in range(num_proteins):
            p_id = protein_ids[j]
            seq = proteins[p_id]
            kd_val = Y[i, j]

            # In Davis, Kd is in nM. Non-binding pairs are labeled 10,000 nM.
            # pKd = -log10(Kd in M) = -log10(Kd_nM * 1e-9) = 9 - log10(Kd_nM)
            pkd_val = float(-np.log10(kd_val / 1e9))

            records.append({
                "pair_idx": i * num_proteins + j,
                "drug_id": c_id,
                "compound_iso_smiles": smi,
                "target_id": p_id,
                "target_sequence": seq,
                "affinity_kd": kd_val,
                "affinity": pkd_val
            })

    full_df = pd.DataFrame(records)

    # Save full dataset
    full_csv = os.path.join(processed_dir, "davis_all.csv")
    full_df.to_csv(full_csv, index=False)

    # If fold splits exist, partition into train and test
    dfs = {"all": full_df}
    if raw_data["train_folds"] is not None and raw_data["test_fold"] is not None:
        # Flatten 5-fold training indices
        train_indices = [idx for fold in raw_data["train_folds"] for idx in fold]
        test_indices = raw_data["test_fold"]

        train_df = full_df.iloc[train_indices].copy().reset_index(drop=True)
        test_df = full_df.iloc[test_indices].copy().reset_index(drop=True)

        train_df.to_csv(os.path.join(processed_dir, "davis_train.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "davis_test.csv"), index=False)
        dfs["train"] = train_df
        dfs["test"] = test_df

    return dfs


class DavisDataset(Dataset):
    """
    Davis Drug-Target Affinity Dataset.
    Each item contains:
    - drug_smiles: str
    - num_atoms: int
    - num_bonds: int
    - edge_index: LongTensor [2, num_directed_edges]
    - protein_seq: str
    - protein_orig_len: int
    - protein_tensor: LongTensor [1200]
    - affinity: float (pKd)
    - py_data: torch_geometric.data.Data object with edge_index, target, and y
    """
    def __init__(
        self,
        split: str = "all",
        raw_dir: str = "data/raw/davis",
        processed_dir: str = "data/processed/davis"
    ):
        super().__init__()
        self.processed_dir = processed_dir
        csv_file = os.path.join(processed_dir, f"davis_{split}.csv")
        
        if not os.path.exists(csv_file):
            print(f"Processed table {csv_file} not found. Generating from raw data...")
            tables = prepare_davis_tables(raw_dir=raw_dir, processed_dir=processed_dir)
            self.df = tables[split]
        else:
            self.df = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        smiles = row["compound_iso_smiles"]
        sequence = row["target_sequence"]
        affinity = float(row["affinity"])

        # 1. Process molecule
        mol = smiles_to_mol(smiles)
        graph_info = extract_graph_structure(mol)

        # 2. Process protein
        protein_tensor, orig_prot_len = preprocess_protein(sequence, max_length=1200)

        # 3. Create PyG Data object (basic topology for now)
        pyg_data = Data(
            edge_index=graph_info["edge_index"],
            target=protein_tensor.unsqueeze(0),
            y=torch.tensor([affinity], dtype=torch.float32)
        )

        return {
            "sample_index": idx,
            "drug_smiles": smiles,
            "num_atoms": graph_info["num_atoms"],
            "num_bonds": graph_info["num_bonds"],
            "edge_index": graph_info["edge_index"],
            "protein_seq": sequence,
            "protein_orig_len": orig_prot_len,
            "protein_tensor": protein_tensor,
            "affinity": affinity,
            "pyg_data": pyg_data
        }
