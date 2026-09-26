"""
Molecular and protein preprocessing for MGraphDTA.

Phase 2 focuses on:
- SMILES validation and molecular graph topology extraction (atoms, bonds, edge_index)
- Protein sequence tokenization with standard VOCAB_PROTEIN, fixed padding/truncation (1200)
"""

from typing import Tuple, Dict, Any, List
import torch
import numpy as np
from rdkit import Chem

# Standard 25-letter amino acid vocabulary mapping (0 reserved for padding)
VOCAB_PROTEIN: Dict[str, int] = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25
}

MAX_PROTEIN_LEN: int = 1200


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Parses a SMILES string into an RDKit Mol object and validates it.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Invalid SMILES input (empty or non-string): {smiles}")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    return mol


def extract_graph_structure(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Extracts basic topological structure of the molecular graph:
    - num_atoms
    - num_bonds (undirected)
    - edge_index: shape [2, num_directed_edges] (bidirectional edges)
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add bidirectional directed edges
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Edge case: single atom molecule without bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "edge_index": edge_index
    }


def preprocess_protein(sequence: str, max_length: int = MAX_PROTEIN_LEN) -> Tuple[torch.Tensor, int]:
    """
    Converts protein amino acid sequence to an integer token tensor.
    - Tokenizes using VOCAB_PROTEIN (unknown characters map to 0 or handled)
    - Captures length before padding
    - Pads with 0 if length < max_length
    - Truncates if length > max_length
    Returns (padded_tensor, original_length).
    """
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Invalid protein sequence (empty or non-string).")

    clean_seq = sequence.strip().upper()
    orig_len = len(clean_seq)

    # Convert characters to integer tokens
    tokens = [VOCAB_PROTEIN.get(char, 0) for char in clean_seq]

    # Pad or truncate
    if orig_len < max_length:
        tokens = tokens + [0] * (max_length - orig_len)
    else:
        tokens = tokens[:max_length]

    protein_tensor = torch.tensor(tokens, dtype=torch.long)
    return protein_tensor, orig_len
