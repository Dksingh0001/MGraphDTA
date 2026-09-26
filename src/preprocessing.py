"""
Molecular and protein preprocessing for MGraphDTA.

Implements:
- SMILES validation and parsing (smiles_to_mol)
- 22-dimensional atom feature extraction (extract_atom_features_22dim)
  as specified in PROJECT_PLAN.md Section 3.1 (Regression Task Featurization)
- Molecular graph topology + node features (extract_graph_structure)
- Protein sequence tokenization with VOCAB_PROTEIN, fixed-length 1200 (preprocess_protein)
"""

import os.path as osp
from typing import Tuple, Dict, Any, List
import torch
from rdkit import Chem

# Standard 25-letter amino acid vocabulary (0 reserved for padding)
VOCAB_PROTEIN: Dict[str, int] = {
    "A": 1,  "C": 2,  "B": 3,  "E": 4,  "D": 5,  "G": 6,
    "F": 7,  "I": 8,  "H": 9,  "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25
}

MAX_PROTEIN_LEN: int = 1200

# Atom types used for one-hot encoding (9 elements)
ATOM_TYPES: List[str] = ["H", "C", "N", "O", "F", "Cl", "S", "Br", "I"]

# Hybridization types used for one-hot encoding (3 types)
HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]

# Feature dimension constant — asserted at runtime
ATOM_FEATURE_DIM: int = 22


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Parses a SMILES string into an RDKit Mol object and validates it.
    Raises ValueError if the SMILES is empty or unparseable.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Invalid SMILES input (empty or non-string): {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    return mol


def _get_donor_acceptor_sets(mol: Chem.Mol):
    """
    Returns (donor_atom_indices, acceptor_atom_indices) sets
    using RDKit's BaseFeatures.fdef chemical feature factory.
    """
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig

    fdef_path = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
    feats = factory.GetFeaturesForMol(mol)

    donors: set = set()
    acceptors: set = set()
    for feat in feats:
        if feat.GetFamily() == "Donor":
            donors.update(feat.GetAtomIds())
        elif feat.GetFamily() == "Acceptor":
            acceptors.update(feat.GetAtomIds())
    return donors, acceptors


def _get_explicit_valence(atom: Chem.Atom) -> float:
    """
    Returns explicit valence using the RDKit 2026 API (avoids deprecation warning).
    Falls back to legacy API for older RDKit versions.
    """
    try:
        return float(atom.GetValence(Chem.ValenceType.EXPLICIT))
    except Exception:
        return float(atom.GetExplicitValence())


def _get_implicit_valence(atom: Chem.Atom) -> float:
    """
    Returns implicit valence using the RDKit 2026 API (avoids deprecation warning).
    Falls back to legacy API for older RDKit versions.
    """
    try:
        return float(atom.GetValence(Chem.ValenceType.IMPLICIT))
    except Exception:
        return float(atom.GetImplicitValence())


def extract_atom_features_22dim(mol: Chem.Mol) -> torch.Tensor:
    """
    Extracts a 22-dimensional atom feature vector for every atom in the molecule.

    Feature vector composition (22 total):
    ┌──────────────────────────────────────────────────────────┬──────┐
    │ Feature                                                  │ Dims │
    ├──────────────────────────────────────────────────────────┼──────┤
    │ 1.  Atom symbol one-hot: H,C,N,O,F,Cl,S,Br,I            │  9   │
    │ 2.  Atomic number (scalar)                               │  1   │
    │ 3.  Hydrogen acceptor (RDKit ChemicalFeatures)           │  1   │
    │ 4.  Hydrogen donor   (RDKit ChemicalFeatures)            │  1   │
    │ 5.  Aromaticity                                          │  1   │
    │ 6.  Hybridization one-hot: SP, SP2, SP3                  │  3   │
    │ 7.  Total Hs (GetTotalNumHs)                             │  1   │
    │ 8.  Explicit valence                                     │  1   │
    │ 9.  Formal charge                                        │  1   │
    │ 10. Implicit valence                                     │  1   │
    │ 11. Num explicit Hs (GetNumExplicitHs)                   │  1   │
    │ 12. Num radical electrons                                │  1   │
    ├──────────────────────────────────────────────────────────┼──────┤
    │ Total                                                    │ 22   │
    └──────────────────────────────────────────────────────────┴──────┘

    Args:
        mol: An RDKit Mol object (must not be None).

    Returns:
        torch.FloatTensor of shape [num_atoms, 22]
    """
    donors, acceptors = _get_donor_acceptor_sets(mol)
    all_features: List[List[float]] = []

    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        h_t: List[float] = []

        # 1. Atom symbol one-hot (9 dims)
        h_t += [float(atom.GetSymbol() == sym) for sym in ATOM_TYPES]

        # 2. Atomic number (1 dim)
        h_t.append(float(atom.GetAtomicNum()))

        # 3. Hydrogen acceptor (1 dim)
        h_t.append(float(idx in acceptors))

        # 4. Hydrogen donor (1 dim)
        h_t.append(float(idx in donors))

        # 5. Aromaticity (1 dim)
        h_t.append(float(atom.GetIsAromatic()))

        # 6. Hybridization one-hot (3 dims)
        h_t += [float(atom.GetHybridization() == htype) for htype in HYBRIDIZATION_TYPES]

        # 7. Total Hs (1 dim)
        h_t.append(float(atom.GetTotalNumHs()))

        # 8. Explicit valence (1 dim)
        h_t.append(_get_explicit_valence(atom))

        # 9. Formal charge (1 dim)
        h_t.append(float(atom.GetFormalCharge()))

        # 10. Implicit valence (1 dim)
        h_t.append(_get_implicit_valence(atom))

        # 11. Num explicit Hs (1 dim)
        h_t.append(float(atom.GetNumExplicitHs()))

        # 12. Num radical electrons (1 dim)
        h_t.append(float(atom.GetNumRadicalElectrons()))

        assert len(h_t) == ATOM_FEATURE_DIM, (
            f"Atom {idx} in molecule: expected {ATOM_FEATURE_DIM} features, got {len(h_t)}"
        )
        all_features.append(h_t)

    node_attr = torch.tensor(all_features, dtype=torch.float32)
    assert node_attr.shape == (mol.GetNumAtoms(), ATOM_FEATURE_DIM), (
        f"Final feature tensor shape mismatch: {node_attr.shape}"
    )
    return node_attr


def extract_graph_structure(mol: Chem.Mol) -> Dict[str, Any]:
    """
    Extracts full molecular graph structure with node features:
      - num_atoms : int
      - num_bonds : int (undirected bond count)
      - x         : FloatTensor [num_atoms, 22]  — 22-dim node feature matrix
      - edge_index : LongTensor [2, 2*num_bonds]  — bidirectional directed edges

    Args:
        mol: An RDKit Mol object (must not be None).

    Returns:
        dict with keys: num_atoms, num_bonds, x, edge_index
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    # 22-dimensional node feature matrix
    x = extract_atom_features_22dim(mol)

    # Bidirectional edge index
    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Single-atom molecule or disconnected graph with no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "x": x,
        "edge_index": edge_index,
    }


def preprocess_protein(sequence: str, max_length: int = MAX_PROTEIN_LEN) -> Tuple[torch.Tensor, int]:
    """
    Converts protein amino acid sequence to an integer token tensor.
      - Tokenizes using VOCAB_PROTEIN (unknown chars → 0)
      - Records original length before padding
      - Pads with 0 to max_length if shorter
      - Truncates to max_length if longer

    Args:
        sequence: Raw protein amino acid sequence string (1-letter codes).
        max_length: Fixed output length (default 1200 as per MGraphDTA paper).

    Returns:
        (protein_tensor [max_length], original_length)
    """
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Invalid protein sequence (empty or non-string).")

    clean_seq = sequence.strip().upper()
    orig_len = len(clean_seq)

    tokens = [VOCAB_PROTEIN.get(char, 0) for char in clean_seq]

    if orig_len < max_length:
        tokens = tokens + [0] * (max_length - orig_len)
    else:
        tokens = tokens[:max_length]

    protein_tensor = torch.tensor(tokens, dtype=torch.long)
    return protein_tensor, orig_len
