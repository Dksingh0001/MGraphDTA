"""Build a standalone 2D residue graph prototype for Davis AKT1."""

import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTEINS_PATH = PROJECT_ROOT / "data/raw/davis/proteins.txt"
OUTPUT_PATH = PROJECT_ROOT / "results/davis/akt1_2d_graph.pt"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_residue_graph(sequence: str) -> dict[str, object]:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("AKT1 protein sequence is empty")

    amino_acid_index = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
    node_features = torch.zeros((len(sequence), len(AMINO_ACIDS)), dtype=torch.float32)
    for residue_index, amino_acid in enumerate(sequence):
        feature_index = amino_acid_index.get(amino_acid)
        if feature_index is not None:
            node_features[residue_index, feature_index] = 1.0

    directed_edges = []
    for residue_index in range(len(sequence) - 1):
        next_index = residue_index + 1
        directed_edges.append((residue_index, next_index))
        directed_edges.append((next_index, residue_index))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "protein_name": "AKT1",
        "sequence": sequence,
        "residue_numbers": list(range(1, len(sequence) + 1)),
        "residues": list(sequence),
        "amino_acid_feature_order": list(AMINO_ACIDS),
        "unknown_residue_indices": [
            index for index, amino_acid in enumerate(sequence)
            if amino_acid not in amino_acid_index
        ],
        "x": node_features,
        "edge_index": edge_index,
        "num_nodes": len(sequence),
        "num_directed_edges": edge_index.shape[1],
        "feature_dim": len(AMINO_ACIDS),
    }


def main() -> None:
    with PROTEINS_PATH.open(encoding="utf-8") as proteins_file:
        proteins = json.load(proteins_file)
    if "AKT1" not in proteins:
        raise KeyError(f"AKT1 was not found in {PROTEINS_PATH}")

    graph = build_residue_graph(proteins["AKT1"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, OUTPUT_PATH)

    print(f"Protein name: {graph['protein_name']}")
    print(f"Sequence length: {len(graph['sequence'])}")
    print(f"Number of graph nodes: {graph['num_nodes']}")
    print(f"Number of directed edges: {graph['num_directed_edges']}")
    print(f"Feature dimension: {graph['feature_dim']}")
    print(f"Unknown/non-standard residues: {len(graph['unknown_residue_indices'])}")
    print("First 10 residues:")
    print(" ".join(
        f"{number}:{residue}"
        for number, residue in zip(graph["residue_numbers"][:10], graph["residues"][:10])
    ))
    print("First edges:")
    print(graph["edge_index"][:, :8].t().tolist())
    print(f"Saved graph: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
