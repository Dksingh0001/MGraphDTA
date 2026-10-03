"""Build standalone 2D residue graphs for unique Davis protein sequences."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTEINS_PATH = PROJECT_ROOT / "data/raw/davis/proteins.txt"
OUTPUT_DIR = PROJECT_ROOT / "results/davis/protein_graphs_2d"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_DIM = len(AMINO_ACIDS)


def build_residue_graph(
    graph_id: str,
    sequence: str,
    target_ids: list[str],
) -> dict[str, Any]:
    if not sequence:
        raise ValueError("sequence is empty after normalization")

    amino_acid_index = {
        amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
    }
    node_features = torch.zeros((len(sequence), FEATURE_DIM), dtype=torch.float32)
    unknown_residue_indices = []
    for residue_index, amino_acid in enumerate(sequence):
        feature_index = amino_acid_index.get(amino_acid)
        if feature_index is None:
            unknown_residue_indices.append(residue_index)
        else:
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
        "graph_id": graph_id,
        "target_ids": list(target_ids),
        "sequence": sequence,
        "residue_numbers": list(range(1, len(sequence) + 1)),
        "residues": list(sequence),
        "amino_acid_feature_order": list(AMINO_ACIDS),
        "unknown_residue_indices": unknown_residue_indices,
        "x": node_features,
        "edge_index": edge_index,
        "num_nodes": len(sequence),
        "num_directed_edges": edge_index.shape[1],
        "feature_dim": FEATURE_DIM,
    }


def validate_graph(graph: dict[str, Any]) -> None:
    sequence_length = len(graph["sequence"])
    features = graph["x"]
    edge_index = graph["edge_index"]
    graph_id = graph["graph_id"]

    if features.shape[0] != sequence_length:
        raise ValueError(
            f"x has {features.shape[0]} rows for {sequence_length} residues"
        )
    if features.shape[1] != FEATURE_DIM:
        raise ValueError(f"x has feature dimension {features.shape[1]}, expected {FEATURE_DIM}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index has invalid shape {tuple(edge_index.shape)}")
    if graph["num_nodes"] != sequence_length:
        raise ValueError("num_nodes does not match the sequence length")
    if graph["num_directed_edges"] != edge_index.shape[1]:
        raise ValueError("num_directed_edges does not match edge_index")

    edge_pairs = [tuple(pair) for pair in edge_index.t().tolist()]
    edge_set = set(edge_pairs)
    if any(source == target for source, target in edge_pairs):
        raise ValueError("edge_index contains a self-loop")

    for residue_index in range(sequence_length - 1):
        forward = (residue_index, residue_index + 1)
        reverse = (residue_index + 1, residue_index)
        if forward not in edge_set or reverse not in edge_set:
            raise ValueError(
                f"missing bidirectional consecutive edge for residues "
                f"{residue_index + 1} and {residue_index + 2}"
            )

    for residue_index in graph["unknown_residue_indices"]:
        if torch.count_nonzero(features[residue_index]).item() != 0:
            raise ValueError(
                f"unknown residue row {residue_index} is not an all-zero vector"
            )


def load_unique_sequences() -> tuple[int, dict[str, list[str]]]:
    with PROTEINS_PATH.open(encoding="utf-8") as proteins_file:
        target_sequences = json.load(proteins_file)

    sequence_targets: dict[str, list[str]] = defaultdict(list)
    for target_id, raw_sequence in target_sequences.items():
        sequence = str(raw_sequence).strip().upper()
        sequence_targets[sequence].append(str(target_id))

    return len(target_sequences), dict(sequence_targets)


def main() -> int:
    total_target_ids, sequence_targets = load_unique_sequences()
    unique_sequences = sorted(sequence_targets)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    failed_graphs = 0
    total_unknown_residues = 0
    total_graph_nodes = 0
    total_directed_edges = 0
    successful_lengths = []

    for graph_index, sequence in enumerate(unique_sequences):
        graph_id = f"protein_{graph_index:04d}"
        target_ids = sorted(sequence_targets[sequence])
        graph_path = OUTPUT_DIR / f"{graph_id}.pt"
        try:
            graph = build_residue_graph(graph_id, sequence, target_ids)
            validate_graph(graph)
            torch.save(graph, graph_path)

            sequence_length = len(sequence)
            unknown_count = len(graph["unknown_residue_indices"])
            metadata_rows.append(
                {
                    "graph_id": graph_id,
                    "target_ids": json.dumps(target_ids, ensure_ascii=True),
                    "sequence_length": sequence_length,
                    "num_nodes": graph["num_nodes"],
                    "num_directed_edges": graph["num_directed_edges"],
                    "num_unknown_residues": unknown_count,
                }
            )
            successful_lengths.append(sequence_length)
            total_unknown_residues += unknown_count
            total_graph_nodes += graph["num_nodes"]
            total_directed_edges += graph["num_directed_edges"]
        except Exception as exc:
            failed_graphs += 1
            print(
                f"FAILED {graph_id} target_ids={json.dumps(target_ids)}: "
                f"{type(exc).__name__}: {exc}"
            )

    metadata_columns = [
        "graph_id",
        "target_ids",
        "sequence_length",
        "num_nodes",
        "num_directed_edges",
        "num_unknown_residues",
    ]
    try:
        with METADATA_PATH.open("w", newline="", encoding="utf-8") as metadata_file:
            writer = csv.DictWriter(metadata_file, fieldnames=metadata_columns)
            writer.writeheader()
            writer.writerows(metadata_rows)
    except Exception as exc:
        print(f"FAILED metadata.csv: {type(exc).__name__}: {exc}")
        failed_graphs += 1

    graphs_created = len(metadata_rows)
    average_length = (
        sum(successful_lengths) / len(successful_lengths)
        if successful_lengths
        else 0.0
    )
    min_length = min(successful_lengths) if successful_lengths else 0
    max_length = max(successful_lengths) if successful_lengths else 0

    print("===== Davis 2D Protein Graph Summary =====")
    print(f"Total Davis target IDs: {total_target_ids}")
    print(f"Total unique sequences: {len(unique_sequences)}")
    print(f"Graphs successfully created: {graphs_created}")
    print(f"Failed graphs: {failed_graphs}")
    print(f"Minimum sequence length: {min_length}")
    print(f"Maximum sequence length: {max_length}")
    print(f"Average sequence length: {average_length:.2f}")
    print(f"Total unknown/non-standard residues: {total_unknown_residues}")
    print(f"Total graph nodes: {total_graph_nodes}")
    print(f"Total directed edges: {total_directed_edges}")
    print(f"Graph directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_PATH}")

    consistency_passed = graphs_created == len(unique_sequences)
    print(
        "Consistency check (generated graphs == unique sequences): "
        f"{'PASSED' if consistency_passed else 'FAILED'}"
    )
    if not consistency_passed or failed_graphs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
