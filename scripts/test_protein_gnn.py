"""Run an untrained Protein GNN prototype on two Davis residue graphs."""

import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from protein_gnn import PROTEIN_EMBEDDING_DIM, ProteinGNN

GRAPH_DIR = PROJECT_ROOT / "results/davis/protein_graphs_2d"
GRAPH_FILES = ("protein_0000.pt", "protein_0378.pt")


def load_graph(filename: str) -> dict[str, Any]:
    path = GRAPH_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Residue graph not found: {path}")
    graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, dict):
        raise TypeError(f"Expected a graph dictionary in {path}")
    return graph


def run_graph(model: ProteinGNN, graph: dict[str, Any]) -> Tensor:
    features = graph["x"]
    edge_index = graph["edge_index"]
    if features.shape != (graph["num_nodes"], 20):
        raise ValueError(
            f"{graph['graph_id']}: invalid node feature shape {tuple(features.shape)}"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"{graph['graph_id']}: invalid edge_index shape {tuple(edge_index.shape)}"
        )

    with torch.no_grad():
        embedding = model(features, edge_index)
    if embedding.shape != (1, PROTEIN_EMBEDDING_DIM):
        raise RuntimeError(
            f"{graph['graph_id']}: expected output shape [1, {PROTEIN_EMBEDDING_DIM}], "
            f"got {tuple(embedding.shape)}"
        )
    return embedding


def main() -> None:
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)
    model = ProteinGNN().eval()

    graph1 = load_graph(GRAPH_FILES[0])
    graph2 = load_graph(GRAPH_FILES[1])
    if graph1["num_nodes"] == graph2["num_nodes"]:
        raise RuntimeError("Test graphs must have different numbers of residues")

    embedding1 = run_graph(model, graph1)
    embedding2 = run_graph(model, graph2)

    print("===== Protein GNN Prototype =====")
    print("Graph 1:")
    print(f"  File: {GRAPH_FILES[0]}")
    print(f"  Target IDs: {', '.join(graph1['target_ids'])}")
    print(f"  Input node feature shape: {list(graph1['x'].shape)}")
    print(f"  Input edge_index shape: {list(graph1['edge_index'].shape)}")
    print(f"  Number of residues: {graph1['num_nodes']}")
    print(f"  Output embedding shape: {list(embedding1.shape)}")
    print(f"  Output embedding: {embedding1.squeeze(0)}")

    print("Graph 2:")
    print(f"  File: {GRAPH_FILES[1]}")
    print(f"  Target IDs: {', '.join(graph2['target_ids'])}")
    print(f"  Input node feature shape: {list(graph2['x'].shape)}")
    print(f"  Input edge_index shape: {list(graph2['edge_index'].shape)}")
    print(f"  Number of residues: {graph2['num_nodes']}")
    print(f"  Output embedding shape: {list(embedding2.shape)}")

    print(f"Protein embedding dimension: {PROTEIN_EMBEDDING_DIM}")
    print("Prototype validation: PASSED")


if __name__ == "__main__":
    main()
