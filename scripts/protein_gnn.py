"""Standalone two-layer Protein GNN for 2D residue graphs."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

INPUT_FEATURE_DIM = 20
HIDDEN_DIM = 64
PROTEIN_EMBEDDING_DIM = 96


class ProteinGNN(nn.Module):
    """Encode one or more residue graphs into 96-dimensional embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConv(INPUT_FEATURE_DIM, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, PROTEIN_EMBEDDING_DIM)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 2 or x.shape[1] != INPUT_FEATURE_DIM:
            raise ValueError(
                f"Expected x with shape [num_residues, {INPUT_FEATURE_DIM}], "
                f"got {tuple(x.shape)}"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Expected edge_index with shape [2, num_edges], "
                f"got {tuple(edge_index.shape)}"
            )
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        elif batch.ndim != 1 or batch.shape[0] != x.shape[0]:
            raise ValueError("batch must contain one graph index per node")

        node_embeddings = F.relu(self.conv1(x, edge_index))
        node_embeddings = F.relu(self.conv2(node_embeddings, edge_index))
        return global_mean_pool(node_embeddings, batch)
