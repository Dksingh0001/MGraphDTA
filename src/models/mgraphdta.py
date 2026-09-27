"""Integrated MGraphDTA model for drug-target affinity prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from src.models.mcnn import MCNN
from src.models.mgnn import MGNN


class MGraphDTA(nn.Module):
    """Combine molecular and protein encoders with the regression head."""

    def __init__(self, vocab_size: int = 26) -> None:
        super().__init__()
        self.drug_encoder = MGNN()
        self.protein_encoder = MCNN(vocab_size=vocab_size)
        self.classifier = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data: PyG Data/Batch with x, edge_index, target, and (for batches) batch.
                 x has shape [total_nodes, 22], target has shape [B, 1200].
        Returns:
            Predicted affinity with shape [B, 1].
        """
        protein_tokens = data.target
        if protein_tokens.ndim == 1:
            protein_tokens = protein_tokens.unsqueeze(0)

        drug_embedding = self.drug_encoder(data)
        protein_embedding = self.protein_encoder(protein_tokens)
        combined = torch.cat((protein_embedding, drug_embedding), dim=1)
        return self.classifier(combined)
