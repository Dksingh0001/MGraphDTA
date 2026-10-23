"""Integrated MGraphDTA model for drug-target affinity prediction."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from src.models.mcnn import MCNN, MCNN_BRANCH_COUNT
from src.models.mgnn import GROWTH_RATE, MGNN, LAYERS_PER_BLOCK, NUM_BLOCKS


class MGraphDTA(nn.Module):
    """Combine molecular and protein encoders with the regression head."""

    def __init__(
        self,
        vocab_size: int = 26,
        *,
        num_blocks: int = NUM_BLOCKS,
        layers_per_block: int = LAYERS_PER_BLOCK,
        mcnn_branch_count: int = MCNN_BRANCH_COUNT,
        hidden_channels: int = GROWTH_RATE,
    ) -> None:
        super().__init__()
        self.drug_encoder = MGNN(
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            hidden_channels=hidden_channels,
        )
        self.protein_encoder = MCNN(
            vocab_size=vocab_size,
            branch_count=mcnn_branch_count,
        )
        classifier_input_dim = (
            self.drug_encoder.drug_embed_dim + self.protein_encoder.output_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
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
