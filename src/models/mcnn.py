"""Multiscale CNN protein encoder used by MGraphDTA."""

from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn


PROTEIN_VOCAB_SIZE = 26
PROTEIN_EMBEDDING_DIM = 128
MCNN_BRANCH_CHANNELS = 96
MCNN_BRANCH_COUNT = 3
MCNN_KERNEL_SIZE = 3


class Conv1dReLU(nn.Module):
    """A one-dimensional convolution followed by ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = MCNN_KERNEL_SIZE,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x))


class StackCNN(nn.Module):
    """A stacked 3-wide Conv1D branch followed by global max pooling."""

    def __init__(
        self,
        layer_num: int,
        in_channels: int,
        out_channels: int = MCNN_BRANCH_CHANNELS,
        kernel_size: int = MCNN_KERNEL_SIZE,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        layers = OrderedDict()
        layers["conv_layer0"] = Conv1dReLU(
            in_channels, out_channels, kernel_size, stride, padding
        )
        for layer_idx in range(layer_num - 1):
            layers[f"conv_layer{layer_idx + 1}"] = Conv1dReLU(
                out_channels, out_channels, kernel_size, stride, padding
            )

        self.conv_layers = nn.Sequential(layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.conv_layers(x)).squeeze(-1)


class MCNN(nn.Module):
    """Encode padded protein token batches as 96-dimensional vectors."""

    def __init__(self, vocab_size: int = PROTEIN_VOCAB_SIZE) -> None:
        super().__init__()
        self.embed = nn.Embedding(
            vocab_size, PROTEIN_EMBEDDING_DIM, padding_idx=0
        )
        self.block_list = nn.ModuleList(
            [
                StackCNN(
                    layer_num=branch_idx + 1,
                    in_channels=PROTEIN_EMBEDDING_DIM,
                )
                for branch_idx in range(MCNN_BRANCH_COUNT)
            ]
        )
        self.linear = nn.Linear(
            MCNN_BRANCH_COUNT * MCNN_BRANCH_CHANNELS,
            MCNN_BRANCH_CHANNELS,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: Protein token IDs with shape [batch_size, sequence_length].
        Returns:
            Protein embeddings with shape [batch_size, 96].
        """
        embedded = self.embed(tokens).permute(0, 2, 1)
        branch_features = [branch(embedded) for branch in self.block_list]
        combined = torch.cat(branch_features, dim=1)
        return self.linear(combined)
