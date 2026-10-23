"""
MGNN: Multiscale Graph Neural Network for Drug Molecular Representation.

Faithfully implements the MGraphDTA drug encoder as described in:
  "MGraphDTA: deep multiscale graph neural network for explainable
   drug-target binding affinity prediction"
  Yang et al., Chemical Science, 2022.

Depth accounting: the paper-reported depth counts each DenseLayer as one unit
and excludes conv0. The implementation has 28 top-level stages including conv0
and 52 GraphConv modules because each DenseLayer contains two graph convolutions.
┌────────────────────────────────────────────────────────────────────┐
│  Step                   │  #units       │  Channels out            │
├────────────────────────────────────────────────────────────────────┤
│  Initial conv0          │      1        │  32                      │
│  Dense Block 1 (8 ly.)  │      8        │  32 + 8×32 = 288         │
│  Transition Layer 1     │      1        │  288 // 2 = 144          │
│  Dense Block 2 (8 ly.)  │      8        │  144 + 8×32 = 400        │
│  Transition Layer 2     │      1        │  400 // 2 = 200          │
│  Dense Block 3 (8 ly.)  │      8        │  200 + 8×32 = 456        │
│  Transition Layer 3     │      1        │  456 // 2 = 228          │
│  Global Mean Pool + FC  │      -        │  96  (drug embedding)    │
├────────────────────────────────────────────────────────────────────┤
│  Paper-reported depth   │     27        │  excludes conv0          │
│  Top-level stages       │     28        │  includes conv0           │
│  GraphConv modules      │     52        │  counts bottleneck steps │
└────────────────────────────────────────────────────────────────────┘

Paper-reported depth = 3 * 8 dense-layer units + 3 transitions = 27.
Top-level stages including conv0 = 1 + 3 * 8 + 3 = 28.
GraphConv modules = 1 + 3 * 8 * 2 + 3 = 52.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool


# --------------------------------------------------------------------------- #
#  Constants from the paper                                                   #
# --------------------------------------------------------------------------- #
ATOM_FEATURE_DIM: int = 22      # 22-dim input node features
GROWTH_RATE: int = 32           # k = 32 output channels per dense layer
BN_SIZE: int = 2                # bottleneck expansion: intermediate = 32*2 = 64
NUM_BLOCKS: int = 3             # 3 multiscale blocks
LAYERS_PER_BLOCK: int = 8       # N = 8 dense layers per block
DRUG_EMBEDDING_DIM: int = 96    # y_G output dimension
INITIAL_CHANNELS: int = 32      # channels after conv0


# --------------------------------------------------------------------------- #
#  NodeLevelBatchNorm (Section 5.2)                                           #
# --------------------------------------------------------------------------- #
class NodeLevelBatchNorm(nn.Module):
    """
    Batch normalization over all nodes in a (possibly batched) molecular graph.

    Unlike standard BatchNorm1d which normalizes per-sample, this module
    treats the entire set of nodes across all graphs in the mini-batch as
    a flat population of shape [total_nodes, channels].

    This is essential for variable-size graphs where standard (B, C) batch
    normalization does not apply directly.

    From paper Section 5.2:
        x_hat_v = (x_v - E[x]) / sqrt(Var[x] + eps)
        y_v = gamma * x_hat_v + beta
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        # Delegate to BatchNorm1d — it normalizes over the *first* dimension
        # (num_nodes treated as the batch dimension), which is exactly what we want.
        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node feature tensor of shape [total_nodes, num_features].
        Returns:
            Normalized tensor of same shape.
        """
        return self.bn(x)

    def __repr__(self) -> str:
        return f"NodeLevelBatchNorm({self.bn.num_features})"


# --------------------------------------------------------------------------- #
#  GraphConvBn: Composite layer unit (Section 5.3)                            #
# --------------------------------------------------------------------------- #
class GraphConvBn(nn.Module):
    """
    Single graph convolutional step with BatchNorm and ReLU.

    Implements:
        x_i^(t+1) = ReLU( BN( W1*x_i^(t) + W2 * sum_{j in N(i)} x_j^(t) ) )

    Using PyG's GraphConv (Morris et al., AAAI 2019) which maintains separate
    weight matrices W1 (self) and W2 (neighbors), matching Eq. 1 of the paper.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x         : Node features [total_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
        Returns:
            Updated node features [total_nodes, out_channels]
        """
        return F.relu(self.norm(self.conv(x, edge_index)))

    def __repr__(self) -> str:
        return (f"GraphConvBn(in={self.conv.in_channels}, "
                f"out={self.conv.out_channels})")


# --------------------------------------------------------------------------- #
#  DenseLayer: One layer inside a Multiscale Block (Section 6.2)              #
# --------------------------------------------------------------------------- #
class DenseLayer(nn.Module):
    """
    One bottleneck dense layer inside a Multiscale Block.

    Receives the concatenation of ALL preceding layer outputs in the block
    (including the block's input), applies a 2-step bottleneck convolution:

        conv1: in_features → growth_rate * bn_size   (compression)
        conv2: growth_rate * bn_size → growth_rate   (projection)

    Only the conv2 output (growth_rate channels) is appended to the
    running feature concatenation; the conv1 intermediate is discarded.

    From Section 6.2:
        growth_rate = 32,  bn_size = 2
        Bottleneck intermediate dim = 32 * 2 = 64
    """

    def __init__(self, num_input_features: int, growth_rate: int = GROWTH_RATE,
                 bn_size: int = BN_SIZE):
        super().__init__()
        bottleneck_dim = growth_rate * bn_size  # 64
        self.conv1 = GraphConvBn(num_input_features, bottleneck_dim)
        self.conv2 = GraphConvBn(bottleneck_dim, growth_rate)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x         : Concatenated features from all prior layers [N, num_input_features]
            edge_index: Graph connectivity [2, num_edges]
        Returns:
            New feature output [N, growth_rate] — to be concatenated to x externally
        """
        out = self.conv1(x, edge_index)
        out = self.conv2(out, edge_index)
        return out

    def __repr__(self) -> str:
        return (f"DenseLayer(in={self.conv1.conv.in_channels}, "
                f"bottleneck={self.conv2.conv.in_channels}, "
                f"out={self.conv2.conv.out_channels})")


# --------------------------------------------------------------------------- #
#  MultiscaleBlock: 8 dense layers per block (Section 6.1)                   #
# --------------------------------------------------------------------------- #
class MultiscaleBlock(nn.Module):
    """
    A single Multiscale Dense Block containing N=8 graph convolutional layers.

    Dense connection rule (Eq. 2-3 of paper):
        x^(n) = H( x^(0) || x^(1) || ... || x^(n-1) ; Theta_n )

    Each layer receives all previously produced feature maps as input
    (concatenated along the feature dimension).

    Input channels  : in_channels
    Output channels : in_channels + num_layers * growth_rate
    """

    def __init__(self, in_channels: int, num_layers: int = LAYERS_PER_BLOCK,
                 growth_rate: int = GROWTH_RATE, bn_size: int = BN_SIZE):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Layer i receives input from block-input + i previous dense outputs
            layer_in_channels = in_channels + i * growth_rate
            self.layers.append(
                DenseLayer(layer_in_channels, growth_rate, bn_size)
            )

        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x         : Node features [total_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        Returns:
            Dense-concatenated output [total_nodes, out_channels]
                where out_channels = in_channels + num_layers * growth_rate
        """
        # Start with the block input; accumulate outputs as new columns
        features = [x]
        for layer in self.layers:
            # Concatenate all feature maps seen so far
            cat_x = torch.cat(features, dim=1)
            new_x = layer(cat_x, edge_index)
            features.append(new_x)

        # Return concatenation of all (input + every layer output)
        return torch.cat(features, dim=1)

    def __repr__(self) -> str:
        return (f"MultiscaleBlock(in={self.layers[0].conv1.conv.in_channels}, "
                f"num_layers={self.num_layers}, "
                f"out={self.out_channels})")


# --------------------------------------------------------------------------- #
#  MGNN: Drug encoder (Sections 5-8)                                         #
# --------------------------------------------------------------------------- #
class MGNN(nn.Module):
    """Multiscale graph encoder with configurable block and dense-layer counts.

    Conceptual depth counts each dense unit and each transition once, excludes
    ``conv0``, and is ``num_blocks * (layers_per_block + 1)``. Physical
    GraphConv modules count both convolutions inside each dense unit, plus
    ``conv0`` and one transition per block.
    """

    def __init__(
        self,
        in_channels: int = ATOM_FEATURE_DIM,
        initial_channels: int = INITIAL_CHANNELS,
        growth_rate: int = GROWTH_RATE,
        bn_size: int = BN_SIZE,
        layers_per_block: int = LAYERS_PER_BLOCK,
        drug_embed_dim: int = DRUG_EMBEDDING_DIM,
        num_blocks: int = NUM_BLOCKS,
        hidden_channels: int | None = None,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if layers_per_block < 1:
            raise ValueError("layers_per_block must be positive")

        # Table S3's hidden-channel search is mapped locally to DenseNet's
        # per-layer growth rate: every DenseLayer appends this many channels.
        # The paper does not define the local bottleneck/transition tensors;
        # retain the existing 2x bottleneck and transition compression.
        if hidden_channels is None:
            effective_hidden_channels = growth_rate
        else:
            if growth_rate != GROWTH_RATE and growth_rate != hidden_channels:
                raise ValueError(
                    "growth_rate and hidden_channels specify conflicting widths"
                )
            effective_hidden_channels = hidden_channels
        if (
            not isinstance(effective_hidden_channels, int)
            or isinstance(effective_hidden_channels, bool)
            or effective_hidden_channels < 1
        ):
            raise ValueError("hidden_channels must be a positive integer")

        self.num_blocks = int(num_blocks)
        self.layers_per_block = int(layers_per_block)
        self.hidden_channels = int(effective_hidden_channels)
        self.drug_embed_dim = int(drug_embed_dim)
        self.conv0 = GraphConvBn(in_channels, initial_channels)
        channels = initial_channels

        # Attribute names intentionally match the legacy default state dict:
        # block1/transition1 through block3/transition3 remain unchanged.
        for block_idx in range(1, self.num_blocks + 1):
            block = MultiscaleBlock(
                channels, self.layers_per_block, self.hidden_channels, bn_size
            )
            self.add_module(f"block{block_idx}", block)
            channels = block.out_channels
            transition = GraphConvBn(channels, channels // 2)
            self.add_module(f"transition{block_idx}", transition)
            channels = transition.conv.out_channels

        self.pre_pool_channels = channels
        self.fc = nn.Linear(channels, self.drug_embed_dim)
        self._gcn_depth = self.num_blocks * (self.layers_per_block + 1)
        actual_conceptual_depth = sum(
            getattr(self, f"block{block_idx}").num_layers + 1
            for block_idx in range(1, self.num_blocks + 1)
        )
        assert self._gcn_depth == actual_conceptual_depth, (
            "MGNN conceptual depth accounting mismatch: "
            f"expected {self._gcn_depth}, got {actual_conceptual_depth}"
        )
        expected_graph_conv_modules = 1 + self.num_blocks * (
            2 * self.layers_per_block + 1
        )
        actual_graph_conv_modules = sum(
            isinstance(module, gnn.GraphConv) for module in self.modules()
        )
        assert actual_graph_conv_modules == expected_graph_conv_modules, (
            "MGNN GraphConv module accounting mismatch: "
            f"expected {expected_graph_conv_modules}, got {actual_graph_conv_modules}"
        )

    @property
    def final_transition(self) -> GraphConvBn:
        """Return the final transition, for attribution hooks and inspection."""
        return getattr(self, f"transition{self.num_blocks}")

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        x = self.conv0(x, edge_index)
        for block_idx in range(1, self.num_blocks + 1):
            x = getattr(self, f"block{block_idx}")(x, edge_index)
            x = getattr(self, f"transition{block_idx}")(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.fc(x)

    def describe_architecture(self) -> str:
        """Return conceptual depth, physical module count, and channel sizes."""
        graph_conv_modules = sum(
            isinstance(module, gnn.GraphConv) for module in self.modules()
        )
        lines = [
            "=" * 62,
            "MGNN Architecture Verification",
            "-" * 62,
            f"Input node feature dim : {ATOM_FEATURE_DIM}",
            f"Initial channels       : {self.conv0.conv.out_channels}",
            f"Dense growth channels  : {self.hidden_channels}",
        ]
        for block_idx in range(1, self.num_blocks + 1):
            block = getattr(self, f"block{block_idx}")
            transition = getattr(self, f"transition{block_idx}")
            lines.append(
                f"Block {block_idx}: {block.num_layers} dense units, "
                f"{block.out_channels} -> {transition.conv.out_channels} transition channels"
            )
        lines.extend([
            f"Readout channels       : {self.pre_pool_channels} -> {self.drug_embed_dim}",
            f"Conceptual depth       : {self._gcn_depth}",
            f"Top-level stages       : {1 + self._gcn_depth} (includes conv0)",
            f"GraphConv modules      : {graph_conv_modules}",
            "=" * 62,
        ])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MGNN(in={ATOM_FEATURE_DIM}, blocks={self.num_blocks}, "
            f"layers_per_block={self.layers_per_block}, "
            f"hidden_channels={self.hidden_channels}, depth={self._gcn_depth}, "
            f"embed={self.drug_embed_dim})"
        )
