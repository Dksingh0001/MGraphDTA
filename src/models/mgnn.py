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
    """
    Multiscale Graph Neural Network (MGNN) — Drug Molecular Encoder.

    Architecture (paper-reported depth 27; top-level stage count 28):

        conv0         : GraphConvBn(22 → 32)          [1 stage; excluded from 27]
        block1        : MultiscaleBlock(32, 8 layers)  [8 DenseLayer units]
            → output : 32 + 8*32 = 288 channels
        transition1   : GraphConvBn(288 → 144)         [1 stage]
        block2        : MultiscaleBlock(144, 8 layers) [8 DenseLayer units]
            → output : 144 + 8*32 = 400 channels
        transition2   : GraphConvBn(400 → 200)         [1 stage]
        block3        : MultiscaleBlock(200, 8 layers) [8 DenseLayer units]
            → output : 200 + 8*32 = 456 channels
        transition3   : GraphConvBn(456 → 228)         [1 stage]
        ─────────────────────────────────────────────────────────
        Paper-reported depth = 3*8 + 3 = 27 (excludes conv0)
        Top-level stages = 1 + 3*8 + 3 = 28 (includes conv0)
        GraphConv modules = 1 + 3*8*2 + 3 = 52 (two per DenseLayer)

        Global Mean Pooling → Linear(228 → 96)
        Output: y_G ∈ R^{96}  (drug embedding)

    Args:
        in_channels     : Input node feature dimension (default 22).
        initial_channels: Channels produced by conv0 (default 32).
        growth_rate     : k, new channels per dense layer (default 32).
        bn_size         : Bottleneck factor (default 2 → 64 intermediate).
        layers_per_block: Dense layers per block (default 8).
        drug_embed_dim  : Final drug embedding size (default 96).
    """

    def __init__(
        self,
        in_channels: int = ATOM_FEATURE_DIM,
        initial_channels: int = INITIAL_CHANNELS,
        growth_rate: int = GROWTH_RATE,
        bn_size: int = BN_SIZE,
        layers_per_block: int = LAYERS_PER_BLOCK,
        drug_embed_dim: int = DRUG_EMBEDDING_DIM,
    ):
        super().__init__()

        self.drug_embed_dim = drug_embed_dim

        # ── Initial convolution (depth 1) ──────────────────────────────────
        self.conv0 = GraphConvBn(in_channels, initial_channels)
        ch = initial_channels  # 32

        # ── Dense Block 1 (depth 8) ────────────────────────────────────────
        self.block1 = MultiscaleBlock(ch, layers_per_block, growth_rate, bn_size)
        ch = self.block1.out_channels  # 32 + 8*32 = 288

        # ── Transition Layer 1 (depth 1): halve channels ──────────────────
        ch_t1 = ch // 2  # 144
        self.transition1 = GraphConvBn(ch, ch_t1)
        ch = ch_t1  # 144

        # ── Dense Block 2 (depth 8) ────────────────────────────────────────
        self.block2 = MultiscaleBlock(ch, layers_per_block, growth_rate, bn_size)
        ch = self.block2.out_channels  # 144 + 8*32 = 400

        # ── Transition Layer 2 (depth 1): halve channels ──────────────────
        ch_t2 = ch // 2  # 200
        self.transition2 = GraphConvBn(ch, ch_t2)
        ch = ch_t2  # 200

        # ── Dense Block 3 (depth 8) ────────────────────────────────────────
        self.block3 = MultiscaleBlock(ch, layers_per_block, growth_rate, bn_size)
        ch = self.block3.out_channels  # 200 + 8*32 = 456

        # ── Transition Layer 3 (depth 1): halve channels ──────────────────
        ch_t3 = ch // 2  # 228
        self.transition3 = GraphConvBn(ch, ch_t3)
        ch = ch_t3  # 228

        # Store the channel dimension entering the readout FC
        self.pre_pool_channels = ch  # 228

        # ── Readout: Global Mean Pool + Linear (Section 8.3) ──────────────
        self.fc = nn.Linear(ch, drug_embed_dim)  # 228 → 96

        # ── Depth accounting (paper definition, Section 8.2) ────────────
        #
        # The paper reports 27 layers as 24 DenseLayer units plus 3
        # transitions, excluding conv0. Each DenseLayer contains two
        # GraphConv operations, so this is not a literal module count.
        # The complete implementation has 28 top-level stages and 52
        # GraphConv modules when conv0 and both bottleneck operations count.
        self._gcn_depth = (
            layers_per_block            # block1 (8)
            + 1                         # transition1
            + layers_per_block          # block2 (8)
            + 1                         # transition2
            + layers_per_block          # block3 (8)
            + 1                         # transition3
        )  # = 3×8 + 3 = 27
        assert self._gcn_depth == 27, (
            f"GCN depth must be 27 per paper formula 3×8+3; got {self._gcn_depth}"
        )

    # ---------------------------------------------------------------------- #
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass through the complete MGNN and its mean-pooling readout.

        Args:
            data: PyG Data object with:
                  - data.x         : [total_nodes, 22]  node features
                  - data.edge_index: [2, num_edges]      bond connectivity
                  - data.batch     : [total_nodes]       graph membership
                                     (None for single graphs, auto-set by Batch)

        Returns:
            y_G: Drug embedding tensor [batch_size, 96]
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # conv0 ── depth 1
        x = self.conv0(x, edge_index)

        # block1 ── depth 8
        x = self.block1(x, edge_index)

        # transition1 ── depth 1
        x = self.transition1(x, edge_index)

        # block2 ── depth 8
        x = self.block2(x, edge_index)

        # transition2 ── depth 1
        x = self.transition2(x, edge_index)

        # block3 ── depth 8
        x = self.block3(x, edge_index)

        # transition3 ── depth 1
        x = self.transition3(x, edge_index)

        # Global mean pooling: [total_nodes, 228] → [batch_size, 228]
        x = global_mean_pool(x, batch)

        # Linear projection: [batch_size, 228] → [batch_size, 96]
        y_g = self.fc(x)

        return y_g

    # ---------------------------------------------------------------------- #
    def describe_architecture(self) -> str:
        """Returns a formatted architecture summary string."""
        t1_in  = self.block1.out_channels        # 288
        t1_out = t1_in // 2                      # 144
        t2_in  = self.block2.out_channels        # 400
        t2_out = t2_in // 2                      # 200
        t3_in  = self.block3.out_channels        # 456
        t3_out = t3_in // 2                      # 228
        lines = [
            "=" * 62,
            "MGNN Architecture Verification",
            "-" * 62,
            f"  Input node feature dim : {ATOM_FEATURE_DIM}",
            f"  Initial convolution    : 1  ({ATOM_FEATURE_DIM} -> {INITIAL_CHANNELS})",
            f"  [conv0 is excluded from paper-reported depth]",
            f"  Dense Block 1          : 8  ({INITIAL_CHANNELS} -> {self.block1.out_channels})",
            f"  Transition Layer 1     : 1  ({t1_in} -> {t1_out})",
            f"  Dense Block 2          : 8  ({t1_out} -> {t2_in})",
            f"  Transition Layer 2     : 1  ({t2_in} -> {t2_out})",
            f"  Dense Block 3          : 8  ({t2_out} -> {t3_in})",
            f"  Transition Layer 3     : 1  ({t3_in} -> {t3_out})",
            f"  Global Mean Pool + FC  : {t3_out} -> {self.drug_embed_dim}",
            "-" * 62,
            f"  Paper-reported depth   : {self._gcn_depth} (24 DenseLayers + 3 transitions)",
            f"  Top-level stages       : {1 + self._gcn_depth} (includes conv0)",
            f"  GraphConv modules      : {sum(isinstance(module, gnn.GraphConv) for module in self.modules())}",
            f"  Drug embedding dim     : {self.drug_embed_dim}",
            "=" * 62,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MGNN(in={ATOM_FEATURE_DIM}, depth={self._gcn_depth}, "
                f"embed={self.drug_embed_dim})")
