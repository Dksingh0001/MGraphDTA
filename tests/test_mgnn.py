"""
tests/test_mgnn.py
Dedicated test suite for the MGNN (Multiscale Graph Neural Network) component.

Verifies:
1. Architecture depth = 27 graph-convolutional layers (paper definition)
2. Correct channel dimensions after each block / transition
3. Forward pass on molecules with different atom counts
4. Output shape = [batch_size, 96]
5. All output values are finite (no NaN / Inf)
6. Batch processing of multiple molecules
7. GPU forwarding when available
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.preprocessing import smiles_to_mol, extract_graph_structure
from src.models.mgnn import (
    MGNN,
    GraphConvBn,
    DenseLayer,
    MultiscaleBlock,
    NodeLevelBatchNorm,
    ATOM_FEATURE_DIM,
    GROWTH_RATE,
    LAYERS_PER_BLOCK,
    NUM_BLOCKS,
    DRUG_EMBEDDING_DIM,
    INITIAL_CHANNELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5 Real Davis molecules with different sizes
# ─────────────────────────────────────────────────────────────────────────────
TEST_MOLECULES = [
    # (label, SMILES)
    ("Staurosporine-like (27 atoms)",
     "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"),
    ("Imatinib (28 atoms)",
     "CN1CCN(CC1)C2=CC=C(C=C2)C(=O)NC3=CC(=NC=C3)C4=CN=CC=C4"),
    ("Erlotinib (29 atoms)",
     "COCCOC1=C(OCCOC)C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C"),
    ("Gefitinib (31 atoms)",
     "COC1=CC2=C(C=C1OCCCN3CCOCC3)NC=NC2=NC4=CC(=C(C=C4)Cl)F"),
    ("Large (40 atoms)",
     "CC(C)(C)C1=CC(=NO1)NC(=O)NC2=CC=C(C=C2)C3=CN4C5=C(C=C(C=C5)"
     "OCCN6CCOCC6)SC4=N3"),
]


def _smiles_to_pyg(smiles: str) -> Data:
    """Convert SMILES → PyG Data with 22-dim node features."""
    mol = smiles_to_mol(smiles)
    graph = extract_graph_structure(mol)
    return Data(x=graph["x"], edge_index=graph["edge_index"])


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Architecture depth verification
# ─────────────────────────────────────────────────────────────────────────────
def test_architecture_depth():
    print("=" * 62)
    print("TEST 1: Architecture Depth Verification")
    print("=" * 62)

    model = MGNN()

    blocks = (model.block1, model.block2, model.block3)
    dense_layer_count = sum(block.num_layers for block in blocks)
    transition_count = NUM_BLOCKS
    paper_reported_depth = dense_layer_count + transition_count
    top_level_stage_count = 1 + paper_reported_depth
    graph_conv_module_count = sum(
        isinstance(module, GraphConvBn) for module in model.modules()
    )
    expected_graph_conv_module_count = (
        1 + 2 * dense_layer_count + transition_count
    )

    print(f"\nMGNN Architecture Verification")
    print(f"  DenseLayer units in blocks : {dense_layer_count}")
    print(f"  Transition layers          : {transition_count}")
    print(f"  Paper-reported depth       : {paper_reported_depth}")
    print(f"  Top-level stages incl conv0: {top_level_stage_count}")
    print(f"  GraphConv modules          : {graph_conv_module_count}")

    assert dense_layer_count == NUM_BLOCKS * LAYERS_PER_BLOCK
    assert paper_reported_depth == 27, (
        f"Expected paper-reported depth 27, got {paper_reported_depth}"
    )
    assert top_level_stage_count == 28, (
        f"Expected 28 top-level stages including conv0, got {top_level_stage_count}"
    )
    assert graph_conv_module_count == expected_graph_conv_module_count == 52, (
        f"Expected 52 GraphConv modules, got {graph_conv_module_count}"
    )
    assert model._gcn_depth == 27, \
        f"model._gcn_depth = {model._gcn_depth}, expected 27"

    print(f"\n  → MGNN depth accounting PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Channel dimensions after each stage
# ─────────────────────────────────────────────────────────────────────────────
def test_channel_dimensions():
    print("\n" + "=" * 62)
    print("TEST 2: Channel Dimensions After Each Stage")
    print("=" * 62)

    model = MGNN()

    # Expected channel counts (derived analytically from paper spec)
    expected = {
        "conv0_out": INITIAL_CHANNELS,                                # 32
        "block1_out": INITIAL_CHANNELS + LAYERS_PER_BLOCK * GROWTH_RATE,  # 288
        "transition1_out": (INITIAL_CHANNELS + LAYERS_PER_BLOCK * GROWTH_RATE) // 2,  # 144
    }
    t1_out = expected["transition1_out"]  # 144
    expected["block2_out"] = t1_out + LAYERS_PER_BLOCK * GROWTH_RATE       # 400
    expected["transition2_out"] = expected["block2_out"] // 2              # 200
    t2_out = expected["transition2_out"]
    expected["block3_out"] = t2_out + LAYERS_PER_BLOCK * GROWTH_RATE       # 456
    expected["transition3_out"] = expected["block3_out"] // 2              # 228

    actual = {
        "conv0_out":       model.conv0.conv.out_channels,
        "block1_out":      model.block1.out_channels,
        "transition1_out": model.transition1.conv.out_channels,
        "block2_out":      model.block2.out_channels,
        "transition2_out": model.transition2.conv.out_channels,
        "block3_out":      model.block3.out_channels,
        "transition3_out": model.transition3.conv.out_channels,
    }

    labels = [
        ("Input node features", None, ATOM_FEATURE_DIM),
        ("conv0 output",        "conv0_out",       expected["conv0_out"]),
        ("Block 1 output",      "block1_out",      expected["block1_out"]),
        ("Transition 1 output", "transition1_out", expected["transition1_out"]),
        ("Block 2 output",      "block2_out",      expected["block2_out"]),
        ("Transition 2 output", "transition2_out", expected["transition2_out"]),
        ("Block 3 output",      "block3_out",      expected["block3_out"]),
        ("Transition 3 output", "transition3_out", expected["transition3_out"]),
        ("Drug embedding (FC)", None,              DRUG_EMBEDDING_DIM),
    ]

    print(f"\n  {'Stage':<25} {'Expected':>10} {'Actual':>10} {'Status':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")

    for name, key, exp_val in labels:
        if key is None:
            # Not a measurable stage in model attrs — just print spec
            print(f"  {name:<25} {exp_val:>10} {'(spec)':>10} {'✓':>8}")
        else:
            act_val = actual[key]
            status = "✓" if act_val == exp_val else "✗ FAIL"
            print(f"  {name:<25} {exp_val:>10} {act_val:>10} {status:>8}")
            assert act_val == exp_val, \
                f"{name}: expected {exp_val}, got {act_val}"

    print(f"\n  → All channel dimensions correct  PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Forward pass on individual molecules
# ─────────────────────────────────────────────────────────────────────────────
def test_single_molecule_forward():
    print("\n" + "=" * 62)
    print("TEST 3: Forward Pass — 5 Molecules (Different Atom Counts)")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGNN().to(device)
    model.eval()

    print(f"\n  Running on: {device}")
    print(f"\n  {'#':<4} {'Label':<30} {'Atoms':>6} {'Edges':>7} "
          f"{'Input shape':>14} {'Output shape':>14} {'Finite':>7} {'Status':>7}")
    print(f"  {'─'*4} {'─'*30} {'─'*6} {'─'*7} {'─'*14} {'─'*14} {'─'*7} {'─'*7}")

    for i, (label, smiles) in enumerate(TEST_MOLECULES):
        data = _smiles_to_pyg(smiles).to(device)
        num_atoms = data.x.size(0)
        num_edges = data.edge_index.size(1)

        with torch.no_grad():
            output = model(data)

        in_shape  = f"[{num_atoms}, 22]"
        out_shape = list(output.shape)
        finite    = torch.isfinite(output).all().item()

        assert output.ndim == 2,           f"Sample {i+1}: output should be 2D"
        assert output.shape[0] == 1,       f"Sample {i+1}: batch dim should be 1"
        assert output.shape[1] == DRUG_EMBEDDING_DIM, \
            f"Sample {i+1}: embed dim should be {DRUG_EMBEDDING_DIM}, got {output.shape[1]}"
        assert finite, f"Sample {i+1}: output contains NaN/Inf"

        status = "PASSED"
        print(f"  {i+1:<4} {label:<30} {num_atoms:>6} {num_edges:>7} "
              f"{in_shape:>14} {str(out_shape):>14} {str(finite):>7} {status:>7}")

    print(f"\n  → All single-molecule forward passes PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Batch processing of multiple molecules
# ─────────────────────────────────────────────────────────────────────────────
def test_batch_processing():
    print("\n" + "=" * 62)
    print("TEST 4: Batch Processing")
    print("=" * 62)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGNN().to(device)
    model.eval()

    # Build a batch from all 5 test molecules
    data_list = [_smiles_to_pyg(smi) for _, smi in TEST_MOLECULES]
    batch = Batch.from_data_list(data_list).to(device)

    batch_size = len(TEST_MOLECULES)  # 5

    print(f"\n  Batch size           : {batch_size}")
    print(f"  Total nodes in batch : {batch.x.size(0)}")
    print(f"  Total edges in batch : {batch.edge_index.size(1)}")

    with torch.no_grad():
        output = model(batch)

    assert output.ndim == 2,              f"Batch output should be 2D"
    assert output.shape[0] == batch_size, \
        f"Expected {batch_size} rows, got {output.shape[0]}"
    assert output.shape[1] == DRUG_EMBEDDING_DIM, \
        f"Expected {DRUG_EMBEDDING_DIM} cols, got {output.shape[1]}"
    assert torch.isfinite(output).all(), "Batch output contains NaN/Inf"

    print(f"\n  Output shape         : {list(output.shape)}")
    print(f"  Expected shape       : [{batch_size}, {DRUG_EMBEDDING_DIM}]")
    print(f"  All values finite    : {torch.isfinite(output).all().item()}")
    print(f"\n  → Batch processing PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Detailed architecture description (from model.describe_architecture)
# ─────────────────────────────────────────────────────────────────────────────
def test_architecture_description():
    print("\n" + "=" * 62)
    print("TEST 5: Architecture Description from Model")
    print("=" * 62)
    model = MGNN()
    desc = model.describe_architecture()
    print(desc)
    assert "27" in desc, "Architecture description must mention depth 27"
    assert "96"  in desc, "Architecture description must mention drug embed dim 96"
    print(f"\n  → Architecture description PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Real Davis compounds (first 5 from ligands_can.txt)
# ─────────────────────────────────────────────────────────────────────────────
def test_real_davis_molecules():
    raw_path = os.path.join("data", "raw", "davis", "ligands_can.txt")
    if not os.path.exists(raw_path):
        print("\nSkipping real Davis test (file not found).")
        return

    print("\n" + "=" * 62)
    print("TEST 6: Forward Pass on Real Davis Compounds")
    print("=" * 62)

    with open(raw_path) as f:
        ligands = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGNN().to(device)
    model.eval()

    smiles_list = list(ligands.values())[:5]
    data_list = [_smiles_to_pyg(smi) for smi in smiles_list]
    batch = Batch.from_data_list(data_list).to(device)

    with torch.no_grad():
        output = model(batch)

    assert output.shape == (5, DRUG_EMBEDDING_DIM)
    assert torch.isfinite(output).all()

    print(f"\n  Batch output shape : {list(output.shape)}")
    print(f"  All values finite  : {torch.isfinite(output).all().item()}")
    print(f"\n  → Real Davis forward pass PASSED")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#" * 62)
    print("  MGNN FULL TEST SUITE")
    print("#" * 62)

    test_architecture_depth()
    test_channel_dimensions()
    test_single_molecule_forward()
    test_batch_processing()
    test_architecture_description()
    test_real_davis_molecules()

    print("\n" + "#" * 62)
    print("  ALL MGNN TESTS PASSED")
    print("#" * 62 + "\n")
