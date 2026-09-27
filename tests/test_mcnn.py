"""Tests for the MGraphDTA multiscale CNN protein encoder."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from src.models.mcnn import MCNN
from src.preprocessing import MAX_PROTEIN_LEN, preprocess_protein


def _record_output(shapes, name):
    def hook(_module, _inputs, output):
        shapes[name] = tuple(output.shape)

    return hook


def _record_input(shapes, name):
    def hook(_module, inputs):
        shapes[name] = tuple(inputs[0].shape)

    return hook


def test_model_and_intermediate_shapes():
    torch.manual_seed(7)
    model = MCNN().eval()
    batch_size = 2
    tokens = torch.arange(batch_size * MAX_PROTEIN_LEN).reshape(
        batch_size, MAX_PROTEIN_LEN
    ) % 26
    shapes = {}
    handles = [model.embed.register_forward_hook(_record_output(shapes, "embedding"))]
    handles.append(
        model.block_list[0].conv_layers[0].register_forward_pre_hook(
            _record_input(shapes, "conv_input")
        )
    )

    for branch_idx, branch in enumerate(model.block_list):
        handles.append(
            branch.conv_layers.register_forward_hook(
                _record_output(shapes, f"branch_{branch_idx + 1}")
            )
        )
        handles.append(
            branch.pool.register_forward_hook(
                _record_output(shapes, f"pool_{branch_idx + 1}")
            )
        )
        handles.append(
            branch.register_forward_hook(
                _record_output(shapes, f"pooled_branch_{branch_idx + 1}")
            )
        )

    handles.append(
        model.linear.register_forward_pre_hook(
            _record_input(shapes, "concatenated")
        )
    )
    try:
        output = model(tokens)
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(model, MCNN)
    assert tokens.shape == (batch_size, MAX_PROTEIN_LEN)
    assert shapes["embedding"] == (batch_size, 1200, 128)
    assert shapes["conv_input"] == (batch_size, 128, 1200)
    assert [len(branch.conv_layers) for branch in model.block_list] == [1, 2, 3]
    assert [shapes[f"branch_{idx}"] for idx in range(1, 4)] == [
        (batch_size, 96, 1198),
        (batch_size, 96, 1196),
        (batch_size, 96, 1194),
    ]
    assert [shapes[f"pool_{idx}"] for idx in range(1, 4)] == [
        (batch_size, 96, 1),
        (batch_size, 96, 1),
        (batch_size, 96, 1),
    ]
    assert [shapes[f"pooled_branch_{idx}"] for idx in range(1, 4)] == [
        (batch_size, 96),
        (batch_size, 96),
        (batch_size, 96),
    ]
    assert shapes["concatenated"] == (batch_size, 288)
    assert output.shape == (batch_size, 96)
    assert torch.isfinite(output).all()

    for branch in model.block_list:
        for layer in branch.conv_layers:
            conv = layer.conv
            assert conv.kernel_size == (3,)
            assert conv.stride == (1,)
            assert conv.padding == (0,)
            assert conv.out_channels == 96


def test_variable_sequence_lengths_after_preprocessing():
    short_tokens, short_length = preprocess_protein("ACD")
    exact_tokens, exact_length = preprocess_protein("A" * MAX_PROTEIN_LEN)
    long_tokens, long_length = preprocess_protein("A" * (MAX_PROTEIN_LEN + 1))

    assert short_length == 3
    assert exact_length == MAX_PROTEIN_LEN
    assert long_length == MAX_PROTEIN_LEN + 1
    assert short_tokens.shape == exact_tokens.shape == long_tokens.shape == (1200,)
    assert torch.equal(short_tokens[3:], torch.zeros(MAX_PROTEIN_LEN - 3, dtype=torch.long))
    assert torch.equal(long_tokens, exact_tokens)

    batch = torch.stack((short_tokens, exact_tokens, long_tokens))
    output = MCNN()(batch)
    assert output.shape == (3, 96)
    assert torch.isfinite(output).all()


def test_backward_produces_gradients():
    torch.manual_seed(11)
    model = MCNN().train()
    tokens = (torch.arange(2 * MAX_PROTEIN_LEN).reshape(2, -1) % 25) + 1
    output = model(tokens)
    output.square().mean().backward()

    parameters = [model.embed.weight, model.linear.weight]
    parameters.extend(
        layer.conv.weight
        for branch in model.block_list
        for layer in branch.conv_layers
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.embed.weight.grad.abs().sum() > 0
    assert model.linear.weight.grad.abs().sum() > 0


def test_cuda_forward_when_available():
    if not torch.cuda.is_available():
        print("Skipping MCNN CUDA test (CUDA is unavailable).")
        return

    model = MCNN().to("cuda").eval()
    tokens = torch.arange(2 * MAX_PROTEIN_LEN, device="cuda").reshape(2, -1) % 26
    with torch.no_grad():
        output = model(tokens)

    assert output.shape == (2, 96)
    assert torch.isfinite(output).all()


if __name__ == "__main__":
    test_model_and_intermediate_shapes()
    test_variable_sequence_lengths_after_preprocessing()
    test_backward_produces_gradients()
    test_cuda_forward_when_available()
    print("All MCNN tests passed.")