"""End-to-end forward-pass tests for the integrated MGraphDTA model."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.data import Batch

from src.dataset import DavisDataset
from src.models.mgraphdta import MGraphDTA


def _record_output(shapes, name):
    def hook(_module, _inputs, output):
        shapes[name] = tuple(output.shape)

    return hook


def _record_input(shapes, name):
    def hook(_module, inputs):
        shapes[name] = tuple(inputs[0].shape)

    return hook


def test_model_construction_and_davis_forward():
    torch.manual_seed(17)
    dataset = DavisDataset(split="all")
    samples = [dataset[index] for index in range(2)]
    data_list = [sample["pyg_data"] for sample in samples]

    assert data_list[0].x.shape == (samples[0]["num_atoms"], 22)
    assert data_list[0].edge_index.shape == (2, samples[0]["num_bonds"] * 2)
    assert data_list[0].target.shape == (1, 1200)
    assert data_list[0].y.shape == (1,)

    model = MGraphDTA().eval()
    shapes = {}
    handles = [
        model.drug_encoder.conv0.register_forward_pre_hook(
            _record_input(shapes, "drug_input")
        ),
        model.drug_encoder.fc.register_forward_hook(
            _record_output(shapes, "drug_embedding")
        ),
        model.protein_encoder.linear.register_forward_hook(
            _record_output(shapes, "protein_embedding")
        ),
        model.classifier.register_forward_pre_hook(
            _record_input(shapes, "fused_embedding")
        ),
    ]
    try:
        with torch.no_grad():
            single_prediction = model(data_list[0])
            batch = Batch.from_data_list(data_list)
            assert batch.target.shape == (2, 1200)
            assert batch.x.shape[1] == 22
            assert batch.edge_index.max().item() < batch.x.shape[0]
            batch_prediction = model(batch)
    finally:
        for handle in handles:
            handle.remove()

    assert single_prediction.shape == (1, 1)
    assert batch_prediction.shape == (2, 1)
    assert shapes["drug_input"][1] == 22
    assert shapes["drug_embedding"] == (2, 96)
    assert shapes["protein_embedding"] == (2, 96)
    assert shapes["fused_embedding"] == (2, 192)
    assert torch.isfinite(single_prediction).all()
    assert torch.isfinite(batch_prediction).all()

    model.train()
    batch_prediction = model(batch)
    batch_prediction.square().mean().backward()
    for module in (
        model.drug_encoder.conv0,
        model.protein_encoder.embed,
        model.classifier[0],
    ):
        parameter = next(module.parameters())
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_cuda_forward_when_available():
    if not torch.cuda.is_available():
        print("Skipping MGraphDTA CUDA test (CUDA is unavailable).")
        return

    dataset = DavisDataset(split="all")
    data_list = [dataset[index]["pyg_data"] for index in range(2)]
    batch = Batch.from_data_list(data_list).to("cuda")
    model = MGraphDTA().to("cuda").eval()

    with torch.no_grad():
        prediction = model(batch)

    assert prediction.shape == (2, 1)
    assert torch.isfinite(prediction).all()


if __name__ == "__main__":
    test_model_construction_and_davis_forward()
    test_cuda_forward_when_available()
    print("All MGraphDTA integration tests passed.")