"""End-to-end forward-pass tests for the integrated MGraphDTA model."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphConv

from src.dataset import DavisDataset
from src.models.mgraphdta import MGraphDTA
from src.scaled_features import DEFAULT_DATASET_DIR
from src.train import make_dataloader


def _record_output(shapes, name):
    def hook(_module, _inputs, output):
        shapes[name] = tuple(output.shape)

    return hook


def _record_input(shapes, name):
    def hook(_module, inputs):
        shapes[name] = tuple(inputs[0].shape)

    return hook


class SyntheticGraphProteinRows(Dataset):
    """In-memory Davis-shaped samples without loading any benchmark labels."""

    def __init__(self):
        self.samples = [
            Data(
                x=torch.linspace(0.0, 1.0, nodes * 22).reshape(nodes, 22),
                edge_index=torch.tensor(
                    [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
                    dtype=torch.long,
                ) if nodes == 3 else torch.tensor(
                    [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
                    dtype=torch.long,
                ),
                target=torch.full((1, 1200), index + 1, dtype=torch.long),
                y=torch.tensor([float(index)], dtype=torch.float32),
            )
            for index, nodes in enumerate((3, 4))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"pyg_data": self.samples[index]}


def _capture_shape(shapes, name):
    def hook(_module, inputs, output=None):
        value = inputs[0] if output is None else output
        shapes[name] = tuple(value.shape)

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


def test_configurable_architecture_keeps_scalar_prediction_and_fusion_dimensions():
    sample = DavisDataset(split="all")[0]["pyg_data"]
    configurations = (
        {"num_blocks": 1, "layers_per_block": 2, "mcnn_branch_count": 2},
        {"num_blocks": 2, "layers_per_block": 4, "mcnn_branch_count": 4},
    )
    for configuration in configurations:
        model = MGraphDTA(**configuration).cpu().eval()
        assert model.classifier[0].in_features == (
            model.drug_encoder.drug_embed_dim + model.protein_encoder.output_dim
        ) == 192
        assert model.protein_encoder.linear.in_features == (
            configuration["mcnn_branch_count"] * 96
        )
        with torch.inference_mode():
            prediction = model(sample)
        assert prediction.shape == (1, 1)
        assert torch.isfinite(prediction).all()
        assert not prediction.requires_grad


def test_default_architecture_retains_legacy_dimensions_and_module_names():
    model = MGraphDTA()
    state_keys = set(model.state_dict())
    assert model.drug_encoder.num_blocks == 3
    assert model.drug_encoder.layers_per_block == 8
    assert model.protein_encoder.branch_count == 3
    assert model.classifier[0].in_features == 192
    assert model.drug_encoder.fc.weight.shape == (96, 228)
    assert model.protein_encoder.linear.weight.shape == (96, 288)
    assert "drug_encoder.block1.layers.0.conv1.conv.lin_rel.weight" in state_keys
    assert "drug_encoder.transition3.conv.lin_rel.weight" in state_keys
    assert "protein_encoder.block_list.2.conv_layers.conv_layer2.conv.weight" in state_keys


def test_s3_hidden_channel_choices_produce_finite_scalar_affinity_predictions():
    dataset = SyntheticGraphProteinRows()
    sample = dataset[0]["pyg_data"]
    for hidden_channels in (32, 64, 96, 128, 160):
        model = MGraphDTA(
            num_blocks=1,
            layers_per_block=2,
            mcnn_branch_count=2,
            hidden_channels=hidden_channels,
        ).cpu().eval()
        expected_readout_channels = (32 + 2 * hidden_channels) // 2
        assert model.drug_encoder.hidden_channels == hidden_channels
        assert model.drug_encoder.fc.in_features == expected_readout_channels
        assert model.classifier[0].in_features == 192, (
            f"hidden_channels={hidden_channels}: drug/protein fusion width changed"
        )
        with torch.inference_mode():
            prediction = model(sample)
        assert prediction.shape == (1, 1), (
            f"hidden_channels={hidden_channels}: expected one scalar affinity prediction"
        )
        assert torch.isfinite(prediction).all(), (
            f"hidden_channels={hidden_channels}: prediction is non-finite"
        )
        assert not prediction.requires_grad


def test_synthetic_graph_protein_batch_runs_real_default_and_configured_models():
    dataset = SyntheticGraphProteinRows()
    batch = next(iter(make_dataloader(dataset, batch_size=2, shuffle=False)))
    assert isinstance(batch, Batch), "Project DataLoader must produce a PyG Batch"
    assert batch.num_graphs == 2
    assert batch.x.shape == (7, 22)
    assert batch.target.shape == (2, 1200)
    assert batch.y.shape == (2,)

    configurations = (
        {},
        {"num_blocks": 1, "layers_per_block": 2, "mcnn_branch_count": 2},
    )
    for configuration in configurations:
        model = MGraphDTA(**configuration).cpu().eval()
        assert not model.training
        assert model.classifier[0].in_features == 192
        assert model.drug_encoder._gcn_depth == model.drug_encoder.num_blocks * (
            model.drug_encoder.layers_per_block + 1
        )
        expected_graph_convs = 1 + model.drug_encoder.num_blocks * (
            2 * model.drug_encoder.layers_per_block + 1
        )
        assert sum(
            isinstance(module, GraphConv) for module in model.modules()
        ) == expected_graph_convs

        shapes = {}
        handles = [
            model.drug_encoder.fc.register_forward_hook(
                _capture_shape(shapes, "drug_embedding")
            ),
            model.protein_encoder.linear.register_forward_hook(
                _capture_shape(shapes, "protein_embedding")
            ),
            model.classifier.register_forward_pre_hook(
                _capture_shape(shapes, "fused_embedding")
            ),
        ]
        try:
            with torch.inference_mode():
                prediction = model(batch)
        finally:
            for handle in handles:
                handle.remove()

        assert prediction.shape == (2, 1)
        assert torch.isfinite(prediction).all()
        assert not prediction.requires_grad
        assert shapes == {
            "drug_embedding": (2, 96),
            "protein_embedding": (2, 96),
            "fused_embedding": (2, 192),
        }


def test_scaled_and_unscaled_davis_samples_pass_model_interface_in_eval_mode():
    """Smoke-test real Davis data variants without training or checkpoint I/O."""
    torch.manual_seed(29)
    model = MGraphDTA().cpu().eval()
    assert not model.training, "Smoke test requires the model to be in evaluation mode"

    for split in ("all", "train", "test"):
        unscaled_dataset = DavisDataset(split=split)
        scaled_dataset = DavisDataset(
            split=split,
            processed_dir=str(DEFAULT_DATASET_DIR),
            atom_features_dir=str(DEFAULT_DATASET_DIR),
        )
        unscaled = unscaled_dataset[0]["pyg_data"]
        scaled = scaled_dataset[0]["pyg_data"]

        assert unscaled.x.shape == scaled.x.shape, (
            f"{split} split: scaled and unscaled atom feature shapes differ"
        )
        assert not torch.equal(unscaled.x, scaled.x), (
            f"{split} split: scaled opt-in did not change atom features"
        )
        assert torch.equal(unscaled.edge_index, scaled.edge_index), (
            f"{split} split: scaling unexpectedly changed molecular connectivity"
        )
        assert torch.equal(unscaled.target, scaled.target), (
            f"{split} split: scaling unexpectedly changed protein tokens"
        )
        assert torch.equal(unscaled.y, scaled.y), (
            f"{split} split: scaling unexpectedly changed affinity labels"
        )

        # The two-item batch is the smallest forward that exercises both
        # feature variants together through the repository's normal PyG API.
        batch = Batch.from_data_list([unscaled, scaled]).cpu()
        with torch.inference_mode():
            prediction = model(batch)

        assert prediction.shape == (2, 1), (
            f"{split} split: expected one scalar prediction per variant, "
            f"got shape {tuple(prediction.shape)}"
        )
        assert torch.isfinite(prediction).all(), (
            f"{split} split: model returned a non-finite affinity prediction"
        )
        assert not prediction.requires_grad, (
            f"{split} split: inference-mode prediction unexpectedly tracks gradients"
        )


if __name__ == "__main__":
    test_model_construction_and_davis_forward()
    test_cuda_forward_when_available()
    print("All MGraphDTA integration tests passed.")
