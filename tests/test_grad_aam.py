"""Focused tests for the authors' Grad-AAM attribution component."""

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from src.explainability.grad_aam import GradAAM, _normalize
from src.dataset import DavisDataset
from src.models.mgraphdta import MGraphDTA


CHANNEL_COUNT = 228


def _authors_normalize(values: Tensor) -> Tensor:
    return (values - values.min()) / (values.max() - values.min() + 1e-10)


class _ToyTransition3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.arange(1, CHANNEL_COUNT + 1, dtype=torch.float64)
        )

    def forward(self, node_features: Tensor, _edge_index: Tensor) -> Tensor:
        return torch.relu(node_features[:, :1] * self.scale)


class _ToyDrugEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transition3 = _ToyTransition3()


class _ToyMGraphDTA(nn.Module):
    """Minimal model exposing the MGraphDTA drug_encoder.transition3 API."""

    def __init__(self) -> None:
        super().__init__()
        self.drug_encoder = _ToyDrugEncoder()
        self.register_buffer(
            "affinity_response",
            torch.arange(1, CHANNEL_COUNT + 1, dtype=torch.float64),
        )
        self.fail_after_transition = False
        self.grad_enabled_in_forward = False
        self.inference_mode_in_forward = True

    def forward(self, data: Data) -> Tensor:
        self.grad_enabled_in_forward = torch.is_grad_enabled()
        self.inference_mode_in_forward = torch.is_inference_mode_enabled()
        activation = self.drug_encoder.transition3(data.x, data.edge_index)
        if self.fail_after_transition:
            raise RuntimeError("toy forward failure")
        return (activation * self.affinity_response).sum().reshape(1, 1)


def _toy_pair() -> Data:
    # Deliberately non-sorted node values make any atom reordering observable.
    return Data(
        x=torch.tensor(
            [[3.0, 0.0], [1.0, 0.0], [4.0, 0.0], [2.0, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        target=torch.ones((1, 1200), dtype=torch.long),
        y=torch.tensor([1.0], dtype=torch.float64),
    )


def test_grad_aam_uses_transition3_and_preserves_atom_order():
    model = _ToyMGraphDTA()
    data = _toy_pair()
    captured: list[Tensor] = []
    observer = model.drug_encoder.transition3.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        result = GradAAM(model).attribute(data)
    finally:
        observer.remove()

    assert len(captured) == 1
    activation = captured[0]
    assert activation.shape == (data.x.shape[0], CHANNEL_COUNT)
    assert result.activation_shape == (data.x.shape[0], CHANNEL_COUNT)
    assert result.atom_importance.shape == (data.x.shape[0],)
    assert result.channel_weights.shape == (1, CHANNEL_COUNT)
    assert torch.isfinite(result.atom_importance).all()
    assert torch.all((result.atom_importance >= 0.0) & (result.atom_importance <= 1.0))

    expected_weights = _authors_normalize(model.affinity_response.unsqueeze(0))
    expected_atom_scores = _authors_normalize(
        torch.sum(activation * expected_weights, dim=-1)
    )
    torch.testing.assert_close(result.channel_weights, expected_weights)
    torch.testing.assert_close(result.atom_importance, expected_atom_scores)
    assert result.atom_importance.tolist() == pytest.approx(
        [2.0 / 3.0, 0.0, 1.0, 1.0 / 3.0]
    )
    assert model.grad_enabled_in_forward
    assert not model.inference_mode_in_forward


def test_normalize_matches_authors_epsilon_formula():
    values = torch.tensor([1.25, 8.75], dtype=torch.float64)
    expected = (values - values.min()) / (
        values.max() - values.min() + 1e-10
    )

    actual = _normalize(values)

    assert torch.equal(actual, expected)
    assert actual[-1] < 1.0


@pytest.mark.parametrize("initial_training", [True, False])
def test_mode_and_hook_are_restored_after_success(initial_training: bool):
    model = _ToyMGraphDTA()
    model.train(initial_training)
    transition = model.drug_encoder.transition3
    hook_count = len(transition._forward_hooks)

    result = GradAAM(model).attribute(_toy_pair())

    assert result.activation_shape == (4, CHANNEL_COUNT)
    assert model.training is initial_training
    assert len(transition._forward_hooks) == hook_count


def test_hook_and_mode_are_restored_after_attribution_exception():
    model = _ToyMGraphDTA()
    model.train()
    model.fail_after_transition = True
    transition = model.drug_encoder.transition3
    hook_count = len(transition._forward_hooks)

    with pytest.raises(RuntimeError, match="toy forward failure"):
        GradAAM(model).attribute(_toy_pair())

    assert model.training
    assert len(transition._forward_hooks) == hook_count


def test_grad_aam_hooks_final_transition_for_nondefault_mgnn_depth():
    data = DavisDataset(split="all")[0]["pyg_data"]
    model = MGraphDTA(num_blocks=1, layers_per_block=2).eval()
    final_transition = model.drug_encoder.final_transition
    assert final_transition is model.drug_encoder.transition1

    result = GradAAM(model).attribute(data)

    assert result.activation_shape[0] == data.x.shape[0]
    assert result.atom_importance.shape == (data.x.shape[0],)
    assert torch.isfinite(result.atom_importance).all()


def test_grad_aam_channel_dimensions_follow_configured_hidden_width():
    data = Data(
        x=torch.rand((4, 22)),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        target=torch.ones((1, 1200), dtype=torch.long),
        y=torch.tensor([5.0]),
    )
    model = MGraphDTA(
        num_blocks=1,
        layers_per_block=2,
        mcnn_branch_count=2,
        hidden_channels=64,
    ).eval()
    result = GradAAM(model).attribute(data)
    expected_channels = model.drug_encoder.pre_pool_channels

    assert result.activation_shape == (4, expected_channels)
    assert result.channel_weights.shape == (1, expected_channels)
    assert torch.isfinite(result.channel_weights).all()
    assert torch.isfinite(result.atom_importance).all()
