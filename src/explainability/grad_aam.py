"""Authors' gradient-weighted affinity activation mapping for MGraphDTA."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from src.models.mgraphdta import MGraphDTA


@dataclass(frozen=True)
class GradAAMResult:
    """Atom attribution and intermediate values from one drug-protein pair."""

    atom_importance: Tensor
    activation_shape: tuple[int, int]
    channel_weights: Tensor


def _normalize(values: Tensor) -> Tensor:
    """Apply the original authors' min-max normalization with epsilon."""
    return (values - values.min()) / (values.max() - values.min() + 1e-10)


class GradAAM:
    """Compute the authors' Grad-AAM atom scores for a single PyG input pair.

    The input must contain one Davis drug-protein pair and be on the same
    device as ``model``. Atom scores retain the input graph's node ordering.
    """

    def __init__(self, model: MGraphDTA) -> None:
        self.model = model

    def attribute(self, data: Data) -> GradAAMResult:
        """Return normalized per-atom importance for one drug-protein pair.

        The model is temporarily switched to evaluation mode while gradients
        remain enabled. Its original train/eval mode is restored afterward.
        """
        activations: list[Tensor] = []

        def capture_activation(
            _module: nn.Module,
            _inputs: tuple[object, ...],
            output: Tensor,
        ) -> None:
            if not isinstance(output, Tensor):
                raise TypeError("transition3 must return an activation Tensor")
            activations.append(output)

        was_training = self.model.training
        handle = self.model.drug_encoder.transition3.register_forward_hook(
            capture_activation
        )
        try:
            self.model.eval()
            prediction = self.model(data)
            if prediction.numel() != 1:
                raise ValueError("Grad-AAM requires exactly one drug-protein pair")
            if len(activations) != 1:
                raise RuntimeError(
                    "Expected one activation from drug_encoder.transition3"
                )

            activation = activations[0]
            if activation.ndim != 2:
                raise ValueError(
                    "transition3 activation must have shape [num_atoms, channels]"
                )
            if activation.shape[0] != data.x.shape[0]:
                raise ValueError(
                    "transition3 node count does not match input atom count"
                )

            gradient = torch.autograd.grad(prediction.sum(), activation)[0]
            channel_weights = _normalize(gradient.mean(dim=0, keepdim=True))
            atom_importance = _normalize(
                torch.sum(activation * channel_weights, dim=-1)
            )

            return GradAAMResult(
                atom_importance=atom_importance.detach(),
                activation_shape=(int(activation.shape[0]), int(activation.shape[1])),
                channel_weights=channel_weights.detach(),
            )
        finally:
            handle.remove()
            self.model.train(was_training)