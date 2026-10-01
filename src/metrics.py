"""Regression metrics for Davis drug-target affinity evaluation.

Each function accepts NumPy arrays, PyTorch tensors, or nested sequences and
returns a scalar float. Inputs must be finite and the same shape; empty inputs
raise a ValueError instead of producing invalid statistics.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def _as_1d_float_array(values: Sequence[float] | np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    """Convert an input to a 1D float64 numpy array with validation."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
    array = np.asarray(values, dtype=np.float64)

    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim > 1:
        array = array.reshape(-1)

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity values.")
    return array.astype(np.float64, copy=False)


def _validate_shapes(y_true, y_pred):
    """Validate target/prediction arrays before metric calculation."""
    true_array = _as_1d_float_array(y_true, name="y_true")
    pred_array = _as_1d_float_array(y_pred, name="y_pred")
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true_array.shape} and {pred_array.shape}."
        )
    return true_array, pred_array


def mse(y_true, y_pred) -> float:
    """Mean Squared Error (MSE).

    Mathematical meaning:
        MSE = (1 / n) * sum_i (y_i - y_hat_i)^2

    It measures the average squared deviation between the true affinity and the
    predicted affinity. Lower values are better.
    """
    y_true_array, y_pred_array = _validate_shapes(y_true, y_pred)
    return float(np.mean(np.square(y_true_array - y_pred_array)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (RMSE).

    Mathematical meaning:
        RMSE = sqrt(MSE) = sqrt((1 / n) * sum_i (y_i - y_hat_i)^2)

    RMSE keeps the same units as the target and is often easier to interpret than
    MSE because it is directly on the affinity scale.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def concordance_index(y_true, y_pred) -> float:
    """Concordance Index (CI).

    Mathematical meaning:
        For all pairs (i, j) with y_i != y_j, CI measures how often the ordering
        of the predicted values matches the ordering of the true values:

        CI = (concordant + 0.5 * ties) / total_comparable_pairs

    This matches standard regression/affinity evaluation practice. A value of 1.0
    means perfect ordering consistency, while 0.5 is random and 0.0 is fully
    inverted ordering.
    """
    y_true_array, y_pred_array = _validate_shapes(y_true, y_pred)
    if y_true_array.size < 2:
        raise ValueError("CI requires at least two samples.")

    concordant = 0
    discordant = 0
    ties = 0

    for i in range(y_true_array.size):
        for j in range(i + 1, y_true_array.size):
            target_diff = y_true_array[i] - y_true_array[j]
            if target_diff == 0.0:
                continue

            pred_diff = y_pred_array[i] - y_pred_array[j]
            if pred_diff == 0.0:
                ties += 1
            elif np.sign(target_diff) == np.sign(pred_diff):
                concordant += 1
            else:
                discordant += 1

    denominator = concordant + discordant + ties
    if denominator == 0:
        raise ValueError("CI is undefined because no comparable target pairs exist.")
    return float((concordant + 0.5 * ties) / denominator)


def r_m2(y_true, y_pred) -> float:
    """Modified squared correlation coefficient (r_m^2).

    Mathematical meaning:
        Let r^2 be the squared Pearson correlation between y and y_hat. Let
        k = sum(y * y_hat) / sum(y_hat^2), and r_0^2 be the coefficient of
        determination from the zero-intercept fit with slope k. Then:

        r_m^2 = r^2 * (1 - sqrt(abs((r^2)^2 - (r_0^2)^2)))

    This is the DTA evaluation metric used to penalize models whose ranking is
    strong but whose calibration is poor. Values closer to 1.0 are better.
    """
    y_true_array, y_pred_array = _validate_shapes(y_true, y_pred)
    if y_true_array.size < 2:
        raise ValueError("r_m^2 requires at least two samples.")

    # Pearson correlation coefficient squared.
    y_true_centered = y_true_array - np.mean(y_true_array)
    y_pred_centered = y_pred_array - np.mean(y_pred_array)
    true_variance = np.sum(np.square(y_true_centered))
    pred_variance = np.sum(np.square(y_pred_centered))
    if true_variance == 0.0 or pred_variance == 0.0:
        raise ValueError("r_m^2 is undefined for constant targets or constant predictions.")

    correlation = (
        np.sum(y_true_centered * y_pred_centered)
        / np.sqrt(true_variance * pred_variance)
    )
    r_squared = float(np.square(correlation))

    slope = np.sum(y_true_array * y_pred_array) / np.sum(np.square(y_pred_array))
    zero_intercept_residual = y_true_array - slope * y_pred_array
    r0_squared = float(
        1.0 - np.sum(np.square(zero_intercept_residual)) / true_variance
    )

    modified = r_squared * (
        1.0 - np.sqrt(abs(np.square(r_squared) - np.square(r0_squared)))
    )
    if not np.isfinite(modified):
        raise ValueError("r_m^2 produced a non-finite result; input values may be invalid.")
    return float(modified)


__all__ = ["mse", "rmse", "concordance_index", "r_m2"]
