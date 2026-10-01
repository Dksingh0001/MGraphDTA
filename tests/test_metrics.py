"""Focused tests for Davis regression evaluation metrics."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
import torch

from src.metrics import concordance_index, mse, r_m2, rmse


def test_mse_and_rmse_known_values():
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([1.0, 3.0, 5.0])
    assert mse(y_true, y_pred) == pytest.approx(0.0)
    assert rmse(y_true, y_pred) == pytest.approx(0.0)

    y_pred = np.array([1.0, 2.0, 4.0])
    assert mse(y_true, y_pred) == pytest.approx(2.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_concordance_index_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])
    assert concordance_index(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    y_pred = np.array([3.0, 2.0, 1.0])
    assert concordance_index(y_true, y_pred) == pytest.approx(0.0)

    y_pred = np.array([1.0, 2.0, 3.0])
    assert concordance_index(y_true, y_pred) == pytest.approx(1.0)


def test_r_m2_known_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_m2(y_true, y_pred) == pytest.approx(1.0)

    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    expected = (27.0 / 28.0) * (1.0 - np.sqrt(185.0) / 49.0)
    assert r_m2(y_true, y_pred) == pytest.approx(expected)


def test_metric_functions_accept_tensors():
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    assert mse(y_true, y_pred) == pytest.approx(0.0)
    assert rmse(y_true, y_pred) == pytest.approx(0.0)
    assert concordance_index(y_true, y_pred) == pytest.approx(1.0)
    assert r_m2(y_true, y_pred) == pytest.approx(1.0)


def test_metric_functions_reject_empty_and_invalid_inputs():
    with pytest.raises(ValueError):
        mse([], [])

    with pytest.raises(ValueError):
        rmse(np.array([1.0, 2.0]), np.array([1.0, np.nan]))

    with pytest.raises(ValueError):
        concordance_index(np.array([1.0]), np.array([2.0]))

    with pytest.raises(ValueError):
        r_m2(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        mse(np.array([1.0, 2.0]), np.array([1.0]))


def test_r_m2_rejects_constant_values():
    with pytest.raises(ValueError):
        r_m2(np.array([2.0, 2.0, 2.0]), np.array([1.0, 3.0, 2.0]))

    with pytest.raises(ValueError):
        r_m2(np.array([1.0, 2.0, 3.0]), np.array([4.0, 4.0, 4.0]))
