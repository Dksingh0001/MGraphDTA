"""Lightweight smoke tests for the MGraphDTA training pipeline."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from unittest.mock import patch

from src.train import (
    DEFAULT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    evaluate,
    fit,
    generate_training_curve,
    load_checkpoint,
    make_dataloader,
    resolve_device,
    save_final_metrics,
    split_train_validation,
    train_davis,
    train_one_epoch,
    validate,
    write_training_history,
)


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, data):
        return data.target[:, :1] * self.weight


def _loader(batch_size=2):
    samples = []
    for value in (1.0, 2.0, 3.0, 4.0):
        samples.append(
            Data(
                x=torch.ones((1, 1)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                target=torch.tensor([[value]]),
                y=torch.tensor([2.0 * value]),
            )
        )
    return make_dataloader(samples, batch_size=batch_size, shuffle=False)


def test_training_validation_and_checkpoint_smoke(tmp_path):
    loader = _loader()
    model = TinyRegressor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    initial_loss = validate(model, loader, device="cpu")
    training_loss = train_one_epoch(
        model, loader, optimizer, device="cpu", max_steps=3
    )
    assert torch.isfinite(torch.tensor(training_loss))
    assert model.weight.item() > 0
    assert validate(model, loader, device="cpu") < initial_loss
    assert model.training

    checkpoint_path = tmp_path / "tiny-model.pt"
    result = fit(
        model,
        loader,
        loader,
        epochs=3,
        steps_per_epoch=1,
        learning_rate=0.01,
        patience=None,
        device="cpu",
        seed=23,
        checkpoint_path=checkpoint_path,
    )
    assert checkpoint_path.is_file()
    assert result["best_epoch"] > 0
    assert torch.isfinite(torch.tensor(result["best_validation_loss"]))

    restored_model = TinyRegressor()
    checkpoint = load_checkpoint(checkpoint_path, restored_model, device="cpu")
    assert checkpoint["epoch"] == result["best_epoch"]
    restored_loss = validate(restored_model, loader, device="cpu")
    assert abs(restored_loss - result["best_validation_loss"]) < 1e-6


def test_auto_device_selects_available_runtime():
    assert resolve_device().type in {"cpu", "cuda"}


def test_train_validation_subsets_are_disjoint_and_reproducible():
    dataset = TensorDataset(torch.arange(100))
    train_a, validation_a = split_train_validation(dataset, 0.2, seed=31)
    train_b, validation_b = split_train_validation(dataset, 0.2, seed=31)

    assert len(validation_a) == 20
    assert set(train_a.indices).isdisjoint(validation_a.indices)
    assert set(train_a.indices) | set(validation_a.indices) == set(range(100))
    assert train_a.indices == train_b.indices
    assert validation_a.indices == validation_b.indices
    assert DEFAULT_VALIDATION_FRACTION == 0.1
    assert DEFAULT_SEED == 42


def test_fit_selects_checkpoint_and_stops_using_validation_loss(tmp_path):
    model = TinyRegressor()
    train_loader = object()
    validation_loader = object()
    validation_losses = iter((5.0, 4.0, 4.5, 4.6, 1.0))
    validated_loaders = []
    training_epochs = []

    def fake_train_one_epoch(model, *_args, **_kwargs):
        training_epochs.append(len(training_epochs) + 1)
        with torch.no_grad():
            model.weight.fill_(training_epochs[-1])
        return 1.0

    def fake_validate(_model, loader, *_args, **_kwargs):
        validated_loaders.append(loader)
        return next(validation_losses)

    checkpoint_path = tmp_path / "best-validation.pt"
    with patch("src.train.train_one_epoch", side_effect=fake_train_one_epoch), patch(
        "src.train.validate", side_effect=fake_validate
    ):
        result = fit(
            model,
            train_loader,
            validation_loader,
            epochs=10,
            steps_per_epoch=1,
            patience=2,
            device="cpu",
            checkpoint_path=checkpoint_path,
        )

    assert len(result["history"]) == 4
    assert result["best_epoch"] == 2
    assert result["best_validation_loss"] == 4.0
    assert result["epochs_completed"] == 4
    assert result["stopped_early"] is True
    assert result["history"][0]["patience_counter"] == 0
    assert result["history"][1]["patience_counter"] == 0
    assert result["history"][2]["patience_counter"] == 1
    assert result["history"][3]["patience_counter"] == 2
    assert result["history"][1]["is_best"] is True
    assert result["history"][2]["is_best"] is False
    assert validated_loaders == [validation_loader] * 4

    best_model = TinyRegressor()
    checkpoint = load_checkpoint(checkpoint_path, best_model, device="cpu")
    assert checkpoint["epoch"] == 2
    assert best_model.weight.item() == 2.0


def test_training_history_csv_and_curve_are_written(tmp_path):
    history = [
        {
            "epoch": 1,
            "training_mse": 1.0,
            "validation_mse": 0.8,
            "best_validation_mse": 0.8,
            "is_best": True,
            "patience_counter": 0,
        },
        {
            "epoch": 2,
            "training_mse": 0.7,
            "validation_mse": 0.5,
            "best_validation_mse": 0.5,
            "is_best": True,
            "patience_counter": 0,
        },
    ]
    csv_path = tmp_path / "training_history.csv"
    curve_path = tmp_path / "training_curve.png"

    write_training_history(history, csv_path)
    generate_training_curve(history, curve_path)

    assert csv_path.exists()
    assert curve_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "epoch,training_mse,validation_mse,best_validation_mse,is_best,patience_counter" in csv_text
    assert "1,1.0,0.8,0.8,True,0" in csv_text
    assert curve_path.stat().st_size > 0


def test_final_metrics_json_is_created(tmp_path):
    metrics = {
        "best_epoch": 3,
        "best_validation_mse": 0.5,
        "final_training_mse": 0.7,
        "final_validation_mse": 0.55,
        "independent_test_mse": 0.6,
        "epochs_completed": 3,
        "stopped_early": True,
        "patience": 2,
        "learning_rate": 5e-4,
        "batch_size": 512,
        "steps_per_epoch": 50,
        "seed": 42,
        "device": "cpu",
    }
    output_path = tmp_path / "final_metrics.json"
    save_final_metrics(output_path, metrics)

    assert output_path.exists()
    payload = output_path.read_text(encoding="utf-8")
    assert '"best_epoch": 3' in payload
    assert '"independent_test_mse": 0.6' in payload


def test_test_data_is_loaded_and_evaluated_only_after_training(tmp_path):
    events = []

    class FakeDavisDataset:
        def __init__(self, split, **_kwargs):
            events.append(f"load_{split}")
            self.split = split

        def __len__(self):
            return 10

        def __getitem__(self, index):
            return index

    def fake_loader(dataset, *_args, **_kwargs):
        base_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset
        events.append(f"loader_{base_dataset.split}")
        return {"split": base_dataset.split, "dataset": dataset}

    def fake_fit(_model, train_loader, validation_loader, **_kwargs):
        assert train_loader["split"] == "train"
        assert validation_loader["split"] == "train"
        train_subset = train_loader["dataset"]
        validation_subset = validation_loader["dataset"]
        assert train_subset.dataset is validation_subset.dataset
        assert set(train_subset.indices).isdisjoint(validation_subset.indices)
        assert "load_test" not in events
        assert "loader_test" not in events
        events.append("fit_complete")
        return {
            "history": [],
            "best_epoch": 1,
            "best_validation_loss": 0.5,
            "device": "cpu",
        }

    def fake_load_checkpoint(_path, _model, **_kwargs):
        events.append("restore_best_checkpoint")
        return {}

    def fake_evaluate(_model, test_loader, **_kwargs):
        assert test_loader["split"] == "test"
        assert test_loader["dataset"].split == "test"
        events.append("evaluate_test")
        return 0.75

    with patch("src.train.DavisDataset", FakeDavisDataset), patch(
        "src.train.make_dataloader", side_effect=fake_loader
    ), patch("src.train.MGraphDTA", TinyRegressor), patch(
        "src.train.fit", side_effect=fake_fit
    ), patch("src.train.load_checkpoint", side_effect=fake_load_checkpoint), patch(
        "src.train.evaluate", side_effect=fake_evaluate
    ):
        result = train_davis(
            checkpoint_path=tmp_path / "unused.pt",
            epochs=1,
            steps_per_epoch=1,
            device="cpu",
        )

    assert result["test_mse"] == 0.75
    assert events.index("fit_complete") < events.index("restore_best_checkpoint")
    assert events.index("restore_best_checkpoint") < events.index("load_test")
    assert events.index("load_test") < events.index("loader_test")
    assert events.index("loader_test") < events.index("evaluate_test")


def test_training_on_cuda_when_available():
    if not torch.cuda.is_available():
        print("Skipping training CUDA test (CUDA is unavailable).")
        return

    model = TinyRegressor()
    result = fit(
        model,
        _loader(),
        _loader(),
        epochs=1,
        steps_per_epoch=1,
        patience=None,
        device="cuda",
    )
    assert result["device"] == "cuda"
    assert next(model.parameters()).device.type == "cuda"