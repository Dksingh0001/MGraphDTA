"""Non-training tests for paper-protocol split and artifact safety helpers."""

from pathlib import Path
from types import SimpleNamespace
import json
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Batch, Data
from src import dataset as dataset_module
from src.train import make_dataloader

from src.paper_protocol import (
    FINAL_SEEDS,
    CVTrainingConfig,
    INDEXED_ESI_SEARCH_CHOICES,
    INDEXED_ESI_SELECTED,
    build_training_cv_subsets,
    evaluate_five_fold_trial,
    load_authoritative_davis_fold_ids,
    load_training_cv_subsets,
    make_davis_optuna_objective,
    refuse_existing_protocol_root,
    run_davis_optuna_study,
    select_final_epoch,
    suggest_indexed_architecture,
    summarize_seed_metrics,
    validate_existing_training_table,
    write_protocol_manifest,
)


class TrainingRows(Dataset):
    def __init__(self, pair_ids):
        self.df = pd.DataFrame({"pair_idx": pair_ids})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return int(self.df.iloc[index]["pair_idx"])


class SyntheticGraphRows(Dataset):
    def __init__(self, pair_ids):
        self.df = pd.DataFrame({"pair_idx": pair_ids})
        self.samples = [
            Data(
                x=torch.full((3, 22), float(index + 1) / 10),
                edge_index=torch.tensor(
                    [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
                    dtype=torch.long,
                ),
                target=torch.full((1, 1200), index + 1, dtype=torch.long),
                y=torch.tensor([float(pair_id)], dtype=torch.float32),
            )
            for index, pair_id in enumerate(pair_ids)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {"pyg_data": self.samples[index]}


def test_training_cv_subsets_use_only_five_established_training_folds():
    folds = [[30, 10], [50], [20, 40], [60], [70, 80]]
    test_ids = [1, 2]
    dataset = TrainingRows([80, 30, 70, 10, 60, 20, 50, 40])

    partitions = build_training_cv_subsets(dataset, folds, test_ids)

    assert len(partitions) == 5
    validation_ids = []
    all_train_ids = set(dataset.df["pair_idx"])
    for fold, partition in zip(folds, partitions):
        assert isinstance(partition["training"], Subset)
        assert isinstance(partition["validation"], Subset)
        assert partition["validation_pair_ids"] == fold
        assert set(partition["training_pair_ids"]).isdisjoint(fold)
        assert set(partition["training_pair_ids"]) | set(fold) == all_train_ids
        assert set(partition["validation_pair_ids"]).isdisjoint(test_ids)
        validation_ids.extend(partition["validation_pair_ids"])
    assert sorted(validation_ids) == sorted(all_train_ids)
    assert [dataset[row] for row in partitions[0]["validation"].indices] == folds[0]
    repeated = build_training_cv_subsets(dataset, folds, test_ids)
    assert [part["training"].indices for part in repeated] == [
        part["training"].indices for part in partitions
    ]
    assert [part["validation"].indices for part in repeated] == [
        part["validation"].indices for part in partitions
    ]


@pytest.mark.parametrize(
    "bad_folds,bad_test,error_text",
    [
        ([[1], [2], [3], [4]], [6, 7], "exactly 5"),
        ([[1], [1], [3], [4], [5]], [6], "duplicates"),
        ([[1], [2], [3], [4], [5]], [5], "overlap"),
    ],
)
def test_training_cv_rejects_invalid_fold_mappings(bad_folds, bad_test, error_text):
    dataset = TrainingRows([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match=error_text):
        build_training_cv_subsets(dataset, bad_folds, bad_test)


def test_training_cv_requires_exact_dataset_fold_coverage():
    folds = [[1], [2], [3], [4], [5]]
    with pytest.raises(ValueError, match="do not exactly match"):
        build_training_cv_subsets(TrainingRows([1, 2, 3, 4, 6]), folds, [7])


def _write_synthetic_split_files(
    raw_dir, training_folds=None, test_fold=None, *, n_ligands=2, n_proteins=3
):
    raw_dir = Path(raw_dir)
    folds_dir = raw_dir / "folds"
    folds_dir.mkdir(parents=True)
    if training_folds is None:
        training_folds = [[0], [1], [2], [3], [4]]
    if test_fold is None:
        test_fold = [5]
    (folds_dir / "train_fold_setting1.txt").write_text(
        json.dumps(training_folds), encoding="utf-8"
    )
    (folds_dir / "test_fold_setting1.txt").write_text(
        json.dumps(test_fold), encoding="utf-8"
    )
    ligands = {f"d{index}": "C" for index in range(n_ligands)}
    proteins = {f"p{index}": "A" for index in range(n_proteins)}
    (raw_dir / "ligands_can.txt").write_text(json.dumps(ligands), encoding="utf-8")
    (raw_dir / "proteins.txt").write_text(json.dumps(proteins), encoding="utf-8")
    return raw_dir


def test_authoritative_split_parser_reads_complete_ids_without_affinity_file(tmp_path):
    raw_dir = _write_synthetic_split_files(tmp_path / "raw")
    folds, test_ids = load_authoritative_davis_fold_ids(raw_dir)
    assert folds == [[0], [1], [2], [3], [4]]
    assert test_ids == [5]
    assert not (raw_dir / "Y").exists()


@pytest.mark.parametrize(
    "training_folds,test_ids,error_text",
    [
        ([[0], [1], [2], [3]], [4, 5], "exactly 5"),
        ([[0], [1], [], [3], [4]], [2, 5], "nonempty"),
        ([[0], [1], [2], [3], [3]], [4, 5], "duplicate"),
        ([[0], [1], [2], [3], [4]], [], "nonempty"),
        ([[0], [1], [2], [3], [4]], [5, 5], "duplicate"),
        ([[0], [1], [2], [3], [4]], [4, 5], "overlap"),
        ([[0], [1], [2], [3], [3]], [5], "duplicate"),
    ],
)
def test_authoritative_split_parser_rejects_incomplete_or_invalid_ids(
    tmp_path, training_folds, test_ids, error_text
):
    raw_dir = _write_synthetic_split_files(
        tmp_path / "raw", training_folds=training_folds, test_fold=test_ids
    )
    with pytest.raises(ValueError, match=error_text):
        load_authoritative_davis_fold_ids(raw_dir)


def test_authoritative_split_parser_rejects_incomplete_fixed_test_set(tmp_path):
    raw_dir = _write_synthetic_split_files(
        tmp_path / "raw", test_fold=[5], n_ligands=3, n_proteins=3
    )
    with pytest.raises(ValueError, match="completely cover"):
        load_authoritative_davis_fold_ids(raw_dir)


def test_processed_training_table_validation_reads_pair_ids_only(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"pair_idx": [3, 1, 4, 0, 2], "affinity": [10, 11, 12, 13, 14]}).to_csv(
        processed_dir / "davis_train.csv", index=False
    )
    assert validate_existing_training_table(processed_dir, [[0], [1], [2], [3], [4]]) == (
        processed_dir / "davis_train.csv"
    )
    with pytest.raises(ValueError, match="duplicate pair_idx"):
        pd.DataFrame({"pair_idx": [0, 1, 2, 3, 3]}).to_csv(
            processed_dir / "davis_train.csv", index=False
        )
        validate_existing_training_table(processed_dir, [[0], [1], [2], [3], [4]])


def test_cv_loader_requests_train_split_only():
    events = []

    def dataset_factory(*, split):
        events.append(split)
        assert split == "train", "CV must never instantiate the fixed test split"
        return TrainingRows([10, 20, 30, 40, 50])

    partitions = load_training_cv_subsets(
        dataset_factory,
        [[10], [20], [30], [40], [50]],
        [90, 91],
    )
    assert events == ["train"]
    assert len(partitions) == 5


def test_final_epoch_is_median_of_five_fold_best_epochs():
    assert select_final_epoch([8, 2, 5, 9, 4]) == 5
    with pytest.raises(ValueError, match="Expected 5"):
        select_final_epoch([4, 6, 8, 9])
    with pytest.raises(ValueError, match="positive"):
        select_final_epoch([1, 2, 0, 4, 5])


def test_indexed_esi_search_values_are_explicit_and_not_guessed():
    assert INDEXED_ESI_SEARCH_CHOICES["num_blocks"] == (1, 2, 3, 4, 5)
    assert INDEXED_ESI_SEARCH_CHOICES["layers_per_block"] == tuple(range(2, 11))
    assert INDEXED_ESI_SEARCH_CHOICES["hidden_channels"] == (32, 64, 96, 128, 160)
    assert INDEXED_ESI_SEARCH_CHOICES["mcnn_branch_count"] == (2, 3, 4, 5)
    assert INDEXED_ESI_SELECTED == {
        "num_blocks": 3,
        "layers_per_block": 8,
        "hidden_channels": 64,
        "mcnn_branch_count": 3,
    }


class FakeTrial:
    def __init__(self):
        self.attrs = {}
        self.reported = []
        self.choices = {}

    def suggest_categorical(self, name, choices):
        self.choices[name] = tuple(choices)
        return INDEXED_ESI_SELECTED[name]

    def report(self, value, step):
        self.reported.append((step, value))

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_optuna_objective_interface_aggregates_only_validation_fold_results():
    trial = FakeTrial()
    dataset = TrainingRows([10, 20, 30, 40, 50])
    partitions = build_training_cv_subsets(
        dataset, [[10], [20], [30], [40], [50]], [90, 91]
    )
    evaluated = []

    def fold_evaluator(params, partition):
        assert params == INDEXED_ESI_SELECTED
        assert isinstance(partition["training"], Subset)
        assert isinstance(partition["validation"], Subset)
        assert partition["training"].dataset is dataset
        assert partition["validation"].dataset is dataset
        assert set(partition["training"].indices).isdisjoint(partition["validation"].indices)
        evaluated.append(partition["fold"])
        return {"best_validation_mse": float(partition["fold"]), "best_epoch": 10 + partition["fold"]}

    params = suggest_indexed_architecture(trial)
    score = evaluate_five_fold_trial(trial, partitions, fold_evaluator)

    assert params == INDEXED_ESI_SELECTED
    assert set(trial.choices) == set(INDEXED_ESI_SEARCH_CHOICES)
    assert set(params) == {
        "num_blocks", "layers_per_block", "hidden_channels", "mcnn_branch_count"
    }
    assert evaluated == [1, 2, 3, 4, 5]
    assert score == pytest.approx(3.0)
    assert trial.attrs["fold_validation_mse"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert trial.attrs["fold_best_epochs"] == [11, 12, 13, 14, 15]
    assert len(trial.reported) == 5


def test_optuna_objective_rejects_missing_folds_and_invalid_validation_score():
    trial = FakeTrial()
    with pytest.raises(ValueError, match="Expected 5"):
        evaluate_five_fold_trial(trial, [{"fold": 1}], lambda *_: {})
    with pytest.raises(ValueError, match="invalid validation MSE"):
        evaluate_five_fold_trial(
            trial,
            build_training_cv_subsets(
                TrainingRows([1, 2, 3, 4, 5]),
                [[1], [2], [3], [4], [5]],
                [9],
            ),
            lambda *_: {"best_validation_mse": float("nan"), "best_epoch": 1},
        )


def test_real_objective_wires_model_loaders_and_fit_without_opening_test_data(tmp_path):
    train_ids = [10, 20, 30, 40, 50]
    raw_dir = _write_synthetic_split_files(
        tmp_path / "raw", training_folds=[[0], [1], [2], [3], [4]], test_fold=[5]
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"pair_idx": [0, 1, 2, 3, 4]}).to_csv(
        processed_dir / "davis_train.csv", index=False
    )
    calls = {"splits": [], "folds": [], "seeds": []}

    def dataset_factory(*, split, raw_dir, processed_dir):
        calls["splits"].append(split)
        assert split == "train", "The objective must never instantiate the test dataset"
        assert Path(processed_dir).joinpath("davis_train.csv").is_file()
        return TrainingRows([0, 1, 2, 3, 4])

    def loader_factory(subset, **kwargs):
        return SimpleNamespace(subset=subset, kwargs=kwargs)

    def model_factory(**params):
        return SimpleNamespace(params=params)

    def seed_fn(seed, *, deterministic):
        calls["seeds"].append((seed, deterministic))

    def fake_fit(model, train_loader, validation_loader, **kwargs):
        train_subset = train_loader.subset
        validation_subset = validation_loader.subset
        assert train_subset.dataset is validation_subset.dataset
        assert set(train_subset.indices).isdisjoint(validation_subset.indices)
        assert kwargs["checkpoint_path"] is None
        assert "test_loader" not in kwargs
        fold = len(calls["folds"]) + 1
        calls["folds"].append((model.params, train_subset.indices, validation_subset.indices, kwargs))
        return {"best_validation_loss": float(fold), "best_epoch": 10 + fold}

    objective = make_davis_optuna_objective(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        training_config=CVTrainingConfig(max_epochs=2, steps_per_epoch=1, patience=1),
        dataset_factory=dataset_factory,
        model_factory=model_factory,
        dataloader_factory=loader_factory,
        fit_fn=fake_fit,
        seed_fn=seed_fn,
    )

    trial = FakeTrial()
    score = objective(trial)

    assert calls["splits"] == ["train"]
    assert len(calls["folds"]) == 5
    assert [seed for seed, _ in calls["seeds"]] == [42, 43, 44, 45, 46]
    assert all(params == INDEXED_ESI_SELECTED for params, *_ in calls["folds"])
    assert all(kwargs["epochs"] == 2 and kwargs["steps_per_epoch"] == 1 for *_, kwargs in calls["folds"])
    assert score == pytest.approx(3.0)  # Arithmetic mean of five fold-best validation MSEs.
    assert trial.attrs["fold_validation_mse"] == [1., 2., 3., 4., 5.]


def test_objective_uses_real_model_and_pyg_loader_with_fit_mocked(tmp_path):
    raw_dir = _write_synthetic_split_files(
        tmp_path / "raw", training_folds=[[0], [1], [2], [3], [4]], test_fold=[5]
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"pair_idx": [0, 1, 2, 3, 4]}).to_csv(
        processed_dir / "davis_train.csv", index=False
    )
    dataset = SyntheticGraphRows([0, 1, 2, 3, 4])
    seen = []

    def dataset_factory(*, split, raw_dir, processed_dir):
        assert split == "train", "Objective factory must request only the training split"
        return dataset

    class SmallArchitectureTrial(FakeTrial):
        def suggest_categorical(self, name, choices):
            self.choices[name] = tuple(choices)
            selected = {
                "num_blocks": 1,
                "layers_per_block": 2,
                "hidden_channels": 64,
                "mcnn_branch_count": 2,
            }[name]
            assert selected in choices
            return selected

    def fake_fit(model, train_loader, validation_loader, **kwargs):
        assert model.__class__.__name__ == "MGraphDTA"
        assert model.drug_encoder.num_blocks == 1
        assert model.drug_encoder.layers_per_block == 2
        assert model.drug_encoder.hidden_channels == 64
        assert model.protein_encoder.branch_count == 2
        assert kwargs["checkpoint_path"] is None
        assert "test_loader" not in kwargs
        train_batch = next(iter(train_loader))
        validation_batch = next(iter(validation_loader))
        assert isinstance(train_batch, Batch)
        assert isinstance(validation_batch, Batch)
        assert train_batch.target.shape == (2, 1200)
        assert validation_batch.target.shape == (1, 1200)
        assert set(train_loader.dataset.dataset.indices).isdisjoint(
            validation_loader.dataset.dataset.indices
        )
        seen.append((train_batch.num_graphs, validation_batch.num_graphs))
        return {"best_validation_loss": float(len(seen)), "best_epoch": len(seen)}

    objective = make_davis_optuna_objective(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        training_config=CVTrainingConfig(batch_size=2, max_epochs=1, steps_per_epoch=1, patience=1),
        dataset_factory=dataset_factory,
        dataloader_factory=make_dataloader,
        fit_fn=fake_fit,
        seed_fn=lambda *_args, **_kwargs: None,
    )
    value = objective(SmallArchitectureTrial())

    assert value == pytest.approx(3.0)
    assert seen == [(2, 1)] * 5


def test_objective_fails_before_dataset_factory_if_training_csv_is_missing(tmp_path, monkeypatch):
    raw_dir = _write_synthetic_split_files(tmp_path / "raw")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    factory_calls = []

    def forbidden_factory(**kwargs):
        factory_calls.append(kwargs)
        pytest.fail("Dataset factory must not run when processed training data is absent")

    monkeypatch.setattr(
        dataset_module, "prepare_davis_tables",
        lambda **_: pytest.fail("Objective must not preprocess Davis tables"),
    )
    monkeypatch.setattr(
        dataset_module, "load_raw_davis_data",
        lambda **_: pytest.fail("Objective must not load raw affinity data"),
    )
    with pytest.raises(FileNotFoundError, match="preprocessing is disabled"):
        make_davis_optuna_objective(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            dataset_factory=forbidden_factory,
            model_factory=lambda **_: None,
            dataloader_factory=lambda *_args, **_kwargs: None,
            fit_fn=lambda *_args, **_kwargs: {},
            seed_fn=lambda *_args, **_kwargs: None,
        )
    assert factory_calls == []


def test_objective_uses_existing_train_table_without_preprocessing_or_test_dataset(
    tmp_path, monkeypatch
):
    raw_dir = _write_synthetic_split_files(tmp_path / "raw")
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pd.DataFrame({"pair_idx": [0, 1, 2, 3, 4]}).to_csv(
        processed_dir / "davis_train.csv", index=False
    )
    monkeypatch.setattr(
        dataset_module, "prepare_davis_tables",
        lambda **_: pytest.fail("Preprocessing must not be called by objective loading"),
    )
    monkeypatch.setattr(
        dataset_module, "load_raw_davis_data",
        lambda **_: pytest.fail("Raw affinity loader must not be called by objective loading"),
    )
    objective = make_davis_optuna_objective(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        model_factory=lambda **_: None,
        dataloader_factory=lambda *_args, **_kwargs: None,
        fit_fn=lambda *_args, **_kwargs: {},
        seed_fn=lambda *_args, **_kwargs: None,
    )
    assert callable(objective)
    assert not (processed_dir / "davis_test.csv").exists()


@pytest.mark.parametrize(
    "kwargs,field",
    [({"batch_size": 0}, "batch_size"), ({"max_epochs": -1}, "max_epochs"),
     ({"steps_per_epoch": 0}, "steps_per_epoch"), ({"patience": 0}, "patience"),
     ({"learning_rate": float("nan")}, "learning_rate")],
)
def test_cv_training_config_rejects_invalid_settings(kwargs, field):
    with pytest.raises(ValueError, match=field):
        CVTrainingConfig(**kwargs)


def test_trial_rejects_architecture_value_outside_supported_search_space():
    class InvalidTrial(FakeTrial):
        def suggest_categorical(self, name, choices):
            return 64 if name == "num_blocks" else choices[0]

    with pytest.raises(ValueError, match="unsupported num_blocks"):
        suggest_indexed_architecture(InvalidTrial())


def test_protocol_output_root_and_manifest_refuse_overwrite(tmp_path: Path):
    root = refuse_existing_protocol_root(tmp_path / "paper_protocol")
    assert root.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        refuse_existing_protocol_root(root)

    manifest = write_protocol_manifest(root / "protocol_manifest.json", {"seeds": list(FINAL_SEEDS)})
    original = manifest.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_protocol_manifest(manifest, {"seeds": [1]})
    assert manifest.read_text(encoding="utf-8") == original


def test_optuna_study_runner_persists_isolated_trial_and_fold_artifacts(tmp_path):
    output_root = tmp_path / "fresh_study"

    def synthetic_objective(trial):
        trial.set_user_attr("fold_validation_mse", [1.0, 2.0, 3.0, 4.0, 5.0])
        trial.set_user_attr("fold_best_epochs", [3, 4, 5, 6, 7])
        trial.set_user_attr("final_epoch_rule", "median_of_five_fold_best_epochs")
        return float(trial.number + 1)

    study = run_davis_optuna_study(
        output_root=output_root,
        trial_count=2,
        sampler_seed=123,
        objective=synthetic_objective,
        training_config=CVTrainingConfig(max_epochs=1, steps_per_epoch=1, patience=1),
        paper_evidence={"learning_rate": "article section 2.8"},
        project_decisions={"synthetic_test_objective": True},
    )

    assert len(study.trials) == 2
    assert all(trial.state.name == "COMPLETE" for trial in study.trials)
    assert (output_root / "optuna_study.db").is_file()
    trial_records = json.loads((output_root / "trials.json").read_text(encoding="utf-8"))
    assert len(trial_records) == 2
    assert trial_records[0]["user_attrs"]["fold_validation_mse"] == [1., 2., 3., 4., 5.]
    assert "objective_mean_validation_mse" in (output_root / "trials.csv").read_text(encoding="utf-8")
    best = json.loads((output_root / "best_parameters.json").read_text(encoding="utf-8"))
    assert best["objective_mean_validation_mse"] == pytest.approx(1.0)
    assert json.loads((output_root / "failures.json").read_text(encoding="utf-8")) == []
    manifest = json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["trial_count"] == 2
    assert manifest["sampler"] == {
        "class": "TPESampler", "seed": 123, "n_startup_trials": 10
    }
    assert manifest["test_partition_policy"] == "not loaded or supplied to the objective"

    with pytest.raises(FileExistsError, match="already exists"):
        run_davis_optuna_study(
            output_root=output_root,
            trial_count=1,
            sampler_seed=1,
            objective=synthetic_objective,
            training_config=CVTrainingConfig(),
            paper_evidence={},
            project_decisions={},
        )


def test_optuna_study_runner_records_failed_trials_and_fails_closed(tmp_path):
    output_root = tmp_path / "failed_study"

    def broken_objective(_trial):
        raise ValueError("synthetic objective failure")

    with pytest.raises(RuntimeError, match="no completed trials"):
        run_davis_optuna_study(
            output_root=output_root,
            trial_count=2,
            sampler_seed=4,
            objective=broken_objective,
            training_config=CVTrainingConfig(max_epochs=1, steps_per_epoch=1, patience=1),
            paper_evidence={},
            project_decisions={"synthetic_test_objective": True},
        )

    failures = json.loads((output_root / "failures.json").read_text(encoding="utf-8"))
    assert [failure["state"] for failure in failures] == ["FAIL", "FAIL"]
    manifest = json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_three_seed_summary_uses_sample_standard_deviation():
    result = summarize_seed_metrics({
        42: {"mse": 1.0, "ci": 0.5},
        43: {"mse": 2.0, "ci": 0.6},
        44: {"mse": 3.0, "ci": 0.7},
    })
    assert result["seeds"] == [42, 43, 44]
    assert result["metrics"]["mse"]["mean"] == pytest.approx(2.0)
    assert result["metrics"]["mse"]["sample_sd"] == pytest.approx(1.0)
    assert result["metrics"]["ci"]["sample_sd"] == pytest.approx(0.1)

    with pytest.raises(ValueError, match="exactly the declared seeds"):
        summarize_seed_metrics({42: {"mse": 1.0}, 43: {"mse": 2.0}})
    with pytest.raises(ValueError, match="non-finite"):
        summarize_seed_metrics({
            42: {"mse": 1.0},
            43: {"mse": float("nan")},
            44: {"mse": 3.0},
        })
