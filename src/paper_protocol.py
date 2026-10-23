"""Safe Davis paper-protocol helpers and an isolated training-only objective.

The objective lazily imports the project model/trainer and fails closed unless
processed training data and the authoritative, label-free split metadata exist.
It never constructs a test dataset or reads test affinity labels.
"""

from __future__ import annotations

import json
import csv
import os
import platform
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


CV_FOLD_COUNT = 5
FINAL_SEEDS = (42, 43, 44)
TPE_STARTUP_TRIALS = 10  # Explicit project choice; not specified in the paper.

# The ranges and selected values below are from official RSC ESI Table S3.
# Local model mapping: hidden_channels controls MGNN DenseLayer growth_rate,
# i.e. appended dense-feature width; bottleneck and transition formulas remain
# the current project architecture and are documented as that mapping choice.
INDEXED_ESI_SEARCH_CHOICES = {
    "num_blocks": (1, 2, 3, 4, 5),
    "layers_per_block": (2, 3, 4, 5, 6, 7, 8, 9, 10),
    "hidden_channels": (32, 64, 96, 128, 160),
    "mcnn_branch_count": (2, 3, 4, 5),
}
INDEXED_ESI_SELECTED = {
    "num_blocks": 3,
    "layers_per_block": 8,
    "hidden_channels": 64,
    "mcnn_branch_count": 3,
}


@dataclass(frozen=True)
class CVTrainingConfig:
    """Explicit settings for CV trials; paper facts and project choices are separated.

    The learning rate and batch size are reported in article §2.8. Epoch limit,
    steps/epoch, patience, seed, device selection and worker count are project
    choices (the first three follow current local defaults). The paper/ESI
    evidence available here does not establish those settings.
    """

    learning_rate: float = 5e-4  # Paper-reported §2.8.
    batch_size: int = 512  # Paper-reported §2.8.
    max_epochs: int = 1000  # Project choice; matches the current local default.
    steps_per_epoch: int = 50  # Project choice; not established as paper protocol.
    patience: int = 400  # Project choice; selection remains validation-only.
    seed: int = 42  # Project choice; each fold derives seed + fold - 1.
    deterministic: bool = True  # Project choice; may need device support.
    device: str = "auto"  # Project choice; resolves to available CUDA or CPU.
    num_workers: int = 0  # Project choice for straightforward reproducibility.

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("batch_size", "max_epochs", "steps_per_epoch", "patience"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not isinstance(self.num_workers, Integral)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ValueError("num_workers must be a nonnegative integer")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a bool")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty device string")
        if self.device != "auto":
            try:
                torch.device(self.device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"invalid device: {self.device!r}") from exc


def build_training_cv_subsets(
    training_dataset: Any,
    training_folds: Sequence[Sequence[int]],
    test_fold: Sequence[int],
) -> List[Dict[str, Any]]:
    """Map original Davis pair indices into five train/validation Subsets.

    `training_dataset` must be the already-loaded `DavisDataset(split="train")`.
    This function accepts no test dataset or test loader, and rejects any
    overlap between the established train and test pair IDs.
    """
    if len(training_folds) != CV_FOLD_COUNT:
        raise ValueError(f"Expected exactly {CV_FOLD_COUNT} Davis training folds")
    if not hasattr(training_dataset, "df") or "pair_idx" not in training_dataset.df:
        raise ValueError("Training dataset must expose its Davis pair_idx column")

    if any(not fold for fold in training_folds):
        raise ValueError("Davis training folds must all be nonempty")
    if not test_fold:
        raise ValueError("Davis fixed-test fold must be nonempty")
    train_ids = [_strict_pair_id(pair_id, "training") for fold in training_folds for pair_id in fold]
    test_ids = [_strict_pair_id(pair_id, "fixed-test") for pair_id in test_fold]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Davis training fold IDs contain duplicates")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Davis test fold IDs contain duplicates")
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        raise ValueError(f"Davis train/test folds overlap at pair_idx values: {sorted(overlap)[:5]}")

    dataset_ids = [int(pair_id) for pair_id in training_dataset.df["pair_idx"].tolist()]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Training dataset pair_idx values are not unique")
    if set(dataset_ids) != set(train_ids):
        missing = set(train_ids) - set(dataset_ids)
        unexpected = set(dataset_ids) - set(train_ids)
        raise ValueError(
            "Training dataset IDs do not exactly match the five established folds "
            f"(missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]})"
        )

    row_for_pair = {pair_id: row for row, pair_id in enumerate(dataset_ids)}
    all_rows = set(range(len(dataset_ids)))
    partitions = []
    validation_coverage: List[int] = []
    for fold_number, held_out_ids in enumerate(training_folds, start=1):
        validation_rows = [row_for_pair[int(pair_id)] for pair_id in held_out_ids]
        validation_row_set = set(validation_rows)
        training_rows = sorted(all_rows - validation_row_set)
        if not validation_row_set or not training_rows:
            raise ValueError(f"Fold {fold_number} has an empty train or validation partition")
        if validation_row_set.intersection(training_rows):
            raise AssertionError(f"Fold {fold_number} train/validation rows overlap")
        partitions.append({
            "fold": fold_number,
            "training": Subset(training_dataset, training_rows),
            "validation": Subset(training_dataset, validation_rows),
            "training_pair_ids": [dataset_ids[row] for row in training_rows],
            "validation_pair_ids": [dataset_ids[row] for row in validation_rows],
        })
        validation_coverage.extend(int(pair_id) for pair_id in held_out_ids)

    if len(validation_coverage) != len(train_ids) or set(validation_coverage) != set(train_ids):
        raise ValueError("The five validation folds must cover every Davis training ID exactly once")
    return partitions


def _strict_pair_id(value: Any, source: str) -> int:
    """Accept integer split IDs only; reject bools and lossy numeric casts."""
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"Davis {source} pair IDs must be integers; got {value!r}")
    return int(value)


def load_authoritative_davis_fold_ids(
    raw_dir: str | Path = "data/raw/davis",
) -> tuple[List[List[int]], List[int]]:
    """Read split metadata and Davis dimensions without opening affinity labels.

    Reads the established fold JSON plus ligand/protein dictionaries to verify
    complete coverage of the flattened interaction-index space. It does not
    call ``load_raw_davis_data`` and never opens the raw ``Y`` affinity matrix.
    """
    root = Path(raw_dir)
    folds_dir = root / "folds"
    train_path = folds_dir / "train_fold_setting1.txt"
    test_path = folds_dir / "test_fold_setting1.txt"
    ligand_path = root / "ligands_can.txt"
    protein_path = root / "proteins.txt"
    required = (train_path, test_path, ligand_path, protein_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot resolve authoritative Davis folds without label-free split metadata; "
            f"missing: {', '.join(missing)}"
        )

    with train_path.open("r", encoding="utf-8") as handle:
        raw_training_folds = json.load(handle)
    with test_path.open("r", encoding="utf-8") as handle:
        raw_test_fold = json.load(handle)
    if not isinstance(raw_training_folds, list) or len(raw_training_folds) != CV_FOLD_COUNT:
        raise ValueError(f"Expected exactly {CV_FOLD_COUNT} Davis training folds in {train_path}")
    if any(not isinstance(fold, list) or not fold for fold in raw_training_folds):
        raise ValueError("Every authoritative Davis training fold must be a nonempty list")
    if not isinstance(raw_test_fold, list) or not raw_test_fold:
        raise ValueError("Authoritative Davis fixed-test IDs must be a nonempty list")

    training_folds = [
        [_strict_pair_id(pair_id, "training") for pair_id in fold]
        for fold in raw_training_folds
    ]
    test_fold = [_strict_pair_id(pair_id, "fixed-test") for pair_id in raw_test_fold]
    training_ids = [pair_id for fold in training_folds for pair_id in fold]
    if len(training_ids) != len(set(training_ids)):
        raise ValueError("Authoritative Davis training folds contain duplicate pair IDs")
    if len(test_fold) != len(set(test_fold)):
        raise ValueError("Authoritative Davis fixed-test fold contains duplicate pair IDs")
    overlap = set(training_ids).intersection(test_fold)
    if overlap:
        raise ValueError(f"Authoritative Davis train/test folds overlap: {sorted(overlap)[:5]}")

    # These dictionaries give only the matrix dimensions; they contain no
    # affinity values. Their counts define valid row-major pair_idx range.
    with ligand_path.open("r", encoding="utf-8") as handle:
        ligands = json.load(handle)
    with protein_path.open("r", encoding="utf-8") as handle:
        proteins = json.load(handle)
    if not isinstance(ligands, dict) or not ligands or not isinstance(proteins, dict) or not proteins:
        raise ValueError("Davis ligand/protein dictionaries must be nonempty JSON objects")
    interaction_count = len(ligands) * len(proteins)
    expected_ids = set(range(interaction_count))
    supplied_ids = set(training_ids).union(test_fold)
    if supplied_ids != expected_ids:
        missing_ids = expected_ids - supplied_ids
        unexpected_ids = supplied_ids - expected_ids
        raise ValueError(
            "Authoritative Davis folds do not completely cover the interaction index space "
            f"(missing={sorted(missing_ids)[:5]}, unexpected={sorted(unexpected_ids)[:5]})"
        )
    return training_folds, test_fold


def validate_existing_training_table(
    processed_dir: str | Path,
    training_folds: Sequence[Sequence[int]],
) -> Path:
    """Require and validate only ``davis_train.csv``'s pair_idx column.

    This check runs before the dataset factory, preventing its ordinary missing
    file fallback from loading raw ``Y`` and materializing both train/test
    tables. Affinity labels and all test files are excluded from this read.
    """
    path = Path(processed_dir) / "davis_train.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Paper-protocol tuning requires an existing processed training table: {path}; "
            "preprocessing is disabled for the tuning objective"
        )
    expected_ids = [_strict_pair_id(pair_id, "training") for fold in training_folds for pair_id in fold]
    try:
        pair_frame = pd.read_csv(path, usecols=["pair_idx"])
    except (ValueError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Processed Davis training table lacks a readable pair_idx column: {path}") from exc
    if pair_frame["pair_idx"].isna().any():
        raise ValueError(f"Processed Davis training table contains missing pair_idx values: {path}")
    table_ids = pair_frame["pair_idx"].tolist()
    normalized_ids = [_strict_pair_id(pair_id, "processed-training") for pair_id in table_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Processed Davis training table contains duplicate pair_idx values")
    if set(normalized_ids) != set(expected_ids):
        raise ValueError("Processed Davis training pair_idx values do not match authoritative training folds")
    return path


def load_training_cv_subsets(
    dataset_factory: Callable[..., Any],
    training_folds: Sequence[Sequence[int]],
    test_fold: Sequence[int],
    **dataset_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Construct CV partitions by requesting only `split="train"`.

    This is the safe entry point for a tuning runner: it never asks the
    `DavisDataset` factory to construct the fixed test split.
    """
    dataset = dataset_factory(split="train", **dataset_kwargs)
    return build_training_cv_subsets(dataset, training_folds, test_fold)


def suggest_indexed_architecture(trial: Any) -> Dict[str, int]:
    """Suggest only indexed Table S3 dimensions with direct model mappings.

    The caller still must supply an explicit reproducible sampler/seed. The
    unimplemented/ambiguous hidden-channel dimension is intentionally excluded.
    """
    suggestions = {}
    for name, choices in INDEXED_ESI_SEARCH_CHOICES.items():
        value = trial.suggest_categorical(name, list(choices))
        if not isinstance(value, Integral) or isinstance(value, bool) or value not in choices:
            raise ValueError(f"Trial suggested unsupported {name}={value!r}")
        suggestions[name] = int(value)
    return suggestions


def make_davis_optuna_objective(
    *,
    raw_dir: str | Path = "data/raw/davis",
    processed_dir: str | Path = "data/processed/davis",
    training_config: CVTrainingConfig = CVTrainingConfig(),
    dataset_factory: Optional[Callable[..., Any]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
    dataloader_factory: Optional[Callable[..., Any]] = None,
    fit_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    seed_fn: Optional[Callable[..., None]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[Any], float]:
    """Build a real five-fold Optuna objective using training data only.

    Official fold IDs are loaded from the existing raw fold files. The helper
    reads only those JSON files and ligand/protein dictionary counts, never raw
    affinity matrix ``Y``. Before invoking the dataset factory, it requires an
    existing processed training CSV and validates only its ``pair_idx`` column.
    It never accepts caller-supplied test IDs, test labels, or test metrics.
    Tests may inject a synthetic dataset/model/loader/fit; no checkpoint or
    history path is passed to `fit`.
    """
    if dataset_factory is None or model_factory is None or dataloader_factory is None or fit_fn is None or seed_fn is None:
        # Lazy imports keep the split helpers lightweight and make all training
        # dependencies replaceable in unit tests.
        from src.dataset import DavisDataset
        from src.models.mgraphdta import MGraphDTA
        from src.train import fit, make_dataloader, set_random_seed

        dataset_factory = dataset_factory or DavisDataset
        model_factory = model_factory or MGraphDTA
        dataloader_factory = dataloader_factory or make_dataloader
        fit_fn = fit_fn or fit
        seed_fn = seed_fn or set_random_seed

    dataset_options = dict(dataset_kwargs or {})
    if "raw_dir" in dataset_options or "processed_dir" in dataset_options:
        raise ValueError("Pass raw_dir and processed_dir as objective arguments, not dataset_kwargs")
    training_folds, test_fold = load_authoritative_davis_fold_ids(raw_dir)
    validate_existing_training_table(processed_dir, training_folds)

    partitions = load_training_cv_subsets(
        dataset_factory,
        training_folds,
        test_fold,
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        **dataset_options,
    )

    def evaluate_fold(params: Mapping[str, int], partition: Mapping[str, Any]) -> Mapping[str, Any]:
        fold_seed = int(training_config.seed + int(partition["fold"]) - 1)
        seed_fn(fold_seed, deterministic=training_config.deterministic)
        generator = torch.Generator().manual_seed(fold_seed)
        train_loader = dataloader_factory(
            partition["training"], batch_size=training_config.batch_size,
            shuffle=True, num_workers=training_config.num_workers, generator=generator,
        )
        validation_loader = dataloader_factory(
            partition["validation"], batch_size=training_config.batch_size,
            shuffle=False, num_workers=training_config.num_workers,
        )
        model = model_factory(**params)
        result = fit_fn(
            model,
            train_loader,
            validation_loader,
            epochs=training_config.max_epochs,
            steps_per_epoch=training_config.steps_per_epoch,
            learning_rate=training_config.learning_rate,
            patience=training_config.patience,
            device=training_config.device,
            seed=fold_seed,
            deterministic=training_config.deterministic,
            checkpoint_path=None,
        )
        return {
            "best_validation_mse": result["best_validation_loss"],
            "best_epoch": result["best_epoch"],
        }

    def objective(trial: Any) -> float:
        return evaluate_five_fold_trial(trial, partitions, evaluate_fold)

    return objective


def evaluate_five_fold_trial(
    trial: Any,
    partitions: Sequence[Mapping[str, Any]],
    evaluate_fold: Callable[[Mapping[str, int], Mapping[str, Any]], Mapping[str, Any]],
) -> float:
    """Evaluate a trial using only supplied train/validation partitions.

    `evaluate_fold` must return `best_validation_mse` and `best_epoch` for each
    fold. No test dataset or test metric is accepted by this interface.
    """
    if len(partitions) != CV_FOLD_COUNT:
        raise ValueError(f"Expected {CV_FOLD_COUNT} training-only fold partitions")
    for fold_number, partition in enumerate(partitions, start=1):
        training = partition.get("training")
        validation = partition.get("validation")
        if not isinstance(training, Subset) or not isinstance(validation, Subset):
            raise ValueError(f"Fold {fold_number} must contain training and validation Subsets")
        if training.dataset is not validation.dataset:
            raise ValueError(f"Fold {fold_number} train/validation datasets do not share one train source")
        if set(training.indices).intersection(validation.indices):
            raise ValueError(f"Fold {fold_number} train/validation row indices overlap")
    params = suggest_indexed_architecture(trial)
    fold_scores: List[float] = []
    fold_epochs: List[int] = []
    for step, partition in enumerate(partitions):
        result = evaluate_fold(params, partition)
        score = float(result["best_validation_mse"])
        epoch = int(result["best_epoch"])
        if not np.isfinite(score) or score < 0:
            raise ValueError(f"Fold {step + 1} returned invalid validation MSE: {score}")
        if epoch < 1:
            raise ValueError(f"Fold {step + 1} returned invalid best epoch: {epoch}")
        fold_scores.append(score)
        fold_epochs.append(epoch)
        if hasattr(trial, "report"):
            trial.report(float(np.mean(fold_scores)), step=step)
            if hasattr(trial, "should_prune") and trial.should_prune():
                # Raise Optuna's pruning exception lazily, so importing this
                # data-only module does not require Optuna.
                import optuna

                raise optuna.TrialPruned()
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("fold_validation_mse", fold_scores)
        trial.set_user_attr("fold_best_epochs", fold_epochs)
        trial.set_user_attr("final_epoch_rule", "median_of_five_fold_best_epochs")
    return float(np.mean(fold_scores))


def select_final_epoch(fold_best_epochs: Sequence[int]) -> int:
    """Choose the median of the five fold-best epochs (a project protocol rule)."""
    if len(fold_best_epochs) != CV_FOLD_COUNT:
        raise ValueError(f"Expected {CV_FOLD_COUNT} fold-best epochs")
    if any(not isinstance(epoch, Integral) or isinstance(epoch, bool) for epoch in fold_best_epochs):
        raise ValueError("Fold-best epochs must be integers")
    epochs = [int(epoch) for epoch in fold_best_epochs]
    if any(epoch < 1 for epoch in epochs):
        raise ValueError("Fold-best epochs must all be positive integers")
    return sorted(epochs)[CV_FOLD_COUNT // 2]


def refuse_existing_protocol_root(path: str | Path) -> Path:
    """Create a new experiment root, refusing any existing path."""
    root = Path(path)
    if root.exists():
        raise FileExistsError(f"Paper-protocol output already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    return root


def summarize_seed_metrics(seed_metrics: Mapping[int, Mapping[str, float]]) -> Dict[str, Any]:
    """Compute per-metric mean and sample SD for exactly three final seeds."""
    if tuple(sorted(int(seed) for seed in seed_metrics)) != tuple(sorted(FINAL_SEEDS)):
        raise ValueError(f"Final summary requires exactly the declared seeds {FINAL_SEEDS}")
    metric_names = set.intersection(*(set(values) for values in seed_metrics.values()))
    if not metric_names:
        raise ValueError("No common metrics are available across final seeds")
    summary: Dict[str, Any] = {"seeds": list(FINAL_SEEDS), "metrics": {}}
    for name in sorted(metric_names):
        values = np.asarray([float(seed_metrics[seed][name]) for seed in FINAL_SEEDS])
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {name!r} contains a non-finite seed result")
        summary["metrics"][name] = {
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
        }
    return summary


def write_protocol_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest into a new location; never overwrite an existing file."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary manifest: {temporary}")
    temporary.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one artifact inside a newly allocated study root atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _study_trial_records(study: Any) -> List[Dict[str, Any]]:
    records = []
    for trial in study.trials:
        records.append({
            "number": int(trial.number),
            "state": trial.state.name,
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
            "intermediate_values": {str(k): float(v) for k, v in trial.intermediate_values.items()},
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        })
    return records


def _persist_study_snapshot(study: Any, root: Path) -> None:
    """Persist trial/fold records after each trial and during finalization."""
    records = _study_trial_records(study)
    _atomic_write_text(
        root / "trials.json",
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    csv_rows = []
    for record in records:
        csv_rows.append({
            "number": record["number"],
            "state": record["state"],
            "objective_mean_validation_mse": record["value"],
            "params_json": json.dumps(record["params"], sort_keys=True),
            "fold_validation_mse_json": json.dumps(record["user_attrs"].get("fold_validation_mse")),
            "fold_best_epochs_json": json.dumps(record["user_attrs"].get("fold_best_epochs")),
        })
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[
        "number", "state", "objective_mean_validation_mse", "params_json",
        "fold_validation_mse_json", "fold_best_epochs_json",
    ])
    writer.writeheader()
    writer.writerows(csv_rows)
    _atomic_write_text(root / "trials.csv", buffer.getvalue())

    complete_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    best = None
    if complete_trials:
        best_trial = study.best_trial
        best = {
            "trial_number": int(best_trial.number),
            "objective_mean_validation_mse": float(best_trial.value),
            "params": dict(best_trial.params),
            "fold_validation_mse": best_trial.user_attrs.get("fold_validation_mse"),
            "fold_best_epochs": best_trial.user_attrs.get("fold_best_epochs"),
            "final_epoch_rule": best_trial.user_attrs.get("final_epoch_rule"),
        }
    _atomic_write_text(
        root / "best_parameters.json",
        json.dumps(best, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    failures = [
        {"number": record["number"], "state": record["state"], "params": record["params"]}
        for record in records if record["state"] in {"FAIL", "PRUNED"}
    ]
    _atomic_write_text(
        root / "failures.json",
        json.dumps(failures, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def run_davis_optuna_study(
    *,
    output_root: str | Path,
    trial_count: int,
    sampler_seed: int,
    objective: Callable[[Any], float],
    training_config: CVTrainingConfig,
    paper_evidence: Mapping[str, Any],
    project_decisions: Mapping[str, Any],
) -> Any:
    """Run a fresh, isolated Optuna study and persist its auditable records.

    Every study-level choice is required at the call site. TPE sampling, its
    seed, trial count, no pruning, and schedule choices are project decisions,
    not paper facts. The output root must not already exist. SQLite retains
    trial state; snapshots are refreshed after each trial and on interruption.
    This helper only orchestrates its supplied objective; calling it starts
    real trials and may train five models per trial.
    """
    if not isinstance(trial_count, Integral) or isinstance(trial_count, bool) or trial_count < 1:
        raise ValueError("trial_count must be a positive integer")
    if not isinstance(sampler_seed, Integral) or isinstance(sampler_seed, bool) or sampler_seed < 0:
        raise ValueError("sampler_seed must be a nonnegative integer")
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not isinstance(training_config, CVTrainingConfig):
        raise TypeError("training_config must be a CVTrainingConfig")
    if not isinstance(paper_evidence, Mapping) or not isinstance(project_decisions, Mapping):
        raise TypeError("paper_evidence and project_decisions must be mappings")
    # Validate caller-supplied metadata before allocating an output root.
    try:
        json.dumps(dict(paper_evidence), allow_nan=False)
        json.dumps(dict(project_decisions), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("paper_evidence and project_decisions must be finite JSON data") from exc

    import optuna

    root = refuse_existing_protocol_root(output_root)
    database_path = (root / "optuna_study.db").resolve()
    manifest = {
        "protocol": "Davis training-only five-fold Optuna tuning",
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "paper_evidence": dict(paper_evidence),
        "project_decisions": dict(project_decisions),
        "trial_count": int(trial_count),
        "sampler": {
            "class": "TPESampler",
            "seed": int(sampler_seed),
            "n_startup_trials": TPE_STARTUP_TRIALS,
        },
        "pruner": {"class": "NopPruner", "reason": "project decision; no pruning policy is specified by the paper"},
        "objective": "arithmetic mean of five fold-best validation MSE values",
        "training_config": {
            field: getattr(training_config, field)
            for field in CVTrainingConfig.__dataclass_fields__
        },
        "fold_seed_rule": "training_config.seed + fold_number - 1",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "search_space": {key: list(values) for key, values in INDEXED_ESI_SEARCH_CHOICES.items()},
        "artifact_files": [
            "optuna_study.db", "trials.csv", "trials.json",
            "best_parameters.json", "failures.json", "protocol_manifest.json",
        ],
        "test_partition_policy": "not loaded or supplied to the objective",
    }
    manifest_path = write_protocol_manifest(root / "protocol_manifest.json", manifest)
    storage_url = "sqlite:///" + database_path.as_posix()
    study = None
    status = "failed"
    try:
        study = optuna.create_study(
            study_name="davis_training_only_five_fold",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=int(sampler_seed), n_startup_trials=TPE_STARTUP_TRIALS
            ),
            pruner=optuna.pruners.NopPruner(),
            storage=storage_url,
            load_if_exists=False,
        )

        def persist_after_trial(_study: Any, _trial: Any) -> None:
            _persist_study_snapshot(_study, root)

        study.optimize(
            objective,
            n_trials=int(trial_count),
            callbacks=[persist_after_trial],
            catch=(Exception,),
        )
        _persist_study_snapshot(study, root)
        if not any(trial.state.name == "COMPLETE" for trial in study.trials):
            raise RuntimeError("Optuna study produced no completed trials; inspect failures.json")
        status = "completed"
        return study
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        if study is not None:
            _persist_study_snapshot(study, root)
        final_manifest = dict(manifest)
        final_manifest["status"] = status
        final_manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        # This manifest belongs to the fresh root allocated above; replacing
        # its initial running record is intentional and isolated to that root.
        _atomic_write_text(
            manifest_path,
            json.dumps(final_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )


__all__ = [
    "CV_FOLD_COUNT",
    "FINAL_SEEDS",
    "TPE_STARTUP_TRIALS",
    "INDEXED_ESI_SEARCH_CHOICES",
    "INDEXED_ESI_SELECTED",
    "CVTrainingConfig",
    "build_training_cv_subsets",
    "load_authoritative_davis_fold_ids",
    "validate_existing_training_table",
    "load_training_cv_subsets",
    "suggest_indexed_architecture",
    "make_davis_optuna_objective",
    "evaluate_five_fold_trial",
    "select_final_epoch",
    "refuse_existing_protocol_root",
    "summarize_seed_metrics",
    "write_protocol_manifest",
    "run_davis_optuna_study",
]
