"""Tests for multi-seed result validation and aggregation."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.aggregate_multiseed import aggregate_runs, write_summary


def _write_run(root: Path, seed: int, mse: float) -> None:
    run_dir = root / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "mgraphdta_best.pt").write_bytes(b"test checkpoint")
    (run_dir / "training_history.csv").write_text("epoch\n1\n2\n", encoding="utf-8")
    (run_dir / "training_curve.png").write_bytes(b"\x89PNG\r\n\x1a\nminimal-test")
    (run_dir / "final_metrics.json").write_text(
        json.dumps({
            "seed": seed,
            "epochs_completed": 2,
            "best_epoch": 1,
            "test_mse": mse,
            "test_rmse": mse + 1,
            "test_ci": 0.7,
            "test_r_m2": 0.2,
        }),
        encoding="utf-8",
    )


def test_aggregator_includes_each_status_and_excludes_baseline(tmp_path):
    runs_dir = tmp_path / "multiseed"
    _write_run(runs_dir, 42, 0.2)
    _write_run(runs_dir, 43, 0.4)
    _write_run(runs_dir, 44, 0.5)
    invalid = runs_dir / "seed_44"
    (invalid / "final_metrics.json").write_text("not json", encoding="utf-8")

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "test_mse": 9,
        "test_rmse": 8,
        "test_ci": 0.1,
        "test_r_m2": 0.01,
    }), encoding="utf-8")

    summary = aggregate_runs(runs_dir, seeds=(42, 43, 44, 45), baseline_path=baseline)
    assert [run["status"] for run in summary["runs"]] == [
        "complete", "complete", "invalid", "incomplete"
    ]
    assert summary["aggregation"]["valid_run_count"] == 2
    assert summary["aggregation"]["excluded_runs"] == [44, 45]
    assert abs(summary["aggregation"]["metrics"]["test_mse"]["mean"] - 0.3) < 1e-12
    assert abs(summary["aggregation"]["metrics"]["test_mse"]["sample_std"] - 0.1414213562) < 1e-9
    assert summary["baseline_included_in_aggregation"] is False
    assert summary["baseline_reference"]["metrics"]["test_mse"] == 9

    csv_path, json_path = write_summary(runs_dir, summary)
    assert json.loads(json_path.read_text(encoding="utf-8"))["aggregation"]["valid_run_count"] == 2
    with csv_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len([row for row in records if row["record_type"] == "seed_run"]) == 4
    assert any(row["record_type"] == "baseline_reference" for row in records)
    try:
        write_summary(runs_dir, summary)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing aggregation files must not be overwritten")


def test_launcher_failure_manifest_excludes_failed_run_even_if_artifacts_look_complete(tmp_path):
    runs_dir = tmp_path / "multiseed"
    _write_run(runs_dir, 42, 0.2)
    (runs_dir / "failures.json").write_text(
        json.dumps({"failures": [{"seed": 42, "exit_code": 1}]}), encoding="utf-8"
    )

    summary = aggregate_runs(runs_dir, seeds=(42,))
    assert summary["runs"][0]["status"] == "failed"
    assert summary["aggregation"]["valid_run_count"] == 0
    assert summary["aggregation"]["excluded_runs"] == [42]


def test_multiseed_launcher_uses_configurable_root_and_nondeterministic_gpu_mode():
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_multiseed.ps1"
    source = launcher.read_text(encoding="utf-8")

    assert '[string]$ExperimentRoot = "results/davis/multiseed_retry_20261016"' in source
    assert '[int[]]$Seeds = @(42, 43, 44, 45, 46)' in source
    assert "--no-deterministic" in source
    assert "--run-dir $runDirectory" in source
    assert 'Join-Path $ExperimentRoot "seed_$seed"' in source
    assert "--runs-dir $ExperimentRoot" in source
    assert 'Set-Content -LiteralPath $failureLog' in source
    assert "choose a new -ExperimentRoot" in source
    assert "Remove-Item" not in source
    assert "failedRuns += [pscustomobject]" in source
