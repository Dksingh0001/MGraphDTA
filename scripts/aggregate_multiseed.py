"""Aggregate completed MGraphDTA multi-seed runs without ranking seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


METRIC_KEYS = {
    "test_mse": "test_mse",
    "test_rmse": "test_rmse",
    "ci": "test_ci",
    "r_m2": "test_r_m2",
}
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_BASELINE = Path("results/davis/run_112_backup/evaluation_metrics_corrected.json")


def _read_run(run_dir: Path, seed: int) -> dict[str, Any]:
    metrics_path = run_dir / "final_metrics.json"
    expected_artifacts = [
        metrics_path,
        run_dir / "training_history.csv",
        run_dir / "training_curve.png",
        run_dir / "checkpoints" / "mgraphdta_best.pt",
    ]
    missing = [str(path) for path in expected_artifacts if not path.is_file()]
    if missing:
        return {"seed": seed, "status": "incomplete", "details": "Missing: " + ", ".join(missing)}

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"seed": seed, "status": "invalid", "details": f"Could not read metrics JSON: {error}"}

    if not isinstance(payload, dict):
        return {"seed": seed, "status": "invalid", "details": "Metrics JSON must contain an object"}

    if payload.get("seed") != seed:
        return {
            "seed": seed,
            "status": "invalid",
            "details": f"Metrics seed is {payload.get('seed')!r}, expected {seed}",
        }

    values: dict[str, float] = {}
    try:
        for output_key, source_key in METRIC_KEYS.items():
            value = float(payload[source_key])
            if not math.isfinite(value):
                raise ValueError(f"{source_key} is not finite")
            values[output_key] = value
    except (KeyError, TypeError, ValueError) as error:
        return {"seed": seed, "status": "invalid", "details": f"Invalid metric data: {error}"}

    try:
        epochs_completed = int(payload.get("epochs_completed", 0))
        best_epoch = int(payload.get("best_epoch", 0))
    except (TypeError, ValueError) as error:
        return {"seed": seed, "status": "invalid", "details": f"Invalid epoch metadata: {error}"}
    if epochs_completed < 1 or best_epoch < 1:
        return {"seed": seed, "status": "incomplete", "details": "Metrics indicate no completed best checkpoint"}

    try:
        with expected_artifacts[1].open(newline="", encoding="utf-8") as handle:
            history_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError) as error:
        return {"seed": seed, "status": "invalid", "details": f"Could not read training history: {error}"}
    if not history_rows:
        return {"seed": seed, "status": "incomplete", "details": "Training history has no epoch rows"}
    try:
        last_epoch = int(history_rows[-1]["epoch"])
    except (KeyError, TypeError, ValueError) as error:
        return {"seed": seed, "status": "invalid", "details": f"Invalid training history: {error}"}
    if len(history_rows) < epochs_completed:
        return {"seed": seed, "status": "incomplete", "details": "Training history has fewer rows than completed epochs"}
    if len(history_rows) != epochs_completed or last_epoch != epochs_completed:
        return {"seed": seed, "status": "invalid", "details": "Training history epoch count disagrees with final metrics"}

    if expected_artifacts[2].stat().st_size < 8 or expected_artifacts[2].read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        return {"seed": seed, "status": "invalid", "details": "Training curve is not a valid PNG file"}
    if expected_artifacts[3].stat().st_size == 0:
        return {"seed": seed, "status": "invalid", "details": "Checkpoint file is empty"}
    return {"seed": seed, "status": "complete", "details": "", "metrics": values}


def _read_baseline(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = {
            output_key: float(payload[source_key])
            for output_key, source_key in METRIC_KEYS.items()
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("baseline contains a non-finite metric")
        return {"status": "available", "path": str(path), "metrics": values}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return {"status": "unavailable", "path": str(path), "details": str(error)}


def aggregate_runs(
    root_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    baseline_path: Path = DEFAULT_BASELINE,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    runs = [_read_run(root_dir / f"seed_{seed}", seed) for seed in seeds]
    failure_manifest = root_dir / "failures.json"
    if failure_manifest.is_file():
        try:
            failure_payload = json.loads(failure_manifest.read_text(encoding="utf-8-sig"))
            failures = failure_payload.get("failures")
            if not isinstance(failures, list):
                raise ValueError("failures must be a list")
            failure_by_seed = {
                int(record["seed"]): record
                for record in failures
                if isinstance(record, dict) and "seed" in record
            }
            manifest_error = ""
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
            failure_by_seed = {}
            manifest_error = str(error)
        for run in runs:
            failure = failure_by_seed.get(run["seed"])
            if failure is not None:
                run["status"] = "failed"
                run["details"] = (
                    f"Training process exited with code {failure.get('exit_code', 'unknown')}; "
                    "artifacts are retained for debugging and excluded from aggregation"
                )
            elif manifest_error:
                run["status"] = "invalid"
                run["details"] = f"Could not read failure manifest: {manifest_error}"
    completed = [run for run in runs if run["status"] == "complete"]
    aggregates: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        values = [run["metrics"][metric] for run in completed]
        aggregates[metric] = {
            "n": len(values),
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }

    summary: dict[str, Any] = {
        "runs_root": str(root_dir),
        "runs": runs,
        "aggregation": {
            "method": "mean and sample standard deviation over complete, valid runs only",
            "valid_run_count": len(completed),
            "excluded_runs": [run["seed"] for run in runs if run["status"] != "complete"],
            "metrics": aggregates,
        },
        "baseline_reference": _read_baseline(Path(baseline_path)),
        "baseline_included_in_aggregation": False,
        "seed_selection_by_test_performance": False,
    }
    return summary


def write_summary(root_dir: Path, summary: dict[str, Any]) -> tuple[Path, Path]:
    root_dir.mkdir(parents=True, exist_ok=True)
    json_path = root_dir / "summary.json"
    csv_path = root_dir / "summary.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing aggregation output: "
            + ", ".join(str(path) for path in existing)
        )
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fields = [
        "record_type", "seed", "status", "details",
        "test_mse", "test_rmse", "ci", "r_m2", "n", "mean", "sample_std",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        baseline = summary["baseline_reference"]
        writer.writerow({
            "record_type": "baseline_reference",
            "status": baseline["status"],
            "details": baseline.get("details", baseline["path"]),
            **baseline.get("metrics", {}),
        })
        for run in summary["runs"]:
            writer.writerow({
                "record_type": "seed_run",
                "seed": run["seed"],
                "status": run["status"],
                "details": run["details"],
                **run.get("metrics", {}),
            })
        for metric, values in summary["aggregation"]["metrics"].items():
            writer.writerow({
                "record_type": f"aggregate_{metric}",
                "status": "computed" if values["n"] else "no_complete_runs",
                "n": values["n"],
                "mean": values["mean"],
                "sample_std": values["sample_std"],
            })
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("results/davis/multiseed"))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    summary = aggregate_runs(args.runs_dir, args.seeds, args.baseline)
    csv_path, json_path = write_summary(args.runs_dir, summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for run in summary["runs"]:
        if run["status"] != "complete":
            print(f"Seed {run['seed']}: {run['status']} — {run['details']}")
    print(f"Included valid runs: {summary['aggregation']['valid_run_count']}/{len(summary['runs'])}")


if __name__ == "__main__":
    main()
