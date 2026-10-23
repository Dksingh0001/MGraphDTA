# Multi-Seed Checkpoint Re-Evaluation Report

**Date:** 2026-10-17
**Scope:** Seeds 44, 45, and 46 in `results/davis/multiseed_retry_20261016/`. This was a read-only checkpoint evaluation, not training. The existing processed Davis test CSV was confirmed to exist before evaluation, so the dataset loader did not need to generate or save processed data. No checkpoint, dataset, metrics JSON, history, failure manifest, or aggregate was changed. No corrected aggregate was generated.

## Result

Each saved checkpoint was loaded in a fresh Python process, and test metrics were recomputed in memory using the existing Davis test CSV, batch size 32, the current `MGraphDTA` implementation, and `evaluate_test_metrics`. The saved JSON values were not used as inputs to this calculation. All four metrics reproduced within absolute tolerance `1e-6`.

These are fresh re-evaluations of the checkpoint artifacts, not an independent reimplementation of the metric code. They verify that the saved checkpoints produce the reported metrics (within GPU numeric variation) on the currently prepared test fold. They do not establish that the current test CSV is byte-for-byte the test fold used by the original process, or explain that process's exit status.

## Per-seed assessment

| Seed | Classification | History/checkpoint consistency | Saved test metrics | Fresh re-evaluation | Process status evidence |
|---:|---|---|---|---|---|
| 44 | **ARTIFACT-COMPLETE, TEST METRICS INDEPENDENTLY RE-EVALUATED** | 150 sequential rows; history minimum validation MSE is 0.4321970014 at epoch 150. Checkpoint metadata: seed 44, epoch 150, same validation loss. Metrics JSON says 150/150 epochs and no early stop. | MSE 0.4241899077; RMSE 0.6512986318; CI 0.8162086968; r_m² 0.4668448174. | MSE 0.4241899065; RMSE 0.6512986308; CI 0.8162090031; r_m² 0.4668448226. All differences ≤ 0.000001. | Current `failures.json` has no seed 44 entry, but `summary.json` and `.csv` call it failed. No process log found; exit outcome unresolved. |
| 45 | **ARTIFACT-COMPLETE, TEST METRICS INDEPENDENTLY RE-EVALUATED** | 150 sequential rows; history minimum validation MSE is 0.4434660903 at epoch 145. Checkpoint metadata: seed 45, epoch 145, same validation loss. Metrics JSON says 150/150 epochs and no early stop. | MSE 0.4374538017; RMSE 0.6614029042; CI 0.8231561607; r_m² 0.4403202479. | MSE 0.4374538071; RMSE 0.6614029083; CI 0.8231555481; r_m² 0.4403202385. All differences ≤ 0.000001. | Current `failures.json` records exit code 1; `summary.json` and `.csv` call it complete. There is no process log explaining the nonzero exit. This conflict does not negate the independently reproduced checkpoint metrics, but the process outcome remains failed as recorded/ambiguous. |
| 46 | **ARTIFACT-COMPLETE, TEST METRICS INDEPENDENTLY RE-EVALUATED** | 150 sequential rows; history minimum validation MSE is 0.3935311230 at epoch 148. Checkpoint metadata: seed 46, epoch 148, same validation loss. Metrics JSON says 150/150 epochs and no early stop. | MSE 0.4012697076; RMSE 0.6334585287; CI 0.8351589640; r_m² 0.4924510674. | MSE 0.4012697098; RMSE 0.6334585305; CI 0.8351598064; r_m² 0.4924510639. All differences ≤ 0.000001. | Current `failures.json` has no seed 46 entry, but `summary.json` and `.csv` call it failed. Its final metrics and curve were written around 23:06, after the summaries. No process log found; exit outcome unresolved. |

The classifications use a more specific status than the initial report’s “metrics not independently verified” option because the approved fresh re-evaluations have now reproduced the saved metrics. The classification concerns the completeness and metric validity of the saved artifacts; it does not silently convert a nonzero process exit into a successful process. Seed 45 remains recorded as a process failure in the current failure manifest, with the reason and timing unknown.

## What is internally consistent

- All three `final_metrics.json` files parse and identify the corresponding seed.
- Each history has epochs 1 through 150; `epochs_completed=150`, `epochs_requested=150`, and `stopped_early=false` agree.
- Each history’s minimum validation MSE and epoch agree with `best_epoch` and `best_validation_mse` in its metrics JSON.
- Each safely loaded checkpoint records the matching seed, best epoch, and best validation loss.
- Each saved RMSE is consistent with the square root of its saved MSE.
- Fresh test evaluation from each saved checkpoint reproduces all four JSON metrics to within absolute tolerance `1e-6`. The largest observed discrepancy is in CI (seed 46, approximately `8.42e-7`), consistent with small GPU numeric differences under nondeterministic execution.

## Process-status and summary conflict

The existing `summary.json` and `summary.csv` have timestamp 19:03 on 2026-10-16. They mark seeds 42, 43, 44, and 46 failed and seed 45 complete. The current `failures.json`, timestamped 23:06 on 2026-10-16, records code 1 for seeds 42, 43, and 45, but not 44 or 46. No stdout/stderr or launcher log was found in the experiment root or scripts directory.

The timestamps and records establish that the summary and current failure manifest represent conflicting status snapshots. They do not establish why seed 45 returned code 1, whether the manifest was updated by a later launch, or what process status seeds 44 and 46 had. Seed 46’s final artifacts postdate the summary, so the summary could not reflect those final artifacts. Seed 44’s metrics existed before the summary but the summary still marks it failed. Keep these process-status conflicts separate from the independently re-evaluated metrics.

No aggregate was generated. The prior summaries and `failures.json` remain untouched. The earlier baseline remains separate from these three seed runs.

## What can and cannot be verified

**Verified now:** each saved best checkpoint reproduces its recorded Davis test metrics, within the stated tolerance, when evaluated against the current processed Davis test fold with the current code. Histories and checkpoint metadata also support best-validation checkpoint selection.

**Still uncertain:** original process exit details for seeds 44 and 46; why seed 45 exited nonzero; whether the on-disk processed test CSV is exactly the one used by the original run; and whether original-process predictions used the same software/hardware state. The fresh evaluations are repeatable evidence for the current artifacts, not a retained audit trail of the original process.

To fully audit historical metric provenance, retain a validation record containing the checkpoint SHA-256, test CSV/fold identity or hash, evaluator/code version, environment details, exact command, per-example prediction and target vectors, and recomputed metrics. The prediction/target vectors are not necessary to reproduce the numbers from the checkpoint now, but are needed to independently inspect the metric calculation and pin down what the original run evaluated.

No aggregate has been authorized in this step. Any later aggregate should be written to a confirmed-unused new directory and should preserve the process exit conflict as metadata instead of overwriting prior run records.
