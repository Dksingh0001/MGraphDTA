# Multi-Seed Artifact Validation Report

**Date:** 2026-10-17
**Scope:** Read-only inspection of `results/davis/multiseed_retry_20261016/`, its per-seed artifacts, the existing summaries, the launcher/aggregator behavior, and `PAPER_COMPLIANCE_AUDIT.md`. No training was launched. No existing artifacts or summaries were changed, and no corrected aggregate was generated.

## Summary

The saved artifacts support that seeds 44, 45, and 46 each reached the configured 150 epochs, wrote a best checkpoint, and wrote final test metrics. The histories agree with the claimed completed epoch count and best validation epoch, and checkpoint metadata agrees with each history’s best epoch and validation loss. This is strong evidence that training and evaluation reached their artifact-writing stages.

The process outcomes are not all established: the current `failures.json` records exit code 1 for seeds 42, 43, and 45. That makes seed 45 a process failure despite its complete-looking outputs. For seeds 44 and 46, the current failure manifest has no failure entry, but the older summary marks them failed. The summary and failure manifest have different modification times and contradictory run states. No per-process stdout/stderr or launcher log was found to resolve why these process statuses differ.

The saved metrics are internally plausible and tied to the best-validation checkpoint by metadata. They are **not independently verified test measurements**: no saved test predictions, per-example labels, or evaluation transcript exists to recompute MSE, CI, and r_m² from the artifacts alone. File presence, JSON values, and matching checkpoint metadata do not independently prove that the metrics were computed from the intended test examples.

## Per-seed evidence

The launcher configuration recorded in each complete metrics JSON is 150 requested epochs, 50 steps per epoch, batch size 32, learning rate 0.0005, patience 30, validation fraction 0.1, CUDA, seed-specific output directory, and deterministic algorithms disabled. This is the capped exploratory setup described in the paper compliance audit, not the full paper reproduction protocol.

| Seed | Process status evidence | Training artifacts and history | Checkpoint selection evidence | Evaluation evidence | Recommended status / next action |
|---:|---|---|---|---|---|
| 42 | `failures.json` records exit code 1. | History has 40 rows (epochs 1–40); checkpoint exists, metadata says epoch 40 and seed 42. No `final_metrics.json` or training curve. | History minimum is validation MSE 0.4799685263 at epoch 40; checkpoint metadata agrees. | No saved final test metrics. | **Failed/incomplete.** Keep as debugging evidence; exclude from any aggregate. Retry only in a new run directory after reviewing the failure log, if available. |
| 43 | `failures.json` records exit code 1. | History has one row (epoch 1); checkpoint exists, metadata says epoch 1 and seed 43. No metrics JSON or curve. | The sole history row is best validation MSE 0.8823934477; checkpoint metadata agrees. | No saved final test metrics. | **Failed/incomplete.** Keep as debugging evidence; exclude from any aggregate. Retry only in a new run directory after reviewing the failure log, if available. |
| 44 | Current `failures.json` has no entry, but `summary.json`/`.csv` mark it failed. No process log was found. | History has 150 rows (1–150), curve exists, checkpoint exists, final metrics JSON exists; `epochs_completed=150`, `stopped_early=false`. | Best history row is epoch 150, validation MSE 0.4321970014; checkpoint metadata reports epoch 150 and the same best loss. | JSON reports MSE 0.4241899077, RMSE 0.6512986318, CI 0.8162086968, r_m² 0.4668448174. It also records an `independent_test_mse` of 0.4241899124, close to `test_mse`. No predictions/transcript to reproduce the values. | **Training appears complete; evaluation artifacts internally consistent; process outcome unresolved.** Preserve artifacts. Verify checkpoint by a fresh read-only evaluation and capture exit status/log before treating as aggregate-valid. |
| 45 | Current `failures.json` records exit code 1, but `summary.json`/`.csv` mark it complete. No process log explains the exit. | History has 150 rows, curve/checkpoint/metrics exist; `epochs_completed=150`, `stopped_early=false`. | Best history row is epoch 145, validation MSE 0.4434660903; checkpoint metadata reports epoch 145 and the same best loss. | JSON reports MSE 0.4374538017, RMSE 0.6614029042, CI 0.8231561607, r_m² 0.4403202479. `independent_test_mse` is 0.4374538026. No saved predictions/transcript. | **Training/evaluation artifacts appear complete, but process is reported failed.** Exclude under a conservative process-success criterion until failure cause is understood and checkpoint evaluation is independently verified. |
| 46 | Current `failures.json` has no entry, but `summary.json`/`.csv` mark it failed. Its artifacts and the failure manifest were updated around 23:06; no process log was found. | History has 150 rows (1–150), curve/checkpoint/metrics exist; `epochs_completed=150`, `stopped_early=false`. | Best history row is epoch 148, validation MSE 0.3935311230; checkpoint metadata reports epoch 148 and the same best loss. | JSON reports MSE 0.4012697076, RMSE 0.6334585287, CI 0.8351589640, r_m² 0.4924510674. `independent_test_mse` is 0.4012697129. No saved predictions/transcript. | **Training appears complete; evaluation artifacts internally consistent; process outcome unresolved.** Preserve artifacts. Verify checkpoint by a fresh read-only evaluation and capture exit status/log before treating as aggregate-valid. |

## Cross-checks and status conflicts

### Histories and checkpoints

The inspected checkpoint files for all five seeds are nonempty and load as PyTorch checkpoints with `weights_only=True`. Their recorded `seed`, `epoch`, and `best_validation_loss` agree with the per-seed histories:

- Seeds 42 and 43 have checkpoint metadata matching their partial history endpoints.
- Seeds 44–46 have best checkpoint epochs matching the minimum validation-MSE row in their 150-row histories.
- For seeds 44–46, `epochs_completed` equals the number of history rows; all requested 150 epochs completed and early stopping is false.

This supports the training-history and checkpoint-selection claims. It does not alone prove that each checkpoint’s tensor state is the exact state saved at that epoch (although the training code saves it when validation improves), nor does it verify test predictions.

### Why the summary statuses conflict

The existing `summary.json` and `summary.csv` were last modified at **19:03 on 2026-10-16**. The current `failures.json` was last modified at **23:06 on 2026-10-16**. The summary says seeds 42, 43, 44, and 46 failed and seed 45 completed. The later failure manifest says seeds 42, 43, and 45 exited with code 1, with no entries for 44 or 46.

The modification times and content establish that the failure manifest and summaries reflect different snapshots of run status (or one was subsequently changed). They do **not** establish exactly who/what changed the manifest or why each process returned code 1. The summary’s seed 46 failure predates seed 46’s final metrics/curve; seed 44’s final metrics existed before the summary, yet the summary still calls it failed. Seed 45 is the inverse conflict: it is complete in the summary, but a later manifest records exit code 1. This cannot be resolved without the original launcher/process logs or an explicit run ledger.

`summary.csv` duplicates the status and one-run aggregate from `summary.json`; it is not an independent source. Both say `valid_run_count=1`, include seed 45’s values in the sole-run aggregate, and exclude 44 and 46. The current failure manifest, if applied to the artifact set, would instead mark seed 45 failed and leave 44/46 without a process failure record. Therefore the current summary is **not a reliable corrected three-seed aggregate**, and the manifest is not enough to prove process success for seeds absent from it.

### Failed process vs. completed training vs. valid evaluation

These are separate claims:

1. **Process failed:** established only where a contemporaneous process exit record exists. The current failure manifest records nonzero exit code 1 for seeds 42, 43, and 45. It does not explain the failure stage or cause.
2. **Training completed:** supported for seeds 44–46 by 150 sequential history rows, `epochs_completed=150`, and checkpoint metadata consistent with each recorded best epoch. It is not supported for 42 or 43.
3. **Evaluation artifacts exist and are internally consistent:** supported for seeds 44–46 by final metrics JSONs, independent-test MSE fields, and history/checkpoint agreement. The RMSE values are consistent with square-rooting the saved MSE to displayed precision. The two saved MSE fields are close but not identical, consistent with two separate evaluation passes on a nondeterministic GPU; this needs actual recomputation to confirm.
4. **Evaluation is independently validated:** not established for any seed. Test predictions and an evaluation log are absent. A valid-looking JSON cannot confirm which checkpoint/data rows were evaluated or independently reproduce CI/r_m². In a conservative aggregate, “artifact-complete” should not be equated with “independently validated.”

## Proposed criteria for a corrected aggregate

Do not aggregate until approved. A safe corrected summary should be written under a **new, previously unused output directory**, leaving `multiseed_retry_20261016/`, its `summary.*`, failure manifest, and baseline untouched. For example, use a fresh directory such as `results/davis/multiseed_validated_20261017/`; first confirm it does not already exist.

Proposed acceptance criteria for a seed to be labeled **validated and included**:

1. The run’s seed and expected configuration in `final_metrics.json` match the directory and experiment manifest.
2. The process has a recorded exit code of 0 from the same run attempt, with no contradictory failure record. A seed recorded as failed remains excluded until the cause is resolved; complete-looking outputs do not erase a nonzero process exit.
3. Training history parses, has sequential epoch numbers, and its row count equals `epochs_completed`. `best_epoch` is in range and equals the minimum validation MSE row. `stopped_early` and requested/completed epoch metadata are coherent.
4. The checkpoint loads safely, and its stored seed, epoch, and best validation loss match the history/metrics.
5. Required curve is a valid PNG; required metrics are finite and internally coherent, including RMSE≈sqrt(MSE).
6. A fresh **read-only** evaluation of that saved checkpoint on the configured independent Davis test fold reproduces MSE, RMSE, CI, and r_m² within a predeclared numeric tolerance. The validation record stores the evaluator command/version, checkpoint hash, test-fold identity, output values, and tolerance. It does not write to the original run directory.

Under these criteria, 42 and 43 are incomplete and excluded; 45 is excluded until its exit-code-1 report is resolved; 44 and 46 are candidates, not yet validated, because their process outcomes conflict with the saved summary and independent test recomputation has not been performed. If process logs cannot be recovered for 44/46, label them **artifact-complete, exit-status-unknown** and keep them out of the strict validated aggregate (or report separately in an explicitly non-validated appendix).

Once approved, perform read-only checkpoint re-evaluation first; then write only the corrected aggregate and its validation manifest under the new root. Never use test performance to choose among seeds.

## Confirmed and unresolved

**Confirmed:** seeds 42 and 43 have partial histories/checkpoints and no final evaluation artifacts; seeds 44, 45, and 46 have 150-epoch histories and corresponding best-checkpoint metadata; their final metrics JSONs agree with history/checkpoint best-validation information; the summaries and failure manifest disagree; baseline references in the existing summary are separate from the multi-seed aggregate.

**Unresolved:** whether seeds 44 and 46 exited successfully; why seed 45 exited with code 1 after producing complete-looking artifacts; whether the manifest was rewritten or came from a later launch; whether saved test metrics reproduce from the saved checkpoints and exact test fold; whether two evaluation passes’ small MSE differences are just GPU numeric variation; and whether any process logs exist outside the searched project/run folders.

## Single next action

Approve the proposed validation criteria and authorize read-only re-evaluation of the seed 44–46 checkpoints. Do not generate a corrected aggregate until that validation is complete and the output directory is confirmed unused.
