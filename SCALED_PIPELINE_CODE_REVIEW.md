# Scaled-Feature Pipeline Code Review

## Scope

Read-only review of `src/dataset.py`, `src/train.py`, `src/scaled_features.py`, `scripts/build_davis_scaled_dataset.py`, `tests/test_davis_scaled_dataset.py`, `SCALED_FEATURE_DATASET.md`, and `data/processed/davis_scaled_v1/manifest.json`. No files other than this report were changed. No tests were rerun, no training was launched, and no checkpoint was evaluated. The implementation's previous verification reported 21 focused tests and 65 full-suite tests passing, with one CUDA determinism warning.

## Findings by severity

### Medium: checkpoint-only evaluation can write to the baseline metrics path

`src/train.py` passes `--atom-features-dir` consistently into both `train_davis` and `evaluate_checkpoint`. However, when `--evaluate-checkpoint` is combined with the scaled feature option, `--evaluation-output` still defaults to `results/davis/run_112_backup/evaluation_metrics.json`. That is a baseline result location. The evaluated checkpoint is also not checked for which feature variant it was trained with. The output JSON records `processed_dir` and `atom_features_dir`, which helps provenance but does not prevent the overwrite or a feature/checkpoint mismatch.

This does **not** block a fresh training comparison if each arm uses a new `--run-dir` and does not use `--evaluate-checkpoint`. For any later checkpoint-only scaled evaluation, specify a new output path and confirm the checkpoint was trained with the matching feature variant.

### Low: runtime RDKit version is recorded but not enforced

The bundle manifest records RDKit `2026.03.6`. At load time, the code verifies file hashes, feature order, tensor shapes, IDs, and finiteness, but it does not compare the runtime RDKit version with the manifest. Current local atom order is consistent because the adapter reconstructs the graph from the same SMILES and current bundle IDs/SMILES are checked. Still, a different RDKit environment could alter feature semantics while passing the current checks. Use the recorded environment for the controlled comparison; a later hardening change could fail on a version mismatch.

### Low: builder checks split partition IDs, not all split-row contents against `davis_all.csv`

`build_scaled_dataset` ensures train/test `pair_idx` sets are disjoint and cover the all-table pair IDs, but it does not compare each split row's remaining fields (label, protein, SMILES, and order) to the corresponding all-table row. For the current bundle, this is not a data discrepancy: the three generated CSVs are byte-identical to the active tables, and the tests compare each scaled-directory table to the active table. The builder could be stricter for future reuse with different source tables.

### Low: default item dictionary gained an additive `drug_id` field

With `atom_features_dir=None`, graph creation and model-facing `x`, `edge_index`, protein target, and label follow the original unscaled path. The returned Python dictionary now also includes `drug_id`. This does not affect the current training/model consumers, but it is a small API-shape change for external callers that require an exact set of keys.

## Verification against the requested checks

1. **Default unscaled path:** no scaled bundle is loaded when `atom_features_dir` is omitted. The original `extract_graph_structure(mol)` output is used as before. The model-facing graph and label path is unchanged; only the returned item mapping has the additive `drug_id` field noted above.
2. **Adapter isolation:** scaled mode first constructs the same molecule graph and edge index, verifies the manifest SMILES and atom count, and then assigns a cloned cached tensor to `graph_info["x"]`. Protein tokenization, `target`, `y`, labels, sample order, and `edge_index` are not modified by the adapter. Tests compare these fields for actual Davis samples across all/train/test splits.
3. **Hash enforcement:** `load_scaled_feature_bundle` checks schema, pinned commit, feature order, exact required bundle file list, and SHA-256 for all three split CSVs plus `ligand_atom_features.pt`, before returning tensors for use. `DavisDataset` separately checks the selected split table against the manifest hash and refuses to regenerate a missing split during scaled opt-in. A changed file fails before an item is returned. Hashes provide integrity against accidental changes; the manifest is not signed, so this is not an authenticity mechanism against coordinated manifest/file replacement.
4. **CLI consistency:** the same `--atom-features-dir` value is wired into the training path and `--evaluate-checkpoint` path. Both use the selected `processed_dir`. The scaled directory can be selected for both. The medium-severity evaluation-output collision remains as described above.
5. **Bad feature records:** the builder rejects unknown IDs, duplicate/negative/non-contiguous atom indices, missing ligand coverage, SMILES mismatches, nonnumeric/non-finite features, feature-order/header mismatches, zero-range source matrices, formula mismatches, and RDKit atom-count mismatches. The loader rejects missing/duplicate manifest IDs, tensor ID mismatches, invalid dtype/shapes, non-finite tensors, and corrupted bundle hashes. Per-item validation also checks ID, SMILES, atom count, feature width, and finiteness.
6. **Exactly-once transformation:** the bundle builder compares scaled CSV values with the pinned author global formula computed from saved unscaled values. The dataset adapter does not call a scaler; it only loads and assigns the stored `[atoms, 22]` tensor. Thus the stored transform is applied once, with no second scaling in the loader or model.
7. **Test quality:** the new tests use the actual local Davis tables, feature CSVs, generated bundle, RDKit graph construction, and `DavisDataset`; they are not solely mocks. They cover row/hash parity, split membership, protein/label/edge preservation, the default and opt-in paths, malformed records, and reproducible hashes. The previous test run was 21 focused and 65 total tests passing.
8. **CUDA warning:** the prior full-suite warning says CUDA had already initialized before deterministic cuBLAS configuration, so bitwise repeatability might not be available in that process. It is a `RuntimeWarning`, not a test failure; the full suite still reported 65 passed. The earlier GPU issue with unsupported deterministic pooling remains a reason to use `--no-deterministic` for GPU runs. RNG seeding still applies, but GPU operations may differ slightly.

## Bundle and transformation status

The manifest identifies the pinned author commit `54386680e8c9562f708a0fe5d381038c62111df6`, 68 ligands, 2,180 atoms, and 22 ordered features. It records the formula `(x - x.min()) / (x.max() - x.min())`, reduced over each complete molecule matrix. The bundle files' declared hashes match the data used in the implementation's passing tests. The safety policy rejects zero-range matrices, while the pinned author code itself has no guard. The official ESI remains unavailable as documented in the prior report, and this ablation would not establish full paper reproduction or prove that scaling improves performance.

## Readiness

**GO for a fresh, controlled training comparison**, provided each arm uses a new, separate run directory, the same seed/settings, and the recorded environment. The checkpoint-only evaluation output collision does not affect a fresh `train_davis` run, but avoid `--evaluate-checkpoint` for this comparison. Use `--no-deterministic` on the known-problem GPU path and describe results as seeded GPU runs rather than bitwise-reproducible runs.

For the first paired comparison at seed 42, using the current CLI baseline settings (1000 max epochs, 50 steps/epoch, batch size 512, learning rate 0.0005, patience 400, validation fraction 0.1), the commands to run **after approval** are:

```powershell
.\.venv\Scripts\python.exe -m src.train --processed-dir data/processed/davis --seed 42 --device cuda --no-deterministic --run-dir results/davis/scaling_ablation_v1/unscaled/seed_42
.\.venv\Scripts\python.exe -m src.train --processed-dir data/processed/davis_scaled_v1 --atom-features-dir data/processed/davis_scaled_v1 --seed 42 --device cuda --no-deterministic --run-dir results/davis/scaling_ablation_v1/scaled/seed_42
```

These commands were not executed. Each run will follow the existing training pipeline, which selects checkpoints by validation loss and subsequently evaluates the fixed test fold. Keep all five paired seeds under the same new experiment root before comparing aggregate results; do not use test performance to choose a seed.
