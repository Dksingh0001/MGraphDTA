# Scaled Davis Model-Input Readiness Review

## Scope

This was a read-only review of `SCALING_EXPERIMENT_PLAN.md`, the scaling manifest and CSVs, and current dataset/model interfaces. No model, dataset, checkpoint, metric, or result was changed, and no training was run. This report is the only new file from this review.

## Artifact verification

The manifest records author commit `54386680e8c9562f708a0fe5d381038c62111df6`, author preprocessing source hash `f181f03a6a73c823dffd6c848c9fddacb2da8b259abfb0c8bbbdf8259f557577`, local ligand dictionary hash `141094d99510a52091d2ea1b05be7c1f66b2c5fcaeeeb261eaf4cbf2128bde6d`, RDKit `2026.03.6`, and PyTorch `2.6.0+cu124`.

I independently checked the generated tables and manifest:

- Both files contain 2,180 atom rows, covering all 68 ligands, with 22 numeric features per atom.
- Their headers use the exact same feature order in the manifest: atom type indicators H/C/N/O/F/Cl/S/Br/I; atomic number; acceptor, donor, aromatic; SP/SP2/SP3 hybridization; total hydrogens; explicit valence; formal charge; implicit valence; explicit hydrogens; radical electrons.
- Drug ID, SMILES, and atom index align row-for-row between the two files. Every numeric value in both files is finite.
- Every ligand has a nonzero complete-matrix range (8 to 53). Recalculation of `(x - min(x)) / (max(x) - min(x))` independently for each ligand matches the stored scaled values within `2.8e-8` maximum absolute error. This small difference is consistent with the pinned implementation calculating in float32 before decimal CSV serialization.
- Both generated SHA-256 values match the manifest:
  - unscaled: `8a4129527ab905f1de5684d3e59396950e4e69adddf9c418e440ac221e0e7a43`
  - author-scaled: `26a1e3ed9a873ea7ce758bcb3708d226476746fdb688aa002ffd45cea5a07726`
- The pinned author formula has no constant-matrix guard. A fully constant `[atoms, 22]` matrix would create non-finite values; the generated Davis ligand matrices avoid this case. A zero-range column alone is not divided by its own range: it undergoes the matrix-wide formula. The guarded helper described in the plan is a safer alternate policy, not exact author behavior.

The generated feature tables are molecule-level feature records, not processed interaction datasets. They do not contain target IDs, protein tensors, labels, or train/test row membership.

## Compatibility with the current input interface

`src/dataset.py` currently loads `davis_{split}.csv` from `processed_dir`, builds a molecule graph inside `DavisDataset.__getitem__`, and constructs PyG `Data(x=..., edge_index=..., target=..., y=...)`. `src/preprocessing.py` creates `x` as `[num_atoms, 22]`; `src/models/mgnn.py` consumes `data.x` and `data.edge_index` and expects 22 input features. The scaled feature matrices retain those shapes, feature identities, and atom ordering, so they are shape-compatible with the MGNN.

The flattened `author_scaled_atom_features.csv` is **not** directly consumable by the model or `DavisDataset`: the loader needs a tensor for the specific molecule in each interaction, with atom rows in precisely the graph's atom order. The required future adapter is localized conceptually at graph construction in `DavisDataset.__getitem__`: continue constructing the molecule and its `edge_index` through the existing graph routine, then replace only `graph_info["x"]` with the scaled `[atoms, 22]` tensor looked up by canonical drug ID. It must check atom count, feature width, and atom order before creating `Data`. No model architecture change is indicated.

The current `prepare_davis_tables` writes CSVs if split files are absent. Therefore the proposed versioned directory must be populated and validated before passing it to training; otherwise the loader may regenerate its ordinary CSV representation. No adapter or loader change is implemented here.

## Proposed isolated dataset layout

Use a new location such as `data/processed/davis_scaled_v1/`. For an auditable, drop-in Davis table layout, it should contain exactly these required files:

1. `davis_all.csv` — a byte-identical copy of the current `data/processed/davis/davis_all.csv`.
2. `davis_train.csv` — a byte-identical copy of the current training-fold CSV.
3. `davis_test.csv` — a byte-identical copy of the current fixed test-fold CSV.
4. `ligand_atom_features.pt` — mapping from canonical string drug ID to the author-scaled float32 tensor `[atoms, 22]`, preserving the atom order from the corresponding SMILES. It should be constructed from the generated artifact, with atom indices checked to be contiguous from zero and counts checked against RDKit graph construction.
5. `manifest.json` — schema/version, feature source and order, scaling formula and zero-range policy, pinned author commit/source hash, source and output hashes, RDKit/PyTorch versions, ligand IDs/counts, and hashes for all three copied split files and the feature tensor.

The scaled directory must not be made the default `processed_dir`. The unscaled comparison arm should continue using the current baseline inputs; the treatment arm should point explicitly to this separate directory. Do not regenerate the train/test splits while preparing it.

## Required tests before model training

The dataset-builder/adapter implementation should add tests that establish a single intended difference:

- **Dataset invariance:** the `all`, `train`, and `test` CSVs in the versioned directory are byte-identical or, if copying format changes are unavoidable, equal in every column and row. Check row count, row order, `pair_idx`, drug and target IDs, SMILES, target sequence, Kd, and pKd. Check original train/test index membership and no overlap are unchanged.
- **Protein and edge invariance:** for each tested sample and split, scaled and unscaled variants have identical protein token tensors, original protein lengths, labels, targets, and `edge_index` including edge ordering.
- **Feature isolation:** feature matrices have equal shapes and atom ordering; unscaled matrices equal current preprocessing output; scaled matrices equal the per-drug table transformed by the pinned formula within a documented float tolerance. Confirm no other `Data` fields differ.
- **Coverage and safety:** every drug used by all/train/test has exactly one feature entry; no extra/missing IDs; each matrix is finite, rank 2, width 22, and has atom count matching its molecule. Reject malformed, duplicate, or non-contiguous atom indices and zero-range matrices under the explicitly chosen policy.
- **Provenance and non-overwrite:** check source hashes and manifest; builder refuses to overwrite an existing version directory and does not write under `data/processed/davis/`.
- **Model-interface smoke test:** batch one or more examples from both variants through the existing model in evaluation mode and verify finite outputs of equal shape. This is an interface test, not a performance result and not training.

## Fair controlled comparison design

Treat this as a **feature-scaling ablation**, not a complete reproduction of the MGraphDTA paper.

- Run paired unscaled and author-scaled experiments for the same five seeds (42–46), using the same baseline configuration and existing Davis fixed test fold.
- For each seed, derive the exact same train/validation partition from the training interactions, initialize the models with the same seed, and use the same explicitly seeded batch-order generator. Run paired arms sequentially with isolated output directories. GPU nondeterministic operations may cause small differences; report that limitation rather than claiming bitwise equality.
- Keep architecture, loss, optimizer, learning rate, batch size, epochs/steps, patience, validation fraction, preprocessing other than atom scaling, and software/hardware fixed. Do not tune either arm on the test data.
- Select each arm's checkpoint using validation MSE only, with the same selection rule. Evaluate the selected checkpoints once on the same untouched fixed test fold.
- Save per run: resolved configuration and feature variant, seed and split-index hashes, checkpoint, history, validation-selected epoch, test predictions keyed by stable `pair_idx`, and test metrics MSE, RMSE, CI, and r_m². Produce paired per-seed deltas and mean/sample standard deviation across the five seeds; do not choose a preferred seed based on test performance.
- Keep all outputs under a new experiment root, separate from baseline and prior multi-seed results. Label conclusions narrowly: this measures the effect of this atom-feature transformation under the chosen local protocol; it does not show paper-protocol parity or establish that scaling improves performance.

## Decision

**GO** for implementing the separate versioned feature dataset and an explicit, opt-in loader adapter, subject to the invariance/provenance tests above. The generated feature artifact itself has passed the read-only integrity checks and its dimensions match the model's expected input. This is not approval to train: first build and validate `data/processed/davis_scaled_v1/`, then review its manifest and tests before launching the paired ablation.

### Blockers and limitations

- The model loader cannot currently consume the flattened comparison CSV; the feature lookup adapter and versioned dataset builder still need implementation.
- The official ESI was not obtained in the prior documented attempt, so any ESI-only details remain unknown.
- Existing local processed splits must be copied and hash-checked to preserve rows and fold membership; that versioned dataset has not yet been created.
- The feature comparison does not test predictive benefit. A training experiment and independent analysis would be needed for that conclusion.
