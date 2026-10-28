# MGraphDTA Project Technical Audit

**Audit date:** 2026-10-27 (Asia/Calcutta)  
**Scope:** Read-only audit of `D:\MGraphDTA`. No source code, training outputs, checkpoints, data files, or Git history were changed. This report is the only new artifact.

## 1. Executive summary

This repository contains an end-to-end MGraphDTA-style drug–target affinity regressor, Davis ingestion and preprocessing, training/evaluation code, a separate 2D protein-graph experiment, Grad-AAM visualizations, tests, and a pinned copy of the authors’ repository. The central model encodes a drug’s atom/bond graph with MGNN and a protein amino-acid sequence with MCNN, then predicts a continuous affinity label.

Verified data facts: 68 compounds × 442 target IDs produce 30,056 interaction records; the IDs contain 379 unique sequences. The project uses the supplied fold indices: 25,046 training and 5,010 test records. It converts Kd in nM to pKd using `9 - log10(Kd_nM)`. Molecules become graphs with 22 raw atom features and bidirected bond connectivity. The default MGNN/MCNN and regression-head dimensions are inspectable in source.

The primary seed-42 experiment (`run_112_backup`) completed 112 of 150 requested epochs and was manually stopped. Best validation epoch was 91. It used batch 32, 50 steps/epoch, learning rate 0.0005, validation fraction 0.1 and patience 30. Test metrics: MSE 0.2570411, RMSE 0.5069922, CI 0.8790452, r_m² 0.6817056. A separate unscaled batch-512 seed-42 run completed 182 epochs and early-stopped with MSE 0.2403023, RMSE 0.4902064, CI 0.8839608, r_m² 0.6712085. Three seed-44–46 checkpoint evaluations reproduce saved metrics on the current processed test table within 1e-6, but historical process-status files conflict. These are not controlled comparable runs: batch size, requested epoch budget/patience, seed, deterministic execution and possibly source/data provenance differ.

A material paper/release difference is atom scaling: pinned author preprocessing min–max scales each molecule’s entire atom-feature matrix; the default local pipeline feeds raw features. A distinct optional scaled path and scaled run exist. The local paper-protocol module describes CV/Optuna settings, but no artifacts establish completion. This project should not be described as a complete reproduction.

Unresolved: `bio_paper.pdf` is present, but local extraction stopped at MiKTeX first-run setup; full ESI retrieval is documented as unsuccessful. Seed process outcomes for 44–46 conflict across files. Original primary-run data hashes and exact validation indices are absent. Environment versions are documented but were not queried live in this audit. Existing docs report a focused 37-test pass, not a fresh full-suite run.

## 2. Purpose and research question

Drug–target binding affinity prediction estimates how strongly a small molecule binds a protein target. A predicted score can prioritize expensive experimental follow-up, but it is not a binding measurement, clinical effect, or proof of efficacy.

Davis is a benchmark matrix of drug–target dissociation constants (Kd). One interaction record pairs one compound with one target ID and its affinity label. The loader constructs the full 68-by-442 grid, then indexes it with the provided folds. Multiple target IDs can have identical sequences: 442 IDs represent 379 unique sequences. The loader retains each pair rather than collapsing by sequence.

The target is **pKd**, not Kd in nM:

`pKd = -log10(Kd in molar) = 9 - log10(Kd in nM)`.

Higher pKd corresponds to lower Kd and nominally stronger binding. pKd is a dimensionless logarithmic measure; RMSE is in pKd units and MSE in squared pKd units. The loader documents 10,000 nM as the Davis non-binding convention (pKd 5). No target normalization is performed in the default loader/trainer.

A **molecular graph** represents atoms as nodes and bonds as edges, with node attributes describing chemistry. A **protein sequence** is an ordered string of amino-acid residue symbols; this project maps symbols to integer IDs and applies 1D convolutions. Sequence input is not a 3D structure. This is regression because the output is a continuous pKd value and the loss is squared numeric error, not a binary binder/non-binder label.

**Verified:** pair construction, transform, representations and labels are in `src/dataset.py` and `src/preprocessing.py`. **Not established:** assay provenance and measurement uncertainty for every cell, or biological interpretation of a particular prediction.

## 3. Structure and stack

| Path | Purpose / connection |
|---|---|
| `src/dataset.py` | Loads ligand/protein dictionaries, affinity matrix and folds; creates interaction tables and PyG samples. |
| `src/preprocessing.py` | RDKit parsing, atom features, graph edges, protein tokenization and fixed-length sequences. |
| `src/models/mgnn.py` | Dense graph encoder: GraphConv, node BatchNorm, ReLU, dense blocks, transitions and global mean pool. |
| `src/models/mcnn.py` | Amino-acid embedding, three Conv1D scales and adaptive max pooling. |
| `src/models/mgraphdta.py` | Fuses encodings and predicts scalar pKd. |
| `src/train.py` | Main trainer, seeded train/validation split, checkpoint/early stopping, evaluation and CLI. |
| `src/paper_protocol.py` | CV/Optuna helpers/configuration; not evidence that the protocol was run. |
| `src/metrics.py` | MSE, RMSE, CI and modified r_m². |
| `src/explainability/grad_aam.py` | Gradient-weighted atom attribution for one pair. |
| `scripts/visualize_grad_aam.py` | Loads checkpoint/example, colors RDKit atoms, saves PNG and JSON. |
| `scripts/run_multiseed.ps1`, `scripts/aggregate_multiseed.py` | Multi-seed launch/aggregation; retained status records conflict. |
| `scripts/check_environment.py` | Runtime environment report; not executed for this audit. |
| `scripts/train_protein_gnn_affinity.py`, `scripts/protein_gnn.py` | Separate experiment replacing sequence MCNN with residue graphs. |
| `tests/` | Tests of data, features, model components, metrics, training, attribution and protocol. Presence is not a passing result. |
| `data/raw/davis/` | Local JSON dictionaries, pickled `Y`, and JSON fold lists. |
| `data/processed/davis/` | Default unscaled `all/train/test` CSV tables. |
| `data/processed/davis_scaled_v1/` | Separate versioned scaled-feature bundle. |
| `results/davis/` | Run histories, checkpoints, evaluations, multi-seed, scaling, protein graph and attribution artifacts. |
| `references/mgraphdta_official/author_repo/` | Author code checkout at `54386680e8c9562f708a0fe5d381038c62111df6`; parent Git reports it as a nested repository. |
| `references/mgraphdta_official/DeepDTA_source/` | Davis data/fold reference checkout. |
| `bio_paper.pdf` | Local cited paper PDF; text extraction prompted for MiKTeX first-run setup and was not completed. |
| `README.md`, `PROJECT_PLAN.md`, `PAPER_*`, `REFERENCE_*`, `MULTISEED_*`, `SCALED_*` | Project documentation and prior audits. Source/artifacts take precedence when checking claims. |
| `report/` | Existing LaTeX drafts, figures and archive. Existing untracked report artifacts were preserved. |

`requirements.txt` pins Python packages including torch 2.6.0+cu124, torchvision 0.21.0+cu124, PyG 2.8.0.post1, RDKit 2026.03.6, NumPy 2.5.2, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2. `PROJECT_PLAN.md` documents Python 3.12, PyTorch 2.6.0+cu124, PyG 2.8.0 and RDKit 2026.03.6. Reconcile the PyG version difference using a live environment capture before publication.

## 4. Data and preprocessing

### Provenance and counts

The loader expects `data/raw/davis/ligands_can.txt` (JSON compound-ID→SMILES), `proteins.txt` (JSON target-ID→sequence), `Y` (pickled Kd matrix), and `folds/train_fold_setting1.txt` / `test_fold_setting1.txt` (JSON flattened indices). Local provenance notes report the matrix/folds match DeepDTA references and the author archive’s train/test interaction records correspond to local folds.

| Quantity | Count |
|---|---:|
| Compounds | 68 |
| Target IDs | 442 |
| Unique sequences | 379 |
| Interactions | 30,056 |
| Training interactions | 25,046 |
| Test interactions | 5,010 |
| Longest sequence | 2,549 residues |

Records are built ligand-major, target-minor; `pair_idx = drug_index * number_of_targets + target_index`. Five training fold lists are flattened into one training table; the separate test indices form test. Default training does not run five-fold CV over those lists. It takes a seeded random 10% validation subset from the training table (configurable), leaving test separate. This is an interaction-level split, not a compound-disjoint or target-disjoint split; compounds and targets recur on both sides.

### Graph construction and exact features

RDKit parses each SMILES; blank or invalid values raise `ValueError`. Atom nodes have 22 floats:

| Feature group | Dims | Meaning |
|---|---:|---|
| Symbol one-hot | 9 | H, C, N, O, F, Cl, S, Br, I; unlisted elements yield all-zero group. |
| Atomic number | 1 | Scalar atomic number. |
| Acceptor/donor | 2 | `BaseFeatures.fdef` membership flags. |
| Aromatic | 1 | Aromatic flag. |
| Hybridization one-hot | 3 | SP, SP2, SP3. |
| Total H | 1 | `GetTotalNumHs()`. |
| Explicit valence | 1 | Explicit valence. |
| Formal charge | 1 | Formal charge. |
| Implicit valence | 1 | Implicit valence. |
| Explicit H count | 1 | `GetNumExplicitHs()`. |
| Radical electrons | 1 | `GetNumRadicalElectrons()`. |

Each bond becomes edges `i→j` and `j→i`. No bond type/conjugation edge attributes are attached by the local graph builder. A zero-bond molecule receives an empty `[2,0]` edge index. The pinned author preprocessing creates bond features too, though the author graph model’s GraphConv path consumes node features and connectivity. The default local path uses raw features; author preprocessing scales each molecule’s complete feature matrix as `(x - x.min())/(x.max()-x.min())`. The optional local scaled bundle and `fast_run_v1/scaled` result are separate conditions.

### Protein/label handling

The protein vocabulary maps 25 residue symbols to IDs 1–25. ID 0 is padding and also the fallback for unknown characters, so unknown and padding are not distinguishable. The sequence is padded to 1,200 or truncated to its first 1,200 residues. Because the longest is 2,549, some residues are dropped. No explicit truncation mask is supplied. `nn.Embedding` uses `padding_idx=0`; convolutions have no mask, so convolution biases may yield nonzero responses in padded positions.

Affinity conversion has no explicit guards for missing, zero, negative or nonfinite Kd. Zero produces infinity under logarithm; invalid labels may propagate. Invalid SMILES raise at item access rather than being dropped. Missing protein values are assumed absent by the dictionary/table pipeline.

### Leakage/reproducibility

In code, test data is held apart from fitting and checkpoint selection. Validation is selected only from the training table, which avoids direct test-driven early stopping. Interaction-level splits still expose the same compounds/targets in training and test; generalization to unseen drugs/targets is not tested. Each epoch is capped at 50 optimizer steps for audited runs, so “epoch” is not a complete pass through all training examples. Some runs disable deterministic algorithms. Original primary-run hashes for processed inputs and explicit validation indices were not saved; exact historical identity is therefore unproven.

## 5. Complete model architecture

| Component | Verified dimensions/operations |
|---|---|
| Drug input | PyG `x=[total_atoms,22]`, `edge_index`, graph assignment vector. |
| MGNN stem | GraphConv 22→32, node BatchNorm, ReLU. |
| Dense blocks | Three blocks × eight dense layers. Every layer concatenates previous block features, then GraphConv+BN+ReLU to 64 (growth rate 32 × bottleneck 2), then GraphConv+BN+ReLU to 32; append output. |
| Transitions | GraphConv+BN+ReLU, compressing to half input channels. Widths: 32→288→144→400→200→456→228. |
| Drug pooling | Global mean pool over atoms, then Linear 228→96. |
| Protein embedding | Token IDs `[B,1200]`; Embedding(26,128,padding_idx=0) → transpose `[B,128,1200]`. |
| MCNN branches | 3 branches with 1/2/3 width-3 Conv1D layers; each layer emits 96 channels, stride 1, padding 0, ReLU. Receptive fields 3/5/7; adaptive max pool to one position per branch. |
| Protein projection | Concatenate 3×96=288; Linear 288→96. |
| Fusion/head | Concatenate drug/protein `[B,192]`; Linear 192→1024→1024→256→1, ReLU after first three, dropout 0.1 after each. Output is predicted pKd. |

No dropout is present in the shown graph or sequence encoders. The MGNN’s 27 conceptual stages count 3×(8 dense units + transition), excluding conv0; 28 top-level stages include conv0; 52 GraphConv modules count two convolutions per dense layer. These are different counting conventions, not a contradiction.

```text
SMILES → RDKit atom graph (22D nodes, bidirected bonds)
       → MGNN stem → [dense block(8) → transition] × 3
       → atom mean pool → 96D drug vector ─┐
                                            ├→ concatenate 192D → MLP → scalar pKd
sequence → residue IDs (length 1200) → Embedding(128)
         → Conv branches (receptive fields 3/5/7) → max pools
         → concatenate/project → 96D protein vector ─┘
```

The pinned author model has the same broad default topology: 22 atom inputs, three 8-layer blocks, growth 32, bottleneck 2, three sequence CNN scales, 128D residue embeddings, 96 channels/outputs, and the same 1024/1024/256 head/dropout. The local MCNN makes branch receptive fields explicit. The author model uses kernel size 3 and default padding 0. Direct extraction of the local paper PDF was blocked; full ESI is documented as unavailable. Thus these are strong code-level correspondences, not a claim that every paper detail has been independently checked.

## 6. Training pipeline

Main `src.train` defaults: LR 0.0005, batch 512, max 1,000 epochs, 50 steps/epoch, patience 400, validation fraction 0.1, seed 42. CLI may override them. Optimizer is Adam with no explicit weight decay. Loss is `nn.MSELoss()`. No scheduler or gradient clipping appears in the main loop. Validation MSE strictly selects best checkpoint; non-improvements increment patience; training stops at patience. `auto` selects available CUDA or CPU. Seed helpers set Python/NumPy/Torch/CUDA RNGs and deterministic settings, although some recorded runs disabled deterministic algorithms.

| Run | Seed | Batch | LR | Steps | Val fraction | Patience | Epochs requested/completed | Best epoch / val MSE | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `run_112_backup` | 42 | 32 | .0005 | 50 | .10 | 30 | 150/112 | 91 / .23236285 | Manual stop; patience 21, no early stop |
| `fast_run_v1/unscaled/seed_42` | 42 | 512 | .0005 | 50 | .10 | 40 | 400/182 | 142 / .21619413 | Early stop at patience 40 |
| `fast_run_v1/scaled/seed_42` | 42 | 512* | .0005* | 50* | .10* | 40* | 400*/122 | 122 / .23312505 | History/evaluation only; stopped status unresolved |
| Retry seeds 44–46 | 44–46 | 32 | .0005 | 50 | .10 | 30 | 150/150 each | 150/.4321970; 145/.4434661; 148/.3935311 | Cap reached, no early stop |
| Historical CPU diagnostic | 42 | 32 | .0005 | 1 | — | 2 | —/1 | 1 / 25.95255 | One epoch/step only |
| Protein-GNN | 42 | 32 | .0005 | 50 | .10 | 10 | 30/30 | 29 / .5732540 | Separate experiment, cap reached |

`*` Scaled-run settings are inferred from its launch-family documentation; its directory lacks complete final run metadata. Recover the exact command before treating those settings as verified. `src.paper_protocol.py` defines five folds, seeds 42/43/44 and LR/batch 0.0005/512; 1,000 epoch cap, 50 steps and patience 400 are explicitly project choices. No result artifact proves completion of that planned protocol.

## 7. Experiments and metrics

### Metric meanings

- **MSE:** mean squared pKd residual, in pKd².
- **RMSE:** square root of MSE, in pKd units.
- **CI:** proportion of comparable true-value pairs ordered correctly, with prediction ties worth half; true ties are excluded. The implementation is O(n²), and CI measures ranking rather than calibration.
- **r_m²:** modified squared correlation. Code computes Pearson `r²`, `k=Σ(y·ŷ)/Σ(ŷ²)`, zero-intercept `r0²`, then `r² × (1 − sqrt(abs((r²)²−(r0²)²)))`. This is not ordinary R². The older `run_112_backup/evaluation_metrics.json` says r_m²=.3037103; corrected metrics/current code say .6817056. Use corrected file and explain that the earlier value is historical/incorrect.

### Results

| Run | Experiment | Seed | Test MSE | RMSE | CI | r_m² | Evidence/status |
|---|---|---:|---:|---:|---:|---:|---|
| `run_112_backup` | Main MGraphDTA, raw features, batch 32 | 42 | .2570411 | .5069922 | .8790452 | .6817056 | Metrics/history agree; manual stop at 112. |
| `fast_run_v1/unscaled/seed_42` | Main MGraphDTA, raw features, batch 512 | 42 | .2403023 | .4902064 | .8839608 | .6712085 | Final JSON/history; early stop at 182. |
| `fast_run_v1/scaled/seed_42` | Main model, scaled features | 42 | .2646931 | .5144834 | .8855522 | .5956434 | Evaluation JSON; launch metadata incomplete. |
| `multiseed_retry.../seed_44` | Main MGraphDTA, raw | 44 | .4241899 | .6512986 | .8162087 | .4668448 | Fresh checkpoint evaluation matches current test table ±1e-6; process status conflict. |
| `.../seed_45` | Main MGraphDTA, raw | 45 | .4374538 | .6614029 | .8231562 | .4403202 | Fresh eval matches ±1e-6; failure manifest says exit code 1. |
| `.../seed_46` | Main MGraphDTA, raw | 46 | .4012697 | .6334585 | .8351590 | .4924511 | Fresh eval matches ±1e-6; summary/failure conflict. |
| Validated aggregate | Seeds 44–46 mean ± sample SD | 44–46 | .4209711 ± .0183055 | .6487200 ± .0141495 | .8248413 ± .0095869 | .4665387 ± .0260668 | Fresh checkpoint reevaluation documented; not paper reproduction. |
| Protein-GNN | Separate drug/residue graph | 42 | .6517305 | .8072983 | .7596737 | .2090045 | 30 epochs, best validation epoch 29; do not mix with main model. |
| Historical one-epoch CPU | Diagnostic | 42 | 26.25162 | 5.123634 | .4997062 | .0035095 | Not a trained benchmark. |

Validation report records the current processed test CSV hash as `5658ff56c9ac6b0a9649fed632d817d58c4a9fcf926919d7307a028184d18638` (5,010 rows). Fresh evaluations reproduce seed 44–46 metrics within 1e-6, but original runs did not record test hashes or prediction vectors; historical test identity is not proven.

`results/davis/multiseed/summary.json` reports zero valid runs, and histories there are empty. Retry 42/43 are incomplete (42 has 40 history rows; 43 is incomplete); failures JSON records exit code 1 for 42, 43 and 45. Seeds 44 and 46 have 150-row histories, matching best-checkpoint metadata and complete-looking metrics, yet status summaries conflict. Seed 45’s complete-looking artifacts also conflict with its nonzero status. Historical process outcomes for 44–46 remain unresolved because process logs were not found. `multiseed_summary_retry` duplicates run data rather than creating new runs. `scaling_ablation_v1/unscaled/seed_42` has only 44 history rows and no final metrics.

### Discrepancy assessment

Primary seed 42 used batch 32, a 150 epoch cap and patience 30, then was manually stopped. Fast unscaled seed 42 used batch 512, a 400 epoch cap and patience 40, and stopped at 182. Changing batch size changes optimizer updates even with the same seed and step count. Seeds 44–46 use batch 32 but different seeds and no early stopping before the 150 cap. Nondeterministic CUDA is recorded for multi-seed runs. These differences plausibly affect results, but **the cause of the performance gap is unresolved**; artifacts lack historical source hashes, validation index lists, exact per-run data hashes for all runs and original process logs. Do not rank configurations or select hyperparameters using these test results; compare only as separately labeled exploratory conditions.

## 8. Explainability

GradAAM registers a forward hook at the MGNN final transition output (node-level activation before global mean pooling and drug projection). It differentiates the scalar predicted affinity with respect to that activation. Gradients are averaged over atoms to make channel weights; weights are min–max normalized with epsilon; activation×weight is summed over channels per atom and min–max normalized again. Atom ordering follows the graph’s node order/RDKit SMILES order.

The visualization loads a selected Davis test row and default run-112 checkpoint, checks atom counts, maps scores through an orange color map onto the RDKit drawing, and writes a PNG plus JSON with sample index, SMILES, sequence, true/predicted affinity, scores and checkpoint metadata. Five example JSONs are present (`davis_sample_0` to `_4`) and a summary CSV exists. These are qualitative model-sensitivity illustrations, not quantitative validation. No causal biochemical interaction follows from a highlighted atom. Exact parity with author Grad-AAM formula needs direct comparison against `visualization/visualization_mgnn.py`; the conceptual last-MGNN-layer approach is consistent with the author README/source.

## 9. Tests, environment and reproducibility

Test modules cover atom features; Davis and scaled datasets; scaling experiment; MGNN, MCNN and integrated model; metrics; training; Grad-AAM; paper protocol; and multi-seed aggregation. `README.md` records a focused protocol/aggregation run of 37 passing tests. No tests were run for this audit; that record is not evidence that the full current suite passes.

`scripts/check_environment.py` is designed to report Python, PyTorch, CUDA, GPU, PyG and RDKit. It was not run. `PROJECT_PLAN.md` documents Python 3.12, PyTorch 2.6.0+cu124, PyG 2.8.0, RDKit 2026.03.6 and CUDA runs. Requirements pins PyG 2.8.0.post1. Run JSON says CUDA but not GPU model, driver or OS. Current installed versions and hardware remain unverified.

Main Git HEAD: `5ee53c404b9b3612fc4783b0dce9aa916b027885`. `git status --short` showed the nested author repository as `?` plus untracked `report/MGraphDTA_Overleaf_Report.zip`, `report/MGraphDTA_Overleaf_Report/`, and `report/report/`; these pre-existing artifacts were preserved. Author checkout HEAD: `54386680e8c9562f708a0fe5d381038c62111df6`. Its nested-repository state does not establish a pristine working tree.

`.gitignore` excludes virtual environments, data, most checkpoints/results and author archive. Selected small result artifacts are re-included, so essential local-only files may not travel in a Git clone. For strict replay, missing/uneven provenance includes exact launch commands, code hashes, validation indices, historical test hashes, environment captures, prediction/target vectors and resolved process logs. Checkpoint files exist for primary and validated seeds; not every folder has complete curves/metadata.

## 10. Paper versus implementation

The local paper PDF exists but extraction was blocked by a first-run MiKTeX prompt. Existing compliance/provenance notes cite the paper and parts of indexed ESI Table S3; those notes say the full ESI PDF was not retrieved. The pinned author implementation is available. “Paper/release” evidence below should be read with that limitation.

| Component/protocol | Paper/release evidence | Project | Evidence | Status / impact |
|---|---|---|---|---|
| Overall model | Graph encoder + multiscale sequence CNN + regression head in author release. | Same broad dual encoder and scalar regressor. | `src/models/mgraphdta.py:11-64`; author `regression/model.py`. | **Matches broadly.** |
| MGNN | Author: 22 inputs, 3×8 dense layers, growth 32, bottleneck 2, transitions, 96 output. | Same default structure, global mean pool, 96D output. | `src/models/mgnn.py:46-52,291-343`; author model. | **Matches code structure;** full ESI check incomplete. |
| Atom data | Author has 22 atom features and six edge features; GraphConv path uses node features/connectivity. | 22 raw node features and bidirected edges; no edge attributes. | `src/preprocessing.py:99-223`; author preprocessing/model. | **Partial;** scaling differs in default path. |
| Atom scaling | Author regression code applies one molecule-wide min/max over all x entries. | Default is unscaled; optional scaled bundle/run separate. | Author `regression/preprocessing.py`; `src/dataset.py:179-220`; reference audit. | **Differs for default runs;** potential material input difference. |
| Protein encoder | Author code: 1200 tokens, vocab, embedding 128, 3 width-3 scales, 96 channels, pooling/projection 96. | Same defaults/operations. | `src/preprocessing.py:17-26,226-...`; `src/models/mcnn.py:10-116`; author source. | **Matches author code substantially;** direct article detail partly unverified. |
| Davis data/split | Author README points to DeepDTA Davis; local provenance audit verifies matrix/folds and author split record correspondence. | Same supplied train/test membership; 25,046/5,010. | `src/dataset.py:33-72,90-139`; `REFERENCE_PROVENANCE_AUDIT.md`. | **Matches data membership** per local audit; validation is project-specific. |
| Target scale | Local loader converts Kd to pKd; author prepared CSV contains affinity on pKd scale. | `9-log10(Kd_nM)`, no extra scaling. | `src/dataset.py:106-118`; reference data audit. | **Matches prepared scale**; PDF check incomplete. |
| Batch/LR | Local protocol notes attribute LR .0005 and batch 512 to article §2.8; author train uses those values. | LR .0005; runs have batch 32 or 512. | `src/paper_protocol.py:49-67`; per-run JSON. | **Partial.** Primary and multi-seed batches differ from cited value. |
| Epochs/validation/stopping | Author regression trainer sets 3,000 epochs, 50 steps/epoch and early-stop threshold >400, and validates on its test loader. | Capped runs (150/400); holdout validation comes from training fold; separate test after fit. | Author `regression/train.py`; `src/train.py:548-647`. | **Differs.** Local test isolation is distinct; complete paper protocol absent. |
| Metrics | Author README describes RMSE, CI and Spearman examples. | MSE, RMSE, CI and r_m². | `src/metrics.py`; result JSON. | **Partial/differs;** report definitions explicitly. |
| Explainability | Author release includes GradAAM and says last MGNN layer used. | Final transition hook and normalized atom scores. | Local GradAAM files; author visualization files. | **Partial.** Exact parity and faithfulness not established. |
| Repeated experiment | Exact reported paper aggregation not checked from PDF here. | Three seeds have fresh checkpoint reevaluation; process records conflict; no completed CV/Optuna evidence. | `src/paper_protocol.py`; validated multi-seed files. | **Incomplete/unverified.** Do not claim paper mean/SD reproduction. |

## 11. Limitations and threats to validity

1. Runs differ in batch size, epoch budget and stopping; multi-seed process records conflict.
2. Default raw atom features differ from author scaling; scaled and unscaled runs are distinct conditions.
3. Interaction holdout includes recurring drugs/targets; it does not establish cold-start generalization.
4. Protein sequences longer than 1,200 are truncated; no explicit convolution mask is supplied.
5. pKd is transformed benchmark data, not a direct clinical or experimental validation; no predictive uncertainty is reported.
6. CI is ranking-only and quadratic; r_m² is not R²; an older primary metrics file contains a conflicting value.
7. Grad-AAM figures are qualitative; no causal claim or faithfulness test is supported.
8. Historical data/source hashes, exact validation indices and process logs are missing for key runs.
9. Full paper/ESI verification was incomplete.
10. Installed package/hardware versions were not live-captured in this audit.

## 12. Completion status

**Complete/evidenced:** data/model/trainer code; Davis fold preprocessing; saved primary run; separate batch-512 run; three checkpoint evaluations reproducible against the current test CSV; scaled feature path; separate Protein-GNN experiment; qualitative Grad-AAM samples; test modules; author-code provenance notes.

**Not complete/established:** exact paper reproduction; finished CV/Optuna and final protocol; clean historical process ledger; original test hashes for every run; full run manifests and source hashes; direct full-ESI check; quantitative explanation validation; fresh full-suite test result.

## 13. Recommended next steps for the academic report

1. Obtain a readable official paper and complete ESI; hash them and verify every supplemental setting.
2. Capture actual environment and reconcile PyG version; record OS, GPU and driver.
3. Maintain a per-run manifest: exact command, code/tree hash, preprocessing, data/fold hashes, validation indices, checkpoint hash, environment, exit status and evaluation command.
4. Keep raw-feature and author-scaled results in separate tables.
5. Resolve seed status only from process logs or a run ledger. Distinguish artifact-complete from successful process completion.
6. For any future run, predeclare validation/model selection and report every planned seed/failure; do not use test results for selection.
7. Label Grad-AAM as qualitative unless a separate quantitative faithfulness/stability study is done.
8. Describe the current work as an implementation and exploratory evaluation inspired by MGraphDTA unless a controlled full reproduction is completed.
9. Explain that the current interaction split does not test unseen-drug/target generalization.

## 14. Files to share with ChatGPT next

Share these instead of raw Davis CSVs, full `Y`, or full checkpoints:

- `report/project_audit/PROJECT_AUDIT.md`
- `README.md`, `PROJECT_PLAN.md`, `PAPER_COMPLIANCE_AUDIT.md`
- `REFERENCE_PROVENANCE_AUDIT.md`, `REFERENCE_DATA_AND_SCALING_AUDIT.md`
- `MULTISEED_VALIDATION_REPORT.md`, `MULTISEED_REEVALUATION_REPORT.md`
- Run-112 `final_metrics.json`, `evaluation_metrics_corrected.json`, `training_history.csv`
- Fast unscaled seed-42 `final_metrics.json`, `training_history.csv`
- `multiseed_validated_20261017/summary.json`, `validation_manifest.json`
- Retry seed 44/45/46 `final_metrics.json` files and `failures.json`
- `protein_gnn_experiment/final_metrics.json`
- `requirements.txt`; `src/dataset.py`, `preprocessing.py`, `train.py`, `metrics.py`, all three model files and `src/explainability/grad_aam.py`
- `results/davis/grad_aam/examples/grad_aam_summary.csv` and only the selected example JSON/PNG pairs to discuss

## 15. Source-evidence index

Line numbers refer to inspected files on 2026-10-27.

| Claim | Evidence |
|---|---|
| Raw inputs and file loading | `src/dataset.py:33-72` |
| Interaction grid, pKd and folds | `src/dataset.py:75-141` |
| PyG item construction | `src/dataset.py:144-243` |
| Vocabulary/constants | `src/preprocessing.py:17-39` |
| SMILES and atom features | `src/preprocessing.py:42-74,99-181` |
| Edge construction | `src/preprocessing.py:184-223` |
| Protein padding/truncation | `src/preprocessing.py:226-...` (continues to file end) |
| MGNN layers/channels | `src/models/mgnn.py:46-52,103-127,137-172,183-231,251-343` |
| MCNN | `src/models/mcnn.py:10-116` |
| Integrated model/head | `src/models/mgraphdta.py:11-64` |
| Trainer defaults and CLI | `src/train.py:28-38,882-917` |
| Train/validation split | `src/train.py:181-202` |
| Adam/MSE, checkpoint, stopping | `src/train.py:548-647` |
| Metrics definitions | `src/metrics.py:46-154` |
| Grad-AAM calculation | `src/explainability/grad_aam.py:21-92` |
| Visualization and JSON | `scripts/visualize_grad_aam.py:29-124` |
| Paper-protocol settings | `src/paper_protocol.py:27-67` |
| Tests | `tests/test_atom_features.py`, `test_davis_dataset.py`, `test_davis_scaled_dataset.py`, `test_davis_scaling_experiment.py`, `test_mgnn.py`, `test_mcnn.py`, `test_mgraphdta.py`, `test_metrics.py`, `test_training.py`, `test_grad_aam.py`, `test_paper_protocol.py`, `test_multiseed_aggregation.py` |
| Primary run | `results/davis/run_112_backup/final_metrics.json`, `evaluation_metrics_corrected.json`, `training_history.csv` |
| Batch-512 run | `results/davis/fast_run_v1/unscaled/seed_42/final_metrics.json`, `training_history.csv` |
| Scaled run | `results/davis/fast_run_v1/scaled/seed_42/evaluation_metrics.json`, `training_history.csv` |
| Validated aggregate/test hash | `results/davis/multiseed_validated_20261017/summary.json`, `validation_manifest.json` |
| Multi-seed conflicts | `results/davis/multiseed_retry_20261016/failures.json`, `summary.json`; `MULTISEED_VALIDATION_REPORT.md`, `MULTISEED_REEVALUATION_REPORT.md` |
| Protein-GNN settings/results | `scripts/train_protein_gnn_affinity.py:39-46`; `results/davis/protein_gnn_experiment/final_metrics.json` |
| Declared stack and environment | `requirements.txt`; `scripts/check_environment.py`; `PROJECT_PLAN.md:11-15` |
| Ignore policy and main Git state | `.gitignore`; main HEAD `5ee53c404b9b3612fc4783b0dce9aa916b027885` |
| Author repository revision | `references/mgraphdta_official/author_repo/README.md`; HEAD `54386680e8c9562f708a0fe5d381038c62111df6` |
| Author scaling/features/model | `references/mgraphdta_official/author_repo/regression/preprocessing.py`, `model.py`, `train.py`; `REFERENCE_DATA_AND_SCALING_AUDIT.md` |
| Project status and focused test note | `README.md:9-50,87-97,146-176,184-213,279`; `PROJECT_PLAN.md` |

**Interpretation note:** This is a local implementation/artifact audit, not an independent scientific replication. Code and saved run metadata support claims about project behavior. Paper-level claims are limited by blocked PDF extraction and unavailable full ESI; author-release correspondence and prior local audits are identified, and uncertain claims remain explicitly unverified.
