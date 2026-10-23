# MGraphDTA Project Plan

## 1. Project Objective

Implement and evaluate a MGraphDTA-style model for drug-target binding affinity prediction from molecular SMILES and protein sequences. This repository is an academic implementation in progress; matching the paper's final reported results has not been established.

Original paper: Ziduo Yang, Weihe Zhong, Lu Zhao, and Calvin Yu-Chian Chen, "MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-Target Binding Affinity Prediction," *Chemical Science*, 2022, 13, 816-833. [DOI](https://doi.org/10.1039/D1SC05095E).

## 2. Completed Work

- Pinned Python dependencies are listed in `requirements.txt`; the checked environment uses Python 3.12, PyTorch 2.6.0+cu124, PyTorch Geometric 2.8.0, and RDKit 2026.03.6.
- Davis CSV tables are prepared from the raw ligand, protein, affinity, and fold files. Kd values in nM are converted to pKd using `9 - log10(Kd_nM)`.
- RDKit preprocessing creates bidirectional molecular graphs with 22-dimensional atom features. Protein sequences are mapped to integer tokens and padded or truncated to 1,200 residues.
- MGNN, MCNN, the MGraphDTA fusion module, and the scalar regression head are implemented and covered by component/integration tests.
- The training pipeline supports Adam/MSE, configurable batch size, CUDA/CPU selection, validation-based early stopping, best-checkpoint restoration, epoch-level training/validation MSE reporting, and final independent-test evaluation. It writes epoch history to `results/davis/training_history.csv`, a curve to `results/davis/training_curve.png`, and the final run summary to `results/davis/final_metrics.json`.
- Training data is split reproducibly into optimization and validation subsets. The independent test dataset is constructed only after training and best-checkpoint restoration.
- Script-style tests cover preprocessing, encoders, model integration, and variable-size PyG graph batching. `tests/test_training.py` covers the split and training protocol.

## 3. Current Implementation Status

| Area | Status | Implementation evidence |
| --- | --- | --- |
| Davis preprocessing | Implemented | `src/dataset.py`, `src/preprocessing.py` |
| Molecular graph representation | Implemented | 22-dimensional atom features and bidirectional `edge_index` |
| Drug encoder | Implemented | `src/models/mgnn.py`; three 8-layer dense blocks, transitions, and 96-dimensional readout |
| Protein encoder | Implemented | `src/models/mcnn.py`; three branches with 1, 2, and 3 kernel-3 convolutions and 96 channels, followed by a 96-dimensional projection |
| Prediction head | Implemented | `src/models/mgraphdta.py`; fused 192-dimensional vector, MLP widths 1024, 1024, 256, 1, ReLU and 0.1 dropout on hidden layers |
| Training and split isolation | Implemented | `src/train.py`; validation MSE controls checkpoint selection/early stopping; best checkpoint restored before test evaluation; MSE history, curve, and final metrics are written |
| Evaluation metrics | Implemented | `src/metrics.py` provides MSE, RMSE, CI, and the corrected authors' `r_m^2` calculation; `tests/test_metrics.py` covers known values |
| Grad-AAM | Implemented | `src/explainability/grad_aam.py` computes per-atom model attributions, with visualization scripts and saved examples. This is implementation-level validation; the attributions are not experimentally validated binding contacts. |
| Protein-GNN experiment | Completed experimental baseline | Separate sequence-adjacency protein-GNN experiment and metrics are stored under `results/davis/protein_gnn_experiment/`; it is not the primary MGraphDTA result or a paper reproduction. |
| Systematic ablations / additional benchmark datasets | Pending | No systematic ablation study or additional benchmark-dataset evaluation is recorded. |

The paper calls MGNN 27 architectural units: 24 DenseLayer units and 3 transitions, excluding `conv0`. This implementation has 28 top-level stages including `conv0` and 52 literal GraphConv modules because each DenseLayer has two GraphConv operations. These are different counting conventions for this implementation, not 52 paper-reported layers.

## 4. Historical Pilot Training Status

A 10-epoch Davis pilot was previously reported with:

| Measure | Reported pilot value |
| --- | ---: |
| Best validation MSE | 0.478949 |
| Independent test MSE | 0.569563 |

These are historical pilot values only. They are superseded by the completed 112-epoch Davis experiment below and are not its final results or a claim of reproducing the paper.

## 5. Current Davis Experiment Status

The primary Davis MGraphDTA run is archived in `results/davis/run_112_backup/`, the reference archive for its checkpoint, 112-epoch history, training curve, and evaluation files. Training ended manually at epoch 112; early stopping did not trigger (the recorded patience counter was 21 of 30). The best-validation checkpoint is from epoch 91, selected by validation MSE 0.23236284946015257, and the independent test fold was evaluated using that checkpoint. Root-level history, curve, and final metrics mirror this primary run.

| Training configuration | Value |
| --- | ---: |
| Learning rate | 0.0005 |
| Batch size | 32 |
| Maximum epochs | 150 |
| Steps per epoch | 50 |
| Validation fraction | 0.1 |
| Early-stopping patience | 30 |
| Seed | 42 |

| Davis result | Value |
| --- | ---: |
| Best epoch | 91 |
| Best validation MSE | 0.23236284946015257 |
| Independent test MSE | 0.2570411 |
| Independent test RMSE | 0.5069922 |
| Independent test CI | 0.8790452 |
| Independent test `r_m^2` | 0.6817056 |
| Device | CUDA |

The final `r_m^2` above uses the corrected implementation, which follows the original MGraphDTA authors' `regression/metrics.py` calculation. The older `results/davis/run_112_backup/evaluation_metrics.json` is retained as a historical result and records the superseded `r_m^2` value 0.3037103; it is not authoritative. Use `results/davis/run_112_backup/evaluation_metrics_corrected.json` for this run's authoritative evaluation. The one-epoch CPU pilot metrics are preserved separately as `results/davis/historical_one_epoch_cpu_final_metrics.json`.

### Separate Protein-GNN Experiment

The experimental sequence-adjacency protein-GNN run is separate from primary MGraphDTA. Its checkpoint, history, curve, and final metrics remain under `results/davis/protein_gnn_experiment/`. Its test metrics (MSE 0.6517305, RMSE 0.8072983, CI 0.7596737, `r_m^2` 0.2090045) are experimental baseline results and must not be substituted for or combined with the primary MGraphDTA metrics.

**This is not an exact reproduction of the paper's Table 3 experimental protocol.** This run used a different training and validation configuration and did not reproduce the paper's full hyperparameter optimization or repeated-experiment protocol. The results above are this project's single-run Davis evaluation, not a paper reproduction claim.

## 6. Evaluation Plan

1. The independent Davis test partition is kept out of optimization, validation, checkpoint selection, and early stopping.
2. Validation MSE selects the best checkpoint; the epoch-91 checkpoint was restored before independent test evaluation.
3. MSE, RMSE, CI, and corrected `r_m^2` are implemented in `src/metrics.py`; the corrected `r_m^2` follows the original authors' calculation.
4. The current run's configuration, checkpoint, and test results are recorded above and in `results/davis/run_112_backup/evaluation_metrics_corrected.json`.
5. Compare with published scores only with explicit consideration of the differing experimental protocols.

This repository's protocol is distinct from the paper's described five-fold cross-validation over five of six partitions plus an independent test partition. The authors' released regression trainer uses prebuilt train/test datasets and evaluates its test dataset during training for checkpoint selection and early stopping. This project instead derives validation from the available training fold and keeps its independent test fold untouched until final evaluation; it does not claim exact reproduction of either procedure.

## 7. Explainability Plan

Grad-AAM is implemented in `src/explainability/grad_aam.py`, with tests and Davis visualization examples. It provides model-derived atom attribution, not scientific validation of physical binding contacts. Future work may assess attribution stability and compare highlighted atoms with independently established structural evidence; do not interpret visualizations alone as proof of binding interactions.

## 8. GitHub and Reproducibility Plan

- Keep source, tests, dependency pins, and documentation in Git.
- Raw and processed Davis files, checkpoints, and generated result files are ignored by `.gitignore`; provide dataset setup instructions without committing those artifacts.
- Record the exact training command, seed, Python/PyTorch/CUDA/PyG/RDKit versions, split settings, best epoch, checkpoint, and evaluation outputs for each run. For this GPU's unsupported deterministic adaptive max pooling backward operation, use seeded RNGs with `--no-deterministic` and document possible small GPU numerical differences.
- Preserve the test-set isolation rule and clearly distinguish pilot, full-run, and paper-reported values.
- Cite the original authors' repository: [guaguabujianle/MGraphDTA](https://github.com/guaguabujianle/MGraphDTA).

## 9. Future Milestones

1. Record the pilot command and environment metadata; confirm the pilot checkpoint and result log are retained as intended.
2. If a paper-level comparison is required, plan and run the paper's full hyperparameter optimization and repeated-experiment protocol, documenting the protocol separately from this completed run.
3. Use the corrected metric implementation for future evaluations and keep results tied to their specific checkpoints and protocols.
4. Assess Grad-AAM attribution stability and compare with independent structural evidence before making scientific claims.
5. Define and run any additional dataset comparisons or ablations with their split protocols documented.
6. Update project documentation with reproducible commands and clearly labeled results without claiming paper reproduction unless protocols and evidence support it.
7. Run the five-seed workflow under a fresh configurable experiment root, using `--no-deterministic` for this GPU, and report aggregate metrics without selecting a seed by test performance. Retry failed seeds only under another unused root so partial outputs remain available for debugging.
