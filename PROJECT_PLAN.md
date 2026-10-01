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
| Grad-AAM | Pending | `src/explainability/grad_aam.py` contains a module description but no implementation |
| Other datasets/experiments | Pending | No additional dataset pipeline or completed ablation is evidenced in the current source tree |

The paper calls MGNN 27 architectural units: 24 DenseLayer units and 3 transitions, excluding `conv0`. This implementation has 28 top-level stages including `conv0` and 52 literal GraphConv modules because each DenseLayer has two GraphConv operations. These are different counting conventions for this implementation, not 52 paper-reported layers.

## 4. Historical Pilot Training Status

A 10-epoch Davis pilot was previously reported with:

| Measure | Reported pilot value |
| --- | ---: |
| Best validation MSE | 0.478949 |
| Independent test MSE | 0.569563 |

These are historical pilot values only. They are superseded by the completed 112-epoch Davis experiment below and are not its final results or a claim of reproducing the paper.

## 5. Current Davis Experiment Status

The Davis training run was stopped manually at epoch 112. The saved best-validation checkpoint is from epoch 91, selected by validation MSE, and is stored at `results/davis/run_112_backup/mgraphdta_best.pt`. The independent test fold was evaluated using that checkpoint.

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
| Independent test MSE | 0.2570411052920772 |
| Independent test RMSE | 0.5069922142322081 |
| Independent test CI | 0.8790452402222367 |
| Independent test `r_m^2` | 0.6817055902984536 |
| Device | CUDA |

The final `r_m^2` above uses the corrected implementation, which follows the original MGraphDTA authors' `regression/metrics.py` calculation. The earlier value 0.303710 was produced by the superseded calculation and must not be reported as the final result. Corrected evaluation metrics are recorded in `results/davis/run_112_backup/evaluation_metrics_corrected.json`.

**This is not an exact reproduction of the paper's Table 3 experimental protocol.** This run used a different training and validation configuration and did not reproduce the paper's full hyperparameter optimization or repeated-experiment protocol. The results above are this project's single-run Davis evaluation, not a paper reproduction claim.

## 6. Evaluation Plan

1. The independent Davis test partition is kept out of optimization, validation, checkpoint selection, and early stopping.
2. Validation MSE selects the best checkpoint; the epoch-91 checkpoint was restored before independent test evaluation.
3. MSE, RMSE, CI, and corrected `r_m^2` are implemented in `src/metrics.py`; the corrected `r_m^2` follows the original authors' calculation.
4. The current run's configuration, checkpoint, and test results are recorded above and in `results/davis/run_112_backup/evaluation_metrics_corrected.json`.
5. Compare with published scores only with explicit consideration of the differing experimental protocols.

This repository's protocol is distinct from the paper's described five-fold cross-validation over five of six partitions plus an independent test partition. The authors' released regression trainer uses prebuilt train/test datasets and evaluates its test dataset during training for checkpoint selection and early stopping. This project instead derives validation from the available training fold and keeps its independent test fold untouched until final evaluation; it does not claim exact reproduction of either procedure.

## 7. Explainability Plan

Grad-AAM is not implemented. Planned work is to capture activations and gradients at the final MGNN transition, compute atom-level importance values, normalize them, and render them on RDKit molecular depictions. Implement and validate this separately after the training/evaluation workflow is stable; do not claim explanations before then.

## 8. GitHub and Reproducibility Plan

- Keep source, tests, dependency pins, and documentation in Git.
- Raw and processed Davis files, checkpoints, and generated result files are ignored by `.gitignore`; provide dataset setup instructions without committing those artifacts.
- Record the exact training command, seed, Python/PyTorch/CUDA/PyG/RDKit versions, split settings, best epoch, checkpoint, and evaluation outputs for each run.
- Preserve the test-set isolation rule and clearly distinguish pilot, full-run, and paper-reported values.
- Cite the original authors' repository: [guaguabujianle/MGraphDTA](https://github.com/guaguabujianle/MGraphDTA).

## 9. Future Milestones

1. Record the pilot command and environment metadata; confirm the pilot checkpoint and result log are retained as intended.
2. If a paper-level comparison is required, plan and run the paper's full hyperparameter optimization and repeated-experiment protocol, documenting the protocol separately from this completed run.
3. Use the corrected metric implementation for future evaluations and keep results tied to their specific checkpoints and protocols.
4. Implement and test Grad-AAM atom visualizations.
5. Define and run any additional dataset comparisons or ablations with their split protocols documented.
6. Update project documentation with reproducible commands and clearly labeled results without claiming paper reproduction unless protocols and evidence support it.
