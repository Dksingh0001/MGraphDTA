# MGraphDTA Project Plan

## 1. Project Objective

Implement and evaluate a MGraphDTA-style model for drug-target binding affinity prediction from molecular SMILES and protein sequences. This repository is an academic implementation in progress; matching the paper's final reported results has not been established.

Original paper: Ziduo Yang, Weihe Zhong, Lu Zhao, and Calvin Yu-Chian Chen, "MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug-Target Binding Affinity Prediction," *Chemical Science*, 2022, 13, 816-833. [DOI](https://doi.org/10.1039/D1SC05095E).

## 2. Completed Work

- Pinned Python dependencies are listed in `requirements.txt`; the checked environment uses Python 3.12, PyTorch 2.6.0+cu124, PyTorch Geometric 2.8.0, and RDKit 2026.03.6.
- Davis CSV tables are prepared from the raw ligand, protein, affinity, and fold files. Kd values in nM are converted to pKd using `9 - log10(Kd_nM)`.
- RDKit preprocessing creates bidirectional molecular graphs with 22-dimensional atom features. Protein sequences are mapped to integer tokens and padded or truncated to 1,200 residues.
- MGNN, MCNN, the MGraphDTA fusion module, and the scalar regression head are implemented and covered by component/integration tests.
- The training pipeline supports Adam/MSE, configurable batch size, CUDA/CPU selection, checkpoint save/load, validation-based selection and early stopping, and final independent-test evaluation.
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
| Training and split isolation | Implemented | `src/train.py`; validation MSE controls checkpoint selection/early stopping; test evaluated after restore |
| Evaluation metrics | Pending | `src/metrics.py` contains a module description but no metric functions |
| Grad-AAM | Pending | `src/explainability/grad_aam.py` contains a module description but no implementation |
| Other datasets/experiments | Pending | No additional dataset pipeline or completed ablation is evidenced in the current source tree |

The paper calls MGNN 27 architectural units: 24 DenseLayer units and 3 transitions, excluding `conv0`. This implementation has 28 top-level stages including `conv0` and 52 literal GraphConv modules because each DenseLayer has two GraphConv operations. These are different counting conventions for this implementation, not 52 paper-reported layers.

## 4. Pilot Training Status

A 10-epoch Davis pilot has been reported with:

| Measure | Reported pilot value |
| --- | ---: |
| Best validation MSE | 0.478949 |
| Independent test MSE | 0.569563 |

These are **pilot results**, not final results and not a claim of reproducing the paper. The exact pilot invocation, seed, environment details, and retained run log are TODOs for reproducibility; no pilot log is currently present under `results/`.

## 5. Full Training Status

The planned full Davis experiment is configured for up to 3,000 epochs, 50 optimizer updates per training epoch, batch size 512, learning rate 0.0005, Adam, MSE, and early-stopping patience 400. Current defaults split the 25,046-row Davis training fold into a 90% optimization subset and 10% validation subset, using seed 42. The independent test fold contains 5,010 rows.

**The full 3,000-epoch experiment has not been completed.** The epoch limit is an upper bound because validation-based early stopping may halt sooner. Do not present the pilot as a full run or paper reproduction.

## 6. Evaluation Plan

1. Keep the independent Davis test partition out of optimization, validation, checkpoint selection, and early stopping.
2. Select the best checkpoint and stop based on validation MSE.
3. Restore the selected checkpoint, then evaluate the independent test set for final metrics.
4. Implement and test additional metrics, including CI, RMSE, and r_m^2, in `src/metrics.py` before reporting them. CI is not currently implemented.
5. Record the split seed, command, environment, checkpoint, and results with each run. Treat comparison with published scores as a separate, protocol-aware analysis.

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
2. Run the planned full Davis experiment with the documented isolated train/validation/test protocol and monitor hardware/resource use.
3. Implement and validate CI, RMSE, and r_m^2; report final independent-test results from the selected checkpoint.
4. Implement and test Grad-AAM atom visualizations.
5. Define and run any additional dataset comparisons or ablations with their split protocols documented.
6. Update project documentation with reproducible commands and clearly labeled results without claiming paper reproduction unless protocols and evidence support it.
