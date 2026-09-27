# MGraphDTA

**Drug–target binding affinity prediction with multiscale graph and convolutional neural networks.**

Original paper: **“MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug–Target Binding Affinity Prediction”** by Ziduo Yang, Weihe Zhong, Lu Zhao, and Calvin Yu-Chian Chen, *Chemical Science* (2022), 13, 816–833. [Paper DOI](https://doi.org/10.1039/D1SC05095E).

This project predicts continuous drug–target binding affinity from a drug's SMILES representation and a target protein sequence. It implements the MGraphDTA forward architecture and a Davis regression training pipeline. Grad-AAM is not implemented.

> **Current status:** The architecture and training pipeline are implemented. A 10-epoch Davis pilot has been reported; the planned full 3,000-epoch experiment has not been completed. Pilot results are not final paper-reproduction results.

## Project Overview

The forward workflow has two input branches:

- **Drug:** SMILES → RDKit molecular graph → MGNN → 96-dimensional drug embedding.
- **Protein:** amino-acid sequence → integer tokens → learned 128-dimensional embedding → MCNN → 96-dimensional protein embedding.

The embeddings are concatenated and passed to the regression head, which returns one predicted affinity per drug–target pair. The short pilot is preliminary and does not establish paper-level performance.

## Current Status

| Component | Status | Repository evidence |
| --- | --- | --- |
| Project setup | ✅ Completed | Python dependencies and environment-check script are present. |
| Davis preprocessing | ✅ Completed | Davis tables and pKd conversion are implemented; the local dataset verification script passes. Dataset files are not committed. |
| Molecular graph construction | ✅ Completed | RDKit parsing, 22-dimensional atom features, and bidirected `edge_index` construction are implemented and passed to MGNN through PyG data. |
| 22-dimensional atom features | ✅ Completed | Feature extraction and tests are implemented; tests pass. |
| MGNN drug encoder | ✅ Completed | Encoder is implemented; shape, depth-accounting, batching, and forward tests pass. |
| MCNN protein encoder | ✅ Completed | Encoder is implemented; branch shapes, pooling, gradients, batching, and available-device tests pass. |
| MGNN tests | ✅ Completed | `tests/test_mgnn.py` passes locally. |
| MCNN tests | ✅ Completed | `tests/test_mcnn.py` passes locally. |
| Full MGraphDTA integration | ✅ Completed | `src/models/mgraphdta.py` connects MGNN, MCNN, and the regression head; end-to-end tests pass on real Davis samples and batched data. |
| Prediction network | ✅ Completed | `192 → 1024 → 1024 → 256 → 1`, with ReLU and dropout 0.1 after each hidden layer. |
| End-to-end MGraphDTA tests | ✅ Completed | `tests/test_mgraphdta.py` verifies sample/batch forward passes, dimensions, finite scalar outputs, gradients, and CUDA when available. |
| Training pipeline | ✅ Completed | Reusable optimization, validation, early stopping, best-checkpoint save/load, and final test evaluation are implemented. No full training or benchmark results are claimed. |
| Training protocol tests | ✅ Completed | `tests/test_training.py` covers reproducible split indices, validation-driven checkpointing/stopping, test isolation, final test evaluation, and CPU/CUDA smoke paths. |
| Davis pilot | ✅ Reported | 10 epochs; best validation MSE 0.478949 and independent test MSE 0.569563. These are pilot-only results. |
| Full training | ⏳ Not completed | The planned run is up to 3,000 epochs; no full run or paper-reproduction result is claimed. |
| Evaluation metrics | ⏳ Planned | `src/metrics.py` currently contains only a module description. |
| Grad-AAM | ⏳ Planned | `src/explainability/grad_aam.py` currently contains only a module description. |
| Experiments and ablations | ⏳ Planned | No project training experiments are implemented or reported. |

The repository implements and tests the forward pass and training pipeline. The reported pilot is preliminary; no final benchmark or paper-reproduction result is claimed.

## Architecture

The complete model forward path and Davis training pipeline are implemented. Grad-AAM and broader experiments remain unimplemented.

```mermaid
flowchart LR
    D[Drug SMILES] --> G[RDKit molecular graph<br/>22-D atom features + edge_index]
    G --> MGNN[MGNN<br/>implemented and tested]
    MGNN --> GD[96-D drug embedding]

    P[Protein sequence] --> T[Tokenization<br/>length 1200]
    T --> E[Learned embedding<br/>128-D]
    E --> MCNN[MCNN<br/>implemented and tested]
    MCNN --> PD[96-D protein embedding]

    GD --> F[Feature fusion<br/>192-D]
    PD --> F
    F --> MLP[Prediction MLP<br/>192 → 1024 → 1024 → 256 → 1]
    MLP --> A[Predicted binding affinity]

    classDef done fill:#e8f5e9,stroke:#2e7d32,color:#102a14;
    classDef planned fill:#fff8e1,stroke:#ed6c02,color:#3b2a00,stroke-dasharray:5 5;
    class MGNN,GD,E,MCNN,PD,F,MLP,A done;
```

## Davis Dataset

Davis preprocessing converts measured $K_d$ values in nanomolar units to pKd using $pK_d = 9 - \log_{10}(K_d\text{ in nM})$. The local verification test currently confirms:

- 30,056 drug–target interactions
- 68 unique compounds
- 442 target IDs
- 379 unique protein sequences
- Protein token sequences padded or truncated to 1,200 positions

The raw dataset and processed CSVs are gitignored and are not included in this repository. `src/dataset.py` expects the raw Davis files under `data/raw/davis/` (including the affinity matrix and fold files). Provide the dataset locally before running `tests/test_davis_dataset.py`; no data-download workflow is included.

### Train/Validation/Test Protocol

The provided Davis training folds are flattened into 25,046 training interactions; the independent fold contains 5,010 test interactions. Training splits only the training dataset into a 90% optimization subset and a 10% validation subset (configurable with `--validation-fraction`, default `0.1`; default seed `42`). The validation subset selects checkpoints and controls early stopping. The best checkpoint is restored before the independent test dataset is loaded and evaluated once; test examples are not used by optimization or model selection.

This is the project's protocol, not an exact reproduction of the paper's six-part cross-validation protocol or the authors' released regression trainer. The released trainer evaluates its test dataset during training; this project deliberately keeps its independent test fold isolated.

Each `DavisDataset` item includes a PyTorch Geometric `pyg_data` object containing `x` (atom features shaped `[number_of_atoms, 22]`), `edge_index`, `target` (protein tokens shaped `[1, 1200]`), and `y` (affinity target). PyG batches these into a combined node matrix and edge list, a graph-membership `batch` vector, protein tokens shaped `[B, 1200]`, and labels. `MGraphDTA` consumes these fields directly.

## Drug Representation

Each SMILES string is parsed with RDKit. Atoms become graph nodes with 22 features, and every undirected chemical bond becomes two directed connections in `edge_index`.

The atom feature vector contains:

- 9-way atom-symbol one-hot encoding: H, C, N, O, F, Cl, S, Br, I
- Atomic number
- Hydrogen acceptor and donor flags
- Aromaticity
- 3-way hybridization one-hot encoding: SP, SP2, SP3
- Total hydrogen count, explicit valence, formal charge, implicit valence, explicit hydrogen count, and radical electron count

Bond-type and bond-attribute feature tensors are not currently returned by `extract_graph_structure`.

## MGNN Drug Encoder

The implemented MGNN accepts node features with shape `[number_of_atoms, 22]` and produces a 96-dimensional graph embedding. Its stages are:

- Initial graph convolution: `22 → 32`
- Three densely connected multiscale blocks, each with 8 DenseLayer units
- Growth rate 32; bottleneck size `bn_size=2` (intermediate width 64)
- A graph-convolution transition after every block, halving channel width
- Global mean pooling across molecular nodes, followed by a linear projection to 96 dimensions

Depth is reported at three distinct counting levels for the same implementation: 27 paper-reported architectural units (24 DenseLayer units plus 3 transitions), 28 top-level stages when `conv0` is included, and 52 literal `GraphConv` modules because each DenseLayer unit contains two graph convolutions. MGNN component tests pass locally, including variable-size molecules and batched graphs.

## MCNN Protein Encoder

The implemented MCNN accepts integer protein tokens shaped `[B, 1200]`. A trainable embedding with padding index 0 maps each token to 128 dimensions. The result is permuted to Conv1D layout `[B, 128, 1200]` and passed to three parallel branches:

| Branch | Conv1D layers | Kernel / stride / padding | Channels | Length before pooling |
| --- | ---: | --- | ---: | ---: |
| 1 | 1 | 3 / 1 / 0 | 96 | 1198 |
| 2 | 2 | 3 / 1 / 0 | 96 | 1196 |
| 3 | 3 | 3 / 1 / 0 | 96 | 1194 |

Each convolution is followed by ReLU. Adaptive max pooling reduces each branch to `[B, 96]`; concatenation yields `[B, 288]`, then `Linear(288, 96)` produces the protein embedding `[B, 96]`. MCNN has no batch normalization.

**Paper and released-code distinction:** The paper describes branch feature maps symbolically with sequence length 1,200, but does not explicitly specify convolution padding. The authors' released implementation omits the padding argument, so PyTorch uses `padding=0`; its actual branch lengths are 1198, 1196, and 1194 before pooling. This project follows the released implementation and does not attribute `padding=0` to an explicit paper setting.

## Integrated Prediction Model

`src/models/mgraphdta.py` concatenates the 96-dimensional protein embedding before the 96-dimensional drug embedding to form a `[B, 192]` joint representation. Its regression head is:

```text
192 → 1024 → 1024 → 256 → 1
```

Each hidden linear layer is followed by ReLU and dropout with probability 0.1. The final layer returns one scalar per input pair. These dimensions agree with the authors' [released regression implementation](https://github.com/guaguabujianle/MGraphDTA/blob/dev/regression/model.py). The paper describes a multilayer perceptron with ReLU and 0.1 dropout; the released implementation supplies the numeric widths.

## Installation

The checked project environment uses Python 3.12, PyTorch 2.6.0 with CUDA 12.4, PyTorch Geometric 2.8.0, and RDKit 2026.03.6. The exact package pins are in `requirements.txt`; its PyTorch packages are CUDA 12.4 builds.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If pip cannot find the CUDA-specific PyTorch wheels through its configured package index, install the pinned PyTorch packages from the CUDA 12.4 wheel index first, then install the remaining requirements. A working NVIDIA driver is required for CUDA; the training code also supports CPU.

For a CUDA 12.4 environment, the explicit PyTorch installation is:

```powershell
python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The raw Davis files and processed tables are ignored by Git. Provide `data/raw/davis/ligands_can.txt`, `proteins.txt`, `Y`, and the two files under `folds/` before running Davis data tests or training. Processed CSV tables are generated by `src/dataset.py` when needed.

## Training

Run the configured full experiment from the repository root:

```powershell
python -m src.train --epochs 3000 --steps-per-epoch 50 --batch-size 512 --learning-rate 0.0005 --patience 400 --validation-fraction 0.1 --seed 42
```

The command trains with Adam and MSE. Each reported epoch contains 50 optimizer updates. The 25,046-row Davis training fold is split into optimization and validation subsets (90%/10%); validation MSE selects the checkpoint and controls early stopping. The best checkpoint is restored before the independent 5,010-row test set is loaded for final evaluation. The test set is not used during optimization, checkpoint selection, or early stopping.

The defaults specify **up to** 3,000 epochs, 50 updates per epoch, batch size 512, learning rate 0.0005, and patience 400. The complete 3,000-epoch experiment has **not** been completed.

### 10-Epoch Pilot Results

The reported pilot completed 10 epochs and obtained:

| Metric | Pilot result |
| --- | ---: |
| Best validation MSE | 0.478949 |
| Independent test MSE | 0.569563 |

These are **PILOT results only**, not final benchmark results and not a reproduction of the paper's reported results. The exact pilot command and run metadata (including seed and environment) should be recorded with the run before treating it as reproducible.

## Tests

The test suite covers:

| Test | What it verifies |
| --- | --- |
| `tests/test_atom_features.py` | 22-dimensional atom feature shapes and finite values for representative molecules and available Davis ligands. |
| `tests/test_davis_dataset.py` | Davis interaction count, SMILES parsing, variable-edge PyG batching, protein tensor length, affinity values, and unique compound/target counts. Requires local Davis data. |
| `tests/test_mgnn.py` | MGNN depth accounting, stage widths, individual and batched forward passes, output dimensions, finite values, and CUDA when available. |
| `tests/test_mcnn.py` | MCNN embedding and branch tensor shapes, pooling and fusion dimensions, preprocessing lengths, gradient flow, batch processing, and CUDA when available. |
| `tests/test_mgraphdta.py` | End-to-end forward pass for one and batched Davis samples, encoder/fusion dimensions, scalar output, finite values, gradients, and CUDA when available. |
| `tests/test_training.py` | Train/validation partition determinism and exclusivity, validation-based checkpoint/early stopping, test isolation/final evaluation order, and CPU/CUDA smoke tests. |

Run the scripts from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe tests\test_atom_features.py
.\.venv\Scripts\python.exe tests\test_davis_dataset.py
.\.venv\Scripts\python.exe tests\test_mgnn.py
.\.venv\Scripts\python.exe tests\test_mcnn.py
.\.venv\Scripts\python.exe tests\test_mgraphdta.py
```

The training tests use pytest-style test functions. Install pytest if it is not already available, then run:

```powershell
python -m pip install pytest
python -m pytest tests\test_training.py
```

Davis tests require the local data files. CUDA checks run only when the selected PyTorch environment can access CUDA.

## Repository Structure

This tree lists tracked project files. Local raw/processed dataset files and the research-paper PDF are ignored by Git and are not included in a clone.

```text
MGraphDTA/
├── .gitignore
├── README.md
├── PROJECT_PLAN.md
├── PROJECT_PLAN.pdf
├── requirements.txt
├── checkpoints/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── scripts/
│   └── check_environment.py
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── train.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── grad_aam.py
│   └── models/
│       ├── __init__.py
│       ├── mcnn.py
│       ├── mgnn.py
│       └── mgraphdta.py
└── tests/
    ├── __init__.py
    ├── test_atom_features.py
    ├── test_davis_dataset.py
    ├── test_mcnn.py
    ├── test_mgnn.py
    ├── test_mgraphdta.py
    └── test_training.py
```

## Future Work

- Complete and record the planned full Davis training run.
- Implement and validate additional evaluation metrics, including CI, RMSE, and $r_m^2$; `src/metrics.py` currently contains no metric functions.
- Implement Grad-AAM atom-level explanations; `src/explainability/grad_aam.py` is currently a placeholder.
- Plan and run further experiments only after the Davis training and evaluation protocol is documented.

This is an academic implementation in progress, not a claim of a completed paper reproduction. The full experiment and paper-level comparisons remain outstanding.

## References

- Yang, Z., Zhong, W., Zhao, L., and Chen, C. Y.-C. “MGraphDTA: Deep Multiscale Graph Neural Network for Explainable Drug–Target Binding Affinity Prediction.” *Chemical Science*, 2022, 13, 816–833. [Paper](https://doi.org/10.1039/D1SC05095E).
- Authors' released implementation: [guaguabujianle/MGraphDTA](https://github.com/guaguabujianle/MGraphDTA).
