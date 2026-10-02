# MGraphDTA — Davis Dataset Project Documentation

**Project scope:** Davis dataset only  
**Model:** MGraphDTA  
**Main components:** MGNN + MCNN + MLP + Grad-AAM

---

## 1. Project Overview

This project implements the MGraphDTA approach for drug–target binding-affinity prediction using the Davis benchmark dataset.

The complete pipeline is:

```text
Drug SMILES → RDKit molecular graph → MGNN → 96-D drug embedding
Protein sequence → tokenization → MCNN → 96-D protein embedding

Drug embedding + Protein embedding
                ↓
             192-D
                ↓
              MLP
                ↓
       Predicted affinity
                ↓
      Compare with true affinity
```

Grad-AAM is additionally used to explain individual drug-side predictions by assigning an importance score to each atom.

This project is a **Davis-only implementation**. It should not be described as an exact reproduction of every experiment in the original paper.

---

## 2. Problem Definition

Drug–target affinity prediction is a regression problem. The model receives:

1. A drug molecule.
2. A target protein sequence.

It predicts a continuous binding-affinity value.

The Davis dataset supplies the experimentally measured affinity values used as the regression targets.

---

## 3. Source Paper and Project Scope

The implementation is based on:

**MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction**, Chemical Science, 2022.

The paper introduces:

- MGNN for molecular graphs.
- MCNN for protein sequences.
- An MLP regression head.
- Grad-AAM for explainability.
- Benchmark experiments including Davis and other datasets.

This project intentionally focuses on Davis.

### Important distinction

The paper's reported benchmark results use its own experimental protocol, hyperparameter optimization and repeated experiments. This project uses a project-specific test-isolated validation protocol and one completed Davis run.

Therefore, the project's numerical result must be reported as **this implementation's result**, not as a reproduction of the paper's Table 3 value.

---

# 4. Davis Dataset

The Davis dataset used by MGraphDTA contains:

- **68 compounds**
- **442 protein targets**
- **30,056 drug–target interactions**

The underlying Davis representation contains files such as:

| File | Meaning | Project use |
|---|---|---|
| `ligands_can.txt` | Drug/ligand information, including SMILES | Build molecular graphs |
| `proteins.txt` | Protein target sequences | Protein input |
| `Y` | Binding-affinity values | Ground-truth regression target |
| `folds` | Split/fold information | Experiment organization |

The project preprocessing verification reported:

- 30,056 interactions
- 68 unique compounds
- 442 target IDs
- 379 unique protein sequences
- Maximum protein length: 1200

The distinction between 442 target IDs and 379 unique sequences is important: different target records can correspond to the same sequence.

---

# 5. Data Flow

```text
                    DAVIS DATASET
                          │
          ┌───────────────┴───────────────┐
          │                               │
      Drug SMILES                   Protein sequence
          │                               │
        RDKit                         Tokenization
          │                               │
   Molecular graph                    1200 tokens
          │                               │
         MGNN                            MCNN
          │                               │
      96-D vector                    96-D vector
          └───────────────┬───────────────┘
                          │
                    Concatenation
                       192-D
                          │
                         MLP
                          │
                 Predicted affinity
                          │
                 Compare with Y
                          │
                    Loss / metrics
```

---

# 6. Drug Preprocessing

## 6.1 SMILES to molecular graph

A drug starts as a SMILES string.

RDKit parses the SMILES and creates a molecular structure. The project converts this structure into a graph:

- Atoms → graph nodes.
- Bonds → graph edges.
- Atom properties → node features.

The current preprocessing represents each undirected bond using two directed entries in the graph edge index.

## 6.2 22-dimensional atom features

The implemented atom representation has 22 dimensions.

The feature groups include:

- 9-way atom-symbol one-hot encoding: H, C, N, O, F, Cl, S, Br, I
- Atomic number
- Hydrogen-bond acceptor
- Hydrogen-bond donor
- Aromaticity
- 3-way hybridization: SP, SP2, SP3
- Total hydrogen count
- Explicit valence
- Formal charge
- Implicit valence
- Explicit hydrogen count
- Radical electrons

The current preprocessing does **not** provide a separate bond-feature tensor; graph connectivity is represented through `edge_index`.

---

# 7. Protein Preprocessing

Protein targets are represented by amino-acid sequences.

The project converts the sequence to integer tokens and uses a maximum length of 1200.

The MCNN begins with a **128-dimensional token embedding**.

---

# 8. Model Architecture

## 8.1 MGNN — Molecular Graph Neural Network

MGNN is the drug encoder.

It uses multiscale/dense graph convolution blocks so information from multiple graph depths can contribute to the molecular representation.

### Main configuration

- Input atom features: 22
- Initial graph convolution: 22 → 32
- 3 multiscale dense blocks
- Each block: N = 8 dense-layer units
- Growth rate: 32
- Bottleneck intermediate size: 64
- 3 transition layers
- Each transition reduces channels by half
- Global mean pooling
- Final projection to 96 dimensions

The implemented channel flow is:

```text
22
 ↓
32
 ↓
288
 ↓
144
 ↓
400
 ↓
200
 ↓
456
 ↓
228
 ↓
Global mean pooling
 ↓
96
```

The paper reports **27 graph convolutional layers**, with three multiscale blocks of N=8 and three transition layers. The project documentation separately discusses the initial `conv0` when explaining layer-count conventions.

---

## 8.2 MCNN — Protein Encoder

MCNN processes the protein sequence at multiple sequence scales.

Input:

```text
Protein tokens
      ↓
Embedding dimension = 128
      ↓
Three parallel convolution branches
```

| Branch | Structure | Effective receptive field |
|---|---|---|
| Branch 1 | 1 × Conv1D, kernel 3 | 3 |
| Branch 2 | 2 × Conv1D, kernel 3 | 5 |
| Branch 3 | 3 × Conv1D, kernel 3 | 7 |

Each convolution produces 96 channels and uses ReLU.

Following the released implementation's no-padding behavior, an input length of 1200 gives branch lengths:

- 1198
- 1196
- 1194

Each branch then uses `AdaptiveMaxPool1d(1)`.

The outputs are concatenated:

```text
96 + 96 + 96 = 288
```

and projected:

```text
288 → 96
```

The final protein embedding is therefore 96-dimensional.

---

# 9. MLP Prediction Head

The drug and protein embeddings are concatenated.

```text
Protein embedding = 96
Drug embedding    = 96
                    ↓
                 192-D
```

The project uses:

```text
192 → 1024 → 1024 → 256 → 1
```

Hidden layers use ReLU and dropout probability 0.1.

The final output is one continuous predicted binding-affinity value.

---

# 10. Training

The completed Davis run used:

| Parameter | Value |
|---|---:|
| Dataset | Davis |
| Maximum epochs | 150 |
| Steps per epoch | 50 |
| Batch size | 32 |
| Learning rate | 0.0005 |
| Validation fraction | 0.10 |
| Patience | 30 |
| Random seed | 42 |
| Device | CUDA |

The project training pipeline:

1. Loads the Davis training portion.
2. Splits it into optimization and validation subsets.
3. Trains using MSE loss.
4. Uses validation MSE for checkpoint selection and early stopping.
5. Restores the best validation checkpoint.
6. Evaluates the independent test set only after model fitting.

### Best checkpoint

- Best epoch: **91**
- Best validation MSE: **0.23236284946015257**

The run was manually stopped at epoch 112 because further training was not desired.

---

# 11. Evaluation Metrics

The project evaluates:

### MSE

Mean squared error between predicted and true affinity.

### RMSE

Square root of MSE. It is expressed in the same target scale as affinity.

### CI

Concordance index. It evaluates how consistently the model preserves the ordering of affinity values.

### r_m²

A regression metric used in DTA research.

The project initially had a different r_m² formula. It was corrected to match the calculation used in the authors' released implementation.

---

# 12. Final Independent Davis Test Result

The corrected evaluation of the best checkpoint produced:

| Metric | Result |
|---|---:|
| MSE | **0.2570411053** |
| RMSE | **0.5069922142** |
| CI | **0.8790452402** |
| r_m² | **0.6817055903** |

These values belong to this project's Davis run and training protocol.

They are **not** the paper's reported mean ± standard deviation.

---

# 13. Grad-AAM Explainability

Grad-AAM explains which molecular atoms contributed strongly to an individual model prediction.

The project follows the authors' approach at the final graph transition layer.

The process is:

```text
Prediction
    ↓
Activation at final graph transition
    ↓
Gradient of prediction with respect to activation
    ↓
Average gradients over atoms
    ↓
Channel weights
    ↓
Weighted activation sum
    ↓
Atom importance
    ↓
Normalize to [0, 1]
```

The target activation has shape:

```text
[number_of_atoms, 228]
```

The resulting attribution has one score for every atom:

```text
[number_of_atoms]
```

The scores are mapped back to RDKit atom indices and visualized on the molecule.

### Interpretation

A high atom score means that the model assigned stronger attribution to that atom for the selected prediction.

It does **not** mean:

- that the atom is experimentally proven to contact the protein;
- that the atom is necessarily a binding site;
- or that Grad-AAM is measuring physical interaction energy.

It is a **model explanation**, not direct experimental evidence.

---

# 14. Grad-AAM Results

Five Davis examples were generated.

For the first five samples:

| Sample | True affinity | Predicted affinity | Absolute error | Atoms |
|---:|---:|---:|---:|---:|
| 0 | 5.000000 | 5.552389 | 0.552389 | 29 |
| 1 | 5.000000 | 5.258000 | 0.258000 | 25 |
| 2 | 5.000000 | 5.100961 | 0.100961 | 40 |
| 3 | 8.823909 | 8.305184 | 0.518724 | 46 |
| 4 | 5.000000 | 5.434842 | 0.434842 | 37 |

The project verifies that each molecule's number of atoms matches the number of Grad-AAM scores and that the scores are finite and normalized.

---

# 15. Testing and Verification

The full project test suite currently reports:

```text
36 passed
```

Major verification areas include:

- Davis dataset preprocessing
- 22-dimensional atom features
- MGNN
- MCNN/model components
- training pipeline
- evaluation metrics
- checkpoint evaluation
- Grad-AAM
- Grad-AAM hook cleanup
- model-mode restoration
- attribution shape and normalization

A Grad-AAM smoke test on an actual Davis sample verified:

```text
Activation:      (29, 228)
Atom importance: (29,)
Channel weights: (1, 228)
```

and confirmed finite attribution values and correct cleanup after the explanation pass.

---

# 16. Paper vs. Current Project

| Topic | Paper / authors | Current project |
|---|---|---|
| Dataset scope | Multiple benchmark datasets | Davis only |
| MGNN | 27 graph convolutional layers reported | Implemented and verified |
| MCNN | 3 multiscale branches | Implemented |
| MLP | Regression head | Implemented |
| Learning rate | 0.0005 in paper setup | 0.0005 |
| Batch size | 512 in paper setup | 32 |
| Experiment protocol | CV/tuning and repeated experiments | 90/10 validation inside training portion |
| Test usage | Paper's benchmark protocol | Independent test isolated from fitting |
| Metrics | MSE, CI, r_m² | MSE, RMSE, CI, r_m² |
| Explainability | Grad-AAM | Implemented for Davis |

The architecture is based on the paper and released implementation, while the training/evaluation protocol is explicitly a project-specific protocol.

---

# 17. Important Limitations

1. This project currently covers Davis, not all seven datasets in the paper.
2. The training protocol differs from the paper's full benchmark protocol.
3. The result is from one completed project run, not the paper's three-seed mean ± SD.
4. Grad-AAM is an attribution method and should not be interpreted as experimental proof of physical binding contacts.
5. The current graph preprocessing does not use separate bond-feature tensors.
6. The Davis dataset has known characteristics and should not be treated as a complete representation of all drug–target interactions.

---

# 18. Repository and Reproducibility

Project repository:

`https://github.com/Dksingh0001/MGraphDTA`

Important repository artifacts include:

```text
src/
tests/
scripts/
results/
bio_paper.pdf
live_plot.py
make_112_graph.py
```

Important result files include:

```text
results/davis/training_history.csv
results/davis/training_curve.png
results/davis/grad_aam/examples/
results/davis/run_112_backup/evaluation_metrics_corrected.json
results/davis/run_112_backup/mgraphdta_best.pt
```

The local virtual environment, cache files and local data directory are intentionally excluded from the repository.

---

# 19. Faculty / Viva Explanation

## 30-second explanation

> “My project implements MGraphDTA for drug–target binding-affinity prediction on the Davis dataset. A drug SMILES string is converted into a molecular graph and processed by an MGNN to obtain a 96-dimensional drug representation. The protein sequence is tokenized and processed by an MCNN to obtain a 96-dimensional protein representation. I concatenate these representations and use an MLP to predict binding affinity. I trained the model using a validation split while keeping the independent test set separate. I also implemented Grad-AAM to visualize which drug atoms contributed most to individual predictions.”

## If asked: What did you implement?

- Davis preprocessing.
- Molecular graph construction.
- 22-dimensional atom features.
- MGNN.
- MCNN.
- MLP prediction head.
- Training and validation pipeline.
- Independent checkpoint evaluation.
- MSE, RMSE, CI and corrected authors-aligned r_m².
- Grad-AAM explainability.
- Automated tests.
- Result generation and documentation.

## If asked: What is the main idea?

> “Instead of representing the drug only as a sequence or a fixed vector, MGraphDTA represents the drug as a molecular graph so the model can learn from atom connectivity, while the protein is represented as a sequence and processed at multiple sequence scales.”

---

# 20. Conclusion

The Davis-focused MGraphDTA implementation has reached a complete implementation-and-analysis milestone.

The project has:

- a verified Davis preprocessing pipeline;
- MGNN, MCNN and MLP;
- CUDA training;
- independent test evaluation;
- corrected evaluation metrics;
- Grad-AAM explainability;
- automated tests with 36 passing tests;
- documented results;
- and a synchronized GitHub repository.

The current best checkpoint is from epoch 91, with validation MSE 0.232363. Its independent Davis test results are MSE 0.257041, RMSE 0.506992, CI 0.879045 and r_m² 0.681706.

The next stage should focus on interpretation, documentation, presentation and research analysis rather than changing the architecture without a specific experimental question.

---

# 21. References

1. MGraphDTA paper: “MGraphDTA: deep multiscale graph neural network for explainable drug–target binding affinity prediction,” Chemical Science, 2022.
2. DOI: 10.1039/D1SC05180F
3. Paper: https://pubs.rsc.org/en-us/content/articlehtml/2022/sc/d1sc05180f
4. Authors' repository: https://github.com/guaguabujianle/MGraphDTA
5. Project repository: https://github.com/Dksingh0001/MGraphDTA
6. Project source material: `Explain MGraphDTA Project.txt`
