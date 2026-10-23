# Versioned Davis Scaled-Feature Dataset

`data/processed/davis_scaled_v1/` is an opt-in input bundle for testing the pinned author-style per-molecule atom-feature scaling. The normal Davis path remains unscaled. No claim is made that scaling improves prediction; no model experiment is included in this dataset preparation.

## Bundle contents

- `davis_all.csv`, `davis_train.csv`, `davis_test.csv`: byte-for-byte copies of the existing processed Davis interaction tables. Labels, row order, protein sequences, and fold membership are unchanged.
- `ligand_atom_features.pt`: CPU float32 dictionary mapping each canonical Davis drug ID to an atom-ordered `[number_of_atoms, 22]` tensor.
- `manifest.json`: schema, feature order, scaling rule, zero-range policy, author commit and source provenance, ligand IDs/SMILES/atom counts, and hashes for the bundle files and source data.

The features use the pinned author's formula `(x - x.min()) / (x.max() - x.min())`, with scalar min/max across each molecule's full `[atoms, 22]` tensor. The pinned source has no zero-range guard. The bundle builder rejects zero-range matrices and malformed/non-finite records instead of emitting NaNs; this safety behavior is explicitly not identical to the unguarded author code.

## Build and validate

Build only into a destination that does not already exist; the builder refuses to overwrite a prior bundle:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_davis_scaled_dataset
```

The source feature tables must already exist at `data/processed/davis_scaling_experiment/`, and the active interaction tables at `data/processed/davis/`. The builder validates source hashes, feature ordering, ligand IDs and SMILES, atom indices/counts, finite values, the scaling formula, and train/test partition membership before writing. The resulting bundle checks its own file hashes when loaded.

To inspect an item without training:

```powershell
@'
from src.dataset import DavisDataset

dataset = DavisDataset(
    split="train",
    processed_dir="data/processed/davis_scaled_v1",
    atom_features_dir="data/processed/davis_scaled_v1",
)
item = dataset[0]
print(item["drug_id"], item["pyg_data"].x.shape, item["pyg_data"].edge_index.shape)
'@ | .\.venv\Scripts\python.exe -
```

The current training CLI exposes the same opt-in through `--processed-dir data/processed/davis_scaled_v1 --atom-features-dir data/processed/davis_scaled_v1`. Omitting `--atom-features-dir` keeps the existing unscaled behavior. Both paths are explicit so copied split tables and the feature tensors are selected together. The adapter rebuilds graph connectivity from the same SMILES and replaces only `Data.x`; it verifies the split-table hash, drug ID/SMILES mapping, tensor shape, and finite values.

## Limitations

The features are transformed using the author implementation pinned at `54386680e8c9562f708a0fe5d381038c62111df6` and the recorded local RDKit environment. The official ESI was not available at the cited RSC URL during the documented retrieval attempt. This bundle does not establish complete paper-protocol parity and says nothing about predictive benefit. Any future comparison should pair the unscaled and scaled inputs with identical split indices, seeds, training settings, and validation-only checkpoint selection, and save outputs to a new experiment directory.
