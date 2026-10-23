# Davis Molecular Feature Scaling Comparison

## Scope and safety

This is a separate preprocessing comparison only. No model training was run, and the active preprocessing code, `data/processed/davis/`, checkpoints, metrics, baseline artifacts, and earlier reports were not changed. The generated outputs live only in `data/processed/davis_scaling_experiment/`.

## Pinned author behavior

The checked-out author repository is pinned at commit `54386680e8c9562f708a0fe5d381038c62111df6`. In `regression/preprocessing.py`, inside `process_data`, the author first obtains `x, edge_index, edge_attr = graph_dict[smi]` and then applies:

```python
x = (x - x.min()) / (x.max() - x.min())
```

At this point `x` is a two-dimensional tensor with shape `[number_of_atoms, 22]`. There is no `dim`/`axis` argument: `min()` and `max()` are scalar reductions over every atom and every feature in the current molecule's matrix. It is therefore one min–max transform per molecule, not one transform per feature column or one transform across the dataset. It is performed after graph/atom-feature construction and before the graph is packaged as a PyG `Data` object for downstream model use.

The pinned formula has no zero-range guard. If every entry in a molecule's complete matrix is equal, the denominator is zero and the result contains non-finite values (`0/0` for a constant matrix). A column with zero range is not independently divided by zero: it receives the same matrix-wide transform as every other column. The generator records the author's behavior and fails closed on an all-constant matrix rather than writing NaNs. The standalone `guarded_global_minmax` helper raises `ValueError` for non-finite or zero-range input; that is a safer policy, but it is explicitly not identical to the author's unguarded formula.

## ESI retrieval attempt

One final request was made to the RSC supplementary-material URL specified in the prior provenance audit:

`https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf`

On 2026-10-17 the endpoint returned HTTP 404 (the response was HTML, not a PDF). No ESI file was saved, so there is no ESI SHA-256 hash. No third-party copy was substituted or treated as authoritative. ESI details therefore remain unverified.

## Generated comparison artifacts

The generator is `scripts/generate_davis_scaling_comparison.py`; it imports the pinned author preprocessing implementation after checking the repository commit. It compares author raw features to the local atom-feature implementation, then writes flattened, molecule-labelled unscaled and author-scaled tables. It does not use or write active processed tensors.

Environment and provenance recorded in `manifest.json`:

- Author commit: `54386680e8c9562f708a0fe5d381038c62111df6`
- Author preprocessing source SHA-256: `f181f03a6a73c823dffd6c848c9fddacb2da8b259abfb0c8bbbdf8259f557577`
- Local ligand dictionary SHA-256: `141094d99510a52091d2ea1b05be7c1f66b2c5fcaeeeb261eaf4cbf2128bde6d`
- RDKit: `2026.03.6`; PyTorch: `2.6.0+cu124`
- Ligands: 68; atom rows: 2,180; features per atom: 22
- For these inputs, each molecule's complete feature-matrix range is nonzero (observed ranges 8 through 53), and both generated tables contain finite values.

| Artifact | SHA-256 |
|---|---|
| `unscaled_atom_features.csv` | `8a4129527ab905f1de5684d3e59396950e4e69adddf9c418e440ac221e0e7a43` |
| `author_scaled_atom_features.csv` | `26a1e3ed9a873ea7ce758bcb3708d226476746fdb688aa002ffd45cea5a07726` |

The manifest also records row counts, ordered feature names, each ligand's atom count and matrix range, input hashes, and output hashes.

## Validation

Added focused tests in `tests/test_davis_scaling_experiment.py`. The tests cover:

- expected `[atoms, 22]` shape, 68-ligand coverage, and finite generated values;
- feature ordering and exact unscaled parity with the pinned author graph builder;
- matrix-wide per-molecule scaling and the effect on a zero-range feature column;
- the author's unguarded all-constant behavior versus the guarded helper's explicit error;
- empty and invalid SMILES rejection;
- output identity, manifest hashes, repeated-generation hash reproducibility, and refusal to overwrite an existing output directory.

Command run:

```powershell
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_davis_scaling_experiment.py
```

Result: **9 passed**. No training or full project test suite was run; these tests are specific to the new feature comparison. The deterministic hashes establish repeatability in the current environment and input order; they do not establish identical RDKit behavior across different versions/platforms.

## Remaining uncertainties and safe next step

The official ESI remains unavailable at the cited RSC endpoint, and the present experiment only compares atom features. It does not establish full preprocessing or paper-protocol parity, and it does not show that scaled features improve prediction. The active training pipeline still uses its prior feature path.

Before any training, review the two generated tables and manifest, then decide whether a separately versioned model-input dataset should be prepared from the scaled table. If proceeding toward paper reproduction, first obtain the ESI from an authoritative RSC route or the authors and record its provenance. Any later model experiment should use a new output directory and explicitly identify which feature variant it consumes; do not redirect the active Davis dataset implicitly.
