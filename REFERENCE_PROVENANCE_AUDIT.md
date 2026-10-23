# MGraphDTA Reference Provenance and Davis Preprocessing Audit

**Date:** 2026-10-17
**Scope:** Reference provenance, local Davis inputs, labels/folds, and atom features. This was a read-only audit of existing project artifacts. No training was launched and no existing model, dataset, checkpoint, metric, report, or result was modified.

## Summary

- Local paper: **bio_paper.pdf**, SHA-256 recorded below. The official ESI PDF is not present locally. The requested RSC ESI URL returned HTTP 404; an NCBI mirror also returned 404 and Europe PMC returned 403. Search indexing exposes selected Table S3 text, but that is not a locally verified ESI document.
- The author repository was cloned into the previously absent **references/mgraphdta_official/author_repo/** directory and pinned at commit **54386680e8c9562f708a0fe5d381038c62111df6**, the exact dev head returned by git ls-remote during this audit. The checkout is detached at that immutable commit.
- Compared with the pinned author preprocessing, all 22 **raw, unscaled** atom-feature values matched for every atom in all 68 local Davis ligands in the installed environment: 2,180 atom rows, maximum absolute difference 0. The confirmed feature discrepancy is scaling: author process_data applies min–max scaling over each molecule’s entire feature matrix; local preprocessing returns raw values.
- Davis raw inputs have the expected dimensions and fold counts. Local processed files are internally consistent with local raw inputs, row-major interaction construction, pKd conversion, and fold-index files. However, the author repository does not contain the Davis raw files; its regression preprocessing reads pre-split data_train.csv/data_test.csv and says they match GraphDTA. Exact upstream bytewise data provenance and equivalence to the authors’ original CSVs remain unverified.

This evidence does not establish complete paper-protocol parity. Keep it separate from existing baseline and multi-seed results.

## Reference materials and pin

Official sources checked: [paper page](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05180f), [requested RSC ESI](https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf), [pinned author preprocessing](https://github.com/guaguabujianle/MGraphDTA/blob/54386680e8c9562f708a0fe5d381038c62111df6/regression/preprocessing.py), [pinned author dataset loader](https://github.com/guaguabujianle/MGraphDTA/blob/54386680e8c9562f708a0fe5d381038c62111df6/regression/dataset.py), and [pinned author trainer](https://github.com/guaguabujianle/MGraphDTA/blob/54386680e8c9562f708a0fe5d381038c62111df6/regression/train.py). The commit object was obtained using git ls-remote; the local repository HEAD matches it.

| Material | Local status / provenance | SHA-256 |
|---|---|---|
| Paper PDF bio_paper.pdf | Present; Yang et al., Chemical Science 13 (2022), DOI 10.1039/D1SC05180F. | 0ed6fc6f669c45f60ccb14ef4c443ad78ef97057d7a34e142db5e2b479ce0cc3 |
| ESI d1sc05180f1.pdf | **Not obtained.** Requested [RSC URL](https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf) returned 404; PMC mirror returned 404 and Europe PMC mirror returned 403. No ESI file/hash exists in the new directory. RSC indexed search results exposed selected Table S3 values only. | Not available |
| Author repository | [Source repository](https://github.com/guaguabujianle/MGraphDTA), cloned under references/mgraphdta_official/author_repo/. Detached checkout at 54386680e8c9562f708a0fe5d381038c62111df6 (dev head observed during this audit). | Commit ID pins tree; source-file hashes below |
| Author regression preprocessor | regression/preprocessing.py at pinned commit. | f181f03a6a73c823dffd6c848c9fddacb2da8b259abfb0c8bbbdf8259f557577 |
| Author regression dataset loader | regression/dataset.py at pinned commit. | 5ffcb4aa0a4135725b53863fc5d9d71ad82c17bf0f6608cf95f8a08fa2600c60 |
| Author regression trainer | regression/train.py at pinned commit. | b5c292b83f9a61e233a2ce3c966e3390a0a2fd303b01fa199b276294246266df |
| Author regression model | regression/model.py at pinned commit. | 04e1696a2633aa6fd97c2b30cfb0bb06fea000f33d4f9d7c9c3a38abac870c1a |

The author README lists the old RDKit version 2009.Q1-1. This audit did not recreate that environment. The exact repository commit is pinned, but the runtime environment used for comparing its source was not the authors’ historical runtime.

The new references/mgraphdta_official/ directory contains the pinned author repository clone. It does **not** contain an ESI PDF: failed downloads did not produce a valid file. No existing path was overwritten; this directory did not exist before this task.

## Local input inventory and SHA-256

| Local file | SHA-256 |
|---|---|
| data/raw/davis/ligands_can.txt | 141094d99510a52091d2ea1b05be7c1f66b2c5fcaeeeb261eaf4cbf2128bde6 |
| data/raw/davis/proteins.txt | 187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291 |
| data/raw/davis/Y | 9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a |
| data/raw/davis/folds/train_fold_setting1.txt | bb7a63b2178c40a4e2417bc701b95511407787fa5eabb6a43a695fe6bd722f56 |
| data/raw/davis/folds/test_fold_setting1.txt | c97589fa2fae318b01870854af0ab6fdd155dd38524cc6ac1d0319418ac6d568 |
| data/processed/davis/davis_all.csv | 957dca7d00b56cf2848cf28d32c65778b84e4563f1260cbea10186b49457655d |
| data/processed/davis/davis_train.csv | a9f7b121fc0d26dd85d9633fd4eee47cd63f2e36d5e068461f8191916d48549d |
| data/processed/davis/davis_test.csv | 5658ff56c9ac6b0a9649fed632d817d58c4a9fcf926919d7307a028184d18638 |
| src/preprocessing.py | d0f6a3a7e1a4a59c2622876143ca37e54552f8343ed11a73b4ca61ed63b77532 |
| src/dataset.py | 482f29f0896d10eb57760202c54b866090d4642d17b248914a8d350525e7e3c9 |
| src/train.py | 81869e4e63c1a423522b446a6ce0f68b7df1fded2cb3385da40e9360acedc758 |

## Davis data, labels, ordering, and folds

### Verified local facts

- ligands_can.txt has 68 ligands; proteins.txt has 442 proteins; Y has shape (68, 442) and 30,056 values. Raw affinities range from 0.016 to 10,000 nM; 20,931 values equal the 10,000 nM sentinel.
- src/dataset.py:load_raw_davis_data reads the ligand/protein dictionaries, Y, and fold JSON files.
- src/dataset.py:prepare_davis_tables iterates ligand dictionary order outside and protein dictionary order inside. It assigns pair_idx = drug_index × 442 + protein_index and reads Y[drug_index, protein_index]. davis_all.csv has 30,056 rows with pair_idx 0–30,055 in sequence. First/last rows are (0, 11314340, AAK1) and (30055, 151194, ZAP70).
- Local label conversion is -log10(Kd_nM / 1e9), equivalent to 9 - log10(Kd_nM). Recomputing from raw Y gave maximum absolute error 1.78e-15; affinity_kd exactly matched Y in order.
- The five training-fold lists have sizes [5010, 5009, 5009, 5009, 5009], total 25,046 unique indices. Test has 5,010 unique indices. There are no duplicates or train/test overlap; the union covers all 30,056 positions.
- davis_train.csv has 25,046 rows and davis_test.csv has 5,010. Their pair_idx sequences exactly equal the flattened training folds and test fold. Their labels match the corresponding davis_all.csv rows.
- The paper identifies Davis, the pKd convention (including 10 μM/pKd 5), and the DeepDTA/GraphDTA data source. The author regression preprocessor says training and test datasets are the same as GraphDTA. Its GNNDataset reads data_train.csv/data_test.csv; it does not construct them from raw Y or fold-index files.

### Exact provenance limitation

The checks prove consistency among the local raw data, local processing code, and local processed CSVs. They do not prove byte identity with the authors’ pre-split CSVs or upstream DeepDTA files. Those files are not in the pinned author repository and were not downloaded or hashed. Matching row counts and a cited source are not bytewise provenance checks.

## Atom-feature comparison

### Feature order in pinned author source

Author regression/preprocessing.py GNNDataset.get_nodes emits 22 columns in this order:

1. Atom-symbol indicators H, C, N, O, F, Cl, S, Br, I (9).
2. Atomic number.
3. Acceptor indicator.
4. Donor indicator.
5. Aromaticity.
6. Hybridization indicators SP, SP2, SP3 (3).
7. Total hydrogen count.
8. Explicit valence.
9. Formal charge.
10. Implicit valence.
11. Explicit hydrogen count.
12. Radical-electron count.

Author mol2graph uses default Chem.MolFromSmiles and RDKit BaseFeatures.fdef. process_data scales the complete node-feature matrix for each molecule with (x - x.min()) / (x.max() - x.min()).

### Element-by-element local comparison

Compared src/preprocessing.py:smiles_to_mol, _get_donor_acceptor_sets, and extract_atom_features_22dim with the pinned author mol2graph/get_nodes using local RDKit 2026.03.6 on every unique Davis ligand:

- Molecules: 68; atom rows: 2,180.
- Feature dimension/order: match (22 columns).
- Raw feature values before scaling: exact match for all values (maximum absolute difference 0).
- Scaling: local code does not scale. Author code applies per-molecule matrix min–max scaling. Maximum absolute difference between local raw features and author-scaled values in this comparison was 52.0.
- Unknown atom symbols: the Davis ligand set contains one phosphorus atom (P), outside the 9-symbol one-hot list. Both implementations produce all-zero symbol indicators for P while retaining atomic number 15. This is consistent behavior, not a mismatch for Davis. Other atom symbols likewise have all-zero symbol indicators; hybridizations other than SP/SP2/SP3 have all-zero hybridization indicators.
- RDKit settings: both source paths use default MolFromSmiles behavior and BaseFeatures.fdef. Current-environment feature values agreed. Local code calls modern GetValence(EXPLICIT/IMPLICIT) with legacy fallback; author code calls GetExplicitValence/GetImplicitValence. They matched under 2026.03.6, though author calls emitted deprecation warnings. Exact results under the historical RDKit release were not checked.
- Author preprocessing also stores six-dimensional bond attributes. Its released GraphConv calls consume node features and edge indices, not edge_attr. Local preprocessing omits edge_attr. This is a stored representation difference; neither audited model passes bond attributes into GraphConv.

## Status

### Verified matches

- Local Davis matrix shape, interaction count, ligand/target counts, label transform, row-major pair_idx construction, and folds are internally consistent.
- Local 22 unscaled atom-feature columns match the pinned author feature order and values for all local Davis ligands under RDKit 2026.03.6.
- Local and author protein vocabularies match. No character in Davis protein sequences is unmapped locally. Local maps other unknown characters to padding token 0; author seqs2int uses direct lookup and would error on unknown characters, but no Davis sequence triggers this difference.
- The author repository commit and relevant source-file hashes are recorded above.

### Confirmed discrepancies

- Local atom features are unscaled while author preprocessing performs per-molecule matrix min–max scaling. Do not apply this in-place to existing processed CSVs or results; use a separate versioned preprocessing output.
- Author source creates bond attributes but local output does not. They are unused in both audited regression GraphConv forward paths.
- The current environment uses RDKit 2026.03.6, not the old RDKit version listed by the author README. Current-runtime output matched; historical-version parity remains untested.

### Unresolved

- Full official ESI PDF unavailable locally; full S1/S2 definitions and complete supplemental detail are unverified. Indexed S3 text is not a hashed ESI artifact.
- Author pre-split Davis CSVs and source hashes are absent; local inputs cannot yet be proven byte-identical to the authors’ inputs.
- Historical RDKit behavior, particularly valence and feature-factory output, is untested.
- The pinned dev commit is a reproducible source snapshot, but no tag/release ties it to the exact paper experiment.

## Next step before any new training

1. Obtain the official ESI PDF and save it as references/mgraphdta_official/d1sc05180f1.pdf; calculate and record its SHA-256. Do not use indexed snippets as a substitute.
2. Obtain the authors’ original Davis data_train.csv/data_test.csv (or a source archive with URL and hash). Compare row identity, SMILES, sequences, labels, and order to the local processed files.
3. Before changing preprocessing, create a new versioned processed-data directory. Apply author scaling to each molecule’s x matrix using (x - x.min()) / (x.max() - x.min()), with a documented policy for constant matrices, then compare generated tensors. Do not regenerate over data/processed/davis/ or overwrite results.
4. Add preprocessing equivalence checks for feature order/value, unknown atoms, protein vocabulary, fold mapping, labels, and scaling. Review those checks before authorizing training.

Read-only commands after obtaining source materials:

    Get-FileHash .\references\mgraphdta_official\d1sc05180f1.pdf -Algorithm SHA256
    Get-ChildItem .\data\raw\davis -Recurse -File | Get-FileHash -Algorithm SHA256
    git -C .\references\mgraphdta_official\author_repo rev-parse HEAD

**No model or dataset change is performed by this report.**

## Reference retrieval update (2026-10-23)

The official supplement is identified in the [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC8768884/) as SC-013-D1SC05180F-s001.pdf; the publisher URL is [RSC d1sc05180f1.pdf](https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf). Retrieval attempts did not yield a valid PDF. The requested local path remains absent, and no file size/hash can be reported. RSC indexed text partially exposes Table S3 selections, but Tables S1–S2 and the full S3 context remain unresolved pending direct PDF inspection.

The author README-linked archive already present at `references/mgraphdta_official/MGraphDTA_Data.zip` is 118,403,071 bytes, SHA-256 46b58cf905e9f803eee3e0df72ec85faf35234bda53054ed165301200565fd3c. It contains the Davis train/test CSVs (member hashes and sizes are recorded in REFERENCE_DATA_AND_SCALING_AUDIT.md). Their rows match local processed train/test records in split order on canonicalized SMILES, sequence, and exact labels (25,046/5,010 rows; max label difference zero). The pinned regression checkout itself lacks these Davis files. This is semantic row correspondence to the shared archive, not byte-identical provenance to an immutable author release.
