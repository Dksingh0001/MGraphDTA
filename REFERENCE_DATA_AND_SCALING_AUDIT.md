# MGraphDTA Reference Data and Atom-Scaling Audit

**Date:** 2026-10-17
**Scope:** Read-only investigation of pinned author references, Davis source records, split files, and atom-feature scaling. No training was launched. No existing code, active data, checkpoint, metric, report, or result was changed. Reference copies were added only under references/mgraphdta_official/.

## Summary

1. The author repository at commit 54386680e8c9562f708a0fe5d381038c62111df6 contains no Davis raw train/test CSV files. Its regression loader expects data_train.csv and data_test.csv; its README links to an author Google Drive archive and the DeepDTA Davis repository.
2. The README-linked author archive was retrieved as references/mgraphdta_official/MGraphDTA_Data.zip (SHA-256 46b58cf905e9f803eee3e0df72ec85faf35234bda53054ed165301200565fd3c). It contains full, train, and test Davis CSVs. The archive is now locally pinned by its hash.
3. The author CSV full dataset contains the same 30,056 interactions as local Davis when comparing canonicalized SMILES, protein sequence, and affinity as a multiset. Its full-table ordering differs from local row-major ordering. The author train and test CSVs match the local train and test files row-for-row by canonicalized SMILES, target sequence, and pKd.
4. The local raw Davis affinity matrix is byte-identical to the DeepDTA repository copy and numerically identical to the academic Davis source matrix. The local compound dictionary has the same parsed values and key order as DeepDTA; its JSON byte representation differs. The local protein JSON and fold JSON files are byte-identical to DeepDTA copies.
5. In the pinned author code, per-molecule scaling uses a single minimum and maximum over the entire two-dimensional node-feature matrix. There is no feature-axis argument and no constant-matrix guard. Scaling occurs in process_data after graph creation and before constructing the PyG data object. Local code does not apply this scaling.
6. The official RSC ESI remained unavailable for local download: the RSC and PMC PDF URLs returned 404; Europe PMC returned 403. The RSC search index exposes selected Table S3 text, but no full ESI was obtained or hashed.

This strengthens data provenance for the current split, but does not establish complete paper reproduction: preprocessing still differs by atom-feature scaling, and the ESI remains unavailable.

## Sources, retrieval, provenance, and hashes

| Source and URL | Retrieval status | Local copy / hash | Provenance notes |
|---|---|---|---|
| [MGraphDTA paper](https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05180f) | Official page accessible through browser search; local paper PDF already exists. | bio_paper.pdf SHA-256: 0ed6fc6f669c45f60ccb14ef4c443ad78ef97057d7a34e142db5e2b479ce0cc3 | Yang et al., Chemical Science 13 (2022), DOI 10.1039/D1SC05180F. |
| [RSC ESI PDF](https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf) | Download attempted; HTTP 404. RSC search index still identifies ESI Tables S1–S3. | No valid file; no hash. | Not treated as locally verified. |
| [PMC supplement listing](https://pmc.ncbi.nlm.nih.gov/articles/PMC8768884/) and listed file SC-013-D1SC05180F-s001.pdf | Listing confirms a supplement attachment; attempted the PMC bin URL and received 404. | No valid file; no hash. | The listing is evidence a supplement exists, not a replacement copy. |
| [Europe PMC supplement route](https://europepmc.org/articles/PMC8768884/bin/SC-013-D1SC05180F-s001.pdf) | Attempted; HTTP 403. | No valid file; no hash. | Not treated as retrieved. |
| [Pinned MGraphDTA author repository](https://github.com/guaguabujianle/MGraphDTA/tree/54386680e8c9562f708a0fe5d381038c62111df6) | Cloned successfully to references/mgraphdta_official/author_repo/ at detached commit 54386680e8c9562f708a0fe5d381038c62111df6. | Source file hashes below. | Exact commit was the dev branch head reported by git ls-remote at retrieval time. |
| Author README’s [Google Drive archive](https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing) | Retrieved after following Google’s virus-scan interstitial’s explicit confirmation form. Archive identified as MGraphDTA_Data.zip, 118,403,071 bytes. | references/mgraphdta_official/MGraphDTA_Data.zip SHA-256: 46b58cf905e9f803eee3e0df72ec85faf35234bda53054ed165301200565fd3c | This is the archive linked from the pinned author README; archive hash pins the retrieved copy. |
| [DeepDTA data repository](https://github.com/hkmztrk/DeepDTA/tree/a546a8433a6822e958f36171c4356ad6f414d623/data/davis) | Cloned successfully to references/mgraphdta_official/DeepDTA_source/ at commit a546a8433a6822e958f36171c4356ad6f414d623. | Davis file hashes below. | This is the Davis/KIBA source explicitly referenced in the pinned MGraphDTA README. |
| [Academic Davis source page](https://staff.cs.utu.fi/~aatapa/data/DrugTarget/) and its linked Davis matrix/ID files | Page and all three relevant linked files retrieved. | Files and hashes below. | Page states that the matrix contains Davis et al. (2011) Kd affinities and explains row/column ID ordering. |

### Pinned code files

| File at the author commit | SHA-256 |
|---|---|
| references/mgraphdta_official/author_repo/regression/preprocessing.py | f181f03a6a73c823dffd6c848c9fddacb2da8b259abfb0c8bbbdf8259f557577 |
| references/mgraphdta_official/author_repo/regression/dataset.py | 5ffcb4aa0a4135725b53863fc5d9d71ad82c17bf0f6608cf95f8a08fa2600c60 |
| references/mgraphdta_official/author_repo/regression/train.py | b5c292b83f9a61e233a2ce3c966e3390a0a2fd303b01fa199b276294246266df |

The author repository root README says the regression folder has no raw data and points to the same Google Drive archive also linked for visualization. Regression preprocessing expects data_train.csv and data_test.csv as input, while its dataset loader reads those names from its raw folder. No script in the pinned regression repository generates those split CSVs from Davis Y/fold files. The separately named filtered_davis folder is a different benchmark task and is not the Davis regression input used here.

### Archive members

| Archive member path | Rows | Size | SHA-256 |
|---|---:|---:|---|
| MGraphDTA_Data/davis/raw/data.csv | 30,056 | 25,816,618 | 133f0682b7b10e7ccf5fdbc9f85d498553e3f5cdb93b97e7055508270883ecc3 |
| MGraphDTA_Data/davis/raw/data_train.csv | 25,046 | 21,383,855 | df64c237a9f046f55c4ae5962e1ca268d381b7fc84d21a59c346310930c4068c |
| MGraphDTA_Data/davis/raw/data_test.csv | 5,010 | 4,293,933 | 62f8a57037a8246d2585465065e4b964f5171baafb2c7944412802f0b16a8603 |

Hashes are of the uncompressed CSV member bytes read directly from the ZIP. Members were inspected in memory; none were extracted into data/processed or another active dataset path.

### Academic Davis source files

| File from staff.cs.utu.fi/~aatapa/data/DrugTarget/ | Local reference copy | SHA-256 |
|---|---|---|
| drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt | references/mgraphdta_official/davis_original_affinity.txt | e2c3d003d49aff61f70be24feafd8d084b0df9404d919338088855828e252d88 |
| drug_PubChem_CIDs.txt | references/mgraphdta_official/davis_drug_pubchem_ids.txt | a4973d1a28b608f28c8e7a111ad049e9093886059751d56abe01bb99951c7bb1 |
| target_gene_names.txt | references/mgraphdta_official/davis_target_gene_names.txt | 68da721a5c5a04da9ba11f3a7cedfe1ffacd87c67f6b39a494514edfc97661a7 |

## Davis record comparison

### Compounds and proteins

- The local raw ligand dictionary has 68 entries. Parsed JSON values and key order exactly match DeepDTA’s ligands_can.txt. The files are not byte-identical because their serialized bytes differ: local SHA-256 is 141094d99510a52091d2ea1b05be7c1f66b2c5fcaeeeb261eaf4cbf2128bde6; DeepDTA SHA-256 is 9789900f1e723226187215235b379998e5735a3ffec0e9487bad27c1c6a64161.
- The 68 original PubChem CID rows are in exactly the same order as the local ligand dictionary keys. The author archive includes 68 unique canonicalized SMILES, and their structures match the local 68 compounds after RDKit canonicalization.
- Local proteins.txt is byte-identical to DeepDTA’s proteins.txt (SHA-256 187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291). Both have 442 target records and 379 unique sequences.
- The academic source’s 442 target gene-name labels are not text-identical to local proteins.json keys at 38 positions, often because variant/phosphorylation labels are abbreviated or normalized (for example, ABL1(E255K)-phosphorylated versus ABL1(E255K)). Do not use exact label-string equality as proof of a sequence mismatch. The author archive carries target sequences, and the split-record comparison below verifies those sequences rowwise against local data.
- The author archive full Davis CSV and local full processed CSV each contain 30,056 rows, 68 compounds, and 379 unique target sequences. All full-dataset records match as a multiset after canonicalizing SMILES, comparing target sequence, and comparing parsed affinity. Their full-table row orders differ.

### Affinity values and pKd

- Local raw Y is byte-identical to the DeepDTA Y file (SHA-256 9855c4f234ec6dd18295b0e0cd6ee54214cafa908c6eb1d447b47d454fc0294a).
- The academic Davis source file parses to a 68×442 matrix numerically identical to local Y; all 30,056 values match exactly. The source reports Kd in nM. There are 20,931 entries equal to 10,000 nM.
- Local src/dataset.py converts to pKd as -log10(Kd_nM / 1e9) = 9 - log10(Kd_nM). A 10,000 nM label becomes pKd 5. Recomputing the local processed labels from Y gave maximum absolute error 1.78e-15.
- The author archive stores affinity already on the pKd scale. For all full records the values match local processed values after canonical-SMILES record matching; train and test match in sequence as described below.

### Ordering and fold assignment

- DeepDTA’s folds/train_fold_setting1.txt and folds/test_fold_setting1.txt are byte-identical to local files. Hashes: training folds bb7a63b2178c40a4e2417bc701b95511407787fa5eabb6a43a695fe6bd722f56; test fold c97589fa2fae318b01870854af0ab6fdd155dd38524cc6ac1d0319418ac6d568.
- Local folds contain five training lists of lengths 5,010, 5,009, 5,009, 5,009, 5,009 and one 5,010-entry test list. Indices are unique, disjoint, and cover all 30,056 matrix positions.
- Local src/dataset.py builds a row-major grid: ligand dictionary order outside, protein dictionary order inside; pair_idx = drug_index × 442 + protein_index. Local all/train/test CSVs have 30,056/25,046/5,010 rows. The split CSV pair_idx sequences exactly follow the flattened train-fold list and test-fold list.
- Author data_train.csv (25,046 rows) and data_test.csv (5,010 rows) match local davis_train.csv and davis_test.csv row-for-row by canonicalized SMILES, protein sequence, and parsed affinity. Thus the author-prepared split membership and order are verified against the local fold-derived splits at the full interaction-record level.
- The author archive data.csv has the same records as local davis_all.csv as a multiset but a different full-dataset ordering. Its index column is sequential 0–30,055 but does not encode local row-major pair_idx. Do not compare these two full tables row-by-row using their index columns.
- The academic Davis page provides the raw matrix and compound/target ID lists, but not the MGraphDTA fold files or protein sequences. Therefore it can verify the raw matrix, compound ID order, and target-label inventory, while DeepDTA and the author archive verify folds and target sequence records.

## Author atom-feature scaling

### Exact implementation

In the pinned author regression/preprocessing.py:

1. mol2graph builds the graph and raw node feature matrix x with shape [number_of_atoms, 22].
2. process_data retrieves x from graph_dict for one molecule.
3. Before creating the PyG Data object, it computes x = (x - x.min()) / (x.max() - x.min()).

Both x.min() and x.max() are scalar reductions over the entire matrix for that one molecule. There is no axis/dimension argument and no per-column normalization. The same molecule-level minimum and maximum are applied to every atom and feature entry. It is not dataset-wide scaling and not feature-wise scaling.

There is no guard for x.max() == x.min(). A constant matrix would divide by zero and produce non-finite values. The implementation does not replace that denominator or specify a fallback. The normal Davis molecule feature matrices are nonconstant; the code itself nevertheless lacks general constant-input handling.

The local src/preprocessing.py creates the same raw 22-feature matrices as the pinned author code in an element-by-element comparison over all 68 Davis compounds (2,180 atoms; maximum absolute difference 0 under installed RDKit 2026.03.6), but it returns them without this scaling. This audit did not write scaled tensors or regenerate CSVs.

## Verified matches, discrepancies, and unresolved items

### Verified

- Official author archive rows match local Davis train/test rows exactly in order on canonicalized SMILES, target sequence, and pKd.
- Author full Davis records and local full records match as a multiset, although their full-table order differs.
- Academic Davis source Kd matrix equals local Y numerically; local Y is byte-identical to DeepDTA’s copy.
- Compound IDs and order match the academic PubChem ID list. Local ligand dictionary order and values match DeepDTA’s parsed dictionary.
- Fold JSON files match DeepDTA byte-for-byte and author train/test CSV memberships match the local folds.
- The source code’s scaling formula, scope, point of application, and lack of constant handling are directly verified in the pinned source.

### Confirmed discrepancies

- Local atom features are raw; author preprocessing scales each molecule’s complete matrix.
- Local ligand JSON bytes differ from DeepDTA’s while parsed mapping and insertion order match.
- 38 target-name labels differ textually between academic source gene-name list and local protein dictionary keys. Author/local sequence records agree in the author split CSV comparison, but the naming conventions are not identical.
- The author archive’s full-table row order differs from local row-major order; local train/test split order agrees with the archived author split files.

### Still unresolved

- The full ESI PDF remains unavailable. Table S1/S2 and all supplemental information should be checked against the actual file once obtained.
- The academic source does not give protein sequences, and target-name normalization differs. The author archive confirms the sequences used per split record, but no independent official sequence-file hash was identified.
- The author Google Drive archive is pinned by the retrieved SHA-256 but is a shared file rather than a versioned release. It could change in the future; retain the local hashed archive.
- The exact historical RDKit runtime from the author README was not recreated. Raw features matched the author source under local RDKit 2026.03.6, not necessarily under the historical package version.

## Safe separate preprocessing experiment (no training)

Use a new path such as data/processed/davis_author_scaled_v1/ and never regenerate over data/processed/davis/. Keep existing checkpoints/results tied to their current unscaled provenance.

Required behavior:

- For each unique Davis molecule, start with the pinned reference raw x matrix and calculate scalar minimum and maximum across all entries.
- If maximum is greater than minimum, apply exactly (x - minimum) / (maximum - minimum) to the entire matrix.
- For a constant matrix, stop with an explicit error and record the affected molecule. This is a deliberate safe guard around the reference code’s undefined divide-by-zero behavior; do not silently claim it is exact author behavior for constant input.
- Preserve SMILES, sequence, labels, pair identifiers, row ordering, and fold membership. Record input hashes, source commit, RDKit version, formula, and output hashes in a preprocessing manifest.

Tests to run before any model training:

1. Check raw feature order and values against the pinned author preprocessing for all 68 Davis ligands.
2. Check scaled matrices against the exact scalar-global formula; verify finite values, observed minimum 0, maximum 1, and equality to the reference output for every nonconstant Davis molecule.
3. Add a synthetic constant-matrix case that verifies the selected explicit error policy.
4. Verify output records have identical row counts, identifiers, sequences, labels, ordering, and train/test assignments to the current data; only atom x values should differ.
5. Verify unknown-symbol behavior, including the Davis phosphorus atom, remains aligned with the author feature vector.
6. Ensure the new output root refuses accidental reuse and existing processed-data hashes remain unchanged.

Do not launch training from this preprocessing experiment. First review the scaled/unscaled tensor comparison and decide whether to adopt the author feature scaling for a future, separately labelled model run.

## Safest next action

The major Davis data blocker is resolved: the author archive’s train/test records match the local folds. The ESI remains the document blocker. Obtain a valid copy of the official ESI from RSC or the corresponding author, record its SHA-256, and verify S1/S2 before changing preprocessing. Then review a new versioned scaled preprocessing output and its tests. Do not overwrite the existing processed files or start training until that review is complete.

## Supplement and author split-file update (2026-10-23)

### ESI access and table evidence

The [PMC article record](https://pmc.ncbi.nlm.nih.gov/articles/PMC8768884/) identifies `SC-013-D1SC05180F-s001.pdf` and describes the supplement's relevant contents. The official [RSC ESI URL](https://www.rsc.org/suppdata/d1/sc/d1sc05180f/d1sc05180f1.pdf) was checked. Attempts on 2026-10-23 did not produce a valid local PDF: PMC returned a non-PDF challenge response and Europe PMC did not complete. The target `references/mgraphdta_official/d1sc05180f1.pdf` is absent. PMC lists 864.8 KB; this is not a locally measured file size, and no local hash exists.

An official RSC search-index rendering of the ESI exposes part of Table S3: selected blocks=3 (range 1–5), graph-convolution layers per block=8 (2–10), intermediate channels=64 (32, 64, 96, 128, 160), and MCNN branches=3 (2–5). This is **PARTIALLY VERIFIED** against the indexed representation only. Tables S1–S2 and the full hyperparameter table cannot be independently checked until the PDF is obtained. The local raw 22-feature match to author code and per-molecule scaling equivalence remain evidence about code, not proof of paper ESI parity.

### Davis author CSV evidence

The author README links the Google Drive archive (`https://drive.google.com/file/d/1K21HJI72fmhryjXka_ijCrSgrCdcq2Or/view?usp=sharing`). The pinned regression source does not contain Davis `data_train.csv`/`data_test.csv`; same-named classification files are unrelated. The already-present archive `references/mgraphdta_official/MGraphDTA_Data.zip` is 118,403,071 bytes, SHA-256 `46b58cf905e9f803eee3e0df72ec85faf35234bda53054ed165301200565fd3c`. Embedded members:

| Archive member | Bytes | SHA-256 |
|---|---:|---|
| `MGraphDTA_Data/davis/raw/data_train.csv` | 21,383,855 | `df64c237a9f046f55c4ae5962e1ca268d381b7fc84d21a59c346310930c4068c` |
| `MGraphDTA_Data/davis/raw/data_test.csv` | 4,293,933 | `62f8a57037a8246d2585465065e4b964f5171baafb2c7944412802f0b16a8603` |
| `MGraphDTA_Data/davis/raw/data.csv` | 25,816,618 | `133f0682b7b10e7ccf5fdbc9f85d498553e3f5cdb93b97e7055508270883ecc3` |

Comparison against the local processed Davis CSVs verified, separately for train and test, row-by-row ordering, canonicalized SMILES, target sequence, and exact affinity labels (25,046/5,010 rows; max label delta 0). Literal SMILES strings differ in some rows because valid alternate serializations were used. Thus **row-level semantic correspondence is VERIFIED** to this shared archive; **byte-identical provenance to an immutable author release is UNRESOLVED**. No datasets were extracted to or written over existing processed paths.
