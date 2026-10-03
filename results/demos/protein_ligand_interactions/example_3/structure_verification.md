# Structure verification — Example 3

**Demonstration only:** this example is independent of the Davis dataset, MGraphDTA training, and all Grad-AAM samples.

- **PDB ID:** 1M17
- **Protein:** EGFR tyrosine kinase domain
- **Organism:** Homo sapiens
- **Ligand:** Erlotinib (Tarceva)
- **Experimental method:** X-ray diffraction
- **Resolution:** 2.60 Å
- **Ligand component ID:** AQ4
- **Protein chain(s) contributing interactions:** A
- **Ligand instance:** author chain A, residue 999
- **Coordinates:** `1M17.cif` (downloaded from RCSB PDB)

## Why this structure is suitable

A 2.60 Å experimental human EGFR kinase complex with the deposited erlotinib component AQ4. The ligand component is present in the downloaded coordinate file and the interactions below are calculated from its deposited atom coordinates and surrounding protein atoms. RCSB entry: https://www.rcsb.org/structure/1M17

## Interaction detection criteria

All distances are Euclidean distances between non-hydrogen atoms in the deposited coordinates. The atom names in the CSV are the deposited atom names.

- **Hydrogen bonds:** donor/acceptor atom roles assigned from the CCD-derived ligand graph and standard protein residue chemistry; donor–acceptor heavy-atom distance ≤3.5 Å; where a bonded heavy-atom antecedent exists, its donor–donor–acceptor surrogate angle must be ≥110°. The deposited structures generally lack hydrogens, so this is a heavy-atom geometric screen, not protonation-aware energy analysis.
- **Hydrophobic contacts:** carbon–carbon distance ≤4.2 Å between ligand nonpolar carbon atoms and side-chain carbons in ALA/VAL/ILE/LEU/MET/PHE/TRP/PRO; one minimum-distance representative per residue.
- **Pi–pi:** ligand and protein aromatic ring centroids ≤5.5 Å; ring-plane angle ≤30° (parallel) or 60–90° (T-shaped).
- **Pi–cation:** Arg/Lys cationic side-chain atom to ligand aromatic-ring centroid ≤6.0 Å.
- **Salt bridges:** only formal charged ligand atoms paired with oppositely charged Asp/Glu or Arg/Lys groups at ≤4.0 Å; no salt bridge is inferred from uncharged atoms.
- **Halogen bonds:** ligand Cl/Br/I to a protein acceptor ≤3.8 Å and C–X···acceptor angle ≥150°. Fluorine is not classified as a halogen-bond donor here.

The criteria are conservative geometric indicators, not a free-energy or binding-strength calculation. The CSV contains all detected interactions. The diagram prioritizes at most 14 distinct interacting residues for readability; it shows only interactions present in the CSV.
