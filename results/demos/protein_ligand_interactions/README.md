# Experimental protein–ligand interaction diagram demonstrations

**These five examples are a separate educational demonstration set. They are not Davis data, are not MGraphDTA results, and did not come from any Grad-AAM sample.**

## What is a protein–ligand interaction diagram?

It is a simplified picture of a small molecule (the ligand) sitting in a protein binding pocket. Nearby amino acids are labeled, and lines show selected contacts such as hydrogen bonds or hydrophobic packing. A diagram makes the chemistry easier to discuss, but it is a summary of measured structure coordinates—not a picture of every atom or a prediction of binding strength.

The diagrams and CSVs here were calculated from experimental X-ray coordinates downloaded from RCSB PDB. The methods and geometric criteria are documented separately in each example’s `structure_verification.md`.

## Examples

### Example 1: ABL1 + imatinib

Shows a targeted kinase inhibitor in the ABL1 kinase pocket; useful for explaining how a drug’s heteroatoms and aromatic groups contact a kinase.
RCSB PDB: https://www.rcsb.org/structure/2HYY
Files: `example_1/interaction_diagram.png`, `example_1/interactions.csv`, `example_1/structure_verification.md`, and the downloaded coordinate mmCIF.

### Example 2: EGFR L858R + gefitinib

Shows gefitinib in a mutant EGFR ATP-binding site and illustrates ligand–kinase contacts in a disease-associated receptor variant.
RCSB PDB: https://www.rcsb.org/structure/2ITZ
Files: `example_2/interaction_diagram.png`, `example_2/interactions.csv`, `example_2/structure_verification.md`, and the downloaded coordinate mmCIF.

### Example 3: EGFR + erlotinib

A second EGFR example with a different inhibitor, useful for comparing ligand scaffolds and their pocket contacts.
RCSB PDB: https://www.rcsb.org/structure/1M17
Files: `example_3/interaction_diagram.png`, `example_3/interactions.csv`, `example_3/structure_verification.md`, and the downloaded coordinate mmCIF.

### Example 4: HIV-1 protease + L-735,524 (indinavir)

Demonstrates an antiviral inhibitor occupying the protease dimer active site. The pocket includes residues from the protease assembly.
RCSB PDB: https://www.rcsb.org/structure/1HSG
Files: `example_4/interaction_diagram.png`, `example_4/interactions.csv`, `example_4/structure_verification.md`, and the downloaded coordinate mmCIF.

### Example 5: Streptavidin + biotin

A classic high-affinity protein–small molecule complex that provides a clear example of shape complementarity and multiple polar/nonpolar contacts.
RCSB PDB: https://www.rcsb.org/structure/1STP
Files: `example_5/interaction_diagram.png`, `example_5/interactions.csv`, `example_5/structure_verification.md`, and the downloaded coordinate mmCIF.
