# MGraphDTA LaTeX report

This directory contains the report source, BibTeX references, and copies of verified project-generated figures. The report is designed for a standard LaTeX installation with `pdflatex` and BibTeX.

## Compile

From this directory, run:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The report uses TikZ/PGFPlots and the packages listed in `main.tex`. A LaTeX distribution with those packages is required. The repository's current shell did not expose `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`, so a compiled PDF could not be generated or checked in this environment.

## Evidence and scope notes

The report cites the original MGraphDTA paper and uses the local source tree, project README/plan, generated outputs, raw Davis files, and structural demonstration CSVs. The requested uploaded `Explain MGraphDTA Project.txt` was not found in the project or accessible attachment directories during this audit. The supplied aggregate RCSB summary is recorded with an explicit provenance caveat because its source CSV/JSON is unavailable for independent recomputation; the saved 9AAA inspection output is also unavailable. The earlier 31-test milestone, current 36-test suite, initial default-run setup errors, and successful workspace-temp rerun are reported in the document.



