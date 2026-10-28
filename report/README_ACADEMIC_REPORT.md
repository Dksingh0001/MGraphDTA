# MGraphDTA academic report

The current report entry point is `main.tex`. The report uses `sections/`, `academic_references.bib` and the assets in `figures/`. Existing alternate LaTeX archives/drafts in this repository have been preserved; compile this root entry point for the current report.

Figures can be regenerated from the saved CSV/JSON artifacts by running from PowerShell:

```powershell
./generate_plots.ps1
```

Build with XeLaTeX and BibTeX in a configured TeX environment:

```text
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

The workspace MiKTeX installation could not initialize its per-user profile, so compilation and page count remain unverified. See `REPORT_VALIDATION.md` before presenting the report.
