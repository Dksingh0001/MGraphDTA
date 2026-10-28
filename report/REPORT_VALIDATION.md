# Academic report validation

**Report source:** `main.tex` and `sections/` in this directory.  
**Build date:** 2026-10-27.  
**Target:** 15 pages including title and references.

## Source and evidence checks

- The Davis, architecture, training, metric, scaling, multi-seed, Protein-GNN and Grad-AAM descriptions were checked against `report/project_audit/PROJECT_AUDIT.md` and the cited project files.
- The original MGraphDTA values are cited as paper results only. The RSC article record reports Davis MSE 0.265 (0.002), CI 0.822 (0.001), and r_m^2 0.701 (0.001); project results are separately labeled.
- `figures/training_histories.png` is generated from the three named epoch-history CSV files by `generate_plots.ps1`; each run’s observed number of rows is used without filling missing epochs.
- `figures/metric_comparison.png` reads the six displayed run metric JSON files. `figures/scaling_comparison.png` reads the raw and scaled seed-42 JSON files. `figures/multiseed_variability.png` reads seed 44–46 JSON files and computes sample SD (ddof=1).
- The molecular images are existing Davis test Grad-AAM renderings. The interaction diagrams are existing independent PDB demonstration outputs; captions distinguish them from model predictions.
- No tests, training, dependency installation, or source-code changes were performed for this report.

## Compilation and page count

**Compilation: blocked. Page count: not determined.** The installed MiKTeX `xelatex` executable failed before document compilation while trying to create its first-run per-user configuration at `C:\Users\Deepak\AppData\Roaming\MiKTeX\2.9`; Windows returned “Access is denied.” A retry with `--disable-installer` and process-local `APPDATA`/`LOCALAPPDATA` redirected to the report build directory reached the same fixed MiKTeX setup path. No package installation or external-directory write was attempted. Consequently citation resolution, layout, PDF rendering and the 15-page target have not been verified. Compile in Overleaf or a configured TeX installation, then record the resulting PDF page count here.

The source uses 10pt type, A4 paper and 18 mm margins, and was structured for a compact report, but this is not a substitute for a compiled-page check. The Overleaf ZIP includes only report source and referenced figure assets; it does not include a PDF because compilation did not complete.
