# Validation record

Validated 2026-10-30.

- `bash build.sh` completed with pdfLaTeX and BibTeX.
- `main.pdf` contains 7 A4 pages with a two-column academic layout.
- Final LaTeX log contains no overfull boxes, undefined references/citations, missing-character messages, or package warnings. Minor underfull line-spacing diagnostics remain.
- Extracted PDF text was checked for unresolved `??` markers; none were found. Performance cells retain explicit `Pending` labels.
- All seven pages were rendered with `pdftoppm` and visually reviewed, including the ground-truth sample montage, pipeline diagram, equations, tables, prediction placeholder, and bibliography.
- `python3 collect_evidence.py` ran successfully using the local dataset files and actual annotation-builder functions, without importing PyTorch or executing training. Structural outputs are saved in `evidence/snapshot.json`.
- The historical October 7 pixel/duplicate audit was read and archived, not rerun. No model training, inference, quantitative model evaluation, or hardware benchmark was performed.
- Work was confined to `reports/`; existing training code, vendored SAM 3 state, and dataset files were not changed.
