# PlantSeg paper report

English, two-column A4 LaTeX paper based on the current PlantSeg files and SAM 3 training implementation. This is a research-paper draft with model results intentionally left pending. Author and affiliation are blank because none were supplied.

**Archive status (2026-11-01):** this draft and its evidence describe the pre-refactor pipeline inspected on October 30. They are retained as research artifacts, not as instructions for the current Transformers trainer. Use [the project README](../README.md) for training/evaluation. The obsolete `collect_evidence.py` depended on removed functions, old dataset paths and the upstream SAM3 checkout; it has been removed. The existing snapshots remain historical records. BibTeX output and PDF text extracts are regenerated when needed instead of being tracked as source.

- `main.pdf`: compiled paper.
- `main.tex`: entry point and formatting.
- `sections/01_introduction.tex`: introduction and related work.
- `sections/02_dataset.tex`: local dataset releases and annotation audit.
- `sections/03_method.tex`: implemented prompts, transforms, loss and optimization.
- `sections/04_protocol.tex`: current validation semantics and proposed final evaluation.
- `sections/05_results.tex`: editable performance tables, runtime records and prediction placeholder.
- `sections/06_discussion.tex`: limitations and conclusion.
- `references.bib`: research references and local evidence citations.
- `evidence/snapshot.json`: refreshed structural counts, source hashes and script defaults.
- `evidence/dataset-audit.md`, `evidence/flagged-files.csv`: archived October 7 audit evidence. Relative dataset links in the archived Markdown retain their original project context; the original is `../docs/plantseg-v6-v7-statistics.md`.
- `SOURCE_NOTES.md`: links from paper claims to implementation and primary literature.

## Build

Requires a TeX installation containing pdfLaTeX, BibTeX, `natbib`, `balance`, `geometry`, `mathptmx`, `microtype`, `tikz`, `caption`, `tabularx`, `booktabs`, and `hyperref`.

```bash
cd reports
bash build.sh
```

The report builds from the files in this directory; neither PyTorch nor dataset access is required. The layout is a generic academic paper, with no claim of compliance with a specific journal or conference template.

## Structural evidence

`evidence/snapshot.json` is the archived October 30 snapshot. It is not refreshed by the current training pipeline. Current runs record model/dataset identity, resolution and configuration in `run-config.json`, and metrics in the evaluation JSON files documented in the project README.

## Add model results later

1. Record the actual model checkpoint, training configuration, software/hardware environment and data manifest. The documented values currently describe script defaults.
2. Decide and freeze the annotation-quality and duplicate policy described in Section 5. The report does not repair data or change the training code.
3. Add a final evaluator for native-resolution PNG references. The existing helper only reports `val_iou`, positive-query response (`val_recall`) and sampled-negative query false alarms (`val_false_positive`), using polygon-derived masks at output resolution. It does not compute Dice, disease macro IoU, pixel recall or a final test result.
4. Populate tables and replace the prediction placeholder in `sections/05_results.tex` with real plots/images under `figures/`. State all evaluation denominators. Raw v7 has 1,561 test masks; the current builder keeps 1,560 polygon-supported test images.
5. Update abstract, discussion and conclusion after interpreting the actual results. Remove or rewrite pending language only where supported by evidence.
6. Add author and affiliation in `main.tex`, rebuild, and inspect the PDF.

No training, inference, or model benchmark was run while preparing this report.
