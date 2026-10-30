#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
for tool in pdflatex bibtex; do
  command -v "$tool" >/dev/null || { echo "Required command missing: $tool" >&2; exit 1; }
done
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build-pass1.log
bibtex main > build-bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build-pass2.log
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex > build-pass3.log
echo "Built $(pwd)/main.pdf"
