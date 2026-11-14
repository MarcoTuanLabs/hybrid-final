#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/pdf
cd paper
for pass in 1 2; do
  "${PDFLATEX:-pdflatex}" -interaction=nonstopmode -halt-on-error \
    -output-directory ../output/pdf -jobname hybrid_memory_working_paper main.tex
done
