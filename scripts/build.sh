#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../paper"
temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT
mkdir -p ../output/pdf
for edition in named review; do
    input='\input{main.tex}'
    if [ "$edition" = named ]; then
        input='\def\NamedWorkingPaper{1}\input{main.tex}'
    fi
    job="hybrid_dai_review"
    if [ "$edition" = named ]; then job="hybrid_memory_working_paper"; fi
    for pass in 1 2; do
        "${PDFLATEX:-pdflatex}" -halt-on-error -interaction=batchmode \
            -output-directory "$temporary_dir" -jobname "$job" "$input" >/dev/null
    done
    if rg -q 'Overfull \\hbox|undefined references|Citation .+ undefined' "$temporary_dir/$job.log"; then
        cat "$temporary_dir/$job.log" >&2
        exit 1
    fi
    cp "$temporary_dir/$job.pdf" "../output/pdf/$job.pdf"
done
