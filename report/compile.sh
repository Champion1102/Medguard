#!/bin/bash
# Compile MedGuard LaTeX report
# Usage: cd report && bash compile.sh

set -e

echo "Compiling MedGuard report..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "Done! Output: main.pdf"
