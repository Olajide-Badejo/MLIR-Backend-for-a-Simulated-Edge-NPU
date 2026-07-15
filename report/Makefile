# Build the main report. Regenerates the results tables from the recorded
# experiments, then compiles with tectonic (which runs the full latex and bibtex
# loop in one invocation).
PYTHON ?= python3

.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib $(wildcard sections/*.tex)
	$(PYTHON) scripts/results_to_tex.py
	tectonic main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
