# Build the main report. Regenerates the results tables from the recorded
# experiments, then compiles with tectonic (which runs the full latex and bibtex
# loop in one invocation).
PYTHON ?= python3

.PHONY: all clean

all: main.pdf

# main.tex does \input{generated/macros}, and those macros are produced from the
# recorded results, so the results are a real dependency of this PDF. They were
# missing from this rule, which is why regenerating the numbers never rebuilt the
# report: make saw main.tex unchanged, said there was nothing to do, and the
# committed PDF stayed byte identical to a build from a month earlier while the
# results under it moved. That is the failure this dependency prevents.
RESULTS := $(wildcard ../experiments/results/*.json)

main.pdf: main.tex references.bib $(wildcard sections/*.tex) $(RESULTS)
	$(PYTHON) scripts/results_to_tex.py
	$(PYTHON) ../experiments/plot_results.py
	tectonic main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
