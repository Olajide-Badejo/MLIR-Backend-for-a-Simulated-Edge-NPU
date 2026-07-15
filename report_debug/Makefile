# Build the debug report with tectonic.
.PHONY: all clean

all: debug_report.pdf

debug_report.pdf: debug_report.tex $(wildcard sections/*.tex)
	tectonic debug_report.tex

clean:
	rm -f debug_report.pdf debug_report.aux debug_report.log debug_report.out
