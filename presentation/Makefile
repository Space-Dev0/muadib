# Makefile for Tectonic Beamer presentation

TECTONIC ?= tectonic
PYTHON ?= python3

PDF = main.pdf
SRC = main.tex
ASSETS = $(wildcard assets/*.png)

.PHONY: all clean help assets

# Default build target: compile the presentation PDF
all: $(PDF)

$(PDF): $(SRC) $(ASSETS)
	$(TECTONIC) $(SRC)

# Optional helper to regenerate illustrative noise sequence images
assets:
	$(PYTHON) generate_noise.py

clean:
	rm -f $(PDF)

help:
	@echo "Available targets:"
	@echo "  make (or make all)   Build the presentation PDF ($(PDF)) using Tectonic"
	@echo "  make assets          Regenerate synthetic noise images via generate_noise.py"
	@echo "  make clean           Remove the compiled PDF ($(PDF))"
	@echo "  make help            Show this help message"
