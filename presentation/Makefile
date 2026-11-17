# Makefile for Tectonic Beamer presentation

TECTONIC ?= tectonic
UV := $(shell which uv 2>/dev/null)
ifneq ($(UV),)
  PYTHON ?= uv run --with python-pptx --with pillow python3
else
  PYTHON ?= python3
endif

PDF = main.pdf
SRC = main.tex
SUBMISSION_PDF = submission.pdf
SUBMISSION_SRC = submission.tex
SUBMISSION_PPTX = submission.pptx
ASSETS = $(wildcard assets/*.png)

.PHONY: all clean help assets submission pptx

# Default build target: compile the presentation PDF
all: $(PDF)

$(PDF): $(SRC) $(ASSETS)
	$(TECTONIC) $(SRC)

submission: $(SUBMISSION_PDF)

$(SUBMISSION_PDF): $(SUBMISSION_SRC) $(ASSETS)
	$(TECTONIC) $(SUBMISSION_SRC)

pptx: $(SUBMISSION_PDF)
	$(PYTHON) ../scripts/pdf_to_pptx.py $(SUBMISSION_PDF) -o $(SUBMISSION_PPTX)

# Optional helper to regenerate illustrative noise sequence images
assets:
	$(PYTHON) generate_noise.py

clean:
	rm -f $(PDF) $(SUBMISSION_PDF) $(SUBMISSION_PPTX)

help:
	@echo "Available targets:"
	@echo "  make (or make all)   Build the presentation PDF ($(PDF)) using Tectonic"
	@echo "  make submission      Build the submission PDF ($(SUBMISSION_PDF))"
	@echo "  make pptx            Build the submission PPTX ($(SUBMISSION_PPTX))"
	@echo "  make assets          Regenerate synthetic noise images via generate_noise.py"
	@echo "  make clean           Remove compiled PDFs and PPTX files"
	@echo "  make help            Show this help message"

