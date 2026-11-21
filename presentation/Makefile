# Build the long Beamer deck. The submission source is the editable PPTX.
TECTONIC ?= tectonic
PDF = main.pdf
SRC = main.tex
ASSETS = $(wildcard assets/*.png)

.PHONY: all clean help assets
all: $(PDF)

$(PDF): $(SRC) $(ASSETS)
	$(TECTONIC) $(SRC)

assets:
	uv run python generate_noise.py

clean:
	rm -f $(PDF)

help:
	@echo 'make        Build the long presentation (main.pdf)'
	@echo 'make assets Regenerate illustrative noise assets'
	@echo 'make clean  Remove the long compiled PDF'
	@echo 'The 12-slide submission is powerpoint/submission-final.pptx'
