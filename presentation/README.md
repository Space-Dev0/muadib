# Project presentations

- **Long deck:** [main.pdf](main.pdf), a 66-page Beamer presentation, built from [main.tex](main.tex). The older [speaking guide](speaking-guide.md) and [detailed notes](detailed-speaker-notes.md) describe an earlier 63-frame version and may not match every current page.
- **Submission deck:** [submission-final.pptx](powerpoint/submission-final.pptx), an editable 12-slide PowerPoint using Computer Modern Unicode (CMU Serif). Install the fonts from [computer-modern-fonts.zip](powerpoint/computer-modern-fonts.zip) before opening it. Some labels in chart images retain their original font. Its [presenter notes](powerpoint/presentation-notes-guide.md) are also embedded in the deck. Slide 5 gives the minimum-KL baseline and the separate exploratory max–min problem.

- **Detailed slide guides:** [Slide 4: exposure and methodology](powerpoint/explanations/slide-04-methodology.md), [Slide 8: experiment and training](powerpoint/explanations/slide-08-experiment-training.md), and [Slide 12: status and next steps](powerpoint/explanations/slide-12-status-next-steps.md). Each guide ends with a short audience script.

The supporting `assets/` directory contains figures used by the decks, and `powerpoint/equations/` contains the LaTeX source for the equation graphics. The long deck can be rebuilt with `make` from this directory, using Tectonic.

## Google Slides sync

GitHub Actions imports `powerpoint/submission-final.pptx` directly into the existing Google Slides file. Push changes to that file on `main`, or run **Sync PowerPoint to Google Slides** manually from the default branch. The workflow verifies 12 slides, key tables and images, and saves rendered previews.

`GDRIVE_PRESENTATION_ID` identifies the target; `GCP_SA_KEY` must contain a service-account key with access to it and Drive API must be enabled. Google converts supported PPTX objects to editable Slides objects, although equation graphics remain images and formatting may change. Each sync replaces the target's contents, so back up any independent Google Slides edits first.
