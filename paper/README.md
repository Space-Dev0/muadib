# Preliminary research paper

`main.pdf` is the named project/preprint version. `anonymous.pdf` uses the ICML 2027 review style and suppresses author names and the repository link; it is a **draft**, not a claim of conference readiness or anonymity certification. `main.tex` is the shared source, with `anonymous.tex` selecting the anonymous version. From this directory, build with:

```sh
tectonic main.tex
tectonic anonymous.tex
```

The Tectonic/XeTeX build explicitly uses the system Nimbus Roman OpenType fonts at `/usr/share/fonts/urw-base35/`, including regular, bold and italic faces. This prevents the supplied `times` package from silently falling back to Latin Modern. Install the URW Base 35 fonts or update the font path when building on a different system. The original ICML style files remain unchanged.

The supplied files in `~/Downloads/icml2027` match the checked-in copies byte for byte. Citations use `references.bib` and the supplied `icml2027.bst`. The unmodified `icml2027.sty`, `icml2027.bst`, `algorithm.sty`, `algorithmic.sty` and `fancyhdr.sty` are from the official [ICML 2027 style archive](https://media.icml.cc/Conferences/ICML2027/styles/icml2027.zip), downloaded 17 November 2026; archive SHA-256 `52384a2a4f32e562755e44ced55a58f857f8706425f2f04904190f613531801e`. The ICML author instructions require at most eight main-body pages, anonymous review submissions and other checks. The authors must review current policies, anonymization, research integrity, and final artifacts before any conference submission.

`generate_figures.py` regenerates the plots with Python, Matplotlib, NumPy and Pillow. It reads `review-aggregate.json`, a compact aggregate of the fixed reviewed score documented in `docs/06-mosaic/mosaic-matched-eval-82000.md`, and bundled fixed images. The illustrative joint-exposure and noise figures are constructed examples, not experimental measurements. Finite-image bounds are not confidence intervals.

`REVIEW.md` records a harsh independent review and unresolved research deficiencies. The paper is a preliminary project report, not a completed causal study. The university guide's request to verify citation counts for at least three papers from a consistent scholarly index remains open.
