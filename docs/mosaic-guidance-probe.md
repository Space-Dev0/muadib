# Classifier-free guidance feasibility probe

The color-conditioned PilotUNet was trained with 10% conditioning dropout,
which permits an exploratory classifier-free-guidance comparison. The probe
combines conditional and unconditional velocity predictions before converting
to noise and running the existing DDPM sampler. At guidance 1, it reduces to
the original conditional sampler; the test suite checks that equivalence
and the velocity combination. [Ho and Salimans](https://arxiv.org/abs/2207.12598)
describe classifier-free guidance as a quality/diversity tradeoff, not a
universal accuracy improvement.

On the frozen 20,000-update checkpoint, two paired seeds and red/blue requests
were sampled at guidance 1, 2 and 4 (12 images). All image hashes matched their
report, and all four scale-1 outputs matched the earlier samples byte-for-byte.
Informal inspection found that for blue seed 81380, the scale-1 result had a
yellow-looking region while scale 2 showed a blue region. At scale 4, the
other objects and their colors also changed; for another seed the blue regions
merged. This observation is not human validation, a quality estimate, a
claim that guidance fixes conditioning, or evidence that the research gate
passes. Higher guidance changes the sampling distribution and may reduce
diversity or introduce artifacts. Results are in ignored
`outputs/mosaic-guidance-81384/`.

The separate 30,000-update checkpoint was compared at guidance 1 and 2
using the same seeds (eight images). All image hashes and the checkpoint hash
matched the report. For blue seed 81380, scale 1 looked yellow while scale 2
showed a blue region; for seed 81381 both appeared blue. The experimental
checker abstained on all eight images for unrecognized foreground colors.
These are informal observations, not independent human labels. The ignored
outputs are `outputs/mosaic-guidance-30k-81385/`. A small adaptively chosen sample cannot establish the
appropriate guidance value; any research protocol must freeze it in advance
and evaluate fresh images with a validated checker or blinded human review.
