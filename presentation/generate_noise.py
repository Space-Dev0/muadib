"""Create fixed illustrative noising images for the beginner presentation."""
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent / "assets"
clean = np.asarray(Image.open(root / "synthetic-example.png").convert("RGB"), dtype=float)
rng = np.random.default_rng(801)
noise = rng.normal(127.5, 60, clean.shape)
for index, fraction in enumerate((0.0, 0.35, 0.70, 1.0)):
    pixels = np.clip(np.rint((1 - fraction) * clean + fraction * noise), 0, 255).astype("uint8")
    Image.fromarray(pixels).save(root / f"noise-{index}.png")
