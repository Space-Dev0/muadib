# Terminal image-generation demo

This demo generates one 64 × 64 image using an existing trained diffusion pilot
checkpoint and its EMA weights. The model accepts **red** or **blue** as a color
condition. It is not a free-form text-to-image model and does not guarantee that
the resulting image contains the requested object.

## Run it here

From the repository root:

```sh
uv run --no-sync python -m exposure.image_demo
```

Type `red` or `blue` when prompted. The image is saved to
`outputs/demo-<color>-0.png`. You can also supply the color directly:

```sh
uv run --no-sync python -m exposure.image_demo blue --seed 42
```

The image is then saved to `outputs/demo-blue-42.png`. Each seed controls the
sampling randomness; changing it generates a different sample from the same
trained model, not a new trained model. The demo refuses to overwrite an
existing output. Choose a different seed or `--output` path to run it again.

The default checkpoint on this machine is
`outputs/model-eval-760k/warmstart-760000.pt`. Checkpoints and generated images
are excluded from Git, so another computer or a fresh clone needs a copy of a
compatible **trusted** checkpoint. Specify it explicitly when needed:

```sh
uv run --no-sync python -m exposure.image_demo red \
  --checkpoint /path/to/warmstart.pt --output outputs/my-image.png
```

The default is a warm-start checkpoint. If using a compatible probe-format
checkpoint, add `--format probe`. Only load checkpoints from sources you trust.

## PyTorch and hardware

The demo needs the project's core dependencies and PyTorch. On this machine,
CPU PyTorch is installed in the current project virtual environment. The
`--no-sync` flag preserves that environment instead of replacing it with the
CUDA package configured for the optional training extra.

On another machine, set up the project with `uv sync --extra train` to install
the configured training dependencies. This project configures a CUDA PyTorch
wheel on Linux and Windows; an appropriate NVIDIA driver is needed to use CUDA.
A CPU-only PyTorch environment can also run the demo, although image generation
may take longer. `--device auto` selects CUDA when available and otherwise CPU;
use `--device cpu` or `--device cuda` to choose explicitly.

Example with a CUDA-capable environment:

```sh
uv run --no-sync python -m exposure.image_demo red --device cuda --seed 12
```

If the checkpoint is absent, the demo reports its path and exits. If PyTorch is
absent, install the training dependencies first. The image is a single pilot
sample, not an accuracy measurement or proof that the project has passed its
research gate.
