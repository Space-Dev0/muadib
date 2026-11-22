"""Generate a 64×64 image from a trained color-conditioned diffusion pilot.

Example: uv run --extra train python -m exposure.image_demo red
Only the colors red and blue are supported by this checkpoint; this is not a
free-form text-to-image model. Checkpoints must come from a trusted source.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CHECKPOINT = Path("outputs/model-eval-760k/warmstart-760000.pt")
COLORS = ("red", "blue")


def nonnegative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer") from exc
    if not 0 <= seed < 2**63:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2^63 - 1")
    return seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an image from the trained diffusion pilot (red or blue condition)."
    )
    parser.add_argument("color", nargs="?", type=str.lower, choices=COLORS,
                        help="target color; prompted for if omitted")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help=f"trusted checkpoint path (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--format", choices=("warmstart", "probe"), default="warmstart",
                        help="checkpoint format (default: warmstart)")
    parser.add_argument("--seed", type=nonnegative_seed, default=0,
                        help="non-negative sampling seed (default: 0)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto uses CUDA when available, otherwise CPU")
    parser.add_argument("--output", type=Path,
                        help="PNG path (default: outputs/demo-<color>-<seed>.png)")
    args = parser.parse_args(argv)

    if args.color is None:
        try:
            color = input("Choose a target color (red or blue): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nNo color selected.")
            return 1
        if color not in COLORS:
            parser.error("choose either red or blue; this model does not support free-form prompts")
    else:
        color = args.color

    output = args.output or Path(f"outputs/demo-{color}-{args.seed}.png")
    if output.suffix.lower() != ".png":
        parser.error("output must be a .png file")
    if output.exists():
        parser.error(f"output already exists: {output}; choose a different --output or --seed")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}; supply a trusted file with --checkpoint")

    try:
        from exposure import mosaic_pilotunet_probe, mosaic_pilotunet_warmstart
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            parser.exit(1, "PyTorch is not installed. Run: uv sync --extra train\n")
        raise

    module = mosaic_pilotunet_warmstart if args.format == "warmstart" else mosaic_pilotunet_probe
    loader = module.load_warmstart_checkpoint if args.format == "warmstart" else module.load_checkpoint
    try:
        checkpoint = loader(args.checkpoint)
        print(f"Generating a 64×64 image for {color!r} (seed {args.seed}); this can take time on CPU...",
              flush=True)
        image = module.sample(checkpoint, color, seed=args.seed, device=args.device)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            image.save(file, format="PNG")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Could not generate image: {exc}\n")

    print(f"Saved: {output}")
    print("This model is conditioned on color only; image contents are not guaranteed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
