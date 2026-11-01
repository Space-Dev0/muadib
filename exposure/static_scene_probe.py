"""Single-render overfit diagnostic; not an online training experiment.

The fixed renderer override is process-wide while training, so this command
must run in its own process. Its checkpoint is sampleable but not resumable as
an ordinary online run. A single-scene outcome is not population evidence.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from exposure import pilot_train
from exposure.measure import unique_object
from exposure.render import render_conditioned


def overfit(definitions: dict, *, rule: str, scene_seed: int, train_seed: int,
            steps: int, batch_size: int, width: int, timesteps: int, device: str,
            deterministic: bool, max_attempts: int) -> tuple:
    """Train on one fixed render and return the target, checkpoint and report."""
    if type(scene_seed) is not int or scene_seed < 0:
        raise ValueError("scene seed must be nonnegative")
    rules = pilot_train.load_rules(definitions)
    if rule not in rules:
        raise ValueError("Unknown rule")
    target, attempts = render_conditioned(scene_seed, rules[rule], max_attempts)
    original = pilot_train.render_conditioned
    try:
        pilot_train.render_conditioned = lambda seed, predicates, max_attempts: (target, 1)
        checkpoint, report = pilot_train.train(
            {rule: definitions[rule]}, None, steps=steps, batch_size=batch_size,
            seed=train_seed, device=device, width=width, timesteps=timesteps,
            lr=1e-4, dropout=0.0, ema_decay=0.999, max_attempts=max_attempts,
            deterministic=deterministic)
    finally:
        pilot_train.render_conditioned = original
    checkpoint["schema_version"] = 0
    checkpoint["diagnostic"] = {"kind": "single_fixed_render", "scene_seed": scene_seed,
                                "target_attempts": attempts, "resumable": False}
    report["note"] = ("Single fixed render overfit diagnostic; not online training, "
                      "population evidence, or a Stage 0 result.")
    return target, checkpoint, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit one fixed render as a diagnostic")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--scene-seed", type=int, default=77)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=10000)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("Output directory already exists; select a new path")
        if args.sample_seed < 0:
            raise ValueError("sample seed must be nonnegative")
        definitions = json.loads(args.rules.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        target, checkpoint, report = overfit(
            definitions, rule=args.rule, scene_seed=args.scene_seed, train_seed=args.train_seed,
            steps=args.steps, batch_size=args.batch_size, width=args.width, timesteps=args.timesteps,
            device=args.device, deterministic=args.deterministic, max_attempts=args.max_attempts)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        target.image.save(args.output_dir / "target.png")
        pilot_train.save_checkpoint(args.output_dir / "checkpoint.pt", checkpoint)
        generated = pilot_train.sample(checkpoint, args.rule, seed=args.sample_seed, device=args.device, guidance=1.0)
        generated.save(args.output_dir / "sample.png")
        target_pixels = np.asarray(target.image, dtype=np.float32) / 255.0
        sample_pixels = np.asarray(generated, dtype=np.float32) / 255.0
        result = {"schema_version": 1, "diagnostic": checkpoint["diagnostic"], "rule": args.rule,
                  "sample_seed": args.sample_seed, "target_objects": [asdict(obj) for obj in target.objects],
                  "sample_mse_0_1": float(np.mean((sample_pixels - target_pixels) ** 2)),
                  "training": report,
                  "note": "One fixed scene and one sample; not a generalization or Stage 0 result."}
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
