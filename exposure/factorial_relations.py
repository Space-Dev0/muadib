"""Build an exploratory fully crossed color–relation–shape template subset.

This is a 24-rule candidate subset, not the final benchmark or a gate result.
"""

import argparse
import json
from pathlib import Path

from exposure.render import COLORS, SHAPES


def build() -> dict[str, list[dict[str, str]]]:
    """Bind color and shape to distinct relation endpoints, crossing all levels."""
    return {
        f"{color}_{relation}_{shape}": [
            {"type": "attribute", "variable": "a", "field": "color", "value": color},
            {"type": "attribute", "variable": "b", "field": "shape", "value": shape},
            {"type": "relation", "subject": "a", "kind": relation, "object": "b"},
        ]
        for color in COLORS for shape in SHAPES for relation in ("left_of", "above")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(build(), file, indent=2)
            file.write("\n")
    except OSError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
