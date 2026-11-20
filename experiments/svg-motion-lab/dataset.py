"""Generate original SVG-to-animation examples and a clearly labelled preview.

Each JSONL row has id, split, family, input_svg, instruction, target_svg,
and specification. Coordinates are SVG user units, time is in seconds.
"""

from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
SEED = 20261120
VIEWBOX_SIZE = 256
SPLIT_COUNTS = {"train": 1000, "validation": 100, "test": 100}
COLORS = ("#f97316", "#38bdf8", "#a78bfa", "#34d399", "#fb7185")
# Family boundaries are enforced before parameter sampling.
SHAPES = {
    "train": {
        "rocket": '<path d="M0 -24 Q20 -5 12 18 L-12 18 Q-20 -5 0 -24Z"/><circle cx="0" cy="-2" r="5" fill="white"/>',
        "ball": '<circle r="19"/><path d="M-13 -13 Q15 0 -13 13" fill="none" stroke="white"/>',
        "diamond": '<path d="M0 -23 L18 0 0 23 -18 0Z"/>',
        "flower": '<circle cx="-12" r="12"/><circle cx="12" r="12"/><circle cy="-12" r="12"/><circle cy="12" r="12"/><circle r="7" fill="white"/>',
        "arrow": '<path d="M-22 -7 H4 V-18 L23 0 4 18 V7 H-22Z"/>',
    },
    "validation": {
        "kite": '<path d="M0 -24 L17 -2 0 22 -17 -2Z"/><path d="M0 22 Q12 30 0 39" fill="none" stroke="white"/>',
        "cloud": '<path d="M-20 10 A10 10 0 0 1 -16 -10 A15 15 0 0 1 12 -12 A12 12 0 0 1 20 10Z"/>',
    },
    "test": {
        "satellite": '<rect x="-8" y="-12" width="16" height="24"/><path d="M-25 -9 H-11 V9 H-25Z M11 -9 H25 V9 H11Z"/>',
        "fish": '<path d="M-19 0 Q0 -24 19 0 Q0 24 -19 0 L-28 -12 V12Z"/><circle cx="10" cy="-3" r="3" fill="white"/>',
        "lantern": '<path d="M-14 -18 H14 L19 12 10 21 H-10 L-19 12Z"/><path d="M-6 -18 V-26 H6 V-18" fill="none" stroke="white"/>',
    },
}
MOTION_PAIRS = {
    "train": (
        ("translate",),
        ("rotate",),
        ("scale",),
        ("opacity",),
        ("translate", "rotate"),
    ),
    "validation": (("translate", "scale"),),
    "test": (("translate", "opacity"), ("rotate", "scale")),
}


@dataclass(frozen=True)
class Motion:
    """One linear out-and-back animation with a pause at its peak."""

    kind: str
    peak: float
    duration: int

    def markup(self) -> str:
        """Return native SVG animation markup, with a held peak."""
        timing = (
            f' dur="{self.duration}s" keyTimes="0;0.4;0.6;1" repeatCount="indefinite"'
        )
        if self.kind == "opacity":
            return f'<animate attributeName="opacity" values="1;{self.peak};{self.peak};1"{timing}/>'
        start = "1" if self.kind == "scale" else "0"
        peak = str(self.peak)
        if self.kind == "translate":
            start, peak = "0 0", f"0 {self.peak}"
        return f'<animateTransform attributeName="transform" type="{self.kind}" values="{start};{peak};{peak};{start}"{timing}/>'

    def instruction(self) -> str:
        """Describe the exact numerical contract without exposing code."""
        action = {
            "translate": f"move upward by {abs(self.peak):g} SVG units",
            "rotate": f"rotate clockwise by {self.peak:g} degrees around its own origin",
            "scale": f"scale uniformly to {self.peak:g} times its original size around its own origin",
            "opacity": f"fade to opacity {self.peak:g}",
        }[self.kind]
        return f"{action} during the first 40% of a {self.duration}-second cycle, hold until 60%, then return to the original state by the end; repeat forever"


@dataclass(frozen=True)
class Example:
    """One instruction and its original, licensed SVG reference pair."""

    id: str
    split: str
    family: str
    input_svg: str
    instruction: str
    target_svg: str
    specification: tuple[Motion, ...]


def make_example(split: str, index: int) -> Example:
    """Generate a reproducible row; split seeds and scene families are disjoint."""
    rng = random.Random(f"{SEED}:{split}:{index}")
    families = tuple(SHAPES[split])
    family = families[index % len(families)]
    duration = rng.choice((3, 4, 5, 6))
    peaks = {
        "translate": -rng.randint(25, 65),
        "rotate": rng.choice((45, 90, 135, 180)),
        "scale": rng.choice((1.2, 1.4, 1.6)),
        "opacity": rng.choice((0.15, 0.3, 0.5)),
    }
    composition = MOTION_PAIRS[split][
        (index // len(families)) % len(MOTION_PAIRS[split])
    ]
    motions = tuple(Motion(kind, peaks[kind], duration) for kind in composition)
    body = f'<g id="drawing" fill="{rng.choice(COLORS)}">{SHAPES[split][family]}</g>'
    prefix = f'<svg xmlns="{SVG_NS}" viewBox="0 0 {VIEWBOX_SIZE} {VIEWBOX_SIZE}"><g id="position" transform="translate(128 154)">'
    suffix = "</g></svg>"
    static = body
    animated = body
    # Distinct nested groups compose transforms without overwriting one another.
    for motion in reversed(motions):
        static = f'<g id="motion-{motion.kind}">{static}</g>'
        animated = f'<g id="motion-{motion.kind}">{motion.markup()}{animated}</g>'
    instruction = (
        "Preserve the drawing and its initial placement. Apply these motions simultaneously: "
        + "; and ".join(m.instruction() for m in motions)
        + ". Return only the complete animated SVG, without scripts, CSS, external resources, or Markdown. Use native SVG animations."
    )
    return Example(
        f"{split}-{index:04d}",
        split,
        family,
        prefix + static + suffix,
        instruction,
        prefix + animated + suffix,
        motions,
    )


def write_dataset(output: Path) -> dict[str, int]:
    """Write unique rows and reference-only HTML; never invent model predictions."""
    output.mkdir(parents=True, exist_ok=True)
    previews = []
    counts = {}
    for split, count in SPLIT_COUNTS.items():
        rows = []
        seen = set()
        index = 0
        while len(rows) < count:
            row = make_example(split, index)
            index += 1
            key = (row.input_svg, row.instruction)
            if key in seen:
                continue
            seen.add(key)
            ET.fromstring(row.target_svg)
            rows.append(asdict(row))
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        counts[split] = len(rows)
        for row in rows[:6]:
            previews.append(
                f"<article><h2>{escape(row['id'])} · {escape(row['family'])}</h2>{row['target_svg']}<p>{escape(row['instruction'])}</p></article>"
            )
    html = (
        '<!doctype html><meta charset="utf-8"><title>SVG Motion Lab — reference targets</title><style>body{background:#101522;color:#edf2ff;font:16px system-ui;margin:40px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}article{background:#1c2538;padding:24px;border-radius:16px}svg{width:100%;height:240px}p{line-height:1.5}</style><h1>SVG Motion Lab</h1><p>Procedural reference targets. These are not outputs from a trained model.</p><main>'
        + "".join(previews)
        + "</main>"
    )
    (output / "reference-preview.html").write_text(html)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "counts": counts,
                "license": "Apache-2.0",
                "families": {s: list(v) for s, v in SHAPES.items()},
                "motion_compositions": MOTION_PAIRS,
            },
            indent=2,
        )
    )
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    print(json.dumps(write_dataset(parser.parse_args().output)))
