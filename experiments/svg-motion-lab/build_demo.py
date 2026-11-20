"""Build a curated replay gallery from real, paired model predictions."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from evaluate import parse_svg

ROOT = Path(__file__).resolve().parent
ARMS = ("base", "finetuned")


def render_source(source: str) -> str | None:
    """Normalize safe SVG for HTML; rejected predictions are never embedded."""
    try:
        return ET.tostring(parse_svg(source), encoding="unicode")
    except ValueError:
        return None


def build(data: Path, run: Path, split: str, output: Path) -> None:
    """Join fixed gallery choices to untouched predictions, preserving provenance."""
    rows = {
        row["id"]: row
        for row in map(json.loads, (data / f"{split}.jsonl").read_text().splitlines())
    }
    arms = {
        arm: {
            row["id"]: row
            for row in map(
                json.loads, (run / f"{arm}-{split}.jsonl").read_text().splitlines()
            )
        }
        for arm in ARMS
    }
    if set(arms["base"]) != set(arms["finetuned"]):
        raise ValueError("Both comparison arms must cover exactly the same examples")
    selections = json.loads((ROOT / "gallery.json").read_text())
    if len({item["id"] for item in selections}) != len(selections):
        raise ValueError("Gallery choices must have unique example IDs")
    examples = []
    for selection in selections:
        identity = selection["id"]
        if identity not in rows or identity not in arms["base"]:
            raise ValueError(f"Missing recorded comparison: {identity}")
        row = rows[identity]
        parse_svg(row["input_svg"])
        views = {}
        for arm in ARMS:
            result = arms[arm][identity]
            source = render_source(result["prediction"])
            views[arm] = {
                **result,
                "renderable": source is not None,
                "render_source": source or "",
            }
        examples.append(
            {
                **selection,
                "family": row["family"],
                "instruction": row["instruction"],
                "input_svg": row["input_svg"],
                "duration": row["specification"][0]["duration"],
                **views,
            }
        )
    payload = (
        json.dumps(examples)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = (ROOT / "web/index.html").read_text().replace("__EXAMPLES__", payload)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(html)
    (output / "comparison.json").write_text(json.dumps(examples, indent=2))
    for name in ("app.js", "style.css"):
        shutil.copyfile(ROOT / "web" / name, output / name)
    print(json.dumps({"examples": len(examples), "ids": [e["id"] for e in examples]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()
    build(args.data, args.run, args.split, args.output)
