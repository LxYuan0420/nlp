"""Evaluate native SVG motion in isolated Chromium pages.

Model outputs are untrusted XML. Only a small SVG vocabulary is accepted;
scripts, CSS, URLs and arbitrary animation targets are rejected before rendering.
The evaluator measures browser transforms and effective opacity against the
procedural reference at fixed times. It does not compare animation source strings.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Self

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
MAX_SVG_BYTES = 65536
MAX_ELEMENTS = 128
ANIMATION_TAGS = {"animate", "animateTransform"}
ALLOWED_TAGS = {
    "svg",
    "g",
    "path",
    "circle",
    "rect",
    "ellipse",
    "line",
    "polygon",
    "polyline",
} | ANIMATION_TAGS
ALLOWED_ATTRIBUTES = {
    "id",
    "viewBox",
    "d",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "points",
    "fill",
    "stroke",
    "stroke-width",
    "opacity",
    "transform",
    "attributeName",
    "type",
    "values",
    "keyTimes",
    "dur",
    "repeatCount",
    "calcMode",
    "begin",
    "additive",
    "from",
    "to",
    "by",
}
SAMPLE_FRACTIONS = (0.0, 0.13, 0.39, 0.5, 0.61, 0.8, 0.99, 1.25)
MATRIX_TOLERANCE = 0.015
POSITION_TOLERANCE = 0.5
OPACITY_TOLERANCE = 0.02


def parse_svg(source: str) -> ET.Element:
    """Validate untrusted SVG without repairing malformed or unsafe output."""
    if len(source.encode()) > MAX_SVG_BYTES or "<!" in source or "<?" in source:
        raise ValueError("SVG exceeds size limit or contains a declaration")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError("Invalid SVG XML") from exc
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError("Expected an SVG root with the standard namespace")
    elements = list(root.iter())
    if len(elements) > MAX_ELEMENTS:
        raise ValueError("Too many SVG elements")
    for element in elements:
        tag = element.tag.removeprefix(f"{{{SVG_NS}}}")
        if tag not in ALLOWED_TAGS:
            raise ValueError(f"Unsupported SVG element: {tag}")
        for name, value in element.attrib.items():
            if (
                name not in ALLOWED_ATTRIBUTES
                or "url" in value.lower()
                or "javascript:" in value.lower()
            ):
                raise ValueError(f"Unsupported SVG attribute: {name}")
        if element.text and element.text.strip():
            raise ValueError("Text content is outside the animation contract")
        if tag == "animate" and element.get("attributeName") != "opacity":
            raise ValueError("Only opacity may use animate")
        if tag == "animateTransform" and (
            element.get("attributeName") != "transform"
            or element.get("type") not in {"translate", "rotate", "scale"}
        ):
            raise ValueError("Unsupported transform animation")
        if tag in ANIMATION_TAGS:
            for time_attribute in ("begin", "dur"):
                value = element.get(time_attribute, "0s")
                if not value.endswith("s"):
                    raise ValueError("Animation times must use seconds")
                try:
                    seconds = float(value[:-1])
                except ValueError as exc:
                    raise ValueError("Animation time must be numeric") from exc
                if not math.isfinite(seconds) or not 0 <= seconds <= 60:
                    raise ValueError("Animation time is outside the allowed range")
    return root


def drawing_structure(root: ET.Element) -> tuple:
    """Compare the original scene tree while ignoring inserted animation nodes."""
    return (
        root.tag,
        tuple(sorted(root.attrib.items())),
        tuple(
            drawing_structure(child)
            for child in root
            if child.tag.removeprefix(f"{{{SVG_NS}}}") not in ANIMATION_TAGS
        ),
    )


@dataclass(frozen=True)
class Score:
    """Task success and explicit failure reason for one generated SVG."""

    valid: bool
    preserves_drawing: bool
    motion_pass: bool
    reason: str


class MotionEvaluator(AbstractContextManager):
    """Own a headless Chromium process; each rendering uses a fresh page."""

    def __enter__(self) -> Self:
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self._browser.close()
        self._playwright.stop()

    def sample(self, source: str, duration: float) -> list:
        """Read CTMs and cumulative opacity for every drawable at fixed times."""
        source = ET.tostring(parse_svg(source), encoding="unicode")
        page = self._browser.new_page(viewport={"width": 256, "height": 256})
        try:
            page.route("**/*", lambda route: route.abort())
            page.set_content(
                "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'none'\"><body>"
                + source
                + "</body>"
            )
            return page.evaluate(
                """({times}) => {
                const svg = document.querySelector('svg');
                svg.setAttribute('width', '256'); svg.setAttribute('height', '256');
                svg.pauseAnimations();
                const shapes = [...svg.querySelectorAll('path,circle,rect,ellipse,line,polygon,polyline')];
                return times.map(t => {
                    svg.setCurrentTime(t);
                    return shapes.map(shape => {
                        const m = shape.getCTM();
                        let opacity = 1;
                        for (let e=shape; e && e instanceof SVGElement; e=e.parentElement) {
                            opacity *= Number(getComputedStyle(e).opacity);
                        }
                        return [m.a,m.b,m.c,m.d,m.e,m.f,opacity];
                    });
                });
            }""",
                {"times": [duration * fraction for fraction in SAMPLE_FRACTIONS]},
            )
        finally:
            page.close()

    def score(self, row: dict, prediction: str) -> Score:
        """Check safety, preservation and rendered motion; invalid output scores zero."""
        try:
            predicted_root = parse_svg(prediction)
        except ValueError as exc:
            return Score(False, False, False, str(exc))
        if drawing_structure(predicted_root) != drawing_structure(
            parse_svg(row["input_svg"])
        ):
            return Score(True, False, False, "Changed the original drawing structure")
        duration = row["specification"][0]["duration"]
        expected = self.sample(row["target_svg"], duration)
        actual = self.sample(prediction, duration)
        for time_index, (expected_shapes, actual_shapes) in enumerate(
            zip(expected, actual, strict=True)
        ):
            for expected_shape, actual_shape in zip(
                expected_shapes, actual_shapes, strict=True
            ):
                tolerances = (
                    [MATRIX_TOLERANCE] * 4
                    + [POSITION_TOLERANCE] * 2
                    + [OPACITY_TOLERANCE]
                )
                if any(
                    not math.isfinite(a) or abs(a - e) > tolerance
                    for a, e, tolerance in zip(
                        actual_shape, expected_shape, tolerances, strict=True
                    )
                ):
                    return Score(
                        True,
                        True,
                        False,
                        f"Motion differs at cycle fraction {SAMPLE_FRACTIONS[time_index]}",
                    )
        return Score(True, True, True, "Passed every sampled motion check")
