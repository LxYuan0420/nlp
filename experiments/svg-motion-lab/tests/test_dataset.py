"""Data separation and real browser checks for the animation task."""

import unittest
import xml.etree.ElementTree as ET
from dataclasses import asdict

from dataset import MOTION_PAIRS, SHAPES, make_example
from evaluate import MotionEvaluator, parse_svg


class DatasetTests(unittest.TestCase):
    def test_each_training_family_covers_every_training_motion(self):
        coverage = {family: set() for family in SHAPES["train"]}
        for index in range(len(SHAPES["train"]) * len(MOTION_PAIRS["train"])):
            row = make_example("train", index)
            coverage[row.family].add(tuple(m.kind for m in row.specification))
        for observed in coverage.values():
            self.assertEqual(observed, set(MOTION_PAIRS["train"]))

    def test_families_and_compositions_do_not_leak(self):
        for first, second in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            self.assertFalse(set(SHAPES[first]) & set(SHAPES[second]))
            self.assertFalse(set(MOTION_PAIRS[first]) & set(MOTION_PAIRS[second]))

    def test_targets_are_well_formed_and_reproducible(self):
        for split in SHAPES:
            for index in range(15):
                row = make_example(split, index)
                self.assertEqual(row, make_example(split, index))
                parse_svg(row.input_svg)
                parse_svg(row.target_svg)

    def test_active_content_is_rejected(self):
        row = make_example("test", 0)
        for unsafe in (
            "<script>alert(1)</script>",
            '<image href="https://example.com/x"/>',
            '<animate attributeName="href" values="javascript:alert(1)"/>',
        ):
            with self.assertRaises(ValueError):
                parse_svg(row.input_svg.replace("</svg>", unsafe + "</svg>"))


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = MotionEvaluator().__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.evaluator.__exit__(None, None, None)

    def test_reference_passes_but_static_and_wrong_timing_fail(self):
        for split in SHAPES:
            for motion_index in range(len(MOTION_PAIRS[split])):
                index = motion_index * len(SHAPES[split])
                row = asdict(make_example(split, index))
                self.assertTrue(
                    self.evaluator.score(row, row["target_svg"]).motion_pass
                )
                self.assertFalse(
                    self.evaluator.score(row, row["input_svg"]).motion_pass
                )
                duration = row["specification"][0]["duration"]
                wrong = row["target_svg"].replace(
                    f'dur="{duration}s"', f'dur="{duration * 2}s"'
                )
                self.assertFalse(self.evaluator.score(row, wrong).motion_pass)

    def test_changed_artwork_fails_even_with_correct_motion(self):
        row = asdict(make_example("test", 0))
        prediction = row["target_svg"].replace('width="16"', 'width="18"')
        self.assertFalse(self.evaluator.score(row, prediction).preserves_drawing)

    def test_equivalent_animation_markup_passes(self):
        row = asdict(make_example("test", 0))
        equivalent = row["target_svg"].replace(
            'keyTimes="0;0.4;0.6;1"', 'keyTimes="0;.4;.6;1"'
        )
        self.assertNotEqual(equivalent, row["target_svg"])
        self.assertTrue(self.evaluator.score(row, equivalent).motion_pass)

    def test_xml_namespace_prefix_does_not_change_rendering(self):
        row = asdict(make_example("test", 0))
        prefixed = ET.tostring(ET.fromstring(row["target_svg"]), encoding="unicode")
        prefixed = prefixed.replace(
            'xmlns="http://www.w3.org/2000/svg"', 'xmlns:s="http://www.w3.org/2000/svg"'
        )
        for tag in ("svg", "g", "path", "rect", "animateTransform", "animate"):
            prefixed = prefixed.replace(f"<{tag} ", f"<s:{tag} ").replace(
                f"</{tag}>", f"</s:{tag}>"
            )
        self.assertTrue(self.evaluator.score(row, prefixed).motion_pass)


if __name__ == "__main__":
    unittest.main()
