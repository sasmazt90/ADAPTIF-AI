import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import localize_v7 as v7  # noqa: E402

ORANGE = (255, 80, 0)
DARK = (26, 26, 26)
WHITE = (250, 250, 250)


def make_creative(path: Path) -> list[v7.OcrLine]:
    """A flat orange panel with a headline, a dark button and a 'logo'."""
    image = Image.new("RGB", (600, 500), ORANGE)
    draw = ImageDraw.Draw(image)
    head = v7.load_font("sans", 850, 80, 64)
    cta = v7.load_font("sans", 600, 100, 26)
    logo = v7.load_font("sans", 900, 100, 48)
    draw.text((300, 140), "RENT PREMIUM.", font=head, fill=DARK, anchor="ms")
    draw.text((300, 210), "LOVE THE DRIVE.", font=head, fill=DARK, anchor="ms")
    draw.rounded_rectangle((180, 300, 420, 360), radius=10, fill=DARK)
    draw.text((300, 340), "Rent now", font=cta, fill=WHITE, anchor="ms")
    draw.text((40, 460), "ACME", font=logo, fill=WHITE, anchor="ls")
    image.save(path)

    def box(text, font, x, y, anchor):
        x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font, anchor=anchor)
        return (int(x0) - 6, int(y0) - 5, int(x1) + 6, int(y1) + 5)

    return [
        v7.OcrLine(1, "RENT PREMIUM", box("RENT PREMIUM.", head, 300, 140, "ms")),
        v7.OcrLine(2, "LOVE THE DRIVE", box("LOVE THE DRIVE.", head, 300, 210, "ms")),
        v7.OcrLine(3, "Rent now", box("Rent now", cta, 300, 340, "ms")),
        v7.OcrLine(4, "ACME", box("ACME", logo, 40, 460, "ls")),
    ]


ANALYSIS = {
    "source_language": "EN",
    "elements": [
        {
            "id": "e1", "box_ids": [1, 2], "role": "headline", "translate": True,
            "source_lines": ["RENT PREMIUM.", "LOVE THE DRIVE."],
            "target_lines": ["Premium kirala.", "Keyfini sür."],
            "align": "center", "font_style": "sans", "font_weight": 850, "uppercase": True, "uniform_size": True,
        },
        {
            "id": "e2", "box_ids": [3], "role": "cta", "translate": True,
            "source_lines": ["Rent now"], "target_lines": ["Şimdi kirala"],
            "align": "center", "font_style": "sans", "font_weight": 600, "uppercase": False, "uniform_size": True,
        },
        {
            "id": "e3", "box_ids": [4], "role": "logo", "translate": False,
            "source_lines": ["ACME"], "target_lines": ["ACME"],
            "align": "left", "font_style": "sans", "font_weight": 900, "uppercase": True, "uniform_size": True,
        },
    ],
}


def fake_caller(system_prompt, payload, images):
    if "elements" in payload and "checks" in payload:
        return {"elements": []}  # copy review: nothing to change
    return {key: value for key, value in ANALYSIS.items()}


class LocalizeV7PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "creative.png"
        cls.lines = make_creative(cls.path)
        cls.result = v7.localize_image(cls.path, "TR", vision_caller=fake_caller, ocr_lines=cls.lines)
        cls.source = np.array(Image.open(cls.path).convert("RGB")).astype(int)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_logo_pixels_are_untouched(self):
        out = np.array(self.result.image).astype(int)
        x0, y0, x1, y1 = self.lines[3].box
        self.assertTrue(np.array_equal(out[y0:y1, x0:x1], self.source[y0:y1, x0:x1]))

    def test_source_text_is_fully_removed_from_flat_panel(self):
        clean = np.array(self.result.clean_base).astype(int)
        for line in self.lines[:2]:
            x0, y0, x1, y1 = line.box
            patch = clean[y0:y1, x0:x1]
            self.assertLess(np.abs(patch - np.array(ORANGE)).max(), 24)
        x0, y0, x1, y1 = self.lines[2].box
        button = clean[y0:y1, x0:x1]
        self.assertLess(np.abs(button - np.array(DARK)).max(), 24, "button text must be erased to the button colour")

    def test_turkish_uppercase_and_colours(self):
        headline = next(e for e in self.result.elements if e.id == "e1")
        self.assertEqual(headline.target_lines, ["PREMİUM KİRALA.", "KEYFİNİ SÜR."])
        self.assertLess(np.abs(np.array(v7._element_color(headline)) - np.array(DARK)).max(), 30)
        cta = next(e for e in self.result.elements if e.id == "e2")
        self.assertGreater(min(v7._element_color(cta)), 200)

    def test_rendered_headline_keeps_source_scale(self):
        render = next(item for item in self.result.report["render"] if item["id"] == "e1")
        for line in render["lines"]:
            self.assertGreater(line["font_size"], 0.8 * line["source_font_size"])
            self.assertGreater(line["weight"], 650, "heavy source must render heavy")

    def test_new_copy_is_painted_inside_the_button(self):
        out = np.array(self.result.image).astype(int)
        x0, y0, x1, y1 = self.lines[2].box
        light = (out[y0:y1, x0:x1].min(axis=2) > 200).sum()
        self.assertGreater(light, 50)

    def test_editor_rerender_round_trip(self):
        states = [v7.element_state(element) for element in self.result.elements]
        rendered, elements = v7.rerender_with_copy(
            Image.open(self.path).convert("RGB"), self.result.clean_base, states, ["Hemen kirala.\nKeyfini çıkar.", "Kirala"], "TR"
        )
        self.assertEqual(rendered.size, (600, 500))
        self.assertEqual(elements[0].target_lines, ["HEMEN KİRALA.", "KEYFİNİ ÇIKAR."])
        self.assertEqual(elements[1].target_lines, ["Kirala"])


class LocalizeV7UnitTests(unittest.TestCase):
    def test_locale_upper(self):
        self.assertEqual(v7.locale_upper("indirim fırsatı", "TR"), "İNDİRİM FIRSATI")
        self.assertEqual(v7.locale_upper("indirim", "DE"), "INDIRIM")

    def test_numeric_line_keeps_emphasis_style(self):
        element = v7.CopyElement(
            id="b", role="badge", translate=True, source_lines=["UP TO", "15%", "OFF*"],
            target_lines=["%15'E", "VARAN", "İNDİRİM*"], ocr_ids=[1, 2, 3], uniform_size=False,
        )
        element.lines = [v7.SourceLine(box=(0, 0, 10, 10)), v7.SourceLine(box=(0, 12, 30, 40)), v7.SourceLine(box=(0, 42, 12, 52))]
        v7._guard_numeric_emphasis(element)
        self.assertEqual(element.target_styles[0], 1)
        self.assertEqual(len(element.target_styles), 3)

    def test_build_elements_assigns_each_box_once_and_detects_unchanged_copy(self):
        lines = [v7.OcrLine(1, "SALE", (0, 0, 50, 20)), v7.OcrLine(2, "Brand", (0, 30, 50, 50))]
        analysis = {
            "elements": [
                {"id": "a", "box_ids": [1, 2], "translate": True, "source_lines": ["SALE"], "target_lines": ["SALE"], "role": "headline"},
                {"id": "b", "box_ids": [2], "translate": True, "source_lines": ["Brand"], "target_lines": ["Marke"], "role": "body"},
            ]
        }
        elements = v7.build_elements(analysis, lines, "DE")
        self.assertEqual(elements[0].ocr_ids, [1, 2])
        self.assertEqual(len(elements), 1, "box 2 is already used by element a")
        self.assertFalse(elements[0].changed, "identical copy must not be repainted")

    def test_merge_words_into_lines(self):
        merged = v7._merge_words_into_lines([
            ("RENT", (10, 10, 60, 40)), ("NOW", (70, 11, 120, 40)), ("Terms", (10, 100, 50, 112)),
        ])
        self.assertEqual([line.text for line in merged], ["RENT NOW", "Terms"])


if __name__ == "__main__":
    unittest.main()
