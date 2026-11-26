import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ADAPTIFAI_RESIZE_V7_OUTPAINT"] = "0"
os.environ["ADAPTIFAI_V7_OPENAI_INPAINT"] = "0"
os.environ.pop("OPENAI_API_KEY", None)

from app import localize_v7 as v7  # noqa: E402
from app import resize_v7 as r7  # noqa: E402

ORANGE = (255, 80, 0)
DARK = (26, 26, 26)
WHITE = (250, 250, 250)


def make_split_creative(path: Path) -> list[v7.OcrLine]:
    """600x1200: studio-gradient scene with a hero and a framed badge on top,
    flat orange copy panel with headline, sub, button and fine print below."""
    w, h = 600, 1200
    image = Image.new("RGB", (w, h), ORANGE)
    scene = np.zeros((700, w, 3), dtype=np.float64)
    ys = np.linspace(0, 1, 700)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    base = 70 + 90 * ys + 30 * xs
    scene[..., 0], scene[..., 1], scene[..., 2] = base, base + 8, base + 14
    image.paste(Image.fromarray(scene.astype(np.uint8)), (0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((110, 330, 490, 640), radius=40, fill=(12, 14, 16))  # hero
    draw.polygon([(330, 180), (560, 180), (545, 310), (315, 310)], outline=WHITE, width=3)  # badge frame
    big = v7.load_font("sans", 850, 80, 70)
    small = v7.load_font("sans", 700, 90, 26)
    head = v7.load_font("sans", 850, 75, 64)
    sub = v7.load_font("sans", 400, 95, 30)
    cta = v7.load_font("sans", 600, 100, 26)
    fine = v7.load_font("sans", 400, 100, 20)
    logo = v7.load_font("sans", 900, 100, 56)
    draw.text((440, 270), "15%", font=big, fill=WHITE, anchor="ms")
    draw.text((440, 205), "UP TO", font=small, fill=WHITE, anchor="ms")
    draw.text((50, 110), "ACME", font=logo, fill=WHITE, anchor="ls")
    draw.text((300, 800), "RENT PREMIUM.", font=head, fill=DARK, anchor="ms")
    draw.text((300, 870), "LOVE THE DRIVE.", font=head, fill=DARK, anchor="ms")
    draw.text((300, 950), "More comfort. More to enjoy.", font=sub, fill=(40, 30, 25), anchor="ms")
    draw.rounded_rectangle((170, 1020, 430, 1090), radius=14, fill=DARK)
    draw.text((300, 1065), "Rent now", font=cta, fill=WHITE, anchor="ms")
    draw.text((50, 1160), "*Terms apply.", font=fine, fill=(60, 30, 10), anchor="ls")
    image.save(path)

    def box(text, font, x, y, anchor, pad=5):
        x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font, anchor=anchor)
        return (int(x0) - pad, int(y0) - pad, int(x1) + pad, int(y1) + pad)

    return [
        v7.OcrLine(1, "ACME", box("ACME", logo, 50, 110, "ls")),
        v7.OcrLine(2, "UP TO", box("UP TO", small, 440, 205, "ms", 3)),
        v7.OcrLine(3, "15%", box("15%", big, 440, 270, "ms", 3)),
        v7.OcrLine(4, "RENT PREMIUM.", box("RENT PREMIUM.", head, 300, 800, "ms")),
        v7.OcrLine(5, "LOVE THE DRIVE.", box("LOVE THE DRIVE.", head, 300, 870, "ms")),
        v7.OcrLine(6, "More comfort. More to enjoy.", box("More comfort. More to enjoy.", sub, 300, 950, "ms")),
        v7.OcrLine(7, "Rent now", box("Rent now", cta, 300, 1065, "ms")),
        v7.OcrLine(8, "*Terms apply.", box("*Terms apply.", fine, 50, 1160, "ls")),
    ]


ANALYSIS = {
    "hero_box_xyxy": [110, 330, 490, 640],
    "elements": [
        {"id": "logo", "box_ids": [1], "role": "logo", "priority": 1, "source_lines": ["ACME"], "font_weight": 900, "uppercase": True},
        {"id": "badge", "box_ids": [2, 3], "role": "badge", "priority": 1, "source_lines": ["UP TO", "15%"], "container_box_xyxy": [312, 176, 564, 314], "uniform_size": False},
        {"id": "head", "box_ids": [4, 5], "role": "headline", "priority": 1, "source_lines": ["RENT PREMIUM.", "LOVE THE DRIVE."], "font_weight": 850, "uppercase": True, "short_lines": ["RENT PREMIUM."]},
        {"id": "sub", "box_ids": [6], "role": "subheadline", "priority": 2, "source_lines": ["More comfort. More to enjoy."], "font_weight": 400},
        {"id": "cta", "box_ids": [7], "role": "cta", "priority": 1, "source_lines": ["Rent now"], "container_box_xyxy": [170, 1020, 430, 1090], "font_weight": 600},
        {"id": "fine", "box_ids": [8], "role": "disclaimer", "priority": 3, "source_lines": ["*Terms apply."], "align": "left", "font_weight": 400},
    ],
}


def fake_caller(system_prompt, payload, images):
    import json

    return json.loads(json.dumps(ANALYSIS))


class ResizeV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "creative.png"
        cls.lines = make_split_creative(cls.path)
        cls.model = r7.decompose(cls.path, vision_caller=fake_caller, ocr_lines=cls.lines)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_decomposition_finds_design_layers(self):
        m = self.model
        self.assertIsNotNone(m.panel)
        self.assertEqual(m.panel.side, "bottom")
        self.assertLess(np.abs(np.array(m.panel.color) - np.array(ORANGE)).max(), 6)
        self.assertIsNotNone(m.logo)
        self.assertEqual(len(m.badges), 1)
        cta = next(t for t in m.texts if t.role == "cta")
        self.assertIsNotNone(cta.button, "the filled CTA shape is detected and redrawn")
        self.assertEqual(m.logo_home, "scene")
        self.assertEqual(m.attached_badges, [False])

    def test_clean_scene_has_no_copy_but_keeps_badge(self):
        clean = np.asarray(self.model.clean).astype(int)
        scene = np.asarray(self.model.scene_clean).astype(int)
        # Headline area on the panel is plain orange again.
        self.assertLess(np.abs(clean[760:880, 80:520] - np.array(ORANGE)).max(), 30)
        # The badge was designed onto the scene: it is still in scene_clean, gone from clean.
        self.assertGreater(scene[230:280, 400:480].max(), 230)
        self.assertLess(clean[230:280, 400:480].max(), 200)

    def test_every_size_is_exact_and_uses_the_right_family(self):
        expected = {
            (300, 600): "same-aspect",
            (1080, 1080): "stack",
            (1080, 1920): "stack",
            (1200, 628): "side",
            (728, 90): "banner",
            (320, 50): "banner",
            (160, 600): "stack",
            (300, 250): "stack",
        }
        results, _ = r7.resize_creative(self.path, [(f"{w}x{h}", w, h) for w, h in expected], model=self.model)
        for (w, h), family in expected.items():
            image, meta = results[f"{w}x{h}"]
            self.assertEqual(image.size, (w, h))
            self.assertEqual(meta["family"], family, f"{w}x{h}")
            self.assertTrue(meta["productionReady"], meta)

    def test_small_sizes_drop_fine_print_but_keep_headline_and_cta(self):
        _image, meta = r7.compose(self.model, 300, 250)
        self.assertIn("head", meta["elements"])
        self.assertIn("cta", meta["elements"])
        self.assertNotIn("fine", meta["elements"])

    def test_upscale_keeps_layout_and_sharp_copy(self):
        image, meta = r7.compose(self.model, 1200, 2400)
        self.assertEqual(meta["family"], "same-aspect")
        arr = np.asarray(image).astype(int)
        # Headline re-set at 2x lands where it was (scaled), in the dark ink colour.
        band = arr[1540:1760, 200:1000]
        self.assertLess(band.min(), 60)

    def test_square_keeps_hero_large(self):
        _image, meta = r7.compose(self.model, 1080, 1080)
        self.assertGreater(meta["sceneHeight"], 1080 * 0.55, "the copy panel stays thin; the hero gets the room")

    def test_hero_cut_by_frame_stays_cut(self):
        model = r7.CreativeModel(**{**self.model.__dict__, "hero_box": (0, 330, 490, 640)})
        self.assertTrue(r7._touching_edges(model)["left"])
        placement = r7.place_scene(model, 900, 500, (0, 0, 900, 500))
        self.assertLessEqual(placement.offset[0], 0.0)

    def test_editor_zoom_and_shift(self):
        base, _ = r7.compose(self.model, 1080, 1080)
        moved, meta = r7.compose_adjusted(self.model, 1080, 1080, zoom=1.2, dx=10, dy=0)
        self.assertEqual(moved.size, (1080, 1080))
        self.assertEqual(meta["manualAdjust"]["zoom"], 1.2)
        self.assertFalse(np.array_equal(np.asarray(base), np.asarray(moved)))

    def test_membrane_fill_is_seamless(self):
        canvas = np.zeros((60, 120, 3))
        known = np.zeros((60, 120), dtype=bool)
        canvas[:, 40:80] = np.linspace(100, 140, 40)[None, :, None]
        known[:, 40:80] = True
        filled = r7._harmonic_fill(canvas, known, grain=False)
        self.assertLess(abs(filled[30, 39, 0] - 100), 4)
        self.assertLess(abs(filled[30, 80, 0] - 140), 4)


if __name__ == "__main__":
    unittest.main()
