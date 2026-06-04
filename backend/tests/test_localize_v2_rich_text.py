import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import block_has_rich_text_segments, normalized_localize_v2_blocks


class LocalizeV2RichTextTests(unittest.TestCase):
    def test_segments_preserve_semantic_styles_after_translation_reorder(self):
        image = Image.new("RGB", (900, 500), "white")
        payload = {
            "source_language": "en",
            "blocks": [
                {
                    "source_text": "-40%\nLESS\nAcne Scars",
                    "translated_text": "Akne Lekelerinde\n-%40\nAZALMA",
                    "x": "10",
                    "y": "10",
                    "w": "80",
                    "h": "70",
                    "align": "center",
                    "font_weight": "bold",
                    "color": "#052b52",
                    "segments": [
                        {
                            "text": "Akne Lekelerinde",
                            "is_bold": True,
                            "color": "#052b52",
                            "is_uppercase": False,
                            "source_segment_hint": "Acne Scars",
                            "semantic_role": "product_condition",
                        },
                        {
                            "text": "-%40",
                            "is_bold": True,
                            "color": "#f58a1a",
                            "is_uppercase": False,
                            "source_segment_hint": "-40%",
                            "semantic_role": "discount",
                        },
                        {
                            "text": "AZALMA",
                            "is_bold": False,
                            "color": "#052b52",
                            "is_uppercase": True,
                            "source_segment_hint": "LESS",
                            "semantic_role": "modifier",
                        },
                    ],
                }
            ],
        }

        block = normalized_localize_v2_blocks(payload, image)[0]
        spans = block.translated_style_spans

        self.assertTrue(block_has_rich_text_segments(block))
        self.assertEqual([span["translatedText"] for span in spans], ["Akne Lekelerinde", "-%40", "AZALMA"])
        self.assertEqual(spans[0]["style"]["color"], "#052b52")
        self.assertGreaterEqual(spans[0]["style"]["fontWeight"], 700)
        self.assertEqual(spans[1]["style"]["color"], "#f58a1a")
        self.assertGreaterEqual(spans[1]["style"]["fontWeight"], 700)
        self.assertEqual(spans[2]["style"]["color"], "#052b52")
        self.assertLess(spans[2]["style"]["fontWeight"], 700)
        self.assertEqual(spans[2]["style"]["casing"], "uppercase")


if __name__ == "__main__":
    unittest.main()
