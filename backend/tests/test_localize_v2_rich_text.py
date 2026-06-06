import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import TextBlock, normalize_cross_line_rich_text_segments, tokenize_style_span


def source_word(word_id: str, text: str, color: str, font_size: int = 20) -> dict:
    return {
        "id": word_id,
        "text": text,
        "color": color,
        "fontWeight": 700,
        "fontSize": font_size,
        "lineHeight": int(font_size * 1.25),
        "fontCategory": "sans-serif",
        "semanticRole": "benefit",
        "isUppercase": text.isupper(),
    }


class LocalizeV2RichTextTests(unittest.TestCase):
    def test_segments_preserve_semantic_styles_after_translation_reorder(self):
        block = TextBlock(
            id="v5-block-test",
            text="-40%\nLESS\nAcne Scars",
            role="headline",
            translate=True,
            bbox=(10, 10, 300, 120),
            clean_box=(10, 10, 300, 120),
            source_word_styles=[
                source_word("discount", "-40%", "#f58a1a"),
                source_word("less", "LESS", "#052b52"),
                source_word("acne", "Acne", "#052b52"),
                source_word("scars", "Scars", "#052b52"),
            ],
        )
        payload = {
            "lines": [
                {
                    "segments": [
                        {"text": "Akne Lekelerinde", "source_word_ids": ["acne", "scars"]},
                        {"text": "-%40", "source_word_ids": ["discount"]},
                        {"text": "AZALMA", "source_word_ids": ["less"]},
                    ]
                }
            ]
        }

        spans = normalize_cross_line_rich_text_segments(payload, block=block)

        self.assertEqual([span["translatedText"] for span in spans], ["Akne Lekelerinde", "-%40", "AZALMA"])
        self.assertEqual(spans[0]["style"]["color"], "#052b52")
        self.assertEqual(spans[1]["style"]["color"], "#f58a1a")
        self.assertEqual(spans[2]["style"]["color"], "#052b52")

    def test_tokenizer_keeps_styles_inside_each_segment_not_global(self):
        span = {
            "translatedText": "Sébium H2O ile",
            "style": {"color": "#bed780", "fontWeight": 700, "fontSize": 20},
            "sourceWordIds": ["with", "sebium", "h2o"],
            "sourceWordStyles": [
                source_word("with", "with", "#707070", 19),
                source_word("sebium", "Sébium", "#515151", 19),
                source_word("h2o", "H2O", "#bed780", 18),
            ],
        }

        tokens = tokenize_style_span(span)

        self.assertEqual([token["text"] for token in tokens], ["Sébium", "H2O", "ile"])
        self.assertEqual([token["style"]["color"] for token in tokens], ["#515151", "#bed780", "#707070"])
        self.assertEqual([token["sourceWordIds"] for token in tokens], [["sebium"], ["h2o"], ["with"]])


if __name__ == "__main__":
    unittest.main()
