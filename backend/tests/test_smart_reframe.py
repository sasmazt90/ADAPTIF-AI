import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.smart_reframe import (
    BackgroundStyle,
    BackgroundType,
    ExpansionStrategy,
    LogicBucket,
    ReframeTarget,
    TargetCanvas,
    VisualAnalysis,
    bucket_for_dimensions,
    choose_expansion_strategy,
)


def analysis(width=1200, height=800, background_type=BackgroundType.PHOTOGRAPHIC, complexity=0.7):
    return VisualAnalysis(
        source_width=width,
        source_height=height,
        background=BackgroundStyle(type=background_type, texture_complexity=complexity),
    )


def target(width, height, bucket):
    return TargetCanvas(
        placement_id="test",
        target=ReframeTarget.CUSTOM,
        width=width,
        height=height,
        logic_bucket=bucket,
    )


class BucketForDimensionsTests(unittest.TestCase):
    def test_standard_display_boundaries(self):
        self.assertEqual(bucket_for_dimensions(728, 90), LogicBucket.NARROW_BANNER)
        self.assertEqual(bucket_for_dimensions(320, 50), LogicBucket.NARROW_BANNER)
        self.assertEqual(bucket_for_dimensions(300, 600), LogicBucket.LARGE_RECTANGLE)
        self.assertEqual(bucket_for_dimensions(160, 600), LogicBucket.LARGE_RECTANGLE)

    def test_banner_boundary_requires_short_height(self):
        self.assertEqual(bucket_for_dimensions(360, 120), LogicBucket.NARROW_BANNER)
        self.assertEqual(bucket_for_dimensions(363, 121), LogicBucket.LANDSCAPE_WIDE)

    def test_ratio_buckets(self):
        self.assertEqual(bucket_for_dimensions(1920, 1080), LogicBucket.LANDSCAPE_WIDE)
        self.assertEqual(bucket_for_dimensions(1080, 1920), LogicBucket.VERTICAL_SQUARE)
        self.assertEqual(bucket_for_dimensions(1200, 1200), LogicBucket.VERTICAL_SQUARE)
        self.assertEqual(bucket_for_dimensions(1200, 900), LogicBucket.LARGE_RECTANGLE)


class ChooseExpansionStrategyTests(unittest.TestCase):
    def test_material_ratio_change_uses_outpaint_not_blur(self):
        decision = choose_expansion_strategy(
            analysis(width=1200, height=800, background_type=BackgroundType.PHOTOGRAPHIC, complexity=0.8),
            target(300, 600, LogicBucket.LARGE_RECTANGLE),
        )
        self.assertEqual(decision.strategy, ExpansionStrategy.OPENAI_OUTPAINT)
        self.assertTrue(decision.requires_ai)

    def test_narrow_banner_uses_hybrid_relayout(self):
        decision = choose_expansion_strategy(
            analysis(width=1200, height=800),
            target(728, 90, LogicBucket.NARROW_BANNER),
        )
        self.assertEqual(decision.strategy, ExpansionStrategy.HYBRID_RELAYOUT)
        self.assertFalse(decision.requires_ai)

    def test_simple_wide_background_can_use_deterministic_extension(self):
        decision = choose_expansion_strategy(
            analysis(width=1200, height=800, background_type=BackgroundType.GRADIENT, complexity=0.1),
            target(1650, 1000, LogicBucket.LANDSCAPE_WIDE),
        )
        self.assertEqual(decision.strategy, ExpansionStrategy.PILLOW_EXTEND)
        self.assertFalse(decision.requires_ai)

    def test_close_square_ratio_uses_non_ai_fit_fallback(self):
        decision = choose_expansion_strategy(
            analysis(width=1080, height=1080, background_type=BackgroundType.PHOTOGRAPHIC, complexity=0.9),
            target(1200, 1200, LogicBucket.VERTICAL_SQUARE),
        )
        self.assertEqual(decision.strategy, ExpansionStrategy.BLURRED_FIT_FALLBACK)
        self.assertFalse(decision.requires_ai)

    def test_textured_background_with_moderate_ratio_change_uses_outpaint(self):
        decision = choose_expansion_strategy(
            analysis(width=1100, height=1000, background_type=BackgroundType.TEXTURED, complexity=0.52),
            target(1200, 1200, LogicBucket.VERTICAL_SQUARE),
        )
        self.assertEqual(decision.strategy, ExpansionStrategy.OPENAI_OUTPAINT)
        self.assertTrue(decision.requires_ai)


if __name__ == "__main__":
    unittest.main()
