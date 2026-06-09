from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ReframeTarget(str, Enum):
    STORY_9_16 = "story_9_16"
    VERTICAL_4_5 = "vertical_4_5"
    LANDSCAPE_1_91_1 = "landscape_1_91_1"
    LANDSCAPE_16_9 = "landscape_16_9"
    SQUARE_1_1 = "square_1_1"
    NARROW_BANNER = "narrow_banner"
    LARGE_RECTANGLE = "large_rectangle"
    CUSTOM = "custom"


class BackgroundType(str, Enum):
    SOLID = "solid"
    GRADIENT = "gradient"
    SOFT_BLUR = "soft_blur"
    PHOTOGRAPHIC = "photographic"
    TEXTURED = "textured"
    PATTERNED = "patterned"
    UNKNOWN = "unknown"


class ExpansionStrategy(str, Enum):
    PILLOW_EXTEND = "pillow_extend"
    OPENCV_INPAINT = "opencv_inpaint"
    OPENAI_OUTPAINT = "openai_outpaint"
    HYBRID_RELAYOUT = "hybrid_relayout"
    BLURRED_FIT_FALLBACK = "blurred_fit_fallback"


class LogicBucket(str, Enum):
    VERTICAL_SQUARE = "vertical_square"
    LANDSCAPE_WIDE = "landscape_wide"
    NARROW_BANNER = "narrow_banner"
    LARGE_RECTANGLE = "large_rectangle"


class LayerRole(str, Enum):
    PRODUCT = "product"
    PERSON = "person"
    LOGO = "logo"
    MARKETING_TEXT = "marketing_text"
    PRODUCT_LABEL_TEXT = "product_label_text"
    CTA = "cta"
    BACKGROUND = "background"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


class BBox1000(BaseModel):
    """Normalized bbox in [ymin, xmin, ymax, xmax], each coordinate 0..1000."""

    ymin: int = Field(ge=0, le=1000)
    xmin: int = Field(ge=0, le=1000)
    ymax: int = Field(ge=0, le=1000)
    xmax: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def validate_order(self) -> "BBox1000":
        if self.ymax <= self.ymin or self.xmax <= self.xmin:
            raise ValueError("bbox must have positive width and height")
        return self

    @classmethod
    def from_list(cls, values: list[int | float]) -> "BBox1000":
        if len(values) != 4:
            raise ValueError("bbox list must be [ymin, xmin, ymax, xmax]")
        return cls(
            ymin=round(values[0]),
            xmin=round(values[1]),
            ymax=round(values[2]),
            xmax=round(values[3]),
        )

    def to_pixel_box(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        left = round(self.xmin / 1000 * image_width)
        top = round(self.ymin / 1000 * image_height)
        right = round(self.xmax / 1000 * image_width)
        bottom = round(self.ymax / 1000 * image_height)
        return (
            max(0, min(image_width, left)),
            max(0, min(image_height, top)),
            max(0, min(image_width, right)),
            max(0, min(image_height, bottom)),
        )

    def area_ratio(self) -> float:
        return ((self.xmax - self.xmin) * (self.ymax - self.ymin)) / 1_000_000

    def overlaps(self, other: "BBox1000") -> bool:
        return self.xmin < other.xmax and self.xmax > other.xmin and self.ymin < other.ymax and self.ymax > other.ymin


class RGBColor(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)

    @classmethod
    def from_list(cls, values: list[int | float]) -> "RGBColor":
        if len(values) != 3:
            raise ValueError("color list must be [R, G, B]")
        return cls(r=round(values[0]), g=round(values[1]), b=round(values[2]))

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


class TextStyle(BaseModel):
    color_rgb: RGBColor
    is_bold: bool = False
    font_type: Literal["serif", "sans-serif", "script", "display", "unknown"] = "unknown"
    estimated_font_size: int | None = Field(default=None, ge=1)
    uppercase: bool = False
    alignment: Literal["left", "center", "right", "unknown"] = "unknown"


class BackgroundStyle(BaseModel):
    type: BackgroundType = BackgroundType.UNKNOWN
    dominant_color_rgb: RGBColor | None = None
    is_gradient: bool = False
    texture_complexity: float = Field(default=0.0, ge=0.0, le=1.0)
    can_extend_without_ai: bool = False


class VisualLayer(BaseModel):
    id: str
    role: LayerRole
    bbox: BBox1000
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    saliency: float = Field(default=0.0, ge=0.0, le=1.0)
    protected: bool = False
    notes: str = ""


class TextLayer(VisualLayer):
    role: LayerRole = LayerRole.MARKETING_TEXT
    original_text: str
    translated_text: str = ""
    text_style: TextStyle | None = None
    translate: bool = True


class ProductLayer(VisualLayer):
    role: LayerRole = LayerRole.PRODUCT
    mask_quality: Literal["none", "bbox_only", "rough_mask", "precise_mask"] = "bbox_only"
    needs_shadow: bool = True


class LogoLayer(VisualLayer):
    role: LayerRole = LayerRole.LOGO
    original_text: str = ""
    translate: bool = False


class SafeZone(BaseModel):
    id: str
    label: str
    bbox: BBox1000
    avoid_roles: list[LayerRole] = Field(default_factory=lambda: [LayerRole.MARKETING_TEXT, LayerRole.CTA, LayerRole.PRODUCT])


class TargetCanvas(BaseModel):
    placement_id: str
    target: ReframeTarget
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    logic_bucket: LogicBucket = LogicBucket.VERTICAL_SQUARE
    safe_zones: list[SafeZone] = Field(default_factory=list)
    preferred_product_zone: BBox1000 | None = None
    preferred_text_zone: BBox1000 | None = None

    @property
    def ratio(self) -> float:
        return self.width / max(1, self.height)


class VisualAnalysis(BaseModel):
    schema_version: str = "smart-reframe-v1"
    source_width: int = Field(ge=1)
    source_height: int = Field(ge=1)
    background: BackgroundStyle
    product_layers: list[ProductLayer] = Field(default_factory=list)
    text_layers: list[TextLayer] = Field(default_factory=list)
    logo_layers: list[LogoLayer] = Field(default_factory=list)
    other_layers: list[VisualLayer] = Field(default_factory=list)
    saliency_summary: str = ""
    quality_warnings: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def require_v1_schema(cls, value: str) -> str:
        if value != "smart-reframe-v1":
            raise ValueError("unsupported smart reframe schema version")
        return value

    @property
    def protected_layers(self) -> list[VisualLayer]:
        return [
            layer
            for layer in [*self.product_layers, *self.text_layers, *self.logo_layers, *self.other_layers]
            if layer.protected
        ]

    @property
    def marketing_text_layers(self) -> list[TextLayer]:
        return [layer for layer in self.text_layers if layer.translate and layer.role in {LayerRole.MARKETING_TEXT, LayerRole.CTA}]


class ExpansionDecision(BaseModel):
    strategy: ExpansionStrategy
    reason: str
    requires_ai: bool = False
    estimated_cost_tier: Literal["none", "low", "medium"] = "none"


class PlacementInstruction(BaseModel):
    layer_id: str
    role: LayerRole
    target_bbox: BBox1000
    z_index: int = 0
    scale: float = Field(default=1.0, gt=0.0)
    preserve_aspect_ratio: bool = True
    add_shadow: bool = False


class ReframePlan(BaseModel):
    schema_version: str = "smart-reframe-plan-v1"
    placement_id: str
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    logic_bucket: LogicBucket
    expansion: ExpansionDecision
    placements: list[PlacementInstruction] = Field(default_factory=list)
    safe_zone_warnings: list[str] = Field(default_factory=list)
    fallback_strategy: ExpansionStrategy = ExpansionStrategy.BLURRED_FIT_FALLBACK


def _bbox(ymin: int, xmin: int, ymax: int, xmax: int) -> BBox1000:
    return BBox1000(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)


def _safe_zone(id: str, label: str, x: int, y: int, width: int, height: int) -> SafeZone:
    return SafeZone(
        id=id,
        label=label,
        bbox=_bbox(y, x, min(1000, y + height), min(1000, x + width)),
    )


def _target(
    placement_id: str,
    target: ReframeTarget,
    width: int,
    height: int,
    bucket: LogicBucket,
    safe_zones: list[SafeZone] | None = None,
) -> TargetCanvas:
    product_zone, text_zone = preferred_zones_for(bucket, width, height)
    return TargetCanvas(
        placement_id=placement_id,
        target=target,
        width=width,
        height=height,
        logic_bucket=bucket,
        safe_zones=safe_zones or [],
        preferred_product_zone=product_zone,
        preferred_text_zone=text_zone,
    )


def preferred_zones_for(bucket: LogicBucket, width: int, height: int) -> tuple[BBox1000, BBox1000]:
    ratio = width / max(1, height)
    if bucket == LogicBucket.NARROW_BANNER:
        return _bbox(90, 755, 910, 980), _bbox(150, 45, 820, 720)
    if bucket == LogicBucket.LANDSCAPE_WIDE:
        return _bbox(130, 585, 880, 955), _bbox(190, 60, 780, 535)
    if bucket == LogicBucket.LARGE_RECTANGLE and ratio < 0.45:
        return _bbox(570, 90, 940, 910), _bbox(110, 85, 500, 915)
    if bucket == LogicBucket.LARGE_RECTANGLE:
        return _bbox(430, 460, 900, 940), _bbox(110, 80, 410, 900)
    if ratio < 0.75:
        return _bbox(360, 90, 835, 910), _bbox(130, 95, 330, 905)
    return _bbox(455, 170, 895, 880), _bbox(80, 80, 390, 920)


TARGET_MAP: dict[str, TargetCanvas] = {
    "social-feed-square": _target("social-feed-square", ReframeTarget.SQUARE_1_1, 1080, 1080, LogicBucket.VERTICAL_SQUARE, [_safe_zone("social-feed-caption", "Caption and actions", 0, 820, 1000, 180)]),
    "social-feed-portrait": _target("social-feed-portrait", ReframeTarget.VERTICAL_4_5, 1080, 1350, LogicBucket.VERTICAL_SQUARE, [_safe_zone("social-portrait-caption", "Caption and actions", 0, 860, 1000, 140)]),
    "story-image": _target("story-image", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("story-header", "Story header", 0, 0, 1000, 120), _safe_zone("story-cta", "CTA/message area", 0, 860, 1000, 140)]),
    "wide-landscape": _target("wide-landscape", ReframeTarget.LANDSCAPE_1_91_1, 1200, 628, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("wide-actions", "Actions and link metadata", 0, 860, 1000, 140)]),
    "google-responsive-landscape": _target("google-responsive-landscape", ReframeTarget.LANDSCAPE_1_91_1, 1200, 628, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("g-rda-landscape-adchoices", "AdChoices corner", 860, 0, 140, 120)]),
    "google-responsive-square": _target("google-responsive-square", ReframeTarget.SQUARE_1_1, 1200, 1200, LogicBucket.VERTICAL_SQUARE, [_safe_zone("g-rda-square-adchoices", "AdChoices corner", 860, 0, 140, 120)]),
    "google-responsive-vertical": _target("google-responsive-vertical", ReframeTarget.STORY_9_16, 900, 1600, LogicBucket.VERTICAL_SQUARE, [_safe_zone("g-rda-vertical-adchoices", "AdChoices corner", 820, 0, 180, 100)]),
    "facebook-feed": _target("facebook-feed", ReframeTarget.SQUARE_1_1, 1080, 1080, LogicBucket.VERTICAL_SQUARE, [_safe_zone("fb-feed-footer", "Reaction + CTA footer", 0, 820, 1000, 180)]),
    "facebook-marketplace": _target("facebook-marketplace", ReframeTarget.SQUARE_1_1, 1080, 1080, LogicBucket.VERTICAL_SQUARE, [_safe_zone("marketplace-meta", "Listing details", 0, 760, 1000, 240)]),
    "facebook-right-column": _target("facebook-right-column", ReframeTarget.LANDSCAPE_1_91_1, 1200, 628, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("fb-column-meta", "Headline + CTA", 0, 700, 1000, 300)]),
    "instagram-feed": _target("instagram-feed", ReframeTarget.SQUARE_1_1, 1080, 1080, LogicBucket.VERTICAL_SQUARE, [_safe_zone("ig-caption", "Caption rail", 0, 820, 1000, 180)]),
    "instagram-story": _target("instagram-story", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("ig-story-header", "Story header", 0, 0, 1000, 120), _safe_zone("ig-story-cta", "CTA zone", 0, 860, 1000, 140)]),
    "instagram-reels": _target("instagram-reels", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("ig-reels-actions", "Action rail", 820, 340, 180, 460), _safe_zone("ig-reels-caption", "Caption area", 0, 800, 1000, 200)]),
    "tiktok-in-feed": _target("tiktok-in-feed", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("tiktok-actions", "Action rail", 830, 380, 170, 400), _safe_zone("tiktok-caption", "Caption + CTA", 0, 780, 1000, 220)]),
    "tiktok-topview": _target("tiktok-topview", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("tiktok-topview-actions", "Action rail", 830, 400, 170, 380), _safe_zone("tiktok-topview-cta", "TopView CTA", 0, 820, 1000, 180)]),
    "tiktok-branded-content": _target("tiktok-branded-content", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("tiktok-brand-label", "Branded disclosure", 0, 0, 1000, 120)]),
    "snap-top-snap": _target("snap-top-snap", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("snap-top-header", "Snap header", 0, 0, 1000, 120), _safe_zone("snap-top-cta", "Swipe CTA", 0, 860, 1000, 140)]),
    "snap-story-ad": _target("snap-story-ad", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("snap-story-header", "Story header", 0, 0, 1000, 120), _safe_zone("snap-story-cta", "CTA", 0, 840, 1000, 160)]),
    "linkedin-single-wide": _target("linkedin-single-wide", ReframeTarget.LANDSCAPE_1_91_1, 1200, 628, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("linkedin-wide-actions", "Action row", 0, 880, 1000, 120)]),
    "linkedin-single-square": _target("linkedin-single-square", ReframeTarget.SQUARE_1_1, 1080, 1080, LogicBucket.VERTICAL_SQUARE, [_safe_zone("linkedin-square-actions", "Action row", 0, 880, 1000, 120)]),
    "linkedin-sponsored": _target("linkedin-sponsored", ReframeTarget.LANDSCAPE_1_91_1, 1200, 628, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("linkedin-sponsored-actions", "Sponsored controls", 0, 860, 1000, 140)]),
    "gdn-300x250": _target("gdn-300x250", ReframeTarget.LARGE_RECTANGLE, 300, 250, LogicBucket.LARGE_RECTANGLE, [_safe_zone("gdn-300x250-adchoices", "AdChoices", 860, 0, 140, 120)]),
    "gdn-728x90": _target("gdn-728x90", ReframeTarget.NARROW_BANNER, 728, 90, LogicBucket.NARROW_BANNER, [_safe_zone("gdn-728x90-adchoices", "AdChoices", 940, 0, 60, 220)]),
    "gdn-160x600": _target("gdn-160x600", ReframeTarget.LARGE_RECTANGLE, 160, 600, LogicBucket.LARGE_RECTANGLE, [_safe_zone("gdn-160x600-adchoices", "AdChoices", 720, 0, 280, 70)]),
    "gdn-320x50": _target("gdn-320x50", ReframeTarget.NARROW_BANNER, 320, 50, LogicBucket.NARROW_BANNER, [_safe_zone("gdn-320x50-adchoices", "AdChoices", 880, 0, 120, 340)]),
    "gdn-300x600": _target("gdn-300x600", ReframeTarget.LARGE_RECTANGLE, 300, 600, LogicBucket.LARGE_RECTANGLE, [_safe_zone("gdn-300x600-adchoices", "AdChoices", 860, 0, 140, 60)]),
    "youtube-instream": _target("youtube-instream", ReframeTarget.LANDSCAPE_16_9, 1920, 1080, LogicBucket.LANDSCAPE_WIDE, [_safe_zone("youtube-controls", "Playback controls", 0, 840, 1000, 160)]),
    "youtube-shorts": _target("youtube-shorts", ReframeTarget.STORY_9_16, 1080, 1920, LogicBucket.VERTICAL_SQUARE, [_safe_zone("youtube-shorts-actions", "Action rail", 820, 360, 180, 450), _safe_zone("youtube-shorts-caption", "Caption area", 0, 820, 1000, 180)]),
    "custom-display": _target("custom-display", ReframeTarget.CUSTOM, 1200, 800, LogicBucket.LANDSCAPE_WIDE),
}


def choose_expansion_strategy(analysis: VisualAnalysis, target: TargetCanvas) -> ExpansionDecision:
    source_ratio = analysis.source_width / max(1, analysis.source_height)
    ratio_delta = abs(source_ratio - target.ratio)
    bg = analysis.background
    has_structured_layers = bool(analysis.product_layers or analysis.marketing_text_layers or analysis.other_layers)

    if target.logic_bucket == LogicBucket.NARROW_BANNER:
        return ExpansionDecision(
            strategy=ExpansionStrategy.HYBRID_RELAYOUT,
            reason="Narrow banners are too short for outpaint; rebuild as product/text/CTA layout.",
            requires_ai=False,
            estimated_cost_tier="none",
        )

    if ratio_delta >= 0.16 and target.logic_bucket != LogicBucket.NARROW_BANNER:
        return ExpansionDecision(
            strategy=ExpansionStrategy.OPENAI_OUTPAINT,
            reason="Aspect ratio changes materially; preserve the protected creative and complete missing canvas with generative outpaint instead of crop, blur, or flat placeholder.",
            requires_ai=True,
            estimated_cost_tier="medium",
        )

    if target.logic_bucket == LogicBucket.LANDSCAPE_WIDE and bg.type in {BackgroundType.SOLID, BackgroundType.GRADIENT, BackgroundType.SOFT_BLUR}:
        return ExpansionDecision(
            strategy=ExpansionStrategy.PILLOW_EXTEND,
            reason="Wide placement with simple background; deterministic edge extension is preferred.",
            requires_ai=False,
            estimated_cost_tier="none",
        )

    if target.logic_bucket == LogicBucket.LANDSCAPE_WIDE and bg.texture_complexity <= 0.58:
        return ExpansionDecision(
            strategy=ExpansionStrategy.OPENCV_INPAINT,
            reason="Wide placement with moderate detail; local inpainting before any AI outpaint.",
            requires_ai=False,
            estimated_cost_tier="low",
        )

    if ratio_delta < 0.08:
        return ExpansionDecision(
            strategy=ExpansionStrategy.BLURRED_FIT_FALLBACK,
            reason="Source and target aspect ratios are already close.",
            requires_ai=False,
            estimated_cost_tier="none",
        )

    if bg.type in {BackgroundType.SOLID, BackgroundType.GRADIENT, BackgroundType.SOFT_BLUR} and bg.texture_complexity <= 0.28:
        return ExpansionDecision(
            strategy=ExpansionStrategy.PILLOW_EXTEND,
            reason="Background is simple enough for deterministic extension.",
            requires_ai=False,
            estimated_cost_tier="none",
        )

    if bg.type in {BackgroundType.PHOTOGRAPHIC, BackgroundType.TEXTURED, BackgroundType.PATTERNED} or bg.texture_complexity > 0.48:
        return ExpansionDecision(
            strategy=ExpansionStrategy.OPENAI_OUTPAINT,
            reason="Background has texture or photographic detail; generative expansion is safer.",
            requires_ai=True,
            estimated_cost_tier="medium",
        )

    return ExpansionDecision(
        strategy=ExpansionStrategy.OPENCV_INPAINT,
        reason="Background is moderately complex; try local inpainting before AI outpaint.",
        requires_ai=False,
        estimated_cost_tier="low",
    )


class SmartReframe:
    target_map = TARGET_MAP

    def __init__(self, analysis: VisualAnalysis):
        self.analysis = analysis

    def target_for(self, placement_id: str, custom_width: int | None = None, custom_height: int | None = None) -> TargetCanvas:
        if placement_id == "custom-display" and custom_width and custom_height:
            bucket = bucket_for_dimensions(custom_width, custom_height)
            return _target("custom-display", ReframeTarget.CUSTOM, custom_width, custom_height, bucket)
        return self.target_map.get(placement_id) or _target(
            placement_id,
            ReframeTarget.CUSTOM,
            custom_width or 1200,
            custom_height or 800,
            bucket_for_dimensions(custom_width or 1200, custom_height or 800),
        )

    async def execute(
        self,
        placement_ids: list[str],
        custom_width: int | None = None,
        custom_height: int | None = None,
    ) -> list[ReframePlan]:
        targets = [self.target_for(placement_id, custom_width, custom_height) for placement_id in placement_ids]
        tasks = [asyncio.to_thread(self.build_plan, target) for target in targets]
        return list(await asyncio.gather(*tasks))

    def execute_sync(
        self,
        placement_ids: list[str],
        custom_width: int | None = None,
        custom_height: int | None = None,
    ) -> list[ReframePlan]:
        return [
            self.build_plan(self.target_for(placement_id, custom_width, custom_height))
            for placement_id in placement_ids
        ]

    def build_plan(self, target: TargetCanvas) -> ReframePlan:
        expansion = choose_expansion_strategy(self.analysis, target)
        placements = self._placement_instructions(target)
        warnings = self._safe_zone_warnings(target, placements)
        return ReframePlan(
            placement_id=target.placement_id,
            source_size=(self.analysis.source_width, self.analysis.source_height),
            target_size=(target.width, target.height),
            logic_bucket=target.logic_bucket,
            expansion=expansion,
            placements=placements,
            safe_zone_warnings=warnings,
        )

    def _placement_instructions(self, target: TargetCanvas) -> list[PlacementInstruction]:
        instructions: list[PlacementInstruction] = []
        product_zone = target.preferred_product_zone or preferred_zones_for(target.logic_bucket, target.width, target.height)[0]
        text_zone = target.preferred_text_zone or preferred_zones_for(target.logic_bucket, target.width, target.height)[1]
        decorative_zones = self._decorative_zones(target, product_zone, text_zone)

        for index, layer in enumerate(self.analysis.product_layers[:2]):
            instructions.append(
                PlacementInstruction(
                    layer_id=layer.id,
                    role=layer.role,
                    target_bbox=product_zone,
                    z_index=20 + index,
                    add_shadow=layer.needs_shadow,
                )
            )

        for index, layer in enumerate(self.analysis.marketing_text_layers[:12]):
            instructions.append(
                PlacementInstruction(
                    layer_id=layer.id,
                    role=layer.role,
                    target_bbox=text_zone,
                    z_index=40 + index,
                    add_shadow=False,
                )
            )

        decorative_layers = sorted(
            [layer for layer in self.analysis.other_layers if layer.role in {LayerRole.DECORATIVE, LayerRole.BACKGROUND, LayerRole.UNKNOWN}],
            key=lambda layer: (layer.saliency, layer.confidence, layer.bbox.area_ratio()),
            reverse=True,
        )
        for index, layer in enumerate(decorative_layers[:3]):
            zone = decorative_zones[index % len(decorative_zones)]
            instructions.append(
                PlacementInstruction(
                    layer_id=layer.id,
                    role=layer.role,
                    target_bbox=zone,
                    z_index=8 + index,
                    scale=max(0.35, min(0.9, 0.42 + layer.saliency * 0.42)),
                    preserve_aspect_ratio=True,
                    add_shadow=False,
                )
            )

        return instructions

    def _decorative_zones(self, target: TargetCanvas, product_zone: BBox1000, text_zone: BBox1000) -> list[BBox1000]:
        if target.logic_bucket == LogicBucket.NARROW_BANNER:
            return [_bbox(100, 650, 900, 960), _bbox(80, 40, 920, 250), _bbox(120, 420, 880, 620)]
        if target.logic_bucket == LogicBucket.LANDSCAPE_WIDE:
            return [_bbox(90, 55, 420, 330), _bbox(560, 80, 910, 390), _bbox(70, 760, 360, 970)]
        if target.logic_bucket == LogicBucket.LARGE_RECTANGLE and target.ratio < 0.45:
            return [_bbox(70, 75, 250, 430), _bbox(90, 565, 320, 940), _bbox(700, 35, 960, 360)]
        if target.logic_bucket == LogicBucket.LARGE_RECTANGLE:
            return [_bbox(70, 70, 360, 360), _bbox(80, 690, 410, 950), _bbox(680, 60, 940, 360)]
        if target.ratio < 0.75:
            return [_bbox(80, 70, 300, 430), _bbox(90, 575, 340, 935), _bbox(705, 60, 935, 360)]
        return [_bbox(70, 70, 335, 360), _bbox(90, 670, 360, 940), _bbox(700, 80, 940, 380)]

    def _safe_zone_warnings(self, target: TargetCanvas, placements: list[PlacementInstruction]) -> list[str]:
        warnings: list[str] = []
        for placement in placements:
            for zone in target.safe_zones:
                if placement.role in zone.avoid_roles and placement.target_bbox.overlaps(zone.bbox):
                    warnings.append(f"{placement.layer_id} overlaps {zone.label} on {target.placement_id}.")
        return warnings


def bucket_for_dimensions(width: int, height: int) -> LogicBucket:
    ratio = width / max(1, height)
    if height <= 120 and ratio >= 3.0:
        return LogicBucket.NARROW_BANNER
    if width in {160, 300} and height in {250, 600}:
        return LogicBucket.LARGE_RECTANGLE
    if ratio >= 1.65:
        return LogicBucket.LANDSCAPE_WIDE
    if ratio <= 1.05:
        return LogicBucket.VERTICAL_SQUARE
    return LogicBucket.LARGE_RECTANGLE


def build_visual_analysis_prompt(target_language: str) -> str:
    return (
        "Analyze this ad creative for Smart Reframe V1. Return JSON only matching schema_version smart-reframe-v1. "
        "Use normalized [ymin, xmin, ymax, xmax] coordinates from 0 to 1000. Identify product/person foreground, "
        "marketing text, CTA, logos, product label text, and background type. Mark product, logos, product label text, "
        "and important foreground as protected. Translate only marketing text and CTA into the target language. "
        "This resize system must recompose the creative, not blindly crop it: the main product/person must remain fully visible, "
        "the campaign theme/background must stay recognizable, and secondary scene props may be resized, moved, or partially visible. "
        "List secondary theme props such as sun, umbrella, lounge chair, sea, sky, hands, tools, furniture, or scenery in other_layers "
        "with role decorative/background/unknown; use notes to state importance:primary/secondary/tertiary, visibility:full/partial, "
        "and theme_element:true/false. Do not mark decorative side props protected unless they are part of the product. "
        f"Target language: {target_language}. "
        "For background, classify type as solid, gradient, soft_blur, photographic, textured, patterned, or unknown; "
        "estimate texture_complexity from 0 to 1 and can_extend_without_ai. Include exact source_width and source_height. "
        "Use these exact top-level keys whenever possible: background, product_layers, text_layers, logo_layers, other_layers, saliency_summary, quality_warnings. "
        "Prefer bbox objects like {\"ymin\": 0, \"xmin\": 0, \"ymax\": 100, \"xmax\": 100}; do not use unlabeled bbox arrays unless unavoidable. "
        "Every layer must include id, role, bbox, confidence, saliency, protected, and notes. Text layers must also include original_text, translated_text, translate, and text_style. "
        "In saliency_summary, describe the main protected subject, visual theme, and which side elements can be partial in narrower formats. "
        "Do not include markdown."
    )


def _stringify_notes(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key}:{val}" for key, val in value.items())
    if value is None:
        return ""
    return str(value)


def _normalize_layer_payload(layer: Any, index: int, prefix: str) -> dict[str, Any] | None:
    if not isinstance(layer, dict):
        return None
    normalized = dict(layer)
    normalized.setdefault("id", f"{prefix}-{index + 1}")
    if "bbox" not in normalized:
        for key in ("box", "bounds", "bounding_box", "boundingBox"):
            if key in normalized:
                normalized["bbox"] = normalized[key]
                break
    if "bbox" not in normalized:
        return None
    if isinstance(normalized.get("bbox"), list):
        try:
            values = [max(0, min(1000, round(float(item)))) for item in normalized["bbox"][:4]]
            # Vision models frequently emit list boxes as [xmin, ymin, xmax, ymax]
            # even when prompted otherwise. Object-shaped bboxes remain authoritative;
            # bare lists are normalized as xyxy to avoid transposed product/text layers.
            normalized["bbox"] = BBox1000(
                ymin=values[1],
                xmin=values[0],
                ymax=values[3],
                xmax=values[2],
            ).model_dump()
        except Exception:
            return None
    elif isinstance(normalized.get("bbox"), dict):
        try:
            bbox = dict(normalized["bbox"])
            for key in ("ymin", "xmin", "ymax", "xmax"):
                if key in bbox:
                    bbox[key] = max(0, min(1000, round(float(bbox[key]))))
            normalized["bbox"] = bbox
        except Exception:
            return None
    normalized["notes"] = _stringify_notes(normalized.get("notes", ""))
    role = normalized.get("role")
    if isinstance(role, str):
        normalized["role"] = role.strip().lower().replace(" ", "_").replace("-", "_")
    elif prefix == "product":
        normalized["role"] = LayerRole.PRODUCT.value
    elif prefix == "logo":
        normalized["role"] = LayerRole.LOGO.value
    else:
        normalized.setdefault("role", LayerRole.UNKNOWN.value)
    return normalized


def _normalize_text_layer_payload(layer: Any, index: int) -> dict[str, Any] | None:
    normalized = _normalize_layer_payload(layer, index, "text")
    if normalized is None:
        return None
    role = str(normalized.get("role") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if role not in {LayerRole.MARKETING_TEXT.value, LayerRole.CTA.value, LayerRole.PRODUCT_LABEL_TEXT.value}:
        normalized["role"] = LayerRole.MARKETING_TEXT.value
    text = normalized.get("original_text") or normalized.get("text") or normalized.get("copy") or ""
    normalized["original_text"] = str(text)
    normalized.setdefault("translated_text", normalized["original_text"])
    text_style = normalized.get("text_style") or normalized.get("textStyle") or normalized.get("style")
    if isinstance(text_style, str):
        lowered = text_style.lower()
        normalized["text_style"] = {
            "color_rgb": normalized.get("color_rgb") or normalized.get("colorRgb") or {"r": 17, "g": 17, "b": 17},
            "is_bold": "bold" in lowered,
            "font_type": "sans-serif",
            "estimated_font_size": normalized.get("estimated_font_size") or normalized.get("font_size"),
            "uppercase": "uppercase" in lowered or "all_caps" in lowered,
            "alignment": normalized.get("alignment") or "unknown",
        }
    elif isinstance(text_style, dict):
        color = text_style.get("color_rgb") or text_style.get("colorRgb") or text_style.get("color")
        if isinstance(color, list):
            color = RGBColor.from_list(color).model_dump()
        if isinstance(color, str) and color.startswith("#") and len(color) in {7, 9}:
            color = {
                "r": int(color[1:3], 16),
                "g": int(color[3:5], 16),
                "b": int(color[5:7], 16),
            }
        normalized["text_style"] = {
            "color_rgb": color or {"r": 17, "g": 17, "b": 17},
            "is_bold": bool(text_style.get("is_bold") or text_style.get("bold") or text_style.get("font_weight") == 700),
            "font_type": str(text_style.get("font_type") or text_style.get("fontCategory") or text_style.get("font_family") or "sans-serif").lower().replace("_", "-"),
            "estimated_font_size": text_style.get("estimated_font_size") or text_style.get("fontSize") or text_style.get("font_size"),
            "uppercase": bool(text_style.get("uppercase") or text_style.get("is_uppercase") or text_style.get("all_caps")),
            "alignment": str(text_style.get("alignment") or text_style.get("align") or "unknown").lower(),
        }
    return normalized


def parse_visual_analysis_payload(payload: dict[str, Any]) -> VisualAnalysis:
    normalized = dict(payload)
    background = (
        normalized.get("background")
        or normalized.get("background_style")
        or normalized.get("backgroundStyle")
        or normalized.get("background_analysis")
    )
    if not isinstance(background, dict):
        background = {}
    normalized["background"] = {
        "type": str(background.get("type") or background.get("background_type") or BackgroundType.UNKNOWN.value).strip().lower(),
        "dominant_color_rgb": background.get("dominant_color_rgb") or background.get("dominantColorRgb"),
        "is_gradient": bool(background.get("is_gradient") or background.get("isGradient", False)),
        "texture_complexity": float(background.get("texture_complexity") or background.get("textureComplexity") or 0.0),
        "can_extend_without_ai": bool(background.get("can_extend_without_ai") or background.get("canExtendWithoutAi", False)),
    }
    if not normalized["background"]["dominant_color_rgb"]:
        normalized["background"]["dominant_color_rgb"] = {"r": 245, "g": 245, "b": 245}

    layer_aliases = {
        "product_layers": (
            "product_layers",
            "productLayers",
            "products",
            "product_foreground",
            "productForeground",
            "foreground_layers",
            "foregroundLayers",
            "main_subject_layers",
            "mainSubjectLayers",
        ),
        "logo_layers": ("logo_layers", "logoLayers", "logos", "brand_layers", "brandLayers"),
        "other_layers": (
            "other_layers",
            "otherLayers",
            "decorative_layers",
            "decorativeLayers",
            "background_layers",
            "backgroundLayers",
            "secondary_theme_elements",
            "secondaryThemeElements",
            "theme_elements",
            "themeElements",
            "props",
        ),
    }
    for key, prefix in (("product_layers", "product"), ("logo_layers", "logo"), ("other_layers", "layer")):
        source_layers = []
        for alias in layer_aliases[key]:
            value = normalized.get(alias)
            if isinstance(value, list):
                source_layers = value
                break
        normalized[key] = [
            item
            for index, layer in enumerate(source_layers if isinstance(source_layers, list) else [])
            if (item := _normalize_layer_payload(layer, index, prefix)) is not None
        ]
    source_text_layers = []
    for alias in (
        "text_layers",
        "textLayers",
        "marketing_text_layers",
        "marketingTextLayers",
        "marketing_copy_layers",
        "marketingCopyLayers",
        "copy_layers",
        "copyLayers",
        "cta_layers",
        "ctaLayers",
        "texts",
        "text",
    ):
        value = normalized.get(alias)
        if isinstance(value, list):
            source_text_layers = value
            break
    normalized["text_layers"] = [
        item
        for index, layer in enumerate(source_text_layers if isinstance(source_text_layers, list) else [])
        if (item := _normalize_text_layer_payload(layer, index)) is not None
    ]

    summary = normalized.get("saliency_summary", "")
    if isinstance(summary, dict):
        summary = json.dumps(summary, ensure_ascii=False)
    normalized["saliency_summary"] = str(summary)
    warnings = normalized.get("quality_warnings", [])
    normalized["quality_warnings"] = warnings if isinstance(warnings, list) else [str(warnings)]
    return VisualAnalysis.model_validate(normalized)
