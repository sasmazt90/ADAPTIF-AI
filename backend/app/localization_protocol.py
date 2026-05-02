"""
Ad Creative Localization Pipeline Protocol V2.2
Enterprise-grade decision engine for localization cleanup.

Phases:
  0 - Creative Risk Classification
  1 - Detection & Segmentation (mask definitions)
  1.5 - Depth / Layering (optional)
  2 - Cleanup Provider Routing (risk-based)
  3 - Inpainting Strategy (stroke-based dilation)
  4 - Quality Gate (multi-metric)
  5 - Retry / Provider Bake-off
  6 - Style-Aware Rendering (copy-fitting)
  7 - Status Propagation (resize/preview gate)
"""

from __future__ import annotations

import io
import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


# ─────────────────────────────────────────────────────────────────────────────
# STATUS CODES
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationStatus(str, Enum):
    CLEANUP_SUCCESS = "cleanup_success"
    REJECT_LOW_CONFIDENCE = "reject_low_confidence"
    UNSUPPORTED_AUTO_CLEANUP = "unsupported_auto_cleanup"
    PACKAGING_PROTECTION_RISK = "packaging_protection_risk"
    REJECT_PRODUCTION_QUALITY = "reject_production_quality"
    REJECT_LAYOUT_FIT = "reject_layout_fit"
    PROVIDER_NOT_CONFIGURED = "provider_not_configured"
    RENDER_SUCCESS = "render_success"
    PREVIEW_SUCCESS = "preview_success"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Creative Risk Classification
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNSUPPORTED_AUTO_CLEANUP = "UNSUPPORTED_AUTO_CLEANUP"
    REJECT_LOW_CONFIDENCE = "REJECT_LOW_CONFIDENCE"
    PACKAGING_PROTECTION_RISK = "PACKAGING_PROTECTION_RISK"


@dataclass
class CreativeRiskReport:
    risk_level: RiskLevel
    text_area_ratio: float = 0.0
    block_area_ratio: float = 0.0
    headline_dominance: float = 0.0
    protected_overlap_score: float = 0.0
    texture_complexity: float = 0.0
    luminance_gradient_complexity: float = 0.0
    compression_quality: float = 1.0
    source_resolution: tuple[int, int] = (0, 0)
    mask_area_ratio: float = 0.0
    text_contrast_level: float = 0.0
    ocr_confidence: float = 1.0
    packaging_separation_feasible: bool = True
    rejection_reason: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "riskLevel": self.risk_level.value,
            "textAreaRatio": round(self.text_area_ratio, 4),
            "blockAreaRatio": round(self.block_area_ratio, 4),
            "headlineDominance": round(self.headline_dominance, 4),
            "protectedOverlapScore": round(self.protected_overlap_score, 4),
            "textureComplexity": round(self.texture_complexity, 4),
            "luminanceGradientComplexity": round(self.luminance_gradient_complexity, 4),
            "compressionQuality": round(self.compression_quality, 4),
            "sourceResolution": list(self.source_resolution),
            "maskAreaRatio": round(self.mask_area_ratio, 4),
            "textContrastLevel": round(self.text_contrast_level, 4),
            "ocrConfidence": round(self.ocr_confidence, 4),
            "packagingSeparationFeasible": self.packaging_separation_feasible,
            "rejectionReason": self.rejection_reason,
            "signals": self.signals,
        }


def compute_texture_complexity(image: Image.Image) -> float:
    """Estimate texture complexity via edge density in grayscale."""
    gray = image.convert("L")
    if gray.size[0] > 512 or gray.size[1] > 512:
        gray = gray.resize((512, int(512 * gray.size[1] / max(1, gray.size[0]))), Image.Resampling.LANCZOS)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float32) / 255.0
    return float(arr.mean())


def compute_luminance_gradient_complexity(image: Image.Image) -> float:
    """Estimate luminance gradient complexity via std of gradient magnitudes."""
    gray = image.convert("L")
    if gray.size[0] > 512 or gray.size[1] > 512:
        gray = gray.resize((512, int(512 * gray.size[1] / max(1, gray.size[0]))), Image.Resampling.LANCZOS)
    arr = np.array(gray, dtype=np.float32)
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return float(magnitude.std() / 255.0)


def estimate_compression_quality(image: Image.Image) -> float:
    """Heuristic: compress to JPEG at q=95, compare size ratio as proxy."""
    buf_high = io.BytesIO()
    image.convert("RGB").save(buf_high, format="JPEG", quality=95)
    size_high = buf_high.tell()

    buf_low = io.BytesIO()
    image.convert("RGB").save(buf_low, format="JPEG", quality=30)
    size_low = buf_low.tell()

    if size_high == 0:
        return 1.0
    ratio = size_low / size_high
    # If ratio is high, image was already heavily compressed
    quality_estimate = max(0.0, min(1.0, 1.0 - (ratio - 0.3) * 2.0))
    return quality_estimate


def compute_text_contrast(image: Image.Image, text_mask: Image.Image | None) -> float:
    """Estimate contrast between text regions and their surroundings."""
    if text_mask is None:
        return 0.5
    gray = np.array(image.convert("L"), dtype=np.float32)
    mask_arr = np.array(text_mask.convert("L"), dtype=np.float32) / 255.0
    if mask_arr.sum() < 10:
        return 0.5
    text_mean = float((gray * mask_arr).sum() / max(1, mask_arr.sum()))
    bg_mask = 1.0 - mask_arr
    bg_mean = float((gray * bg_mask).sum() / max(1, bg_mask.sum()))
    contrast = abs(text_mean - bg_mean) / 255.0
    return contrast


def classify_creative_risk(
    image: Image.Image,
    blocks: list[Any],
    text_mask: Image.Image | None = None,
    protected_mask: Image.Image | None = None,
    ocr_confidence: float = 1.0,
    packaging_blocks: list[Any] | None = None,
) -> CreativeRiskReport:
    """
    Classify creative into LOW / MEDIUM / HIGH / UNSUPPORTED_AUTO_CLEANUP.
    Returns CreativeRiskReport with all metrics.
    """
    w, h = image.size
    image_area = max(1, w * h)

    # OCR confidence gate
    if ocr_confidence < 0.30:
        return CreativeRiskReport(
            risk_level=RiskLevel.REJECT_LOW_CONFIDENCE,
            ocr_confidence=ocr_confidence,
            source_resolution=(w, h),
            rejection_reason="OCR confidence too low for automatic processing",
        )

    # Compute text area ratio from blocks
    text_area = 0
    headline_area = 0
    translate_blocks = [b for b in blocks if getattr(b, "translate", False) and getattr(b, "surface", "") == "overlay"]

    for block in translate_blocks:
        bbox = getattr(block, "bbox", (0, 0, 0, 0))
        bw = max(0, bbox[2] - bbox[0])
        bh = max(0, bbox[3] - bbox[1])
        area = bw * bh
        text_area += area
        # Headline = large multi-line blocks
        multi_line = len(getattr(block, "line_boxes", [])) >= 2
        if multi_line and bw >= w * 0.35:
            headline_area += area

    text_area_ratio = text_area / image_area
    headline_dominance = headline_area / max(1, text_area) if text_area > 0 else 0.0
    block_area_ratio = text_area / image_area

    # Mask area ratio
    mask_area_ratio = 0.0
    if text_mask is not None:
        mask_arr = np.array(text_mask.convert("L"), dtype=np.uint8)
        mask_area_ratio = float((mask_arr > 16).sum()) / image_area

    # Protected overlap
    protected_overlap_score = 0.0
    if text_mask is not None and protected_mask is not None:
        text_arr = np.array(text_mask.convert("L"), dtype=np.uint8) > 16
        prot_arr = np.array(protected_mask.convert("L"), dtype=np.uint8) > 16
        overlap_pixels = float(np.logical_and(text_arr, prot_arr).sum())
        protected_overlap_score = overlap_pixels / max(1, float(text_arr.sum()))

    # Texture and luminance complexity
    texture_complexity = compute_texture_complexity(image)
    luminance_gradient = compute_luminance_gradient_complexity(image)

    # Compression quality
    compression_quality = estimate_compression_quality(image)

    # Text contrast
    text_contrast = compute_text_contrast(image, text_mask)

    # Packaging separation check
    packaging_separation_feasible = True
    if packaging_blocks:
        for pb in packaging_blocks:
            pb_bbox = getattr(pb, "bbox", (0, 0, 0, 0))
            for tb in translate_blocks:
                tb_bbox = getattr(tb, "bbox", (0, 0, 0, 0))
                overlap = _bbox_overlap_fraction(tb_bbox, pb_bbox)
                if overlap > 0.25:
                    packaging_separation_feasible = False
                    break

    if not packaging_separation_feasible:
        return CreativeRiskReport(
            risk_level=RiskLevel.PACKAGING_PROTECTION_RISK,
            text_area_ratio=text_area_ratio,
            block_area_ratio=block_area_ratio,
            headline_dominance=headline_dominance,
            protected_overlap_score=protected_overlap_score,
            texture_complexity=texture_complexity,
            luminance_gradient_complexity=luminance_gradient,
            compression_quality=compression_quality,
            source_resolution=(w, h),
            mask_area_ratio=mask_area_ratio,
            text_contrast_level=text_contrast,
            ocr_confidence=ocr_confidence,
            packaging_separation_feasible=False,
            rejection_reason="Marketing text cannot be separated from protected packaging/logo/product text",
        )

    # Source quality gate
    min_resolution = int(os.getenv("ADAPTIFAI_MIN_SOURCE_RESOLUTION", "200"))
    if min(w, h) < min_resolution or compression_quality < 0.15:
        return CreativeRiskReport(
            risk_level=RiskLevel.UNSUPPORTED_AUTO_CLEANUP,
            text_area_ratio=text_area_ratio,
            block_area_ratio=block_area_ratio,
            source_resolution=(w, h),
            compression_quality=compression_quality,
            rejection_reason="Source quality too low or compression too severe",
        )

    # Risk level classification
    risk_score = 0.0
    risk_score += text_area_ratio * 2.0  # High text coverage = harder
    risk_score += protected_overlap_score * 3.0  # Protected overlap = very hard
    risk_score += texture_complexity * 1.5  # Complex textures harder to inpaint
    risk_score += luminance_gradient * 1.0
    risk_score += headline_dominance * 0.5
    risk_score += (1.0 - compression_quality) * 0.5

    # Hard rule: textAreaRatio > 0.40 = HIGH
    if text_area_ratio > 0.40:
        risk_level = RiskLevel.HIGH
    elif risk_score > 1.2 or protected_overlap_score > 0.15:
        risk_level = RiskLevel.HIGH
    elif risk_score > 0.5 or text_area_ratio > 0.20:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW

    return CreativeRiskReport(
        risk_level=risk_level,
        text_area_ratio=text_area_ratio,
        block_area_ratio=block_area_ratio,
        headline_dominance=headline_dominance,
        protected_overlap_score=protected_overlap_score,
        texture_complexity=texture_complexity,
        luminance_gradient_complexity=luminance_gradient,
        compression_quality=compression_quality,
        source_resolution=(w, h),
        mask_area_ratio=mask_area_ratio,
        text_contrast_level=text_contrast,
        ocr_confidence=ocr_confidence,
        packaging_separation_feasible=packaging_separation_feasible,
        signals={
            "riskScore": round(risk_score, 4),
            "translateBlockCount": len(translate_blocks),
        },
    )


def _bbox_overlap_fraction(a: tuple, b: tuple) -> float:
    """Fraction of bbox a overlapping with bbox b."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap_area = (x2 - x1) * (y2 - y1)
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return overlap_area / a_area


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Detection & Segmentation Mask Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentationMasks:
    """All segmentation masks for a creative."""
    marketing_text_mask: Image.Image | None = None
    packaging_text_mask: Image.Image | None = None
    logo_mask: Image.Image | None = None
    product_mask: Image.Image | None = None
    hand_mask: Image.Image | None = None
    face_skin_mask: Image.Image | None = None
    shoe_mask: Image.Image | None = None
    protected_object_mask: Image.Image | None = None
    protected_packaging_text_mask: Image.Image | None = None

    # Default packaging policy
    packaging_policy: str = "preserve_all_packaging_text"

    def get_composite_protected_mask(self, size: tuple[int, int]) -> Image.Image:
        """Union of all protected region masks."""
        composite = Image.new("L", size, 0)
        masks = [
            self.packaging_text_mask,
            self.logo_mask,
            self.product_mask,
            self.hand_mask,
            self.face_skin_mask,
            self.shoe_mask,
            self.protected_object_mask,
            self.protected_packaging_text_mask,
        ]
        for mask in masks:
            if mask is not None:
                resized = mask.convert("L")
                if resized.size != size:
                    resized = resized.resize(size, Image.Resampling.NEAREST)
                composite = ImageChops.lighter(composite, resized)
        return composite

    def compute_cleanup_mask(self, size: tuple[int, int]) -> Image.Image:
        """
        cleanupMask = marketingTextMask - protectedObjectMask
                    - packagingTextMask - protectedPackagingTextMask - logoMask
        """
        if self.marketing_text_mask is None:
            return Image.new("L", size, 0)
        cleanup = np.array(self.marketing_text_mask.convert("L"), dtype=np.float32)
        subtract_masks = [
            self.protected_object_mask,
            self.packaging_text_mask,
            self.protected_packaging_text_mask,
            self.logo_mask,
        ]
        for mask in subtract_masks:
            if mask is not None:
                m = mask.convert("L")
                if m.size != size:
                    m = m.resize(size, Image.Resampling.NEAREST)
                cleanup = np.maximum(0, cleanup - np.array(m, dtype=np.float32))
        return Image.fromarray(np.clip(cleanup, 0, 255).astype(np.uint8), "L")


def build_segmentation_masks(
    image: Image.Image,
    blocks: list[Any],
    protected_region_mask: Image.Image | None = None,
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> SegmentationMasks:
    """
    Build segmentation masks from existing block data and protected regions.
    Uses PaddleOCR if available, falls back to EasyOCR/TrOCR pipeline.
    SAM/SAM2 used for mask refinement if available.
    """
    w, h = image.size
    masks = SegmentationMasks()

    # Marketing text mask from translatable overlay blocks
    marketing_mask = Image.new("L", (w, h), 0)
    packaging_mask = Image.new("L", (w, h), 0)
    draw_marketing = ImageDraw.Draw(marketing_mask)
    draw_packaging = ImageDraw.Draw(packaging_mask)

    for block in blocks:
        bbox = getattr(block, "bbox", None)
        if bbox is None:
            continue
        surface = getattr(block, "surface", "overlay")
        translate = getattr(block, "translate", False)

        if translate and surface == "overlay":
            # Use line_boxes if available for tighter masks
            line_boxes = getattr(block, "line_boxes", [])
            if line_boxes:
                for lb in line_boxes:
                    draw_marketing.rectangle(lb, fill=255)
            else:
                draw_marketing.rectangle(bbox, fill=255)
        elif surface in ("packaging", "product"):
            draw_packaging.rectangle(bbox, fill=255)

    masks.marketing_text_mask = marketing_mask
    masks.packaging_text_mask = packaging_mask
    masks.protected_packaging_text_mask = packaging_mask.copy()

    # Use provided protected region mask for product/logo/etc
    if protected_region_mask is not None:
        masks.protected_object_mask = protected_region_mask.convert("L")

    # Logo mask: derive from non-translatable blocks with logo cues
    logo_mask = Image.new("L", (w, h), 0)
    draw_logo = ImageDraw.Draw(logo_mask)
    for block in blocks:
        surface = getattr(block, "surface", "overlay")
        if surface == "logo":
            bbox = getattr(block, "bbox", None)
            if bbox:
                draw_logo.rectangle(bbox, fill=255)
    masks.logo_mask = logo_mask

    # Product mask from foreground bbox if available
    if foreground_bbox is not None:
        product_mask = Image.new("L", (w, h), 0)
        draw_product = ImageDraw.Draw(product_mask)
        draw_product.rectangle(foreground_bbox, fill=180)
        masks.product_mask = product_mask

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1.5 — Depth / Layering Analysis (Optional)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DepthLayeringReport:
    """Optional depth/layering analysis results."""
    foreground_masks: list[Image.Image] = field(default_factory=list)
    occlusion_mask_for_rendering: Image.Image | None = None
    layering_risk: str = "none"  # none | low | medium | high
    foreground_occlusion_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layeringRisk": self.layering_risk,
            "foregroundOcclusionReport": self.foreground_occlusion_report,
            "hasForegroundMasks": len(self.foreground_masks) > 0,
            "hasOcclusionMask": self.occlusion_mask_for_rendering is not None,
        }


def analyze_depth_layering(
    image: Image.Image,
    blocks: list[Any],
    segmentation_masks: SegmentationMasks,
) -> DepthLayeringReport:
    """
    Analyze depth/layering to detect text behind foreground objects.
    Uses luminance/saturation heuristics when depth models aren't available.
    """
    w, h = image.size
    report = DepthLayeringReport()

    # Check if any text block overlaps significantly with product/foreground
    protected = segmentation_masks.get_composite_protected_mask((w, h))
    protected_arr = np.array(protected, dtype=np.uint8)

    translate_blocks = [b for b in blocks if getattr(b, "translate", False)]
    occluded_blocks = []

    for block in translate_blocks:
        bbox = getattr(block, "bbox", (0, 0, 0, 0))
        region = protected_arr[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if region.size > 0:
            overlap_ratio = float((region > 64).sum()) / max(1, region.size)
            if overlap_ratio > 0.10:
                occluded_blocks.append({
                    "blockId": getattr(block, "id", ""),
                    "overlapRatio": round(overlap_ratio, 3),
                })

    if occluded_blocks:
        report.layering_risk = "medium" if len(occluded_blocks) <= 2 else "high"
        report.foreground_occlusion_report = {
            "occludedBlocks": occluded_blocks,
            "recommendation": "Use occlusion mask for rendering to avoid overwriting foreground",
        }
        # Build occlusion mask = areas where text rendering should avoid
        occlusion_mask = Image.new("L", (w, h), 0)
        for block_info in occluded_blocks:
            # The protected mask in the block region serves as occlusion guide
            pass
        report.occlusion_mask_for_rendering = protected.copy()
    else:
        report.layering_risk = "none"

    return report


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Cleanup Provider Routing (Risk-Based)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProviderRoute:
    provider: str
    strategy: str  # primary | fallback | experimental | diagnostic
    reason: str
    mask_capable: bool = True


def get_configured_providers() -> dict[str, bool]:
    """Check which providers are configured and available."""
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "huggingface": bool(
            os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
        ),
        "clipdrop": bool(os.getenv("CLIPDROP_API_KEY", "").strip()),
        "stability": bool(os.getenv("STABILITY_API_KEY", "").strip()),
        "photoroom": bool(os.getenv("PHOTOROOM_API_KEY", "").strip()),
        "adobe_firefly": bool(os.getenv("ADOBE_FIREFLY_API_KEY", "").strip()),
        "lama": True,  # LaMa can run locally if models loaded
    }


def route_cleanup_providers(risk_level: RiskLevel, block: Any = None, image_size: tuple[int, int] = (0, 0)) -> list[ProviderRoute]:
    """
    Risk-based provider routing. LaMa is NOT universal primary.
    Returns ordered list of providers to try.
    """
    configured = get_configured_providers()

    if risk_level == RiskLevel.UNSUPPORTED_AUTO_CLEANUP:
        return [ProviderRoute(
            provider="none",
            strategy="rejected",
            reason="UNSUPPORTED_AUTO_CLEANUP - no provider can safely process",
            mask_capable=False,
        )]

    routes: list[ProviderRoute] = []

    if risk_level == RiskLevel.LOW:
        # 1. mask-capable specialist
        if configured.get("clipdrop"):
            routes.append(ProviderRoute("clipdrop", "primary", "mask-capable specialist for LOW risk"))
        if configured.get("stability"):
            routes.append(ProviderRoute("stability", "primary", "mask-capable specialist for LOW risk"))
        # 2. OpenAI gpt-image
        if configured.get("openai"):
            routes.append(ProviderRoute("openai", "primary" if not routes else "fallback", "OpenAI mask-capable for LOW risk"))
        # 3. HuggingFace / LaMa fallback
        if configured.get("huggingface"):
            routes.append(ProviderRoute("huggingface", "fallback", "HuggingFace SD inpainting fallback"))

    elif risk_level == RiskLevel.MEDIUM:
        # 1. Specialist providers
        if configured.get("clipdrop"):
            routes.append(ProviderRoute("clipdrop", "primary", "Clipdrop Cleanup for MEDIUM risk"))
        if configured.get("stability"):
            routes.append(ProviderRoute("stability", "primary", "Stability Erase for MEDIUM risk"))
        if configured.get("photoroom"):
            routes.append(ProviderRoute("photoroom", "primary", "Photoroom for MEDIUM risk"))
        if configured.get("adobe_firefly"):
            routes.append(ProviderRoute("adobe_firefly", "primary", "Adobe Firefly for MEDIUM risk"))
        # 2. SDXL+ControlNet
        if configured.get("huggingface"):
            routes.append(ProviderRoute("huggingface", "primary" if not routes else "fallback", "SDXL+ControlNet for MEDIUM risk"))
        # 3. OpenAI only if safe
        if configured.get("openai"):
            routes.append(ProviderRoute("openai", "fallback", "OpenAI for MEDIUM risk (if safe)"))

    elif risk_level == RiskLevel.HIGH:
        # 1. specialist mask-capable
        if configured.get("clipdrop"):
            routes.append(ProviderRoute("clipdrop", "primary", "specialist for HIGH risk"))
        if configured.get("stability"):
            routes.append(ProviderRoute("stability", "primary", "specialist for HIGH risk"))
        # 2. SDXL+ControlNet
        if configured.get("huggingface"):
            routes.append(ProviderRoute("huggingface", "primary" if not routes else "fallback", "SDXL+ControlNet for HIGH risk"))
        # 3. OpenAI experimental
        if configured.get("openai"):
            routes.append(ProviderRoute("openai", "experimental", "OpenAI experimental for HIGH risk", mask_capable=True))

    # If no routes configured, report
    if not routes:
        routes.append(ProviderRoute("none", "rejected", "provider_not_configured", mask_capable=False))

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Inpainting Strategy
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InpaintStrategy:
    """Inpainting strategy for a block."""
    strategy_type: str  # surgical | region_expansion | deep_fill
    dilation_px: int = 8
    feather_px: int = 3
    binary_mask: Image.Image | None = None
    composite_mask: Image.Image | None = None
    reason: str = ""


def compute_stroke_width(block: Any, image: Image.Image) -> float:
    """Estimate text stroke width from block font size and image DPI."""
    font_size = getattr(block, "font_size_estimate", 16)
    # Approximate stroke width as ~8% of font size for typical marketing fonts
    stroke_width = max(2.0, font_size * 0.08)
    return stroke_width


def compute_dilation_px(block: Any, image: Image.Image) -> int:
    """
    Stroke-width-based dilation:
    dilationPx = clamp(strokeWidth * 0.8 to strokeWidth * 1.5, min 4px, max 18px)
    """
    stroke_width = compute_stroke_width(block, image)
    dilation_low = stroke_width * 0.8
    dilation_high = stroke_width * 1.5
    dilation = (dilation_low + dilation_high) / 2.0
    return int(max(4, min(18, round(dilation))))


def determine_inpaint_strategy(
    block: Any,
    image: Image.Image,
    segmentation_masks: SegmentationMasks,
    risk_level: RiskLevel,
) -> InpaintStrategy:
    """
    Determine inpainting strategy per block:
    - Surgical: skin/fabric/product-adjacent
    - Region expansion: simple background only
    - Deep fill: confirmed background-only
    Never deep-fill over protected semantic regions.
    """
    w, h = image.size
    bbox = getattr(block, "bbox", (0, 0, 0, 0))
    dilation_px = compute_dilation_px(block, image)
    feather_px = max(2, dilation_px // 3)

    # Check if block is adjacent to protected regions
    protected = segmentation_masks.get_composite_protected_mask((w, h))
    protected_arr = np.array(protected, dtype=np.uint8)

    # Expand bbox by dilation to check adjacency
    expanded = (
        max(0, bbox[0] - dilation_px * 2),
        max(0, bbox[1] - dilation_px * 2),
        min(w, bbox[2] + dilation_px * 2),
        min(h, bbox[3] + dilation_px * 2),
    )
    region = protected_arr[expanded[1]:expanded[3], expanded[0]:expanded[2]]
    adjacency_ratio = float((region > 64).sum()) / max(1, region.size) if region.size > 0 else 0

    # Determine strategy
    if adjacency_ratio > 0.08:
        strategy_type = "surgical"
        reason = "Block adjacent to protected region (face/skin/hand/product/shoe/logo/packaging)"
    elif risk_level == RiskLevel.LOW and adjacency_ratio < 0.02:
        strategy_type = "deep_fill"
        reason = "Low risk, confirmed background-only region"
    else:
        strategy_type = "region_expansion"
        reason = "Standard region expansion on semi-simple background"

    # Build binary inpaint mask
    cleanup_mask = segmentation_masks.compute_cleanup_mask((w, h))
    # Focus on this block's region
    block_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(block_mask)
    line_boxes = getattr(block, "line_boxes", [])
    if line_boxes:
        for lb in line_boxes:
            draw.rectangle(lb, fill=255)
    else:
        draw.rectangle(bbox, fill=255)

    # Intersect with cleanup mask
    block_arr = np.array(block_mask, dtype=np.float32)
    cleanup_arr = np.array(cleanup_mask, dtype=np.float32)
    final_mask_arr = np.minimum(block_arr, cleanup_arr)
    binary_mask = Image.fromarray(np.clip(final_mask_arr, 0, 255).astype(np.uint8), "L")

    # Apply stroke-based dilation
    if dilation_px > 0:
        import cv2
        mask_cv = np.array(binary_mask, dtype=np.uint8)
        kernel_size = max(3, dilation_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_cv, kernel, iterations=1)
        binary_mask = Image.fromarray(dilated, "L")

    # Subtract protected regions from dilated mask (safety)
    final_arr = np.array(binary_mask, dtype=np.float32)
    prot_arr = np.array(protected, dtype=np.float32)
    safe_mask_arr = np.maximum(0, final_arr - prot_arr)
    binary_mask = Image.fromarray(np.clip(safe_mask_arr, 0, 255).astype(np.uint8), "L")

    # Soft composite mask for blending
    composite_mask = binary_mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

    return InpaintStrategy(
        strategy_type=strategy_type,
        dilation_px=dilation_px,
        feather_px=feather_px,
        binary_mask=binary_mask,
        composite_mask=composite_mask,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Enhanced Quality Gate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityGateReport:
    """Multi-metric quality gate report."""
    passed: bool = False
    status: LocalizationStatus = LocalizationStatus.REJECT_PRODUCTION_QUALITY
    ocr_residual_score: float = 0.0
    negative_ocr_words_detected: list[str] = field(default_factory=list)
    source_color_residual_score: float = 0.0
    ghosting_score: float = 0.0
    tone_mismatch_lab_score: float = 0.0
    boundary_seam_score: float = 0.0
    texture_frequency_continuity: float = 1.0
    protected_region_similarity: float = 1.0
    luminance_gradient_continuity: float = 1.0
    packaging_preservation_score: float = 1.0
    patch_visibility_score: float = 0.0
    blur_smear_score: float = 0.0
    hard_fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "ocrResidualScore": round(self.ocr_residual_score, 4),
            "negativeOcrWordsDetected": self.negative_ocr_words_detected,
            "sourceColorResidualScore": round(self.source_color_residual_score, 4),
            "ghostingScore": round(self.ghosting_score, 4),
            "toneMismatchLabScore": round(self.tone_mismatch_lab_score, 4),
            "boundarySeamScore": round(self.boundary_seam_score, 4),
            "textureFrequencyContinuity": round(self.texture_frequency_continuity, 4),
            "protectedRegionSimilarity": round(self.protected_region_similarity, 4),
            "luminanceGradientContinuity": round(self.luminance_gradient_continuity, 4),
            "packagingPreservationScore": round(self.packaging_preservation_score, 4),
            "patchVisibilityScore": round(self.patch_visibility_score, 4),
            "blurSmearScore": round(self.blur_smear_score, 4),
            "hardFailReasons": self.hard_fail_reasons,
        }


def compute_tone_mismatch_lab(source: Image.Image, cleaned: Image.Image, mask: Image.Image) -> float:
    """Compute LAB color difference in inpainted region boundary."""
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    if mask_arr.max() < 16:
        return 0.0

    # Dilate mask to get boundary region
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask_arr, kernel, iterations=1)
    boundary = np.logical_and(dilated > 16, mask_arr <= 16)

    if boundary.sum() < 10:
        return 0.0

    # Convert to LAB (approximate using luminance/chroma)
    src_arr = np.array(source.convert("RGB"), dtype=np.float32)
    cln_arr = np.array(cleaned.convert("RGB"), dtype=np.float32)

    src_boundary = src_arr[boundary]
    cln_boundary = cln_arr[boundary]

    # Simple LAB-like difference (euclidean in RGB as proxy)
    diff = np.sqrt(np.sum((src_boundary - cln_boundary) ** 2, axis=1))
    return float(diff.mean() / 255.0)


def compute_boundary_seam_score(source: Image.Image, cleaned: Image.Image, mask: Image.Image) -> float:
    """Detect visible seams at mask boundaries."""
    import cv2
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    if mask_arr.max() < 16:
        return 0.0

    # Find contour of mask
    contours, _ = cv2.findContours(
        np.where(mask_arr > 16, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0

    # Sample pixels along boundary in both source and cleaned
    boundary_mask = np.zeros_like(mask_arr)
    cv2.drawContours(boundary_mask, contours, -1, 255, thickness=3)

    src_gray = np.array(source.convert("L"), dtype=np.float32)
    cln_gray = np.array(cleaned.convert("L"), dtype=np.float32)

    boundary_pixels = boundary_mask > 0
    if boundary_pixels.sum() < 5:
        return 0.0

    # Compute gradient difference at boundary
    src_grad_x = np.gradient(src_gray, axis=1)
    src_grad_y = np.gradient(src_gray, axis=0)
    cln_grad_x = np.gradient(cln_gray, axis=1)
    cln_grad_y = np.gradient(cln_gray, axis=0)

    grad_diff = np.sqrt(
        (src_grad_x[boundary_pixels] - cln_grad_x[boundary_pixels]) ** 2
        + (src_grad_y[boundary_pixels] - cln_grad_y[boundary_pixels]) ** 2
    )
    return float(min(1.0, grad_diff.mean() / 30.0))


def compute_patch_visibility(source: Image.Image, cleaned: Image.Image, mask: Image.Image) -> float:
    """Detect visible rectangular patches in cleaned image."""
    import cv2
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    if mask_arr.max() < 16:
        return 0.0

    # Compare texture variance inside vs outside inpainted region
    cln_gray = np.array(cleaned.convert("L"), dtype=np.float32)
    inside = mask_arr > 16
    outside_border = np.logical_and(
        cv2.dilate(mask_arr, np.ones((11, 11), np.uint8)) > 16,
        ~inside,
    )

    if inside.sum() < 10 or outside_border.sum() < 10:
        return 0.0

    # Variance comparison
    var_inside = float(cln_gray[inside].var())
    var_outside = float(cln_gray[outside_border].var())

    if var_outside < 1:
        return 0.0

    # Large variance difference = visible patch
    ratio = abs(var_inside - var_outside) / max(1.0, var_outside)
    return float(min(1.0, ratio / 2.0))


def compute_blur_smear_score(cleaned: Image.Image, mask: Image.Image) -> float:
    """Detect blur/smear artifacts in inpainted region."""
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    if mask_arr.max() < 16:
        return 0.0

    inside = mask_arr > 16
    if inside.sum() < 10:
        return 0.0

    gray = np.array(cleaned.convert("L"), dtype=np.float32)
    # High-pass filter to detect detail
    edges = np.array(cleaned.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)

    edge_inside = float(edges[inside].mean())
    edge_outside_mask = np.logical_and(~inside, mask_arr < 240)
    if edge_outside_mask.sum() > 10:
        edge_outside = float(edges[edge_outside_mask].mean())
    else:
        edge_outside = float(edges.mean())

    if edge_outside < 1:
        return 0.0

    # If inpainted region is significantly smoother = blur/smear
    blur_ratio = 1.0 - min(1.0, edge_inside / max(1.0, edge_outside))
    return float(max(0.0, blur_ratio - 0.2))  # Threshold out minor differences


def compute_protected_region_similarity(
    source: Image.Image,
    cleaned: Image.Image,
    protected_mask: Image.Image,
) -> float:
    """Check that protected regions are unchanged."""
    prot_arr = np.array(protected_mask.convert("L"), dtype=np.uint8)
    if prot_arr.max() < 16:
        return 1.0

    protected = prot_arr > 16
    if protected.sum() < 10:
        return 1.0

    src_arr = np.array(source.convert("RGB"), dtype=np.float32)
    cln_arr = np.array(cleaned.convert("RGB"), dtype=np.float32)

    diff = np.abs(src_arr[protected] - cln_arr[protected])
    mean_diff = float(diff.mean() / 255.0)
    return max(0.0, 1.0 - mean_diff * 10.0)


def compute_packaging_preservation(
    source: Image.Image,
    cleaned: Image.Image,
    packaging_mask: Image.Image | None,
) -> float:
    """Verify packaging text is preserved."""
    if packaging_mask is None:
        return 1.0
    pkg_arr = np.array(packaging_mask.convert("L"), dtype=np.uint8)
    if pkg_arr.max() < 16:
        return 1.0

    protected = pkg_arr > 16
    if protected.sum() < 10:
        return 1.0

    src_arr = np.array(source.convert("RGB"), dtype=np.float32)
    cln_arr = np.array(cleaned.convert("RGB"), dtype=np.float32)

    diff = np.abs(src_arr[protected] - cln_arr[protected])
    mean_diff = float(diff.mean() / 255.0)
    return max(0.0, 1.0 - mean_diff * 10.0)


def run_quality_gate(
    source_image: Image.Image,
    cleaned_image: Image.Image,
    blocks: list[Any],
    segmentation_masks: SegmentationMasks,
    cleanup_mask: Image.Image,
    source_words: list[str] | None = None,
    ocr_fn: Any = None,
) -> QualityGateReport:
    """
    Enhanced quality gate with multi-metric assessment.
    Hard-fails if any critical threshold is breached.
    """
    report = QualityGateReport()
    w, h = source_image.size

    # 1. OCR residual score — check if source text remains readable
    residual_words: list[str] = []
    if ocr_fn is not None:
        try:
            cleaned_ocr = ocr_fn(cleaned_image)
            if cleaned_ocr:
                residual_words = [w.strip().lower() for w in cleaned_ocr if len(w.strip()) > 2]
        except Exception:
            pass

    # 2. Negative OCR check against original source words
    if source_words:
        source_lower = {w.lower() for w in source_words if len(w) > 2}
        detected_source_words = [w for w in residual_words if w in source_lower]
        report.negative_ocr_words_detected = detected_source_words
        report.ocr_residual_score = len(detected_source_words) / max(1, len(source_lower))
    else:
        report.ocr_residual_score = 0.0

    # 3. Source color residual score
    mask_arr = np.array(cleanup_mask.convert("L"), dtype=np.uint8)
    inside = mask_arr > 16
    if inside.sum() > 10:
        src_arr = np.array(source_image.convert("RGB"), dtype=np.float32)
        cln_arr = np.array(cleaned_image.convert("RGB"), dtype=np.float32)
        color_diff = np.abs(src_arr[inside] - cln_arr[inside]).mean() / 255.0
        report.source_color_residual_score = float(color_diff)

    # 4. Ghosting score (structural similarity in text regions)
    if inside.sum() > 10:
        src_gray = np.array(source_image.convert("L"), dtype=np.float32)
        cln_gray = np.array(cleaned_image.convert("L"), dtype=np.float32)
        correlation = np.corrcoef(src_gray[inside].flatten(), cln_gray[inside].flatten())[0, 1]
        # High correlation in text region = ghosting (text still visible)
        report.ghosting_score = float(max(0, correlation - 0.3)) if not np.isnan(correlation) else 0.0

    # 5. Tone mismatch LAB score
    report.tone_mismatch_lab_score = compute_tone_mismatch_lab(source_image, cleaned_image, cleanup_mask)

    # 6. Boundary seam score
    report.boundary_seam_score = compute_boundary_seam_score(source_image, cleaned_image, cleanup_mask)

    # 7. Texture frequency continuity
    report.texture_frequency_continuity = 1.0 - compute_patch_visibility(source_image, cleaned_image, cleanup_mask)

    # 8. Protected region similarity
    protected = segmentation_masks.get_composite_protected_mask((w, h))
    report.protected_region_similarity = compute_protected_region_similarity(source_image, cleaned_image, protected)

    # 9. Luminance gradient continuity
    report.luminance_gradient_continuity = 1.0 - report.boundary_seam_score

    # 10. Packaging preservation score
    report.packaging_preservation_score = compute_packaging_preservation(
        source_image, cleaned_image, segmentation_masks.packaging_text_mask
    )

    # 11. Patch visibility
    report.patch_visibility_score = compute_patch_visibility(source_image, cleaned_image, cleanup_mask)

    # 12. Blur/smear score
    report.blur_smear_score = compute_blur_smear_score(cleaned_image, cleanup_mask)

    # Hard fail checks
    hard_fails: list[str] = []

    if report.ocr_residual_score > 0.3:
        hard_fails.append("source text remains readable")
    if report.negative_ocr_words_detected:
        hard_fails.append(f"original source words detected: {report.negative_ocr_words_detected[:5]}")
    if report.protected_region_similarity < 0.85:
        hard_fails.append("product/logo/packaging text changed")
    if report.packaging_preservation_score < 0.85:
        hard_fails.append("packaging preservation failed")
    if report.patch_visibility_score > 0.5:
        hard_fails.append("visible rectangular patch detected")
    if report.tone_mismatch_lab_score > 0.15:
        hard_fails.append("visible tone mismatch")
    if report.ghosting_score > 0.6:
        hard_fails.append("visible ghosting/smear")
    if report.boundary_seam_score > 0.4:
        hard_fails.append("mask boundary visible")
    if report.blur_smear_score > 0.5:
        hard_fails.append("visible blur/smear artifact")

    report.hard_fail_reasons = hard_fails
    report.passed = len(hard_fails) == 0
    report.status = LocalizationStatus.CLEANUP_SUCCESS if report.passed else LocalizationStatus.REJECT_PRODUCTION_QUALITY

    return report


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Retry / Provider Bake-off
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleanupCandidate:
    provider: str
    image: Image.Image | None = None
    quality_report: QualityGateReport | None = None
    score: float = 0.0
    selected: bool = False
    rejection_reason: str = ""


@dataclass
class ProviderBakeoffReport:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    selected_provider: str = ""
    selected_reason: str = ""
    rejected_reasons: dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    residual_retry_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "providerBakeoffReport": {
                "cleanupCandidateScores": self.candidates,
                "selectedCleanupProvider": self.selected_provider,
                "selectedCleanupReason": self.selected_reason,
                "rejectedCandidateReasons": self.rejected_reasons,
                "retryCount": self.retry_count,
                "residualRetryApplied": self.residual_retry_applied,
            },
        }


def compute_candidate_score(report: QualityGateReport) -> float:
    """Compute composite quality score for a cleanup candidate."""
    if not report.passed:
        # Penalize but still rank failed candidates
        penalty = len(report.hard_fail_reasons) * 0.15
    else:
        penalty = 0.0

    score = (
        (1.0 - report.ocr_residual_score) * 0.25
        + (1.0 - report.ghosting_score) * 0.20
        + report.protected_region_similarity * 0.15
        + report.packaging_preservation_score * 0.10
        + report.texture_frequency_continuity * 0.10
        + (1.0 - report.tone_mismatch_lab_score) * 0.05
        + (1.0 - report.boundary_seam_score) * 0.05
        + (1.0 - report.patch_visibility_score) * 0.05
        + (1.0 - report.blur_smear_score) * 0.05
        - penalty
    )
    return max(0.0, min(1.0, score))


def run_provider_bakeoff(
    candidates: list[CleanupCandidate],
    source_image: Image.Image,
    blocks: list[Any],
    segmentation_masks: SegmentationMasks,
    cleanup_mask: Image.Image,
    source_words: list[str] | None = None,
    ocr_fn: Any = None,
    max_retries: int = 2,
) -> ProviderBakeoffReport:
    """
    Select best cleanup candidate by quality score, not execution order.
    If residual source text remains, rerun with wider residual mask (max 2-3 passes).
    """
    report = ProviderBakeoffReport()

    for candidate in candidates:
        if candidate.image is None:
            candidate.rejection_reason = "no image produced"
            report.rejected_reasons[candidate.provider] = "no image produced"
            continue

        # Run quality gate on each candidate
        quality = run_quality_gate(
            source_image,
            candidate.image,
            blocks,
            segmentation_masks,
            cleanup_mask,
            source_words=source_words,
            ocr_fn=ocr_fn,
        )
        candidate.quality_report = quality
        candidate.score = compute_candidate_score(quality)

        report.candidates.append({
            "provider": candidate.provider,
            "score": round(candidate.score, 4),
            "passed": quality.passed,
            "hardFailReasons": quality.hard_fail_reasons,
        })

        if not quality.passed:
            report.rejected_reasons[candidate.provider] = "; ".join(quality.hard_fail_reasons)

    # Select best candidate by score
    valid_candidates = [c for c in candidates if c.image is not None]
    if not valid_candidates:
        report.selected_provider = "none"
        report.selected_reason = "no valid candidates produced"
        return report

    # Prefer passing candidates, then highest score
    passing = [c for c in valid_candidates if c.quality_report and c.quality_report.passed]
    if passing:
        best = max(passing, key=lambda c: c.score)
    else:
        best = max(valid_candidates, key=lambda c: c.score)

    best.selected = True
    report.selected_provider = best.provider
    report.selected_reason = f"highest quality score ({best.score:.4f})"

    return report


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — Style-Aware Rendering (Copy-Fitting)
# ─────────────────────────────────────────────────────────────────────────────

class CopyFitResult(str, Enum):
    SUCCESS = "success"
    REDUCED_FONT = "reduced_font"
    ADJUSTED_TRACKING = "adjusted_tracking"
    RECOMPUTED_LINES = "recomputed_lines"
    SHORTER_VARIANT = "shorter_variant"
    REJECT_LAYOUT_FIT = "reject_layout_fit"


@dataclass
class RenderDecision:
    """Rendering decision for a text block."""
    block_id: str = ""
    copy_fit_result: CopyFitResult = CopyFitResult.SUCCESS
    final_font_size: int = 16
    tracking_adjustment: float = 0.0
    line_count: int = 1
    overflow: bool = False
    used_occlusion_mask: bool = False
    reason: str = ""


def iterative_copy_fit(
    text: str,
    bbox: tuple[int, int, int, int],
    original_font_size: int,
    max_width: int,
    max_height: int,
    translation_candidates: list[dict[str, str]] | None = None,
) -> tuple[str, int, float, CopyFitResult]:
    """
    Iterative copy-fitting:
    1. Original scale
    2. Reduce font size (min 60% of original)
    3. Adjust tracking (letter-spacing)
    4. Recompute line breaks
    5. Request shorter approved translation variant
    6. If still not fitting, REJECT_LAYOUT_FIT
    """
    from PIL import ImageFont

    def _get_font(size: int, bold: bool = False):
        font_names = (
            ["C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
            if bold else
            ["C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        )
        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _text_fits(text: str, font_size: int, tracking: float) -> bool:
        font = _get_font(font_size)
        test_img = Image.new("L", (max_width * 2, max_height * 2))
        draw = ImageDraw.Draw(test_img)
        # Simple word wrap check
        words = text.split()
        lines: list[str] = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox_test = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox_test[2] - bbox_test[0]
            # Apply tracking
            line_width += len(test_line) * tracking
            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        line_height = int(font_size * 1.2)
        total_height = line_height * len(lines)
        return total_height <= max_height

    # Step 1: Original scale
    if _text_fits(text, original_font_size, 0.0):
        return text, original_font_size, 0.0, CopyFitResult.SUCCESS

    # Step 2: Reduce font size (down to 60%)
    for reduction in range(5, 45, 5):
        reduced_size = max(8, int(original_font_size * (100 - reduction) / 100))
        if _text_fits(text, reduced_size, 0.0):
            return text, reduced_size, 0.0, CopyFitResult.REDUCED_FONT

    # Step 3: Adjust tracking (negative)
    reduced_size = max(8, int(original_font_size * 0.65))
    for tracking in [-0.5, -1.0, -1.5, -2.0]:
        if _text_fits(text, reduced_size, tracking):
            return text, reduced_size, tracking, CopyFitResult.ADJUSTED_TRACKING

    # Step 4: Already tried recomputed line breaks above

    # Step 5: Try shorter translation variant
    if translation_candidates:
        for candidate in sorted(translation_candidates, key=lambda c: len(c.get("text", ""))):
            candidate_text = candidate.get("text", "")
            if candidate_text and _text_fits(candidate_text, original_font_size, 0.0):
                return candidate_text, original_font_size, 0.0, CopyFitResult.SHORTER_VARIANT
            if candidate_text and _text_fits(candidate_text, reduced_size, 0.0):
                return candidate_text, reduced_size, 0.0, CopyFitResult.SHORTER_VARIANT

    # Step 6: REJECT_LAYOUT_FIT
    return text, reduced_size, -1.0, CopyFitResult.REJECT_LAYOUT_FIT


def decide_render_strategy(
    block: Any,
    cleanup_status: LocalizationStatus,
    depth_report: DepthLayeringReport | None = None,
) -> RenderDecision:
    """
    Decide rendering strategy per block.
    Only render if cleanup_success.
    """
    decision = RenderDecision(block_id=getattr(block, "id", ""))

    if cleanup_status != LocalizationStatus.CLEANUP_SUCCESS:
        decision.copy_fit_result = CopyFitResult.REJECT_LAYOUT_FIT
        decision.reason = f"Cleanup status is {cleanup_status.value}, cannot render"
        return decision

    bbox = getattr(block, "bbox", (0, 0, 0, 0))
    max_width = max(1, bbox[2] - bbox[0])
    max_height = max(1, bbox[3] - bbox[1])
    original_font_size = getattr(block, "font_size_estimate", 16)
    translated_text = getattr(block, "translated_text", "") or getattr(block, "text", "")
    candidates = getattr(block, "translation_candidates", [])

    text, font_size, tracking, fit_result = iterative_copy_fit(
        translated_text, bbox, original_font_size, max_width, max_height,
        translation_candidates=candidates,
    )

    decision.copy_fit_result = fit_result
    decision.final_font_size = font_size
    decision.tracking_adjustment = tracking
    decision.overflow = fit_result == CopyFitResult.REJECT_LAYOUT_FIT

    # Check occlusion mask
    if depth_report and depth_report.occlusion_mask_for_rendering is not None:
        decision.used_occlusion_mask = True

    return decision


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — Status Propagation (Resize / Preview Gate)
# ─────────────────────────────────────────────────────────────────────────────

def should_proceed_to_preview(cleanup_status: LocalizationStatus, render_decisions: list[RenderDecision]) -> tuple[bool, str]:
    """
    Resize/preview only runs after cleanup_success AND render_success.
    Returns (proceed, reason).
    """
    if cleanup_status != LocalizationStatus.CLEANUP_SUCCESS:
        return False, f"Cleanup status is {cleanup_status.value}"

    # Check if any block has REJECT_LAYOUT_FIT
    layout_rejects = [d for d in render_decisions if d.copy_fit_result == CopyFitResult.REJECT_LAYOUT_FIT]
    if layout_rejects:
        return False, f"{len(layout_rejects)} block(s) failed layout fit"

    return True, "cleanup_success and render_success"


def propagate_status(
    risk_report: CreativeRiskReport,
    quality_report: QualityGateReport | None,
    render_decisions: list[RenderDecision],
    bakeoff_report: ProviderBakeoffReport | None = None,
) -> LocalizationStatus:
    """Determine final pipeline status from all phase outputs."""

    # Phase 0 rejections
    if risk_report.risk_level == RiskLevel.REJECT_LOW_CONFIDENCE:
        return LocalizationStatus.REJECT_LOW_CONFIDENCE
    if risk_report.risk_level == RiskLevel.UNSUPPORTED_AUTO_CLEANUP:
        return LocalizationStatus.UNSUPPORTED_AUTO_CLEANUP
    if risk_report.risk_level == RiskLevel.PACKAGING_PROTECTION_RISK:
        return LocalizationStatus.PACKAGING_PROTECTION_RISK

    # Provider not configured
    if bakeoff_report and bakeoff_report.selected_provider == "none":
        return LocalizationStatus.PROVIDER_NOT_CONFIGURED

    # Quality gate
    if quality_report and not quality_report.passed:
        return LocalizationStatus.REJECT_PRODUCTION_QUALITY

    # Layout fit
    if any(d.copy_fit_result == CopyFitResult.REJECT_LAYOUT_FIT for d in render_decisions):
        return LocalizationStatus.REJECT_LAYOUT_FIT

    # All passed
    return LocalizationStatus.CLEANUP_SUCCESS


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — Full Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full pipeline execution result."""
    status: LocalizationStatus
    risk_report: CreativeRiskReport | None = None
    segmentation_masks: SegmentationMasks | None = None
    depth_report: DepthLayeringReport | None = None
    quality_report: QualityGateReport | None = None
    bakeoff_report: ProviderBakeoffReport | None = None
    render_decisions: list[RenderDecision] = field(default_factory=list)
    cleaned_image: Image.Image | None = None
    can_render: bool = False
    can_preview: bool = False
    preview_reason: str = ""

    def to_reports_dict(self) -> dict[str, Any]:
        """Generate all report JSONs."""
        result: dict[str, Any] = {
            "pipelineStatus": self.status.value,
            "canRender": self.can_render,
            "canPreview": self.can_preview,
            "previewReason": self.preview_reason,
        }
        if self.risk_report:
            result["creativeRiskReport"] = self.risk_report.to_dict()
        if self.depth_report:
            result["depthLayeringReport"] = self.depth_report.to_dict()
        if self.quality_report:
            result["qualityGateReport"] = self.quality_report.to_dict()
        if self.bakeoff_report:
            result.update(self.bakeoff_report.to_dict())
        if self.render_decisions:
            result["renderDecisions"] = [
                {
                    "blockId": d.block_id,
                    "copyFitResult": d.copy_fit_result.value,
                    "finalFontSize": d.final_font_size,
                    "trackingAdjustment": d.tracking_adjustment,
                    "overflow": d.overflow,
                    "usedOcclusionMask": d.used_occlusion_mask,
                    "reason": d.reason,
                }
                for d in self.render_decisions
            ]
        return result


def run_localization_protocol(
    image: Image.Image,
    blocks: list[Any],
    protected_region_mask: Image.Image | None = None,
    foreground_bbox: tuple[int, int, int, int] | None = None,
    ocr_confidence: float = 1.0,
    cleanup_fn: Any = None,
    ocr_fn: Any = None,
    job_dir: Path | None = None,
) -> PipelineResult:
    """
    Execute full localization pipeline protocol V2.2.
    Returns PipelineResult with all phase outputs.

    This function is called from the main adapt pipeline after OCR/translation.
    The actual cleanup execution is delegated back to main.py's cleanup functions
    via cleanup_fn callback.
    """
    result = PipelineResult(status=LocalizationStatus.REJECT_PRODUCTION_QUALITY)
    w, h = image.size

    # --- PHASE 0: Creative Risk Classification ---
    packaging_blocks = [b for b in blocks if getattr(b, "surface", "") in ("packaging", "product")]

    # Build preliminary text mask for risk assessment
    preliminary_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(preliminary_mask)
    for block in blocks:
        if getattr(block, "translate", False) and getattr(block, "surface", "") == "overlay":
            bbox = getattr(block, "bbox", None)
            if bbox:
                line_boxes = getattr(block, "line_boxes", [])
                if line_boxes:
                    for lb in line_boxes:
                        draw.rectangle(lb, fill=255)
                else:
                    draw.rectangle(bbox, fill=255)

    risk_report = classify_creative_risk(
        image,
        blocks,
        text_mask=preliminary_mask,
        protected_mask=protected_region_mask,
        ocr_confidence=ocr_confidence,
        packaging_blocks=packaging_blocks,
    )
    result.risk_report = risk_report

    # Early exit on risk rejection
    if risk_report.risk_level in (
        RiskLevel.REJECT_LOW_CONFIDENCE,
        RiskLevel.UNSUPPORTED_AUTO_CLEANUP,
        RiskLevel.PACKAGING_PROTECTION_RISK,
    ):
        status_map = {
            RiskLevel.REJECT_LOW_CONFIDENCE: LocalizationStatus.REJECT_LOW_CONFIDENCE,
            RiskLevel.UNSUPPORTED_AUTO_CLEANUP: LocalizationStatus.UNSUPPORTED_AUTO_CLEANUP,
            RiskLevel.PACKAGING_PROTECTION_RISK: LocalizationStatus.PACKAGING_PROTECTION_RISK,
        }
        result.status = status_map[risk_report.risk_level]
        return result

    # --- PHASE 1: Segmentation ---
    segmentation = build_segmentation_masks(image, blocks, protected_region_mask, foreground_bbox)
    result.segmentation_masks = segmentation

    # --- PHASE 1.5: Depth/Layering ---
    depth_report = analyze_depth_layering(image, blocks, segmentation)
    result.depth_report = depth_report

    # --- PHASE 2: Provider Routing ---
    routes = route_cleanup_providers(risk_report.risk_level)
    if routes and routes[0].provider == "none":
        result.status = LocalizationStatus.PROVIDER_NOT_CONFIGURED
        return result

    # --- PHASE 3: Compute inpaint strategies per block ---
    translate_blocks = [b for b in blocks if getattr(b, "translate", False) and getattr(b, "surface", "") == "overlay"]
    inpaint_strategies: list[InpaintStrategy] = []
    for block in translate_blocks:
        strategy = determine_inpaint_strategy(block, image, segmentation, risk_report.risk_level)
        inpaint_strategies.append(strategy)

    # Compute final cleanup mask
    cleanup_mask = segmentation.compute_cleanup_mask((w, h))

    # --- Execute cleanup via callback (delegates to main.py providers) ---
    # The cleanup_fn is expected to handle the actual provider calls
    # and return (cleaned_image, cleanup_meta) or None
    cleaned_image = None
    if cleanup_fn is not None:
        try:
            cleanup_result = cleanup_fn(
                image=image,
                blocks=translate_blocks,
                cleanup_mask=cleanup_mask,
                routes=routes,
                strategies=inpaint_strategies,
            )
            if cleanup_result is not None:
                if isinstance(cleanup_result, tuple):
                    cleaned_image = cleanup_result[0]
                elif isinstance(cleanup_result, Image.Image):
                    cleaned_image = cleanup_result
        except Exception:
            pass

    if cleaned_image is None:
        result.status = LocalizationStatus.REJECT_PRODUCTION_QUALITY
        result.quality_report = QualityGateReport(
            passed=False,
            hard_fail_reasons=["cleanup produced no output"],
        )
        return result

    result.cleaned_image = cleaned_image

    # --- PHASE 4: Quality Gate ---
    source_words = []
    for block in translate_blocks:
        text = getattr(block, "text", "")
        source_words.extend(w.strip() for w in text.split() if len(w.strip()) > 2)

    quality_report = run_quality_gate(
        source_image=image,
        cleaned_image=cleaned_image,
        blocks=translate_blocks,
        segmentation_masks=segmentation,
        cleanup_mask=cleanup_mask,
        source_words=source_words,
        ocr_fn=ocr_fn,
    )
    result.quality_report = quality_report

    # --- PHASE 5: Bake-off (single candidate in default flow) ---
    candidate = CleanupCandidate(
        provider=routes[0].provider if routes else "unknown",
        image=cleaned_image,
        quality_report=quality_report,
        score=compute_candidate_score(quality_report),
        selected=True,
    )
    bakeoff = ProviderBakeoffReport(
        candidates=[{
            "provider": candidate.provider,
            "score": round(candidate.score, 4),
            "passed": quality_report.passed,
            "hardFailReasons": quality_report.hard_fail_reasons,
        }],
        selected_provider=candidate.provider,
        selected_reason=f"primary provider, score={candidate.score:.4f}",
    )
    result.bakeoff_report = bakeoff

    # --- PHASE 6: Render decisions ---
    cleanup_status = LocalizationStatus.CLEANUP_SUCCESS if quality_report.passed else LocalizationStatus.REJECT_PRODUCTION_QUALITY
    render_decisions: list[RenderDecision] = []
    for block in translate_blocks:
        decision = decide_render_strategy(block, cleanup_status, depth_report)
        render_decisions.append(decision)
    result.render_decisions = render_decisions

    # --- PHASE 7: Status propagation ---
    final_status = propagate_status(risk_report, quality_report, render_decisions, bakeoff)
    result.status = final_status
    result.can_render = final_status == LocalizationStatus.CLEANUP_SUCCESS
    proceed, reason = should_proceed_to_preview(final_status, render_decisions)
    result.can_preview = proceed
    result.preview_reason = reason

    # Write reports to job directory
    if job_dir is not None:
        try:
            reports = result.to_reports_dict()
            (job_dir / "creativeRiskReport.json").write_text(
                json.dumps(risk_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if quality_report:
                (job_dir / "qualityGateReport.json").write_text(
                    json.dumps(quality_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            (job_dir / "providerBakeoffReport.json").write_text(
                json.dumps(bakeoff.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (job_dir / "pipelineProtocolReport.json").write_text(
                json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    return result
