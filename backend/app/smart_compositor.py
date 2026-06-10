from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

try:
    from app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer
except ImportError:
    from backend.app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer


ImageRenderer = Callable[[Image.Image, int, int, ReframePlan, VisualAnalysis], tuple[Image.Image, dict[str, Any]]]
FallbackRenderer = Callable[[Image.Image, int, int], Image.Image]
TextRenderer = Callable[[Image.Image, list[Any]], Image.Image]
_REMBG_SESSION: Any | None = None


def _placement_preserves_creative_brand_layers(placement_id: str | None) -> bool:
    """Keep original creative logos only for formats where the ad unit has no native account chrome."""
    pid = (placement_id or "").strip().lower()
    return pid.startswith("gdn-") or pid.startswith("google-responsive") or pid == "custom-display"


def _placement_is_display_ad(placement_id: str | None) -> bool:
    return _placement_preserves_creative_brand_layers(placement_id)


def _resize_provider_must_not_see_brand_layers(placement_id: str | None) -> bool:
    """AI background/outpaint providers should never rasterize brand marks for social placements."""
    return not _placement_preserves_creative_brand_layers(placement_id)


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(left)))
    top = max(0, min(height - 1, int(top)))
    right = max(left + 1, min(width, int(right)))
    bottom = max(top + 1, min(height, int(bottom)))
    return left, top, right, bottom


def _prepare_foreground_rgba_crop(crop: Image.Image) -> Image.Image:
    return crop.convert("RGBA")


def _layer_by_id(analysis: VisualAnalysis) -> dict[str, VisualLayer]:
    layers: list[VisualLayer] = [
        *analysis.product_layers,
        *analysis.text_layers,
        *analysis.logo_layers,
        *analysis.other_layers,
    ]
    return {layer.id: layer for layer in layers}


def build_overlay_text_mask(source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    masked_layers = 0
    for layer in analysis.marketing_text_layers:
        left, top, right, bottom = layer.bbox.to_pixel_box(source.width, source.height)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        pad_x = max(3, min(22, int(round(box_w * 0.08))))
        pad_y = max(2, min(18, int(round(box_h * 0.22))))
        draw.rectangle(
            _clip_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), source.width, source.height),
            fill=255,
        )
        masked_layers += 1
    if masked_layers:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.55))
    return mask, {"maskedTextLayers": masked_layers, "textMaskMode": "bbox-dilated-overlay-text"}


def build_compositor_background_mask(source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    mask, mask_meta = build_overlay_text_mask(source, analysis)
    draw = ImageDraw.Draw(mask)
    removed_foreground = 0
    for layer in [*analysis.product_layers, *analysis.logo_layers]:
        left, top, right, bottom = layer.bbox.to_pixel_box(source.width, source.height)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        pad_x = max(4, min(30, int(round(box_w * 0.05))))
        pad_y = max(4, min(30, int(round(box_h * 0.05))))
        draw.rectangle(
            _clip_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), source.width, source.height),
            fill=255,
        )
        removed_foreground += 1
    if removed_foreground:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.75))
    return mask, {
        **mask_meta,
        "removedForegroundLayers": removed_foreground,
        "backgroundMaskMode": "text-product-logo-removal",
    }


def clean_overlay_text_only(source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    mask, mask_meta = build_overlay_text_mask(source, analysis)
    if not mask_meta["maskedTextLayers"]:
        return source.convert("RGB"), mask, {**mask_meta, "foregroundTextCleanup": "skipped_no_overlay_text"}
    try:
        import cv2

        source_np = np.array(source.convert("RGB"), dtype=np.uint8)
        mask_np = (np.array(mask, dtype=np.uint8) > 24).astype(np.uint8) * 255
        cleaned = cv2.inpaint(source_np, mask_np, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB"), Image.fromarray(mask_np, "L"), {
            **mask_meta,
            "foregroundTextCleanup": "opencv_telea_text_only",
        }
    except Exception as exc:
        return source.convert("RGB"), mask, {
            **mask_meta,
            "foregroundTextCleanup": "failed_passthrough",
            "foregroundTextCleanupError": str(exc),
        }


def clean_compositor_background_source(source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    mask, mask_meta = build_compositor_background_mask(source, analysis)
    if not mask_meta["maskedTextLayers"] and not mask_meta["removedForegroundLayers"]:
        return source.convert("RGB"), mask, {**mask_meta, "backgroundCleanup": "skipped_no_foreground"}
    try:
        import cv2

        source_np = np.array(source.convert("RGB"), dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 24).astype(np.uint8) * 255
        cleaned = cv2.inpaint(source_np, mask_np, 7, cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB"), Image.fromarray(mask_np, "L"), {
            **mask_meta,
            "backgroundCleanup": "opencv_telea_background_only",
        }
    except Exception as exc:
        return source.convert("RGB"), mask, {
            **mask_meta,
            "backgroundCleanup": "failed_passthrough",
            "backgroundCleanupError": str(exc),
        }


def build_resize_provider_safe_seed(
    source: Image.Image,
    analysis: VisualAnalysis,
    *,
    placement_id: str | None,
    remove_products: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Return an image seed that is safe for generative background completion.

    Resize output must never ask an image model to draw typography or brand marks.
    The provider can complete scene/background pixels, while product, brand and
    marketing layers are composited deterministically afterwards.
    """
    visual_box, _visual_meta = _detect_role_aware_visual_box(source, analysis)
    parts = _partition_resize_layers(source, analysis, visual_box)
    removal_boxes: list[tuple[int, int, int, int]] = [
        *parts["primary"],
        *parts["secondary"],
        *parts.get("trust_badge", []),
        *[
            _expand_text_box_line_region(source, layer.bbox.to_pixel_box(source.width, source.height))
            for layer in analysis.marketing_text_layers
        ],
    ]
    if _resize_provider_must_not_see_brand_layers(placement_id):
        removal_boxes.extend(parts["brand"])
    if remove_products:
        removal_boxes.extend(_collect_visual_element_boxes(source, analysis, visual_box))
    if not removal_boxes:
        return source.convert("RGB"), {
            "resizeProviderSeed": "source_no_raster_text_or_brand_removal_needed",
            "resizeProviderSeedRemovedLayerCount": 0,
        }
    seed = _inpaint_rectangular_overlays(source, removal_boxes)
    return seed.convert("RGB"), {
        "resizeProviderSeed": "text_and_social_brand_removed_before_ai_background",
        "resizeProviderSeedRemovedLayerCount": len(removal_boxes),
        "resizeProviderSeedRemovedBrandLayers": 0 if not _resize_provider_must_not_see_brand_layers(placement_id) else len(parts["brand"]),
        "resizeProviderSeedRemovedMarketingLayers": len(parts["primary"]) + len(parts["secondary"]),
        "resizeProviderSeedRemovedTrustBadges": len(parts.get("trust_badge", [])),
    }


def render_background(
    clean_source: Image.Image,
    width: int,
    height: int,
    plan: ReframePlan,
    analysis: VisualAnalysis,
    *,
    outpaint_renderer: ImageRenderer | None,
    fallback_renderer: FallbackRenderer,
) -> tuple[Image.Image, dict[str, Any]]:
    should_outpaint = plan.expansion.strategy == ExpansionStrategy.OPENAI_OUTPAINT or plan.expansion.requires_ai
    texture = float(getattr(analysis.background, "texture_complexity", 0.0) or 0.0)
    if not should_outpaint and texture < float(__import__("os").getenv("ADAPTIFAI_COMPOSITOR_AI_TEXTURE_THRESHOLD", "0.46")):
        return build_deterministic_background_canvas(clean_source, width, height), {
            "provider": "local",
            "strategy": "deterministic_background_canvas",
            "backgroundSource": "clean_base",
            "productionReady": True,
            "textureComplexity": round(texture, 4),
        }
    if should_outpaint and outpaint_renderer is not None:
        try:
            rendered, meta = outpaint_renderer(clean_source, width, height, plan, analysis)
            if rendered.size != (width, height):
                rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
            return rendered.convert("RGB"), {**meta, "backgroundSource": "clean_base_outpaint"}
        except Exception as exc:
            fallback = fallback_renderer(clean_source, width, height)
            return fallback.convert("RGB"), {
                "provider": "local",
                "strategy": "deterministic_clean_base_fallback_after_outpaint_failed",
                "backgroundSource": "clean_base",
                "productionReady": False,
                "fallbackReason": str(exc),
            }
    fallback = fallback_renderer(clean_source, width, height)
    return fallback.convert("RGB"), {
        "provider": "local",
        "strategy": "deterministic_clean_base_fallback",
        "backgroundSource": "clean_base",
        "productionReady": not should_outpaint,
    }


def build_deterministic_background_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
    source = clean_source.convert("RGB")
    arr = np.array(source, dtype=np.uint8)
    strips = [
        arr[: max(1, source.height // 8), :, :],
        arr[-max(1, source.height // 8) :, :, :],
        arr[:, : max(1, source.width // 10), :],
        arr[:, -max(1, source.width // 10) :, :],
    ]
    samples = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    top_color = np.percentile(strips[0].reshape(-1, 3), 72, axis=0)
    bottom_color = np.percentile(strips[1].reshape(-1, 3), 72, axis=0)
    base_color = np.median(samples, axis=0)
    if np.linalg.norm(top_color - bottom_color) < 14:
        top_color = base_color * 0.96 + np.array([255, 255, 255]) * 0.04
        bottom_color = base_color * 0.92 + np.array([255, 255, 255]) * 0.08
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        color = top_color * (1 - t) + bottom_color * t
        gradient[y, :, :] = np.clip(color, 0, 255)
    canvas = Image.fromarray(gradient, "RGB")
    cover_scale = max(width / max(1, source.width), height / max(1, source.height))
    cover_w = max(1, int(round(source.width * cover_scale)))
    cover_h = max(1, int(round(source.height * cover_scale)))
    cover = source.resize((cover_w, cover_h), Image.Resampling.LANCZOS)
    left = max(0, (cover_w - width) // 2)
    top = max(0, (cover_h - height) // 2)
    texture = cover.crop((left, top, left + width, top + height))
    soft_radius = max(0.55, min(4.0, min(width, height) * 0.006))
    texture = texture.filter(ImageFilter.GaussianBlur(radius=soft_radius))
    # Keep the source's cream/wave texture instead of collapsing to a flat edge color.
    # The gradient only harmonizes exposure; it must not flatten the creative.
    return Image.blend(texture, canvas.filter(ImageFilter.GaussianBlur(radius=0.45)), 0.16)


def build_low_artifact_background_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
    """Texture-aware canvas for portrait/story targets where product remnants are costly."""
    source = clean_source.convert("RGB")
    arr = np.array(source, dtype=np.uint8)
    strips = [
        arr[: max(1, source.height // 8), :, :],
        arr[-max(1, source.height // 8) :, :, :],
        arr[:, : max(1, source.width // 10), :],
        arr[:, -max(1, source.width // 10) :, :],
    ]
    samples = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    top_color = np.percentile(strips[0].reshape(-1, 3), 72, axis=0)
    bottom_color = np.percentile(strips[1].reshape(-1, 3), 72, axis=0)
    base_color = np.median(samples, axis=0)
    if np.linalg.norm(top_color - bottom_color) < 14:
        top_color = base_color * 0.96 + np.array([255, 255, 255]) * 0.04
        bottom_color = base_color * 0.92 + np.array([255, 255, 255]) * 0.08
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        color = top_color * (1 - t) + bottom_color * t
        gradient[y, :, :] = np.clip(color, 0, 255)
    canvas = Image.fromarray(gradient, "RGB")
    cover_scale = max(width / max(1, source.width), height / max(1, source.height))
    cover_w = max(1, int(round(source.width * cover_scale)))
    cover_h = max(1, int(round(source.height * cover_scale)))
    cover = source.resize((cover_w, cover_h), Image.Resampling.LANCZOS)
    left = max(0, (cover_w - width) // 2)
    top = max(0, (cover_h - height) // 2)
    texture = cover.crop((left, top, left + width, top + height)).filter(
        ImageFilter.GaussianBlur(radius=max(8.0, min(width, height) * 0.040))
    )
    texture_arr = np.array(texture, dtype=np.float32)
    gradient_arr = np.array(canvas.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32)
    # Preserve the source's soft cream/degrade atmosphere without letting removed
    # foreground remnants dominate the canvas.
    mixed = np.clip((texture_arr * 0.34) + (gradient_arr * 0.66), 0, 255).astype(np.uint8)
    return Image.fromarray(mixed, "RGB").filter(ImageFilter.GaussianBlur(radius=0.28))


def _edge_median_rgb(image: Image.Image) -> np.ndarray:
    source = image.convert("RGB")
    arr = np.array(source, dtype=np.uint8)
    strips = [
        arr[: max(1, source.height // 10), :, :],
        arr[-max(1, source.height // 10) :, :, :],
        arr[:, : max(1, source.width // 12), :],
        arr[:, -max(1, source.width // 12) :, :],
    ]
    samples = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def _edge_touch_sides(box: tuple[int, int, int, int], width: int, height: int, *, tolerance: int = 2) -> list[str]:
    sides: list[str] = []
    if box[0] <= tolerance:
        sides.append("left")
    if box[1] <= tolerance:
        sides.append("top")
    if box[2] >= width - tolerance:
        sides.append("right")
    if box[3] >= height - tolerance:
        sides.append("bottom")
    return sides


def _expand_product_seed_box(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    seed_w = max(1, box[2] - box[0])
    seed_h = max(1, box[3] - box[1])
    return _clip_box(
        (
            box[0] - max(int(source.width * 0.16), int(seed_w * 0.78)),
            box[1] - max(int(source.height * 0.34), int(seed_h * 0.70)),
            box[2] + max(int(source.width * 0.13), int(seed_w * 0.62)),
            box[3] + max(int(source.height * 0.65), int(seed_h * 1.35)),
        ),
        source.width,
        source.height,
    )


def _fit_inside(source_size: tuple[int, int], target_box: tuple[int, int, int, int]) -> tuple[int, int, int, int, float]:
    source_w, source_h = source_size
    left, top, right, bottom = target_box
    target_w = max(1, right - left)
    target_h = max(1, bottom - top)
    scale = min(target_w / max(1, source_w), target_h / max(1, source_h))
    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))
    paste_x = left + (target_w - new_w) // 2
    paste_y = top + (target_h - new_h) // 2
    return paste_x, paste_y, paste_x + new_w, paste_y + new_h, scale


def _extract_layer_crop(source: Image.Image, layer: VisualLayer) -> tuple[Image.Image, tuple[int, int, int, int]]:
    box = _clip_box(layer.bbox.to_pixel_box(source.width, source.height), source.width, source.height)
    crop = source.crop(box).convert("RGBA")
    alpha = Image.new("L", crop.size, 255)
    crop.putalpha(alpha)
    return crop, box


def _box_area(box: tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_overlap(box: tuple[int, int, int, int], other: tuple[int, int, int, int]) -> int:
    left = max(box[0], other[0])
    top = max(box[1], other[1])
    right = min(box[2], other[2])
    bottom = min(box[3], other[3])
    return _box_area((left, top, right, bottom))


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _union_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    valid = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def _pad_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    return _clip_box((box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y), width, height)


def _merge_overlapping_boxes(boxes: list[tuple[int, int, int, int]], *, pad: int = 0) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for raw_box in boxes:
        box = raw_box
        if _box_area(box) <= 0:
            continue
        did_merge = True
        while did_merge:
            did_merge = False
            next_merged: list[tuple[int, int, int, int]] = []
            for existing in merged:
                expanded_existing = (existing[0] - pad, existing[1] - pad, existing[2] + pad, existing[3] + pad)
                expanded_box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
                if _box_overlap(expanded_existing, expanded_box) > 0:
                    box = (
                        min(box[0], existing[0]),
                        min(box[1], existing[1]),
                        max(box[2], existing[2]),
                        max(box[3], existing[3]),
                    )
                    did_merge = True
                else:
                    next_merged.append(existing)
            merged = next_merged
        merged.append(box)
    return merged


def _layer_box(layer: VisualLayer, source: Image.Image) -> tuple[int, int, int, int]:
    return _clip_box(layer.bbox.to_pixel_box(source.width, source.height), source.width, source.height)


def _tighten_marketing_text_box(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    try:
        import cv2

        box = _clip_box(box, source.width, source.height)
        if _box_area(box) <= 0:
            return box
        crop = np.array(source.crop(box).convert("RGB"), dtype=np.uint8)
        h, w = crop.shape[:2]
        if h < 5 or w < 5:
            return box
        edge_samples = np.concatenate(
            [
                crop[: max(1, h // 8), :, :].reshape(-1, 3),
                crop[-max(1, h // 8) :, :, :].reshape(-1, 3),
                crop[:, : max(1, w // 10), :].reshape(-1, 3),
                crop[:, -max(1, w // 10) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(edge_samples, axis=0)
        dist = np.linalg.norm(crop.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 45, 140)
        mask = ((dist > 24) | (edges > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        kept: list[tuple[int, int, int, int]] = []
        crop_area = max(1, w * h)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < max(3, crop_area * 0.00035):
                continue
            if area > crop_area * 0.28:
                continue
            if bh > h * 0.88 and bw > w * 0.22:
                continue
            if bw > w * 0.70 and bh > h * 0.55:
                continue
            kept.append((x, y, x + bw, y + bh))
        union = _union_boxes(kept)
        if not union:
            return box
        tightened = (
            box[0] + max(0, union[0] - 2),
            box[1] + max(0, union[1] - 2),
            box[0] + min(w, union[2] + 2),
            box[1] + min(h, union[3] + 2),
        )
        original_area = max(1, _box_area(box))
        if _box_area(tightened) < original_area * 0.08:
            return box
        return _clip_box(tightened, source.width, source.height)
    except Exception:
        return box


def _collect_visual_element_boxes(
    source: Image.Image,
    analysis: VisualAnalysis,
    visual_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    w, h = source.size
    source_area = max(1, w * h)
    boxes: list[tuple[int, int, int, int]] = []
    person_layers = list(getattr(analysis, "person_layers", []) or [])
    text_boxes_for_filter = [layer.bbox.to_pixel_box(w, h) for layer in analysis.marketing_text_layers]
    for layer in [*analysis.product_layers, *person_layers]:
        box = _layer_box(layer, source)
        text_overlap_ratio = sum(_box_overlap(box, text_box) for text_box in text_boxes_for_filter) / max(1, _box_area(box))
        area_ratio = _box_area(box) / source_area
        _, cy = _box_center(box)
        box_h_ratio = (box[3] - box[1]) / max(1, h)
        if w / max(1, h) > 1.35 and (box_h_ratio < 0.38 or cy < h * 0.24):
            box = _expand_product_seed_box(source, box)
            area_ratio = _box_area(box) / source_area
            text_overlap_ratio = sum(_box_overlap(box, text_box) for text_box in text_boxes_for_filter) / max(1, _box_area(box))
        if text_overlap_ratio > 0.18 and w / max(1, h) > 1.35:
            continue
        if 0.01 <= area_ratio <= 0.58:
            boxes.append(_pad_box(box, w, h, pad_x=max(4, w // 120), pad_y=max(4, h // 100)))
    for layer in analysis.other_layers:
        box = _layer_box(layer, source)
        area_ratio = _box_area(box) / source_area
        cx, cy = _box_center(box)
        notes = str(getattr(layer, "notes", "") or "").lower()
        negative_theme_note = any(token in notes for token in ("theme_element:false", "theme:false", "importance:tertiary"))
        visual_note = (
            any(token in notes for token in ("protected", "theme_element:true", "side_component", "product", "material"))
            and not negative_theme_note
        )
        top_badge_like = cy < h * 0.22 and (box[2] - box[0]) < w * 0.32 and (box[3] - box[1]) < h * 0.25
        if 0.006 <= area_ratio <= 0.55 and visual_note and not top_badge_like:
            boxes.append(_pad_box(box, w, h, pad_x=max(4, w // 130), pad_y=max(4, h // 110)))
    foreground_boxes = _foreground_component_visual_boxes(source)
    if not boxes and foreground_boxes:
        boxes = foreground_boxes
    if not boxes:
        return [visual_box]
    if foreground_boxes and w / max(1, h) > 1.35:
        # Vision models often return only the visible label/top band for tall
        # products or faces in wide creatives. If deterministic foreground
        # components are materially taller, use them as the visual elements.
        avg_detected_height = sum(max(1, box[3] - box[1]) for box in boxes) / max(1, len(boxes))
        avg_foreground_height = sum(max(1, box[3] - box[1]) for box in foreground_boxes) / max(1, len(foreground_boxes))
        if avg_foreground_height >= avg_detected_height * 1.35:
            boxes = foreground_boxes
    merged = _merge_overlapping_boxes(boxes, pad=max(8, min(w, h) // 60))
    if len(merged) == 1 and _box_area(merged[0]) / source_area > 0.72:
        return [visual_box]
    return merged or [visual_box]


def _foreground_component_visual_boxes(source: Image.Image) -> list[tuple[int, int, int, int]]:
    """Return deterministic product/person foreground boxes from alpha segmentation."""
    try:
        import cv2

        rgba = _apply_foreground_alpha(source.convert("RGBA"))
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        h, w = alpha.shape[:2]
        source_area = max(1, w * h)
        mask = (alpha > 32).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        min_area = max(900, int(source_area * 0.018))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            box = _clip_box((x, y, x + bw, y + bh), w, h)
            box_area_ratio = _box_area(box) / source_area
            if box_area_ratio > 0.42:
                continue
            if bh < h * 0.22 and bw < w * 0.22:
                continue
            boxes.append(_pad_box(box, w, h, pad_x=max(4, w // 140), pad_y=max(4, h // 120)))
        return boxes
    except Exception:
        return []


def _visual_focus_from_product_label_union(
    source: Image.Image,
    label_union: tuple[int, int, int, int] | None,
    visual_box: tuple[int, int, int, int],
    *,
    compact_target: bool,
) -> tuple[int, int, int, int] | None:
    """Derive a tighter product crop from readable package-label geometry.

    Compact resize targets must keep the product name readable. If a broad
    visual fallback includes too much neutral background, use the detected
    package-label union as the mathematical core and expand around it.
    """
    if not label_union or not compact_target:
        return None
    if _box_overlap(label_union, visual_box) <= 0:
        return None
    label_w = max(1, label_union[2] - label_union[0])
    label_h = max(1, label_union[3] - label_union[1])
    if _box_area(label_union) < max(16, source.width * source.height * 0.002):
        return None

    pad_x_ratio = 0.28 if compact_target else 0.42
    pad_x = max(int(round(label_w * pad_x_ratio)), source.width // (44 if compact_target else 32), 8)
    pad_y = max(int(round(label_h * 0.55)), source.height // 18, 8)
    candidate = _clip_box(
        (
            label_union[0] - pad_x,
            label_union[1] - pad_y,
            label_union[2] + pad_x,
            label_union[3] + pad_y,
        ),
        source.width,
        source.height,
    )

    # If the readable label sits near a truncated product edge, preserve that
    # edge so the package remains visually plausible instead of over-cropped.
    edge_margin_x = max(6, source.width // 90)
    edge_margin_y = max(6, source.height // 70)
    if visual_box[1] <= edge_margin_y or label_union[1] <= edge_margin_y:
        candidate = (candidate[0], visual_box[1], candidate[2], candidate[3])
    if visual_box[3] >= source.height - edge_margin_y or label_union[3] >= source.height - edge_margin_y:
        candidate = (candidate[0], candidate[1], candidate[2], visual_box[3])
    if visual_box[0] <= edge_margin_x or label_union[0] <= edge_margin_x:
        candidate = (visual_box[0], candidate[1], candidate[2], candidate[3])
    if visual_box[2] >= source.width - edge_margin_x or label_union[2] >= source.width - edge_margin_x:
        candidate = (candidate[0], candidate[1], visual_box[2], candidate[3])

    candidate = _clip_box(candidate, source.width, source.height)
    if _box_area(candidate) < _box_area(label_union) * 1.15:
        return None
    if _box_area(candidate) >= _box_area(visual_box) * 0.96:
        return None
    return candidate


def _detect_role_aware_visual_box(source: Image.Image, analysis: VisualAnalysis) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Detect the main product/visual region without trusting over-wide analysis boxes."""
    product_boxes = []
    source_ratio = source.width / max(1, source.height)
    text_boxes_for_filter = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
    for layer in analysis.product_layers:
        box = _layer_box(layer, source)
        _, cy = _box_center(box)
        box_h_ratio = (box[3] - box[1]) / max(1, source.height)
        if source_ratio > 1.35 and (box_h_ratio < 0.38 or cy < source.height * 0.24):
            box = _expand_product_seed_box(source, box)
        text_overlap_ratio = sum(_box_overlap(box, text_box) for text_box in text_boxes_for_filter) / max(1, _box_area(box))
        if source_ratio > 1.35 and text_overlap_ratio > 0.18:
            continue
        if _box_area(box) / max(1, source.width * source.height) <= 0.62:
            product_boxes.append(box)
    product_union = _union_boxes(product_boxes)
    if product_union:
        return _pad_box(product_union, source.width, source.height, pad_x=max(6, source.width // 36), pad_y=max(6, source.height // 28)), {
            "visualBoxMethod": "product_layer_union"
        }
    foreground_boxes = _foreground_component_visual_boxes(source)
    if foreground_boxes:
        foreground_union = _union_boxes(foreground_boxes)
        if foreground_union:
            return _pad_box(foreground_union, source.width, source.height, pad_x=max(6, source.width // 44), pad_y=max(6, source.height // 34)), {
                "visualBoxMethod": "foreground_component_union"
            }
    try:
        import cv2

        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        text_mask, _ = build_overlay_text_mask(source, analysis)
        text = np.array(text_mask, dtype=np.uint8) > 16
        edge_samples = np.concatenate(
            [
                rgb[: max(1, h // 12), :, :].reshape(-1, 3),
                rgb[-max(1, h // 12) :, :, :].reshape(-1, 3),
                rgb[:, : max(1, w // 14), :].reshape(-1, 3),
                rgb[:, -max(1, w // 14) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(edge_samples, axis=0)
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 38, 130) > 0
        candidate = ((dist > 18) | edges) & (~text)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate_u8 = cv2.morphologyEx((candidate.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate_u8 = cv2.dilate(candidate_u8, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate_u8 > 0).astype(np.uint8), connectivity=8)
        components: list[tuple[float, tuple[int, int, int, int]]] = []
        min_area = max(80, int(w * h * 0.004))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            box = (x, y, x + bw, y + bh)
            box_area = _box_area(box)
            if box_area <= 0 or box_area > int(w * h * 0.88):
                continue
            cx, cy = _box_center(box)
            if y <= h * 0.20 and bh <= h * 0.30 and bw <= w * 0.30:
                continue
            right_bias = cx / max(1, w)
            vertical_presence = bh / max(1, h)
            score = area * (0.8 + right_bias * 0.7 + vertical_presence * 0.45)
            if cy > h * 0.20 or right_bias > 0.36:
                components.append((score, box))
        if components:
            components.sort(reverse=True, key=lambda item: item[0])
            seed = components[0][1]
            seed_cx, _ = _box_center(seed)
            keep = [seed]
            for _, box in components[1:8]:
                cx, _ = _box_center(box)
                if abs(cx - seed_cx) <= w * 0.34 or _box_overlap(box, seed) > 0:
                    keep.append(box)
            union = _union_boxes(keep)
            if union:
                pad_x = max(8, int((union[2] - union[0]) * 0.04))
                pad_y = max(8, int((union[3] - union[1]) * 0.06))
                return _pad_box(union, w, h, pad_x=pad_x, pad_y=pad_y), {
                    "visualBoxMethod": "foreground_components_minus_text",
                    "visualComponentCount": len(keep),
                }
        raise ValueError("no usable visual components")
    except Exception as exc:
        source_ratio = source.width / max(1, source.height)
        package_label_candidates: list[tuple[int, int, int, int]] = []
        for layer in analysis.marketing_text_layers:
            box = _layer_box(layer, source)
            cx, cy = _box_center(box)
            rel_x = cx / max(1, source.width)
            rel_y = cy / max(1, source.height)
            area_ratio = _box_area(box) / max(1, source.width * source.height)
            if 0.42 <= rel_x <= 0.78 and rel_y >= 0.26 and area_ratio <= 0.035:
                package_label_candidates.append(box)
        label_union = _union_boxes(package_label_candidates)
        if label_union:
            label_w = max(1, label_union[2] - label_union[0])
            label_h = max(1, label_union[3] - label_union[1])
            # A package label is only an inner detail of the sellable product.
            # Expand from the mathematically detected label toward the probable
            # pack/body bounds so the resize layer carries the product, not only
            # the printed label rectangle.
            expand_left = max(int(source.width * 0.16), int(label_w * 0.78))
            expand_right = max(int(source.width * 0.13), int(label_w * 0.62))
            expand_top = max(int(source.height * 0.34), int(label_h * 0.70))
            expand_bottom = max(int(source.height * 0.30), int(label_h * 0.55))
            return _clip_box(
                (
                    label_union[0] - expand_left,
                    label_union[1] - expand_top,
                    label_union[2] + expand_right,
                    label_union[3] + expand_bottom,
                ),
                source.width,
                source.height,
            ), {
                "visualBoxMethod": "package_label_expanded_fallback",
                "packageLabelCandidateCount": len(package_label_candidates),
                "visualBoxError": str(exc),
            }
        text_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
        non_top_text = [box for box in text_boxes if _box_center(box)[1] > source.height * 0.22]
        if source_ratio >= 1.35 and non_top_text:
            left_weight = sum(_box_area(box) for box in non_top_text if _box_center(box)[0] <= source.width * 0.50)
            right_weight = sum(_box_area(box) for box in non_top_text if _box_center(box)[0] > source.width * 0.50)
            if left_weight >= right_weight:
                x0 = int(source.width * 0.42)
                return (x0, 0, source.width, source.height), {
                    "visualBoxMethod": "opposite_text_side_fallback",
                    "visualTextDensitySide": "left",
                    "visualBoxError": str(exc),
                }
            x1 = int(source.width * 0.58)
            return (0, 0, x1, source.height), {
                "visualBoxMethod": "opposite_text_side_fallback",
                "visualTextDensitySide": "right",
                "visualBoxError": str(exc),
            }
        product_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.product_layers]
        union = _union_boxes(product_boxes)
        if union:
            return _clip_box(union, source.width, source.height), {
                "visualBoxMethod": "analysis_product_fallback",
                "visualBoxError": str(exc),
            }
        return (0, 0, source.width, source.height), {
            "visualBoxMethod": "full_source_fallback",
            "visualBoxError": str(exc),
        }


def _partition_resize_layers(
    source: Image.Image,
    analysis: VisualAnalysis,
    visual_box: tuple[int, int, int, int],
) -> dict[str, list[tuple[int, int, int, int]]]:
    w, h = source.size
    brand: list[tuple[int, int, int, int]] = []
    trust_badge: list[tuple[int, int, int, int]] = []
    primary: list[tuple[int, int, int, int]] = []
    secondary: list[tuple[int, int, int, int]] = []
    product_label: list[tuple[int, int, int, int]] = []
    visual_area = max(1, _box_area(visual_box))

    def is_trust_badge_candidate(box: tuple[int, int, int, int], *, text: str = "", notes: str = "", role: str = "") -> bool:
        cx, cy = _box_center(box)
        box_w = max(1, box[2] - box[0])
        box_h = max(1, box[3] - box[1])
        area_ratio = _box_area(box) / max(1, w * h)
        descriptor = f"{text} {notes} {role}".lower()
        explicit_trust = any(token in descriptor for token in ("badge", "seal", "award", "trust", "tavsiye", "recommended", "dermatolog"))
        top_right_free_asset = cy <= h * 0.26 and cx >= w * 0.58 and box_w <= w * 0.34 and box_h <= h * 0.24
        not_product_overlap = _box_overlap(box, visual_box) / max(1, _box_area(box)) < 0.18
        return not_product_overlap and 0.001 <= area_ratio <= 0.08 and (explicit_trust or top_right_free_asset)

    for layer in analysis.marketing_text_layers:
        raw_box = _layer_box(layer, source)
        # Use the real text-pixel box for layer relocation. The expanded same-line
        # helper is useful for cleanup, but it can swallow nearby product/secondary
        # regions and then force false clipping against the visual box.
        box = _tighten_marketing_text_box(source, raw_box)
        box = _pad_box(box, w, h, pad_x=max(2, w // 240), pad_y=max(2, h // 240))
        cx, cy = _box_center(raw_box)
        overlap_visual = _box_overlap(raw_box, visual_box) / max(1, _box_area(raw_box))
        near_top = cy <= h * 0.24
        near_edge = cx <= w * 0.32 or cx >= w * 0.68
        original_text = str(getattr(layer, "original_text", "") or "")
        has_alpha = any(ch.isalpha() for ch in original_text)
        if is_trust_badge_candidate(raw_box, text=original_text, role=str(getattr(layer, "role", ""))):
            trust_badge.append(box)
            continue
        if near_top and near_edge and has_alpha:
            brand.append(box)
            continue
        if overlap_visual > 0.45 or (visual_box[0] <= cx <= visual_box[2] and visual_box[1] <= cy <= visual_box[3]):
            if cx >= w * 0.74 and raw_box[0] >= visual_box[0] + (visual_box[2] - visual_box[0]) * 0.42:
                secondary.append(box)
            else:
                product_label.append(raw_box)
            continue
        if cx < visual_box[0] or cx < w * 0.48:
            primary.append(box)
        else:
            secondary.append(box)
    for layer in analysis.logo_layers:
        box = _layer_box(layer, source)
        overlap_visual = _box_overlap(box, visual_box) / max(1, _box_area(box))
        box_w = max(1, box[2] - box[0])
        box_h = max(1, box[3] - box[1])
        # A large or heavily visual-overlapping "logo" is usually text printed
        # on a product/person region, not a floating brand mark to relocate.
        if overlap_visual > 0.36 or box_h > h * 0.24 or box_w > w * 0.42:
            product_label.append(box)
            continue
        if is_trust_badge_candidate(box, notes=str(getattr(layer, "notes", "")), role=str(getattr(layer, "role", ""))):
            trust_badge.append(box)
            continue
        brand.append(box)
    for layer in analysis.other_layers:
        box = _layer_box(layer, source)
        cx, cy = _box_center(box)
        notes = str(getattr(layer, "notes", "") or "").lower()
        role = str(getattr(layer, "role", "") or "").lower()
        is_explicit_brand_mark = any(token in f"{role} {notes}" for token in ("logo", "brand", "badge", "seal", "award", "mark"))
        if is_trust_badge_candidate(box, notes=notes, role=role):
            trust_badge.append(box)
            continue
        if is_explicit_brand_mark and cy <= h * 0.25 and (cx <= w * 0.35 or cx >= w * 0.65) and _box_overlap(box, visual_box) / visual_area < 0.12:
            if is_trust_badge_candidate(box, notes=notes, role=role):
                trust_badge.append(box)
            else:
                brand.append(box)
    deduped_trust_badge: list[tuple[int, int, int, int]] = []
    for box in sorted(trust_badge, key=_box_area, reverse=True):
        if any(_box_overlap(box, existing) / max(1, min(_box_area(box), _box_area(existing))) > 0.60 for existing in deduped_trust_badge):
            continue
        deduped_trust_badge.append(box)
    return {
        "brand": brand,
        "trust_badge": deduped_trust_badge,
        "primary": primary,
        "secondary": secondary,
        "product_label": product_label,
    }


def _expand_text_box_line_region(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Expand a text box to nearby same-line foreground text pixels."""
    try:
        import cv2

        w, h = source.size
        x1, y1, x2, y2 = _clip_box(box, w, h)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        band = (
            max(0, x1 - int(w * 0.08)),
            max(0, y1 - max(4, int(bh * 0.55))),
            min(w, x2 + max(int(w * 0.16), int(bw * 1.2))),
            min(h, y2 + max(4, int(bh * 0.55))),
        )
        crop = np.array(source.crop(band).convert("RGB"), dtype=np.uint8)
        if crop.size == 0:
            return box
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        edge_samples = np.concatenate(
            [
                crop[: max(1, crop.shape[0] // 8), :, :].reshape(-1, 3),
                crop[-max(1, crop.shape[0] // 8) :, :, :].reshape(-1, 3),
                crop[:, : max(1, crop.shape[1] // 12), :].reshape(-1, 3),
                crop[:, -max(1, crop.shape[1] // 12) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(edge_samples, axis=0)
        dist = np.linalg.norm(crop.astype(np.float32) - bg.astype(np.float32), axis=2)
        mask = ((dist > 22) | ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] < 250)) | (gray < 190)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        seed_local = (x1 - band[0], y1 - band[1], x2 - band[0], y2 - band[1])
        candidates: list[tuple[int, int, int, int]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            lx = int(stats[label, cv2.CC_STAT_LEFT])
            ly = int(stats[label, cv2.CC_STAT_TOP])
            lw = int(stats[label, cv2.CC_STAT_WIDTH])
            lh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < max(12, int(bh * 0.10)) or lh < 2 or lw < 2:
                continue
            component_area_ratio = area / max(1, lw * lh)
            component_band_ratio = area / max(1, crop.shape[0] * crop.shape[1])
            looks_like_visual_object = (
                lh > max(bh * 1.85, bh + 18)
                or (lw > bw * 1.75 and lh > bh * 1.25)
                or (component_band_ratio > 0.18 and component_area_ratio > 0.42)
            )
            if looks_like_visual_object:
                continue
            comp = (lx, ly, lx + lw, ly + lh)
            y_overlap = _box_overlap((seed_local[0], seed_local[1], seed_local[2], seed_local[3]), (seed_local[0], comp[1], seed_local[2], comp[3]))
            if y_overlap <= 0:
                continue
            close_x = comp[0] <= seed_local[2] + max(int(w * 0.18), bw) and comp[2] >= seed_local[0] - max(int(w * 0.06), bw // 2)
            if close_x:
                candidates.append((band[0] + comp[0], band[1] + comp[1], band[0] + comp[2], band[1] + comp[3]))
        union = _union_boxes(candidates + [box])
        if not union:
            return box
        local_y1 = max(0, union[1] - band[1])
        local_y2 = min(mask.shape[0], union[3] - band[1])
        local_x2 = min(mask.shape[1] - 1, union[2] - band[0])
        if local_y2 > local_y1:
            col_counts = np.count_nonzero(mask[local_y1:local_y2, :] > 0, axis=0)
            threshold = max(2, int((local_y2 - local_y1) * 0.05))
            gap_limit = max(8, int(w * 0.045))
            gap = 0
            scan_right = local_x2
            for cx in range(local_x2, mask.shape[1]):
                if col_counts[cx] >= threshold:
                    scan_right = cx
                    gap = 0
                else:
                    gap += 1
                    if gap >= gap_limit and cx > local_x2 + 4:
                        break
            union = (union[0], union[1], max(union[2], band[0] + scan_right + 1), union[3])
        if _box_area(union) > _box_area(box) * 12.0:
            return box
        return _clip_box(union, w, h)
    except Exception:
        return box


def _inpaint_rectangular_overlays(source: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    if not boxes:
        return source
    try:
        import cv2

        arr = np.array(source.convert("RGB"), dtype=np.uint8)
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        for box in boxes:
            left, top, right, bottom = _clip_box(box, source.width, source.height)
            pad_x = max(6, min(42, int((right - left) * 0.16)))
            pad_y = max(4, min(30, int((bottom - top) * 0.28)))
            mask[max(0, top - pad_y): min(source.height, bottom + pad_y), max(0, left - pad_x): min(source.width, right + pad_x)] = 255
        if not np.any(mask):
            return source
        cleaned = cv2.inpaint(arr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        local_fill = cleaned.copy()
        for box in boxes:
            left, top, right, bottom = _clip_box(box, source.width, source.height)
            pad_x = max(6, min(42, int((right - left) * 0.16)))
            pad_y = max(4, min(30, int((bottom - top) * 0.28)))
            x1 = max(0, left - pad_x)
            y1 = max(0, top - pad_y)
            x2 = min(source.width, right + pad_x)
            y2 = min(source.height, bottom + pad_y)
            sx1 = max(0, x1 - max(8, pad_x))
            sy1 = max(0, y1 - max(8, pad_y))
            sx2 = min(source.width, x2 + max(8, pad_x))
            sy2 = min(source.height, y2 + max(8, pad_y))
            sample_region = arr[sy1:sy2, sx1:sx2]
            sample_mask = mask[sy1:sy2, sx1:sx2] == 0
            samples = sample_region[sample_mask]
            if samples.size:
                fill_color = np.median(samples.reshape(-1, 3), axis=0).astype(np.uint8)
                local_fill[y1:y2, x1:x2] = fill_color
        feather = cv2.GaussianBlur(mask, (0, 0), sigmaX=7.0, sigmaY=7.0)
        alpha = (feather.astype(np.float32) / 255.0)[..., None]
        blended = (cleaned.astype(np.float32) * (1.0 - alpha) + local_fill.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended, "RGB")
    except Exception:
        return source


def _remove_foreground_visuals_for_background(source: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    """Remove product/person foreground pixels while preserving nearby theme/background details."""
    if not boxes:
        return source
    try:
        import cv2

        rgb = source.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        for box in boxes:
            left, top, right, bottom = _clip_box(box, source.width, source.height)
            if right <= left or bottom <= top:
                continue
            crop = rgb.crop((left, top, right, bottom)).convert("RGBA")
            extracted = _apply_foreground_alpha(crop)
            alpha = np.array(extracted.getchannel("A"), dtype=np.uint8)
            if not np.any(alpha > 12):
                continue
            local = (alpha > 12).astype(np.uint8) * 255
            kernel_size = max(5, min(19, int(round(min(source.width, source.height) * 0.018))))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            local = cv2.dilate(local, kernel, iterations=1)
            mask[top:bottom, left:right] = np.maximum(mask[top:bottom, left:right], local)
        if not np.any(mask):
            return source
        cleaned = cv2.inpaint(arr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB")
    except Exception:
        return source


def _apply_foreground_alpha(crop: Image.Image) -> Image.Image:
    if os.getenv("ADAPTIFAI_RESIZE_REMBG_FOREGROUND", "1").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            global _REMBG_SESSION
            from rembg import new_session, remove

            rgba = crop.convert("RGBA")
            if _REMBG_SESSION is None:
                _REMBG_SESSION = new_session(os.getenv("ADAPTIFAI_REMBG_MODEL", "u2net"))
            extracted = remove(rgba, session=_REMBG_SESSION).convert("RGBA")
            alpha = np.array(extracted.getchannel("A"), dtype=np.uint8)
            alpha_ratio = float(np.count_nonzero(alpha > 8)) / max(1, alpha.size)
            if 0.045 <= alpha_ratio <= 0.88:
                return _prepare_foreground_rgba_crop(extracted)
        except Exception:
            pass
    try:
        import cv2

        rgba = crop.convert("RGBA")
        rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        if h < 8 or w < 8:
            return rgba

        grabcut_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(2, min(w, h) // 18)
        grabcut_mask[:border, :] = cv2.GC_BGD
        grabcut_mask[-border:, :] = cv2.GC_BGD
        grabcut_mask[:, :border] = cv2.GC_BGD
        grabcut_mask[:, -border:] = cv2.GC_BGD
        rect = (
            max(1, border),
            max(1, border),
            max(2, w - border * 2),
            max(2, h - border * 2),
        )
        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, grabcut_mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            keep = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=2)
            keep = cv2.dilate(keep, kernel, iterations=1)
            keep_area = int(np.count_nonzero(keep))
            if keep_area >= max(60, int(w * h * 0.05)) and keep_area <= int(w * h * 0.62):
                if keep_area < int(w * h * 0.18):
                    ys, xs = np.where(keep > 0)
                    if xs.size and ys.size:
                        left, right = int(xs.min()), int(xs.max()) + 1
                        top, bottom = int(ys.min()), int(ys.max()) + 1
                        bw = max(1, right - left)
                        bh = max(1, bottom - top)
                        expanded = np.zeros_like(keep)
                        x0 = max(0, left - int(bw * 0.85))
                        x1 = min(w, right + int(bw * 0.62))
                        y0 = max(0, top - int(bh * 0.58))
                        y1 = min(h, bottom + int(bh * 0.22))
                        cv2.rectangle(expanded, (x0, y0), (x1, y1), 255, thickness=-1)
                        round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, w // 10), max(9, h // 10)))
                        expanded = cv2.morphologyEx(expanded, cv2.MORPH_OPEN, round_kernel, iterations=1)
                        if int(np.count_nonzero(expanded)) >= int(w * h * 0.18):
                            keep = np.maximum(keep, expanded)
                keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=1.2, sigmaY=1.2)
                rgba.putalpha(Image.fromarray(keep, "L"))
                return rgba
        except Exception:
            pass

        edge_samples = np.concatenate(
            [
                rgb[: max(1, h // 10), :, :].reshape(-1, 3),
                rgb[-max(1, h // 10) :, :, :].reshape(-1, 3),
                rgb[:, : max(1, w // 12), :].reshape(-1, 3),
                rgb[:, -max(1, w // 12) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(edge_samples, axis=0)
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 34, 120)
        mask = ((dist > 13) | (edges > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros((h, w), dtype=np.uint8)
        min_area = max(40, int(w * h * 0.015))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[labels == label] = 255
        if int(np.count_nonzero(keep)) < min_area:
            keep = mask
        keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=1.2, sigmaY=1.2)
        rgba.putalpha(Image.fromarray(keep, "L"))
        return _prepare_foreground_rgba_crop(rgba)
    except Exception:
        return crop.convert("RGBA")


def _apply_artwork_alpha(crop: Image.Image) -> Image.Image:
    """Extract text/highlight artwork from a mostly flat creative background."""
    try:
        import cv2

        rgba = crop.convert("RGBA")
        rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        if h < 4 or w < 4:
            return rgba
        border = max(2, min(w, h) // 10)
        samples = np.concatenate(
            [
                rgb[:border, :, :].reshape(-1, 3),
                rgb[-border:, :, :].reshape(-1, 3),
                rgb[:, :border, :].reshape(-1, 3),
                rgb[:, -border:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(samples, axis=0)
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        mask = ((dist > 18) | ((saturation > 45) & (value > 70))).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        filtered = np.zeros_like(mask)
        crop_area = max(1, w * h)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area_ratio = area / crop_area
            aspect = bw / max(1, bh)
            horizontal_highlight = aspect >= 2.2 and bh <= h * 0.36 and area_ratio <= 0.28
            text_like = area_ratio <= 0.09 and bh <= h * 0.72
            small_detail = area_ratio <= 0.018
            if horizontal_highlight or text_like or small_detail:
                filtered[labels == label] = 255
        if np.count_nonzero(filtered) >= max(8, int(crop_area * 0.006)):
            mask = filtered
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.65, sigmaY=0.65)
        alpha_ratio = float(np.count_nonzero(mask > 12)) / max(1, mask.size)
        if not (0.01 <= alpha_ratio <= 0.72):
            return rgba
        rgba.putalpha(Image.fromarray(mask, "L"))
        return rgba
    except Exception:
        return crop.convert("RGBA")


def _trim_rgba_to_alpha(crop: Image.Image, *, pad_ratio: float = 0.06) -> Image.Image:
    rgba = crop.convert("RGBA")
    alpha_box = rgba.getchannel("A").getbbox()
    if not alpha_box:
        return rgba
    left, top, right, bottom = alpha_box
    content_w = max(1, right - left)
    content_h = max(1, bottom - top)
    if content_w * content_h < rgba.width * rgba.height * 0.08:
        return rgba
    pad_x = max(2, int(round(content_w * pad_ratio)))
    pad_y = max(2, int(round(content_h * pad_ratio)))
    return rgba.crop(
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(rgba.width, right + pad_x),
            min(rgba.height, bottom + pad_y),
        )
    )


def _paste_crop_fit(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    mode: str = "contain",
    anchor: tuple[float, float] = (0.5, 0.5),
    feather: int = 0,
    foreground_alpha: bool = False,
    artwork_alpha: bool = False,
    alpha_cut_source_boxes: list[tuple[int, int, int, int]] | None = None,
    anchor_bottom_if_source_truncated: bool = False,
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    target_box = _clip_box(target_box, output.width, output.height)
    crop = source.crop(source_box).convert("RGBA")
    if artwork_alpha:
        crop = _apply_artwork_alpha(crop)
    elif foreground_alpha:
        original_crop = crop
        crop = _source_crop_with_global_foreground_alpha(source, source_box)
        if alpha_cut_source_boxes:
            crop = _cut_source_boxes_from_alpha(crop, source_box, alpha_cut_source_boxes)
        crop = _trim_rgba_to_alpha(crop)
        alpha = np.array(crop.getchannel("A"), dtype=np.uint8)
        visible_alpha_ratio = float(np.count_nonzero(alpha > 16)) / max(1, crop.width * crop.height)
        if (
            not alpha_cut_source_boxes
            and (visible_alpha_ratio < 0.08 or crop.width < max(8, original_crop.width * 0.22) or crop.height < max(8, original_crop.height * 0.22))
        ):
            crop = original_crop
    target_w = max(1, target_box[2] - target_box[0])
    target_h = max(1, target_box[3] - target_box[1])
    scale = max(target_w / max(1, crop.width), target_h / max(1, crop.height)) if mode == "cover" else min(target_w / max(1, crop.width), target_h / max(1, crop.height))
    scaled_w = max(1, int(round(crop.width * scale)))
    scaled_h = max(1, int(round(crop.height * scale)))
    resized = crop.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    if mode == "cover":
        crop_left = int(round(max(0, min(scaled_w - target_w, scaled_w * anchor[0] - target_w * anchor[0]))))
        crop_top = int(round(max(0, min(scaled_h - target_h, scaled_h * anchor[1] - target_h * anchor[1]))))
        patch = resized.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))
        paste_left, paste_top = target_box[0], target_box[1]
        paste_box = [paste_left, paste_top, paste_left + target_w, paste_top + target_h]
    else:
        paste_left = target_box[0] + int(round((target_w - scaled_w) * anchor[0]))
        paste_top = target_box[1] + int(round((target_h - scaled_h) * anchor[1]))
        if anchor_bottom_if_source_truncated and source_box[3] >= source.height - max(2, int(source.height * 0.012)):
            paste_top = output.height - scaled_h
        patch = resized
        paste_box = [paste_left, paste_top, paste_left + scaled_w, paste_top + scaled_h]
    if feather > 0:
        alpha = patch.getchannel("A")
        gradient = Image.new("L", patch.size, 255)
        px = gradient.load()
        fw = max(1, min(feather, patch.width // 3, patch.height // 3))
        for y in range(patch.height):
            for x in range(patch.width):
                edge = min(x, y, patch.width - 1 - x, patch.height - 1 - y)
                if edge < fw:
                    px[x, y] = min(px[x, y], int(round(255 * edge / fw)))
        patch.putalpha(ImageChops.multiply(alpha, gradient))
    output.alpha_composite(patch, (paste_box[0], paste_box[1]))
    return {
        "layerId": layer_id,
        "role": role,
        "sourceBox": list(source_box),
        "targetBox": list(target_box),
        "pasteBox": paste_box,
        "scale": round(scale, 4),
        "fitMode": mode,
    }


def _cut_source_boxes_from_alpha(
    crop: Image.Image,
    source_box: tuple[int, int, int, int],
    cut_boxes: list[tuple[int, int, int, int]],
) -> Image.Image:
    if not cut_boxes:
        return crop
    rgba = crop.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_np = np.array(alpha, dtype=np.uint8)
    rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = source_box
    for cut_box in cut_boxes:
        ix1 = max(sx1, cut_box[0])
        iy1 = max(sy1, cut_box[1])
        ix2 = min(sx2, cut_box[2])
        iy2 = min(sy2, cut_box[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        local_x1 = max(0, ix1 - sx1)
        local_y1 = max(0, iy1 - sy1)
        local_x2 = min(rgba.width, ix2 - sx1)
        local_y2 = min(rgba.height, iy2 - sy1)
        x1 = max(0, local_x1)
        y1 = max(0, local_y1)
        x2 = min(rgba.width, local_x2)
        y2 = min(rgba.height, local_y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cut_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, rgba.width * rgba.height)
        if cut_area_ratio <= 0.22:
            alpha_np[y1:y2, x1:x2] = 0
            continue
        try:
            import cv2

            region = rgb[y1:y2, x1:x2]
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            text_pixels = (((hsv[:, :, 1] > 76) & (hsv[:, :, 2] > 80)) | (gray < 78)).astype(np.uint8) * 255
            kernel_w = max(3, min(11, (x2 - x1) // 22))
            kernel_h = max(3, min(9, (y2 - y1) // 8))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
            text_pixels = cv2.dilate(text_pixels, kernel, iterations=1)
            alpha_region = alpha_np[y1:y2, x1:x2]
            alpha_region[text_pixels > 0] = 0
            alpha_np[y1:y2, x1:x2] = alpha_region
        except Exception:
            continue
    rgba.putalpha(Image.fromarray(alpha_np, "L").filter(ImageFilter.GaussianBlur(radius=0.45)))
    return rgba


def _source_crop_with_global_foreground_alpha(source: Image.Image, source_box: tuple[int, int, int, int]) -> Image.Image:
    """Crop source RGB while applying a foreground alpha mask computed on the full image."""
    source_box = _clip_box(source_box, source.width, source.height)
    crop = source.crop(source_box).convert("RGBA")
    try:
        full_foreground = _apply_foreground_alpha(source.convert("RGBA"))
        alpha_crop = full_foreground.getchannel("A").crop(source_box)
        alpha_crop = _remove_text_like_alpha_components(crop, alpha_crop)
        if np.count_nonzero(np.array(alpha_crop, dtype=np.uint8) > 16) >= max(64, crop.width * crop.height * 0.04):
            crop.putalpha(alpha_crop)
            return _prepare_foreground_rgba_crop(crop)
    except Exception:
        pass
    return _apply_foreground_alpha(crop)


def _remove_text_like_alpha_components(crop: Image.Image, alpha: Image.Image) -> Image.Image:
    try:
        import cv2

        rgba = crop.convert("RGBA")
        rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
        mask = (np.array(alpha.convert("L"), dtype=np.uint8) > 16).astype(np.uint8)
        if not np.any(mask):
            return alpha
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = mask.copy()
        crop_area = max(1, crop.width * crop.height)
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            aspect = w / max(1, h)
            area_ratio = area / crop_area
            comp_mask = labels == label
            sat = float(np.median(hsv[:, :, 1][comp_mask])) if np.any(comp_mask) else 0.0
            value = float(np.median(hsv[:, :, 2][comp_mask])) if np.any(comp_mask) else 0.0
            text_like = (
                (aspect >= 2.4 and h <= crop.height * 0.22 and area_ratio <= 0.18)
                or (sat > 70 and value > 90 and area_ratio <= 0.08 and h <= crop.height * 0.30)
            )
            if text_like:
                filtered[comp_mask] = 0
        cleaned = (filtered * 255).astype(np.uint8)
        cleaned = cv2.GaussianBlur(cleaned, (0, 0), sigmaX=0.8, sigmaY=0.8)
        return Image.fromarray(cleaned, "L")
    except Exception:
        return alpha


def _trim_visual_box_away_from_text_edges(
    visual_box: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = visual_box
    vcx, vcy = _box_center(visual_box)
    min_w = max(24, int((right - left) * 0.42))
    min_h = max(24, int((bottom - top) * 0.42))
    for text_box in text_boxes:
        overlap = _box_overlap(visual_box, text_box)
        if overlap <= 0:
            continue
        tx1, ty1, tx2, ty2 = text_box
        tcx, tcy = _box_center(text_box)
        if tcx < vcx and tx1 <= left + max(10, int((right - left) * 0.22)) and right - max(left, tx2 + 2) >= min_w:
            left = max(left, tx2 + 2)
        elif tcx > vcx and tx2 >= right - max(10, int((right - left) * 0.22)) and min(right, tx1 - 2) - left >= min_w:
            right = min(right, tx1 - 2)
        if tcy < vcy and ty1 <= top + max(10, int((bottom - top) * 0.22)) and bottom - max(top, ty2 + 2) >= min_h:
            top = max(top, ty2 + 2)
        elif tcy > vcy and ty2 >= bottom - max(10, int((bottom - top) * 0.22)) and min(bottom, ty1 - 2) - top >= min_h:
            bottom = min(bottom, ty1 - 2)
    return _clip_box((left, top, right, bottom), source_width, source_height)


def _trim_visual_box_to_product_label_side(
    visual_box: tuple[int, int, int, int],
    product_label_union: tuple[int, int, int, int] | None,
    floating_copy_union: tuple[int, int, int, int] | None,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    if not product_label_union or not floating_copy_union or _box_overlap(visual_box, floating_copy_union) <= 0:
        return visual_box
    left, top, right, bottom = visual_box
    label_left, _label_top, label_right, _label_bottom = product_label_union
    copy_left, _copy_top, copy_right, _copy_bottom = floating_copy_union
    pad = max(4, int((label_right - label_left) * 0.08))
    min_w = max(32, int((label_right - label_left) * 1.18))
    if copy_left >= label_right and copy_left < right:
        candidate_right = max(label_right + pad, copy_left - pad)
        if candidate_right - left >= min_w:
            right = min(right, candidate_right)
    if copy_right <= label_left and copy_right > left:
        candidate_left = min(label_left - pad, copy_right + pad)
        if right - candidate_left >= min_w:
            left = max(left, candidate_left)
    return _clip_box((left, top, right, bottom), source_width, source_height)


def _map_portrait_text_box_to_safe_copy_zone(
    source_box: tuple[int, int, int, int],
    source_union: tuple[int, int, int, int],
    copy_zone: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Map a source text box into a portrait-safe copy zone without locking to source pixels."""
    su_w = max(1, source_union[2] - source_union[0])
    su_h = max(1, source_union[3] - source_union[1])
    cz_w = max(1, copy_zone[2] - copy_zone[0])
    cz_h = max(1, copy_zone[3] - copy_zone[1])
    rel_x1 = (source_box[0] - source_union[0]) / su_w
    rel_y1 = (source_box[1] - source_union[1]) / su_h
    rel_x2 = (source_box[2] - source_union[0]) / su_w
    rel_y2 = (source_box[3] - source_union[1]) / su_h
    mapped = (
        copy_zone[0] + int(round(rel_x1 * cz_w)),
        copy_zone[1] + int(round(rel_y1 * cz_h)),
        copy_zone[0] + int(round(rel_x2 * cz_w)),
        copy_zone[1] + int(round(rel_y2 * cz_h)),
    )
    min_w = max(56, int((source_box[2] - source_box[0]) * 0.62))
    min_h = max(18, int((source_box[3] - source_box[1]) * 0.72))
    if mapped[2] - mapped[0] < min_w:
        mapped = (mapped[0], mapped[1], min(copy_zone[2], mapped[0] + min_w), mapped[3])
    if mapped[3] - mapped[1] < min_h:
        mapped = (mapped[0], mapped[1], mapped[2], min(copy_zone[3], mapped[1] + min_h))
    return _clip_box(mapped, canvas_width, canvas_height)


def _merge_redraw_blocks_by_inline_rows(blocks: list[Any], canvas_width: int) -> list[Any]:
    """Merge OCR-fragmented same-row text blocks into a single inline render block."""
    if len(blocks) <= 1:
        return blocks

    def source_order_box(item: Any) -> tuple[int, int, int, int]:
        box = getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0))
        try:
            if len(box) == 4:
                return tuple(int(value) for value in box)  # type: ignore[arg-type]
        except Exception:
            pass
        return tuple(getattr(item, "bbox", (0, 0, 0, 0)))

    sorted_blocks = sorted(blocks, key=lambda item: (source_order_box(item)[1], source_order_box(item)[0]))
    rows: list[list[Any]] = []
    for block in sorted_blocks:
        box = source_order_box(block)
        if len(box) != 4:
            rows.append([block])
            continue
        cx, cy = _box_center(box)
        placed = False
        for row in rows:
            row_anchor_box = source_order_box(row[0])
            if not row_anchor_box:
                continue
            _, row_cy = _box_center(row_anchor_box)
            row_h = max(1, row_anchor_box[3] - row_anchor_box[1])
            box_h = max(1, box[3] - box[1])
            row_gap_union = _union_boxes([source_order_box(item) for item in row]) or row_anchor_box
            gap = max(0, box[0] - row_gap_union[2], row_gap_union[0] - box[2])
            vertical_overlap = max(0, min(box[3], row_gap_union[3]) - max(box[1], row_gap_union[1]))
            vertical_overlap_ratio = vertical_overlap / max(1, min(box_h, row_h))
            same_baseline = abs(cy - row_cy) <= max(row_h, box_h) * 0.35
            same_text_row = vertical_overlap_ratio >= 0.58 or same_baseline
            if same_text_row and gap <= max(int(canvas_width * 0.055), max(row_h, box_h) * 4):
                row.append(block)
                placed = True
                break
        if not placed:
            rows.append([block])

    merged: list[Any] = []
    for row in rows:
        if len(row) == 1:
            merged.append(row[0])
            continue
        row = sorted(row, key=lambda item: source_order_box(item)[0])
        try:
            base = row[0].model_copy(deep=True)
        except Exception:
            base = row[0]
        target_row_boxes = [tuple(getattr(item, "bbox", (0, 0, 0, 0))) for item in row]
        union = _union_boxes(target_row_boxes)
        if not union:
            merged.extend(row)
            continue
        texts = [str(getattr(item, "translated_text", None) or getattr(item, "text", "")).strip() for item in row if str(getattr(item, "translated_text", None) or getattr(item, "text", "")).strip()]
        base.text = " ".join(texts)
        base.translated_text = base.text
        base.bbox = union
        base.clean_box = union
        base.line_boxes = [union]
        base.line_texts = [base.text]
        base.align = "left"
        source_word_styles: list[dict[str, Any]] = []
        translated_style_spans: list[dict[str, Any]] = []
        for item_index, item in enumerate(row):
            item_words = [dict(word) for word in (getattr(item, "source_word_styles", []) or []) if isinstance(word, dict)]
            item_spans = [dict(span) for span in (getattr(item, "translated_style_spans", []) or []) if isinstance(span, dict)]
            if item_index > 0 and item_spans:
                first_span = item_spans[0]
                first_span["translatedText"] = " " + str(first_span.get("translatedText") or "")
                style = first_span.setdefault("style", {})
                style["text"] = first_span["translatedText"]
            source_word_styles.extend(item_words)
            translated_style_spans.extend(item_spans)
        if source_word_styles:
            base.source_word_styles = source_word_styles
        if translated_style_spans:
            base.translated_style_spans = translated_style_spans
        merged.append(base)
    return sorted(merged, key=lambda item: (source_order_box(item)[1], source_order_box(item)[0]))


def _sort_redraw_blocks_by_source_yx(blocks: list[Any]) -> list[Any]:
    def source_order_box(item: Any) -> tuple[int, int, int, int]:
        box = getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0))
        try:
            if len(box) == 4:
                return tuple(int(value) for value in box)  # type: ignore[arg-type]
        except Exception:
            pass
        return (0, 0, 0, 0)

    return sorted(blocks, key=lambda item: (source_order_box(item)[1], source_order_box(item)[0]))


def _build_display_copy_stack_blocks(
    blocks: list[Any],
    copy_zone: tuple[int, int, int, int],
    *,
    stack_role: str = "primary",
    social: bool = False,
) -> list[Any]:
    ordered = _sort_redraw_blocks_by_source_yx(
        _merge_redraw_blocks_by_inline_rows(blocks, max(1, copy_zone[2] - copy_zone[0]))
    )
    if not ordered:
        return []
    try:
        base = ordered[0].model_copy(deep=True)
    except Exception:
        base = ordered[0]

    copy_w = max(1, copy_zone[2] - copy_zone[0])
    copy_h = max(1, copy_zone[3] - copy_zone[1])
    pad_x = max(4, int(round(copy_w * 0.10)))
    stack_zone = (
        min(copy_zone[2] - 1, copy_zone[0] + pad_x),
        copy_zone[1],
        max(copy_zone[0] + pad_x + 1, copy_zone[2] - pad_x),
        copy_zone[3],
    )
    stack_w = max(1, stack_zone[2] - stack_zone[0])
    stack_h = max(1, stack_zone[3] - stack_zone[1])
    source_texts = [
        str(getattr(block, "translated_text", None) or getattr(block, "text", "") or "").strip()
        for block in ordered
    ]
    source_texts = [text for text in source_texts if text]

    def wrapped_line_count(text: str, font: ImageFont.ImageFont) -> int:
        words = [word for word in text.split() if word]
        if not words:
            return 0
        line_count = 1
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            try:
                width_px = font.getlength(candidate)
            except Exception:
                width_px = len(candidate) * max(1, getattr(font, "size", 10)) * 0.58
            if current and width_px > stack_w:
                line_count += 1
                current = word
            else:
                current = candidate
        return line_count

    spacious_stack = copy_w >= 320 and copy_h >= 220
    is_secondary = stack_role == "secondary"
    max_font_cap = 39 if social and not is_secondary else (30 if is_secondary else 46)
    width_factor = 0.22 if is_secondary else (0.30 if spacious_stack else 0.26)
    height_factor = 0.24 if is_secondary else (0.36 if spacious_stack else 0.46)
    max_font = max(8, min(max_font_cap, int(stack_h * height_factor), int(stack_w * width_factor)))
    stack_font_size = 8
    best_fit: tuple[int, float] | None = None
    for candidate_size in range(max_font, 7, -1):
        font = _load_cta_font(candidate_size)
        total_lines = sum(wrapped_line_count(text, font) for text in source_texts)
        line_height = max(candidate_size + 2, int(round(candidate_size * 1.22)))
        if total_lines * line_height <= stack_h:
            widest = 0.0
            for text in source_texts:
                words = [word for word in text.split() if word]
                current = ""
                for word in words:
                    candidate = word if not current else f"{current} {word}"
                    try:
                        width_px = font.getlength(candidate)
                    except Exception:
                        width_px = len(candidate) * max(1, getattr(font, "size", 10)) * 0.58
                    if current and width_px > stack_w:
                        try:
                            widest = max(widest, font.getlength(current))
                        except Exception:
                            widest = max(widest, len(current) * candidate_size * 0.58)
                        current = word
                    else:
                        current = candidate
                if current:
                    try:
                        widest = max(widest, font.getlength(current))
                    except Exception:
                        widest = max(widest, len(current) * candidate_size * 0.58)
            fill_ratio = widest / max(1, stack_w)
            if best_fit is None:
                best_fit = (candidate_size, fill_ratio)
            if fill_ratio >= (0.62 if is_secondary else 0.70):
                best_fit = (candidate_size, fill_ratio)
                break
    if best_fit is not None:
        stack_font_size = best_fit[0]
    readable_floor = max(9, int(min(copy_w, copy_h) * (0.040 if is_secondary else 0.045)))
    if is_secondary and stack_font_size < readable_floor:
        return []
    stack_line_height = max(stack_font_size + 2, int(round(stack_font_size * 1.22)))

    spans: list[dict[str, Any]] = []
    source_word_styles: list[dict[str, Any]] = []
    texts: list[str] = []
    for block_index, block in enumerate(ordered):
        block_text = str(getattr(block, "translated_text", None) or getattr(block, "text", "") or "").strip()
        if not block_text:
            continue
        texts.append(block_text)
        block_spans = [dict(span) for span in (getattr(block, "translated_style_spans", []) or []) if isinstance(span, dict)]
        if not block_spans:
            block_spans = [
                {
                    "translatedText": block_text,
                    "sourceText": block_text,
                    "style": {
                        "fontSize": stack_font_size,
                        "fontWeight": int(getattr(block, "font_weight", 700) or 700),
                        "fontCategory": "sans-serif",
                        "color": getattr(block, "color", None) or "#111111",
                    },
                }
            ]
        for span_index, span in enumerate(block_spans):
            text = str(span.get("translatedText") or span.get("sourceText") or "").strip()
            if not text:
                continue
            span["translatedText"] = text
            span["sourceText"] = text
            span["forceBreakAfter"] = span_index == len(block_spans) - 1
            style = span.setdefault("style", {})
            style["fontSize"] = stack_font_size
            style["lineHeight"] = stack_line_height
            style["alignment"] = "left"
            updated_source_styles: list[dict[str, Any]] = []
            for source_style in span.get("sourceWordStyles", []) or []:
                if not isinstance(source_style, dict):
                    continue
                cloned_style = dict(source_style)
                cloned_style["fontSize"] = stack_font_size
                cloned_style["peerRowFontSize"] = stack_font_size
                cloned_style["bbox"] = [
                    stack_zone[0],
                    stack_zone[1] + block_index * stack_line_height,
                    stack_zone[2],
                    stack_zone[1] + (block_index + 1) * stack_line_height,
                ]
                cloned_style["lineIndex"] = block_index
                updated_source_styles.append(cloned_style)
                source_word_styles.append(cloned_style)
            if updated_source_styles:
                span["sourceWordStyles"] = updated_source_styles
                span["sourceWordIds"] = [str(item.get("id")) for item in updated_source_styles if item.get("id")]
            spans.append(span)

    if not spans:
        return ordered
    base.text = "\n".join(texts)
    base.translated_text = base.text
    base.bbox = stack_zone
    base.clean_box = stack_zone
    base.line_boxes = [stack_zone]
    base.line_texts = texts
    base.font_size_estimate = stack_font_size
    base.line_height_estimate = stack_line_height
    base.align = "left"
    base.resize_target_fill = 0.62 if is_secondary else 0.74
    base.resize_min_font_size = readable_floor if is_secondary else max(8, int(min(copy_w, copy_h) * 0.034))
    base.resize_max_font_size = max_font_cap
    base.resize_stack_role = stack_role
    base.source_word_styles = source_word_styles
    base.translated_style_spans = spans
    base.render_strategy = "resize_display_copy_stack"
    return [base]


def _paste_crop_exact(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    paste_box: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    feather: int = 0,
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    paste_box = _clip_box(paste_box, output.width, output.height)
    target_w = max(1, paste_box[2] - paste_box[0])
    target_h = max(1, paste_box[3] - paste_box[1])
    crop = source.crop(source_box).convert("RGBA").resize((target_w, target_h), Image.Resampling.LANCZOS)
    if feather > 0:
        alpha = crop.getchannel("A")
        gradient = Image.new("L", crop.size, 255)
        px = gradient.load()
        fw = max(1, min(feather, crop.width // 3, crop.height // 3))
        for y in range(crop.height):
            for x in range(crop.width):
                edge = min(x, y, crop.width - 1 - x, crop.height - 1 - y)
                if edge < fw:
                    px[x, y] = min(px[x, y], int(round(255 * edge / fw)))
        crop.putalpha(ImageChops.multiply(alpha, gradient))
    output.alpha_composite(crop, (paste_box[0], paste_box[1]))
    return {
        "layerId": layer_id,
        "role": role,
        "sourceBox": list(source_box),
        "targetBox": list(paste_box),
        "pasteBox": list(paste_box),
        "scale": round(target_w / max(1, source_box[2] - source_box[0]), 4),
        "fitMode": "relative_visual_element",
    }


def _build_focus_cover_scene(
    clean_source: Image.Image,
    width: int,
    height: int,
    *,
    source_focus_box: tuple[int, int, int, int],
    target_focus_box: tuple[int, int, int, int],
) -> Image.Image:
    source = clean_source.convert("RGB")
    scale = max(width / max(1, source.width), height / max(1, source.height))
    scaled_w = max(1, int(round(source.width * scale)))
    scaled_h = max(1, int(round(source.height * scale)))
    resized = source.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    source_cx, source_cy = _box_center(source_focus_box)
    target_cx, target_cy = _box_center(target_focus_box)
    scaled_focus_x = source_cx * scale
    scaled_focus_y = source_cy * scale
    crop_left = int(round(scaled_focus_x - target_cx))
    crop_top = int(round(scaled_focus_y - target_cy))
    crop_left = max(0, min(max(0, scaled_w - width), crop_left))
    crop_top = max(0, min(max(0, scaled_h - height), crop_top))
    crop = resized.crop((crop_left, crop_top, crop_left + width, crop_top + height))
    if crop.size != (width, height):
        crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    # Very light denoise only; this is not a blurred-fit background.
    return crop.filter(ImageFilter.GaussianBlur(radius=0.18))


def extract_foreground_group(source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, tuple[int, int, int, int], dict[str, Any]]:
    try:
        import cv2

        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        edge_samples = np.concatenate(
            [
                rgb[: max(1, h // 12), :, :].reshape(-1, 3),
                rgb[-max(1, h // 12) :, :, :].reshape(-1, 3),
                rgb[:, : max(1, w // 14), :].reshape(-1, 3),
                rgb[:, -max(1, w // 14) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(edge_samples, axis=0)
        dist = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 35, 120)
        candidate = ((dist > 22) | (edges > 0)).astype(np.uint8) * 255
        text_mask, _ = build_overlay_text_mask(source, analysis)
        candidate[np.array(text_mask, dtype=np.uint8) > 16] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate = cv2.dilate(candidate, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros((h, w), dtype=np.uint8)
        min_area = max(80, int(w * h * 0.0025))
        product_boxes = [layer.bbox.to_pixel_box(w, h) for layer in analysis.product_layers]
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            box = (x, y, x + bw, y + bh)
            overlaps_product = any(not (box[2] < pb[0] or box[0] > pb[2] or box[3] < pb[1] or box[1] > pb[3]) for pb in product_boxes)
            lower_visual = y + bh > int(h * 0.34) and area >= min_area
            if overlaps_product or lower_visual:
                keep[labels == label] = 255
        for box in product_boxes:
            left, top, right, bottom = _clip_box(box, w, h)
            keep[top:bottom, left:right] = np.maximum(keep[top:bottom, left:right], candidate[top:bottom, left:right])
        if int(np.count_nonzero(keep)) < min_area or int(np.count_nonzero(keep)) < int(w * h * 0.045):
            raise ValueError("foreground mask too small")
        ys, xs = np.where(keep > 0)
        left, top, right, bottom = _clip_box((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), w, h)
        pad = max(2, min(18, int(max(right - left, bottom - top) * 0.025)))
        left, top, right, bottom = _clip_box((left - pad, top - pad, right + pad, bottom + pad), w, h)
        crop = source.crop((left, top, right, bottom)).convert("RGBA")
        alpha = Image.fromarray(keep[top:bottom, left:right], "L").filter(ImageFilter.GaussianBlur(radius=0.8))
        crop.putalpha(alpha)
        return crop, (left, top, right, bottom), {
            "foregroundExtraction": "deterministic_group_mask",
            "foregroundPixelCount": int(np.count_nonzero(keep)),
        }
    except Exception as exc:
        if analysis.product_layers:
            crops = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.product_layers]
            left = min(box[0] for box in crops)
            top = min(box[1] for box in crops)
            right = max(box[2] for box in crops)
            bottom = max(box[3] for box in crops)
            box = _clip_box((left, top, right, bottom), source.width, source.height)
            crop = source.crop(box).convert("RGBA")
            crop.putalpha(Image.new("L", crop.size, 255))
            return crop, box, {"foregroundExtraction": "product_bbox_union_fallback", "foregroundExtractionError": str(exc)}
        crop = source.convert("RGBA")
        crop.putalpha(Image.new("L", crop.size, 255))
        return crop, (0, 0, source.width, source.height), {"foregroundExtraction": "full_source_fallback", "foregroundExtractionError": str(exc)}


def crop_text_clean_visual_area(foreground_source: Image.Image, analysis: VisualAnalysis) -> tuple[Image.Image, tuple[int, int, int, int], dict[str, Any]]:
    ratio = foreground_source.width / max(1, foreground_source.height)
    if ratio > 1.65:
        box = _clip_box((int(foreground_source.width * 0.48), 0, foreground_source.width, foreground_source.height), foreground_source.width, foreground_source.height)
        crop = foreground_source.crop(box).convert("RGBA")
        crop.putalpha(Image.new("L", crop.size, 255))
        return crop, box, {"foregroundExtraction": "landscape_visual_region_crop"}

    text_boxes = [layer.bbox.to_pixel_box(foreground_source.width, foreground_source.height) for layer in analysis.marketing_text_layers]
    product_boxes = [layer.bbox.to_pixel_box(foreground_source.width, foreground_source.height) for layer in analysis.product_layers]
    if text_boxes:
        text_bottom = max(box[3] for box in text_boxes)
        visual_top = max(0, min(text_bottom - int(foreground_source.height * 0.08), int(foreground_source.height * 0.34)))
        if visual_top > int(foreground_source.height * 0.58):
            visual_top = int(foreground_source.height * 0.32)
        box = _clip_box((0, visual_top, foreground_source.width, foreground_source.height), foreground_source.width, foreground_source.height)
    elif product_boxes:
        top = min(box[1] for box in product_boxes)
        bottom = max(box[3] for box in product_boxes)
        left = min(box[0] for box in product_boxes)
        right = max(box[2] for box in product_boxes)
        pad_x = int(foreground_source.width * 0.14)
        pad_y = int(foreground_source.height * 0.10)
        box = _clip_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), foreground_source.width, foreground_source.height)
    else:
        box = _clip_box((0, int(foreground_source.height * 0.28), foreground_source.width, foreground_source.height), foreground_source.width, foreground_source.height)
    crop = foreground_source.crop(box).convert("RGBA")
    crop.putalpha(Image.new("L", crop.size, 255))
    return crop, box, {"foregroundExtraction": "text_clean_visual_area_crop"}


def composite_relayout_layers(
    canvas: Image.Image,
    source: Image.Image,
    foreground_source: Image.Image,
    plan: ReframePlan,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    layers = _layer_by_id(analysis)
    composited = []
    product_placements = [placement for placement in plan.placements if placement.role in {LayerRole.PRODUCT, LayerRole.PERSON}]
    if product_placements:
        target_union = (
            min(p.target_bbox.to_pixel_box(output.width, output.height)[0] for p in product_placements),
            min(p.target_bbox.to_pixel_box(output.width, output.height)[1] for p in product_placements),
            max(p.target_bbox.to_pixel_box(output.width, output.height)[2] for p in product_placements),
            max(p.target_bbox.to_pixel_box(output.width, output.height)[3] for p in product_placements),
        )
        crop, source_box, extraction_meta = crop_text_clean_visual_area(foreground_source, analysis)
        paste_left, paste_top, paste_right, paste_bottom, scale = _fit_inside(crop.size, target_union)
        resized = crop.resize((paste_right - paste_left, paste_bottom - paste_top), Image.Resampling.LANCZOS)
        output.alpha_composite(resized, (paste_left, paste_top))
        composited.append(
            {
                "layerId": "foreground-group",
                "role": "foreground_group",
                "sourceBox": list(source_box),
                "targetBox": list(target_union),
                "pasteBox": [paste_left, paste_top, paste_right, paste_bottom],
                "scale": round(scale, 4),
                **extraction_meta,
            }
        )
    for placement in sorted(plan.placements, key=lambda item: item.z_index):
        if placement.role not in {LayerRole.LOGO}:
            continue
        layer = layers.get(placement.layer_id)
        if layer is None:
            continue
        crop, source_box = _extract_layer_crop(foreground_source, layer)
        target_box = placement.target_bbox.to_pixel_box(output.width, output.height)
        paste_left, paste_top, paste_right, paste_bottom, scale = _fit_inside(crop.size, target_box)
        resized = crop.resize((paste_right - paste_left, paste_bottom - paste_top), Image.Resampling.LANCZOS)
        if placement.add_shadow:
            shadow = Image.new("RGBA", output.size, (0, 0, 0, 0))
            shadow_alpha = resized.getchannel("A").filter(ImageFilter.GaussianBlur(radius=max(2, output.width // 180)))
            shadow_patch = Image.new("RGBA", resized.size, (0, 0, 0, 58))
            shadow_patch.putalpha(shadow_alpha)
            shadow.alpha_composite(shadow_patch, (paste_left + max(2, output.width // 140), paste_top + max(2, output.height // 140)))
            output = Image.alpha_composite(output, shadow)
        output.alpha_composite(resized, (paste_left, paste_top))
        composited.append(
            {
                "layerId": placement.layer_id,
                "role": placement.role.value,
                "sourceBox": list(source_box),
                "targetBox": list(target_box),
                "pasteBox": [paste_left, paste_top, paste_right, paste_bottom],
                "scale": round(scale, 4),
            }
        )
    return output.convert("RGB"), {"compositedLayers": composited, "compositedLayerCount": len(composited)}


def should_use_vertical_band_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    if target_ratio >= 0.75 and not (target_ratio <= 1.25 and source_ratio > 1.35):
        return False
    if not analysis.marketing_text_layers:
        return False
    text_area = sum(layer.bbox.area_ratio() for layer in analysis.marketing_text_layers)
    return abs(source_ratio - target_ratio) >= 0.42 or text_area >= 0.10


def should_use_source_fit(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    if abs(source_ratio - target_ratio) <= 0.18:
        return True
    if target_ratio >= 0.75 and target_ratio <= 1.25 and abs(source_ratio - target_ratio) <= 0.32:
        return True
    return not analysis.marketing_text_layers and abs(source_ratio - target_ratio) <= 0.45


def should_use_landscape_width_anchor(source: Image.Image, width: int, height: int) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    return source_ratio > 1.35 and target_ratio > 1.35 and source_ratio > target_ratio and abs(source_ratio - target_ratio) > 0.18


def suppress_generated_foreground_below_anchor(output: Image.Image, source: Image.Image, paste_h: int, width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
    if paste_h >= height - 4:
        return output, {"generatedForegroundSuppression": "skipped_no_gap"}
    try:
        import cv2

        start_y = max(0, min(height - 1, paste_h - max(8, height // 80)))
        region = np.array(output.crop((0, start_y, width, height)).convert("RGB"), dtype=np.uint8)
        if region.size == 0:
            return output, {"generatedForegroundSuppression": "skipped_empty_region"}

        replacement = build_deterministic_background_canvas(source, width, height).convert("RGBA").crop((0, start_y, width, height))
        replacement_np = np.array(replacement.convert("RGB"), dtype=np.uint8)
        color_delta = np.linalg.norm(region.astype(np.float32) - replacement_np.astype(np.float32), axis=2)
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 45, 130)
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        high_detail = ((edges > 0) | (saturation > 62) | (color_delta > 30)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        high_detail = cv2.dilate(high_detail, kernel, iterations=2)
        high_detail = cv2.morphologyEx(high_detail, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((high_detail > 0).astype(np.uint8), connectivity=8)
        mask = np.zeros_like(high_detail)
        min_area = max(90, int(width * height * 0.0012))
        suppressed_components = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            comp_y = int(stats[label, cv2.CC_STAT_TOP])
            comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < min_area:
                continue
            if comp_y + comp_h < max(10, region.shape[0] // 5):
                continue
            mask[labels == label] = 255
            suppressed_components += 1
        if not suppressed_components:
            return output, {"generatedForegroundSuppression": "no_generated_foreground_detected"}

        soft_mask = Image.fromarray(mask, "L").filter(ImageFilter.GaussianBlur(radius=max(3, width // 260)))
        cleaned = output.convert("RGBA")
        cleaned_region = cleaned.crop((0, start_y, width, height))
        cleaned_region.paste(replacement, (0, 0), soft_mask)
        cleaned.paste(cleaned_region, (0, start_y))
        return cleaned, {
            "generatedForegroundSuppression": "removed_high_detail_generated_area",
            "generatedForegroundComponents": suppressed_components,
            "generatedForegroundStartY": start_y,
        }
    except Exception as exc:
        return output, {"generatedForegroundSuppression": "failed_passthrough", "generatedForegroundSuppressionError": str(exc)}


def composite_landscape_width_anchor(
    canvas: Image.Image,
    source: Image.Image,
    width: int,
    height: int,
    *,
    fill_source: Image.Image | None = None,
    preserve_canvas_fill: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    source_rgba = source.convert("RGBA")
    cover_scale = max(width / max(1, source.width), height / max(1, source.height))
    crop_left = 0
    crop_top = 0
    crop_right = width
    crop_bottom = height
    if preserve_canvas_fill:
        output = canvas.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
        fill_mode = "generative_canvas_underlay"
    else:
        fill_rgba = (fill_source or source).convert("RGBA")
        cover_w = max(1, int(round(source.width * cover_scale)))
        cover_h = max(1, int(round(source.height * cover_scale)))
        cover = fill_rgba.resize((cover_w, cover_h), Image.Resampling.LANCZOS)
        crop_left = max(0, (cover_w - width) // 2)
        crop_right = min(cover_w, crop_left + width)
        crop_bottom = min(cover_h, crop_top + height)
        output = cover.crop((crop_left, crop_top, crop_right, crop_bottom))
        if output.size != (width, height):
            output = output.resize((width, height), Image.Resampling.LANCZOS)
        fill_mode = "cover_clean_content_underlay"

    scale = width / max(1, source.width)
    paste_w = width
    paste_h = max(1, int(round(source.height * scale)))
    resized = source_rgba.resize((paste_w, paste_h), Image.Resampling.LANCZOS)
    if paste_h >= height:
        crop_top = 0
        resized = resized.crop((0, crop_top, paste_w, crop_top + height))
        paste_h = height
    elif paste_h < height:
        seam = max(10, min(36, height // 24, paste_h // 5))
        alpha = Image.new("L", resized.size, 255)
        alpha_px = alpha.load()
        for y in range(max(0, paste_h - seam), paste_h):
            t = (y - (paste_h - seam)) / max(1, seam - 1)
            value = int(round(255 * (1.0 - t)))
            for x in range(paste_w):
                alpha_px[x, y] = value
        resized.putalpha(alpha)
    suppression_meta: dict[str, Any] = {}
    if preserve_canvas_fill and paste_h < height:
        output, suppression_meta = suppress_generated_foreground_below_anchor(output, source, paste_h, width, height)
    output.alpha_composite(resized, (0, 0))
    return output.convert("RGB"), {
        "compositedLayers": [
            {
                "layerId": "source-landscape-width-anchor",
                "role": "preserved_landscape_width_anchor",
                "sourceBox": [0, 0, source.width, source.height],
                "targetBox": [0, 0, width, height],
                "pasteBox": [0, 0, paste_w, paste_h],
                "scale": round(scale, 4),
                "fillScale": round(cover_scale, 4),
                "fillCropBox": [crop_left, crop_top, crop_right, crop_bottom],
                "fillMode": fill_mode,
            }
        ],
        "compositedLayerCount": 1,
        "landscapeWidthAnchor": True,
        **suppression_meta,
    }


def composite_source_fit(canvas: Image.Image, source: Image.Image, width: int, height: int) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    paste_left, paste_top, paste_right, paste_bottom, scale = _fit_inside(source.size, (0, 0, width, height))
    resized = source.convert("RGBA").resize((paste_right - paste_left, paste_bottom - paste_top), Image.Resampling.LANCZOS)
    output.alpha_composite(resized, (paste_left, paste_top))
    return output.convert("RGB"), {
        "compositedLayers": [
            {
                "layerId": "source-fit",
                "role": "preserved_source_fit",
                "sourceBox": [0, 0, source.width, source.height],
                "targetBox": [0, 0, width, height],
                "pasteBox": [paste_left, paste_top, paste_right, paste_bottom],
                "scale": round(scale, 4),
            }
        ],
        "compositedLayerCount": 1,
        "sourceFit": True,
    }


def composite_full_creative_preserve(
    background_source: Image.Image,
    source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    target_area_smaller = width * height < source.width * source.height
    if target_area_smaller:
        edge_rgb = tuple(int(value) for value in _edge_median_rgb(source))
        underlay = Image.new("RGBA", (width, height), (*edge_rgb, 255))
    elif source_ratio <= 1.25 and target_ratio < 0.78:
        underlay = build_deterministic_background_canvas(background_source, width, height).convert("RGBA")
    else:
        underlay = _build_focus_cover_scene(
            background_source,
            width,
            height,
            source_focus_box=visual_box,
            target_focus_box=(0, 0, width, height),
        ).convert("RGBA")
    if target_ratio < 0.78 and source_ratio > 1.25:
        scale = min(width / max(1, source.width), height / max(1, source.height))
        max_band_h = int(height * 0.34)
        if int(round(source.height * scale)) > max_band_h:
            scale = max_band_h / max(1, source.height)
        paste_w = max(1, int(round(source.width * scale)))
        paste_h = max(1, int(round(source.height * scale)))
        paste_x = (width - paste_w) // 2
        paste_y = int(round(height * 0.36))
    else:
        scale = min(width / max(1, source.width), height / max(1, source.height))
        paste_w = max(1, int(round(source.width * scale)))
        paste_h = max(1, int(round(source.height * scale)))
        paste_x = (width - paste_w) // 2
        paste_y = (height - paste_h) // 2
    paste_y = max(0, min(height - paste_h, paste_y))
    preserved = source.convert("RGBA").resize((paste_w, paste_h), Image.Resampling.LANCZOS)
    underlay.alpha_composite(preserved, (paste_x, paste_y))
    return underlay.convert("RGB"), {
        "compositedLayers": [
            {
                "layerId": "source-full-creative-preserve",
                "role": "full_creative_preserved",
                "sourceBox": [0, 0, source.width, source.height],
                "targetBox": [0, 0, width, height],
                "pasteBox": [paste_x, paste_y, paste_x + paste_w, paste_y + paste_h],
                "scale": round(scale, 4),
                "fitMode": "full_creative_over_focus_cover_underlay",
            }
        ],
        "compositedLayerCount": 1,
        "fullCreativePreserve": True,
        **visual_meta,
    }


def _target_area_is_smaller(source: Image.Image, width: int, height: int) -> bool:
    return width * height < source.width * source.height


def _role_aware_layout_zones(
    width: int,
    height: int,
    *,
    target_ratio: float,
    source_ratio: float,
    target_smaller: bool,
) -> dict[str, tuple[int, int, int, int]]:
    margin_x = max(10, int(width * 0.055))
    margin_y = max(8, int(height * 0.045))
    if target_smaller and 0.92 <= target_ratio <= 1.08 and source_ratio > 1.25:
        split_x = width // 2
        left_pad = max(14, int(width * 0.06))
        right_pad = max(12, int(width * 0.045))
        top_pad = max(14, int(height * 0.07))
        bottom_pad = max(14, int(height * 0.07))
        return {
            "brand": (left_pad, top_pad, split_x - left_pad, int(height * 0.16)),
            "badge": (left_pad, top_pad, split_x - left_pad, int(height * 0.19)),
            "copy": (left_pad, int(height * 0.18), split_x - left_pad, int(height * 0.57)),
            "secondary": (left_pad, int(height * 0.61), split_x - left_pad, height - bottom_pad),
            "cta": (left_pad, int(height * 0.77), split_x - left_pad, height - bottom_pad),
            "visual": (split_x + right_pad, top_pad, width - right_pad, height - bottom_pad),
        }
    if target_ratio < 0.78:
        if width <= 420 or target_ratio < 0.45:
            usable_top = margin_y
            usable_bottom = height - margin_y
            gap = max(8, int(height * 0.018))
            brand_bottom = usable_top + max(20, int(height * 0.08))
            copy_top = brand_bottom + gap
            copy_bottom = copy_top + max(70, int(height * 0.22))
            visual_top = copy_bottom + gap
            visual_bottom = visual_top + max(120, int(height * 0.29))
            secondary_top = visual_bottom + gap
            secondary_bottom = secondary_top + max(64, int(height * 0.16))
            cta_top = secondary_bottom + gap
            cta_bottom = cta_top + max(36, int(height * 0.09))
            return {
                "brand": (margin_x, usable_top, width - margin_x, min(brand_bottom, usable_bottom)),
                "badge": (margin_x, usable_top, width - margin_x, min(brand_bottom, usable_bottom)),
                "copy": (margin_x, min(copy_top, usable_bottom - 1), width - margin_x, min(copy_bottom, usable_bottom)),
                "visual": (margin_x, min(visual_top, usable_bottom - 1), width - margin_x, min(visual_bottom, usable_bottom)),
                "cta": (margin_x, min(cta_top, usable_bottom - 1), width - margin_x, min(cta_bottom, usable_bottom)),
                "secondary": (margin_x, min(secondary_top, usable_bottom - 1), width - margin_x, min(secondary_bottom, usable_bottom)),
            }
        return {
            "brand": (margin_x, margin_y, width - margin_x, int(height * 0.145)),
            "badge": (margin_x, margin_y, width - margin_x, int(height * 0.145)),
            "copy": (margin_x, int(height * 0.10), width - margin_x, int(height * 0.30)),
            "cta": (margin_x, int(height * 0.30), width - margin_x, int(height * 0.38)),
            "visual": (int(width * 0.035), int(height * 0.32), width - int(width * 0.035), int(height * 0.975)),
            "secondary": (margin_x, int(height * 0.70), width - margin_x, int(height * 0.92)),
        }
    if target_smaller and source_ratio > 1.25 and target_ratio >= 0.86:
        if width <= 360 or height <= 280:
            return {
                "brand": (margin_x, margin_y, int(width * 0.56), int(height * 0.22)),
                "badge": (int(width * 0.66), margin_y, width - margin_x, int(height * 0.23)),
                "copy": (margin_x, int(height * 0.24), int(width * 0.59), int(height * 0.68)),
                "cta": (margin_x, int(height * 0.70), int(width * 0.62), height - margin_y),
                "visual": (int(width * 0.60), int(height * 0.16), width - margin_x, int(height * 0.62)),
                "secondary": (int(width * 0.60), int(height * 0.64), width - margin_x, height - margin_y),
            }
        return {
            "brand": (margin_x, margin_y, int(width * 0.46), int(height * 0.20)),
            "badge": (int(width * 0.66), margin_y, width - margin_x, int(height * 0.22)),
            "copy": (margin_x, int(height * 0.22), int(width * 0.48), height - margin_y),
            "cta": (margin_x, int(height * 0.76), int(width * 0.50), height - margin_y),
            "visual": (int(width * 0.47), int(height * 0.12), width - margin_x, height - margin_y),
            "secondary": (int(width * 0.62), int(height * 0.52), width - margin_x, height - margin_y),
        }
    if target_ratio <= 1.25:
        return {
            "brand": (margin_x, margin_y, width - margin_x, int(height * 0.16)),
            "badge": (int(width * 0.64), margin_y, width - margin_x, int(height * 0.18)),
            "copy": (margin_x, int(height * 0.16), width - margin_x, int(height * 0.38)),
            "cta": (margin_x, int(height * 0.40), int(width * 0.58), int(height * 0.53)),
            "visual": (int(width * 0.16), int(height * 0.38), width - int(width * 0.16), height - margin_y),
            "secondary": (margin_x, int(height * 0.70), width - margin_x, height - margin_y),
        }
    if source_ratio > 1.25:
        return {
            "brand": (margin_x, margin_y, width - margin_x, int(height * 0.17)),
            "badge": (int(width * 0.68), margin_y, width - margin_x, int(height * 0.19)),
            "copy": (margin_x, int(height * 0.18), int(width * (0.50 if not target_smaller else 0.48)), int(height * 0.86)),
            "cta": (margin_x, int(height * 0.76), int(width * 0.52), int(height * 0.92)),
            "visual": (int(width * (0.47 if not target_smaller else 0.50)), int(height * 0.10), width - margin_x, int(height * 0.93)),
            "secondary": (int(width * 0.58), int(height * 0.52), width - margin_x, int(height * 0.90)),
        }
    return {
        "brand": (margin_x, margin_y, width - margin_x, int(height * 0.16)),
        "badge": (int(width * 0.68), margin_y, width - margin_x, int(height * 0.18)),
        "copy": (margin_x, int(height * 0.16), int(width * 0.50), int(height * 0.88)),
        "cta": (margin_x, int(height * 0.76), int(width * 0.52), int(height * 0.92)),
        "visual": (int(width * 0.45), int(height * 0.12), width - margin_x, int(height * 0.90)),
        "secondary": (int(width * 0.58), int(height * 0.56), width - margin_x, int(height * 0.90)),
    }


def _fit_box_has_room(target: tuple[int, int, int, int], *, min_w: int, min_h: int) -> bool:
    return (target[2] - target[0]) >= min_w and (target[3] - target[1]) >= min_h


def _load_cta_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=max(6, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_display_cta_button(
    output: Image.Image,
    zone: tuple[int, int, int, int],
    *,
    label: str = "Learn More",
) -> dict[str, Any] | None:
    zone = _clip_box(zone, output.width, output.height)
    zone_w = max(1, zone[2] - zone[0])
    zone_h = max(1, zone[3] - zone[1])
    if zone_w < 54 or zone_h < 18:
        return None
    button_w = min(zone_w, max(58, int(zone_w * 0.88)))
    button_h = min(zone_h, max(20, int(zone_h * 0.68)))
    left = zone[0] + (zone_w - button_w) // 2
    top = zone[1] + (zone_h - button_h) // 2
    right = left + button_w
    bottom = top + button_h
    draw = ImageDraw.Draw(output)
    radius = max(5, min(button_h // 2, 14))
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=(0, 80, 145, 255))
    stroke = max(1, button_h // 18)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, outline=(255, 255, 255, 235), width=stroke)
    font_size = max(7, min(int(button_h * 0.42), 18))
    font = _load_cta_font(font_size)
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    while (text_w > button_w * 0.78 or text_h > button_h * 0.72) and font_size > 6:
        font_size -= 1
        font = _load_cta_font(font_size)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
    text_x = left + (button_w - text_w) // 2
    text_y = top + (button_h - text_h) // 2 - text_box[1]
    draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)
    return {
        "layerId": "display-cta-button",
        "role": "display_cta_button",
        "sourceBox": [],
        "targetBox": list(zone),
        "pasteBox": [left, top, right, bottom],
        "scale": None,
        "fitMode": "deterministic_cta",
    }


def _resolve_display_cta_zone(
    zone: tuple[int, int, int, int],
    *,
    drawn_text_boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    margin_x: int,
    margin_y: int,
) -> tuple[int, int, int, int] | None:
    zone = _clip_box(zone, width, height)
    zone_w = max(1, zone[2] - zone[0])
    zone_h = max(1, zone[3] - zone[1])
    if not drawn_text_boxes:
        return zone
    blocking_boxes = [
        _clip_box(box, width, height)
        for box in drawn_text_boxes
        if box[2] > box[0] and box[3] > box[1]
    ]
    if not blocking_boxes:
        return zone
    safe_gap = max(6, int(round(height * 0.035)))

    def has_collision(candidate: tuple[int, int, int, int]) -> bool:
        expanded = (candidate[0], candidate[1] - safe_gap, candidate[2], candidate[3] + safe_gap)
        return any(_box_overlap(expanded, box) > 0 for box in blocking_boxes)

    if not has_collision(zone):
        return zone

    max_text_bottom = max(box[3] for box in blocking_boxes)
    below_top = max(zone[1], max_text_bottom + safe_gap)
    below = _clip_box((zone[0], below_top, zone[2], below_top + zone_h), width, height)
    if below[3] <= height - margin_y and not has_collision(below):
        return below

    min_text_top = min(box[1] for box in blocking_boxes)
    above_bottom = min(zone[3], min_text_top - safe_gap)
    above = _clip_box((zone[0], above_bottom - zone_h, zone[2], above_bottom), width, height)
    if above[1] >= margin_y and not has_collision(above):
        return above

    fallback_left = margin_x
    fallback_right = min(width - margin_x, fallback_left + zone_w)
    fallback_top = max_text_bottom + safe_gap
    fallback = _clip_box((fallback_left, fallback_top, fallback_right, fallback_top + zone_h), width, height)
    if fallback[3] <= height - margin_y and not has_collision(fallback):
        return fallback
    return None


def _localized_display_cta_label(analysis: VisualAnalysis) -> str:
    text = " ".join(str(getattr(layer, "original_text", "") or "") for layer in analysis.marketing_text_layers)
    upper = text.upper()
    turkish_signals = (
        "İ",
        "Ş",
        "Ğ",
        "Ü",
        "Ö",
        "Ç",
        " DAHIL",
        " DAHİL",
        " CILT",
        " CİLT",
        " GUNES",
        " GÜNEŞ",
        " KORUMA",
        " HIZLI",
        " SIVILCE",
        " SİVİLCE",
    )
    if any(signal in upper for signal in turkish_signals):
        return "İncele"
    return "Learn More"


def _draw_programmatic_rtb_guides(
    output: Image.Image,
    product_box: tuple[int, int, int, int] | None,
    rtb_box: tuple[int, int, int, int] | None,
    *,
    color: tuple[int, int, int] = (20, 86, 128),
) -> dict[str, Any] | None:
    if not product_box or not rtb_box:
        return None
    product_box = _clip_box(product_box, output.width, output.height)
    rtb_box = _clip_box(rtb_box, output.width, output.height)
    if product_box[2] <= product_box[0] or product_box[3] <= product_box[1] or rtb_box[2] <= rtb_box[0] or rtb_box[3] <= rtb_box[1]:
        return None
    product_cx, product_cy = _box_center(product_box)
    rtb_cx, rtb_cy = _box_center(rtb_box)
    line_width = 1
    max_len = max(14, int(round(min(output.width, output.height) * 0.12)))
    horizontal_overlap = not (rtb_box[2] <= product_box[0] or rtb_box[0] >= product_box[2])
    vertical_overlap = not (rtb_box[3] <= product_box[1] or rtb_box[1] >= product_box[3])
    if horizontal_overlap and rtb_box[1] >= product_box[3]:
        gap = rtb_box[1] - product_box[3]
        if gap <= 4:
            return None
        x = int(max(rtb_box[0], min(rtb_box[2], product_cx)))
        start = (x, rtb_box[1])
        end = (x, max(product_box[3] + 2, rtb_box[1] - min(max_len, gap - 2)))
    elif horizontal_overlap and rtb_box[3] <= product_box[1]:
        gap = product_box[1] - rtb_box[3]
        if gap <= 4:
            return None
        x = int(max(rtb_box[0], min(rtb_box[2], product_cx)))
        start = (x, rtb_box[3])
        end = (x, min(product_box[1] - 2, rtb_box[3] + min(max_len, gap - 2)))
    elif rtb_cx >= product_cx:
        gap = rtb_box[0] - product_box[2]
        if gap <= 4:
            return None
        start = (rtb_box[0], int(rtb_cy))
        end = (max(product_box[2] + 2, rtb_box[0] - min(max_len, gap - 2)), int(rtb_cy))
    else:
        gap = product_box[0] - rtb_box[2]
        if gap <= 4:
            return None
        start = (rtb_box[2], int(rtb_cy))
        end = (min(product_box[0] - 2, rtb_box[2] + min(max_len, gap - 2)), int(rtb_cy))
    line_length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    if line_length < max(8, int(min(output.width, output.height) * 0.018)):
        return None
    draw = ImageDraw.Draw(output)
    draw.line([start, end], fill=color + (210,), width=line_width)
    return {
        "layerId": "programmatic-rtb-guide",
        "role": "programmatic_rtb_connector",
        "sourceBox": [],
        "targetBox": list(rtb_box),
        "pasteBox": [min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1])],
        "scale": None,
        "fitMode": "vector_line",
    }


def composite_priority_layer_resize(
    background_source: Image.Image,
    source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
    *,
    text_blocks: list[Any] | None = None,
    draw_text: TextRenderer | None = None,
    preserve_brand_layers: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    """General resize compositor: background first, product/text first, secondary only if room remains."""
    source_ratio = source.width / max(1, source.height)
    target_ratio = width / max(1, height)
    target_smaller = _target_area_is_smaller(source, width, height)
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    parts = _partition_resize_layers(source, analysis, visual_box)
    visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
    meaningful_visuals = [
        box for box in visual_elements if _box_area(box) / max(1, source.width * source.height) >= 0.014
    ] or [visual_box]
    compact_target = target_smaller and (width <= 360 or height <= 280)
    product_label_focus = _visual_focus_from_product_label_union(
        source,
        _union_boxes(parts["product_label"]),
        visual_box,
        compact_target=compact_target,
    )
    if product_label_focus:
        product_label_focus = _trim_visual_box_to_product_label_side(
            product_label_focus,
            _union_boxes(parts["product_label"]),
            _union_boxes(parts["secondary"]),
            source.width,
            source.height,
        )
        meaningful_visuals = [product_label_focus]

    text_remove_boxes = [
        *parts["brand"],
        *parts.get("trust_badge", []),
        *parts["primary"],
        *parts["secondary"],
    ]
    background_clean = _remove_foreground_visuals_for_background(
        _inpaint_rectangular_overlays(source, text_remove_boxes),
        meaningful_visuals,
    )
    if background_source.size == (width, height):
        output = background_source.convert("RGBA")
    else:
        output = build_deterministic_background_canvas(background_clean, width, height).convert("RGBA")

    zones = _role_aware_layout_zones(
        width,
        height,
        target_ratio=target_ratio,
        source_ratio=source_ratio,
        target_smaller=target_smaller,
    )
    composited: list[dict[str, Any]] = []

    brand_scale = max(0.20, min(1.35, (width * height / max(1, source.width * source.height)) ** 0.5))
    if preserve_brand_layers:
        for index, box in enumerate(parts["brand"][:4]):
            if not _fit_box_has_room(zones["brand"], min_w=max(12, width // 14), min_h=max(8, height // 34)):
                break
            composited.append(
                _paste_layer_relative(
                    output,
                    source,
                    box,
                    target_bounds=zones["brand"],
                    layer_id=f"brand-{index}",
                    role="brand_or_badge_preserved",
                    scale=brand_scale,
                    preserve_source_position=True,
                )
            )
    badge_floor = 0.34 if preserve_brand_layers else 0.52
    badge_scale = max(badge_floor, min(1.32, (width * height / max(1, source.width * source.height)) ** 0.5))
    for index, box in enumerate(parts.get("trust_badge", [])[:3]):
        if not _fit_box_has_room(zones["badge"], min_w=max(12, width // 16), min_h=max(8, height // 42)):
            break
        composited.append(
            _paste_layer_relative(
                output,
                source,
                box,
                target_bounds=zones["badge"],
                layer_id=f"trust-badge-{index}",
                role="trust_badge_preserved",
                scale=badge_scale,
                preserve_source_position=True,
            )
        )

    display_cta_zone = zones["cta"] if preserve_brand_layers else None

    redraw_blocks: list[Any] = []
    # Very narrow canvases need programmatic redraw for legibility. Provider
    # outpaint still never sees text, and redraw order is locked to source Y/X.
    use_redraw_for_small_text = bool(text_blocks and draw_text) and target_smaller
    if target_smaller and not use_redraw_for_small_text:
        text_artwork_source = _remove_foreground_visuals_for_background(source, meaningful_visuals)
        primary_union = _union_boxes(parts["primary"])
        if primary_union:
            padded = _pad_box(
                primary_union,
                source.width,
                source.height,
                pad_x=max(3, int((primary_union[2] - primary_union[0]) * 0.06)),
                pad_y=max(2, int((primary_union[3] - primary_union[1]) * 0.14)),
            )
            composited.append(
                _paste_crop_contain_limited(
                    output,
                    text_artwork_source,
                    padded,
                    zones["copy"],
                    layer_id="primary-copy-artwork",
                    role="primary_marketing_copy_artwork_preserved",
                    max_scale=1.0,
                    artwork_alpha=False,
                )
            )
        secondary_union = _union_boxes(parts["secondary"])
        compact_display = target_smaller and (width < 360 or height < 280)
        if (
            secondary_union
            and not compact_display
            and target_ratio >= 1.1
            and _fit_box_has_room(zones["secondary"], min_w=max(60, width // 5), min_h=max(18, height // 12))
        ):
            padded = _pad_box(
                secondary_union,
                source.width,
                source.height,
                pad_x=max(3, int((secondary_union[2] - secondary_union[0]) * 0.06)),
                pad_y=max(2, int((secondary_union[3] - secondary_union[1]) * 0.14)),
            )
            composited.append(
                _paste_crop_contain_limited(
                    output,
                    text_artwork_source,
                    padded,
                    zones["secondary"],
                    layer_id="secondary-copy-artwork",
                    role="secondary_marketing_copy_artwork_preserved",
                    max_scale=1.0,
                    artwork_alpha=False,
                )
            )
    if (not target_smaller or use_redraw_for_small_text) and text_blocks and draw_text:
        layer_boxes = {layer.id: _layer_box(layer, source) for layer in analysis.marketing_text_layers}
        primary_union = _union_boxes(parts["primary"])
        secondary_union = _union_boxes(parts["secondary"])
        product_label_union = _union_boxes(parts["product_label"])
        brand_union = _union_boxes(parts["brand"])
        trust_badge_union = _union_boxes(parts.get("trust_badge", []))
        text_copy_union = primary_union or _union_boxes([box for box in layer_boxes.values() if product_label_union is None or _box_overlap(box, product_label_union) / max(1, _box_area(box)) < 0.42])
        for block in text_blocks:
            layer_id = str(getattr(block, "id", "")).replace("v5-resize-", "")
            source_box = layer_boxes.get(layer_id)
            if not source_box:
                continue
            brand_overlap = (
                _box_overlap(source_box, brand_union) / max(1, _box_area(source_box))
                if brand_union
                else 0.0
            )
            if brand_overlap >= 0.35:
                continue
            trust_badge_overlap = (
                _box_overlap(source_box, trust_badge_union) / max(1, _box_area(source_box))
                if trust_badge_union
                else 0.0
            )
            if trust_badge_overlap >= 0.35:
                continue
            product_label_overlap = (
                _box_overlap(source_box, product_label_union) / max(1, _box_area(source_box))
                if product_label_union
                else 0.0
            )
            if product_label_overlap >= 0.42:
                continue
            if secondary_union and _box_overlap(source_box, secondary_union) / max(1, _box_area(source_box)) >= 0.35:
                target_box = _map_box_between_unions(source_box, secondary_union, zones["secondary"], width, height)
            elif text_copy_union:
                target_box = _map_box_between_unions(source_box, text_copy_union, zones["copy"], width, height)
            else:
                continue
            try:
                cloned = block.model_copy(deep=True)
            except Exception:
                cloned = block
            target_box = _clip_box(target_box, width, height)
            cloned.bbox = target_box
            cloned.clean_box = target_box
            cloned.line_boxes = [target_box]
            source_h = max(1, source_box[3] - source_box[1])
            canvas_scale = max(0.22, min(1.65, (width * height / max(1, source.width * source.height)) ** 0.5))
            font_size = max(7, min(160, int(round(source_h * 0.72 * canvas_scale))))
            available_h = max(1, target_box[3] - target_box[1])
            if font_size > available_h * 0.92:
                font_size = max(7, int(available_h * 0.92))
            if target_smaller:
                font_size = max(6, min(font_size, int(max(7, min(width, height) * 0.044))))
            if not preserve_brand_layers:
                font_size = max(font_size, int(max(14, min(width, height) * (0.036 if target_ratio >= 0.78 else 0.040))))
            cloned.font_size_estimate = font_size
            cloned.line_height_estimate = int(round(font_size * 1.18))
            for style in getattr(cloned, "source_word_styles", []) or []:
                if isinstance(style, dict):
                    style["fontSize"] = font_size
                    style["peerRowFontSize"] = font_size
                    style["bbox"] = [target_box[0], target_box[1], target_box[2], target_box[3]]
            for span in getattr(cloned, "translated_style_spans", []) or []:
                if isinstance(span, dict):
                    span_style = span.setdefault("style", {})
                    span_style["fontSize"] = font_size
                    span_style["lineHeight"] = cloned.line_height_estimate
                    for source_style in span.get("sourceWordStyles", []) or []:
                        if isinstance(source_style, dict):
                            source_style["fontSize"] = font_size
                            source_style["peerRowFontSize"] = font_size
                            source_style["bbox"] = [target_box[0], target_box[1], target_box[2], target_box[3]]
            redraw_blocks.append(cloned)

    visual_bounds = zones["visual"]
    visual_alpha_cut_boxes = [*parts["brand"], *parts.get("trust_badge", []), *parts["primary"], *parts["secondary"]]
    if target_ratio < 0.45 and len(meaningful_visuals) > 1:
        meaningful_visuals = sorted(meaningful_visuals, key=_box_area, reverse=True)[:1]

    visual_paste_box: tuple[int, int, int, int] | None = None
    if len(meaningful_visuals) == 1:
        visual_layer = _paste_crop_fit(
                output,
                source,
                meaningful_visuals[0],
                visual_bounds,
                layer_id="visual-main",
                role="primary_visual_product_preserved_readable",
                mode="contain",
                anchor=(0.5, 0.5),
                foreground_alpha=True,
                alpha_cut_source_boxes=visual_alpha_cut_boxes,
            )
        composited.append(visual_layer)
        if visual_layer.get("pasteBox"):
            visual_paste_box = tuple(int(value) for value in visual_layer["pasteBox"])
    else:
        visual_count = min(3, len(meaningful_visuals))
        gap = max(6, int(width * 0.025))
        slot_w = max(1, (visual_bounds[2] - visual_bounds[0] - gap * (visual_count - 1)) // visual_count)
        for index, box in enumerate(meaningful_visuals[:visual_count]):
            slot = (
                visual_bounds[0] + index * (slot_w + gap),
                visual_bounds[1],
                visual_bounds[0] + index * (slot_w + gap) + slot_w,
                visual_bounds[3],
            )
            composited.append(
                _paste_crop_fit(
                    output,
                    source,
                    box,
                    slot,
                    layer_id=f"visual-{index}",
                    role="visual_element_preserved_readable",
                    mode="contain",
                    anchor=(0.5, 0.5),
                    foreground_alpha=True,
                    alpha_cut_source_boxes=visual_alpha_cut_boxes,
                )
            )

    if redraw_blocks and draw_text:
        if use_redraw_for_small_text and preserve_brand_layers:
            secondary_source_union = _union_boxes(parts["secondary"])
            primary_redraw: list[Any] = []
            secondary_redraw: list[Any] = []
            for block in redraw_blocks:
                layer_id = str(getattr(block, "id", "")).replace("v5-resize-", "")
                source_box = layer_boxes.get(layer_id, (0, 0, 0, 0))
                if secondary_source_union and _box_overlap(source_box, secondary_source_union) / max(1, _box_area(source_box)) >= 0.35:
                    secondary_redraw.append(block)
                else:
                    primary_redraw.append(block)
            blocks_to_draw = [
                *_build_display_copy_stack_blocks(primary_redraw, zones["copy"], stack_role="primary", social=not preserve_brand_layers),
                *_build_display_copy_stack_blocks(secondary_redraw, zones["secondary"], stack_role="secondary", social=not preserve_brand_layers),
            ]
        else:
            blocks_to_draw = _sort_redraw_blocks_by_source_yx(
                _merge_redraw_blocks_by_inline_rows(redraw_blocks, width)
            )
        output = draw_text(output.convert("RGB"), blocks_to_draw).convert("RGBA")
        composited.append(
            {
                "layerId": "resize-redrawn-marketing-copy",
                "role": "phase1_typography_redraw",
                "sourceBox": [],
                "targetBox": [],
                "pasteBox": [],
                "scale": None,
                "fitMode": "text_redraw",
                "blockCount": len(redraw_blocks),
            }
        )
        secondary_layer_boxes = [
            tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))
            for block in blocks_to_draw
            if getattr(block, "render_strategy", "") == "resize_display_copy_stack"
            and _box_overlap(tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0))), zones["secondary"])
            / max(1, _box_area(tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0))))) >= 0.55
        ]
        guide = _draw_programmatic_rtb_guides(output, visual_paste_box, _union_boxes(secondary_layer_boxes))
        if guide:
            composited.append(guide)
    display_cta_layer = _draw_display_cta_button(output, display_cta_zone, label=_localized_display_cta_label(analysis)) if display_cta_zone else None
    if display_cta_layer:
        composited.append(display_cta_layer)

    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "priorityLayerResize": True,
        "targetAreaSmaller": target_smaller,
        "sourceRatio": round(source_ratio, 4),
        "targetRatio": round(target_ratio, 4),
        "visualRenderBounds": list(visual_bounds),
        "textRedrawBlocks": len(redraw_blocks),
        **visual_meta,
    }


def _paste_layer_relative(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    *,
    target_bounds: tuple[int, int, int, int],
    layer_id: str,
    role: str,
    scale: float,
    preserve_source_position: bool = True,
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    target_bounds = _clip_box(target_bounds, output.width, output.height)
    box_w = max(1, source_box[2] - source_box[0])
    box_h = max(1, source_box[3] - source_box[1])
    paste_w = max(1, int(round(box_w * scale)))
    paste_h = max(1, int(round(box_h * scale)))
    if preserve_source_position:
        rel_cx = ((source_box[0] + source_box[2]) / 2) / max(1, source.width)
        rel_cy = ((source_box[1] + source_box[3]) / 2) / max(1, source.height)
        cx = int(round(target_bounds[0] + rel_cx * max(1, target_bounds[2] - target_bounds[0])))
        cy = int(round(target_bounds[1] + rel_cy * max(1, target_bounds[3] - target_bounds[1])))
        paste_x = cx - paste_w // 2
        paste_y = cy - paste_h // 2
    else:
        paste_x = target_bounds[0] + (max(1, target_bounds[2] - target_bounds[0]) - paste_w) // 2
        paste_y = target_bounds[1] + (max(1, target_bounds[3] - target_bounds[1]) - paste_h) // 2
    paste_x = max(target_bounds[0], min(max(target_bounds[0], target_bounds[2] - paste_w), paste_x))
    paste_y = max(target_bounds[1], min(max(target_bounds[1], target_bounds[3] - paste_h), paste_y))
    return _paste_crop_exact(
        output,
        source,
        source_box,
        (paste_x, paste_y, paste_x + paste_w, paste_y + paste_h),
        layer_id=layer_id,
        role=role,
        feather=0,
    )


def _paste_crop_contain_limited(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    max_scale: float,
    anchor: tuple[float, float] = (0.5, 0.5),
    foreground_alpha: bool = False,
    artwork_alpha: bool = False,
    anchor_bottom_if_source_truncated: bool = False,
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    target_box = _clip_box(target_box, output.width, output.height)
    crop: Image.Image | None = None
    if artwork_alpha or foreground_alpha:
        crop = source.crop(source_box).convert("RGBA")
        if artwork_alpha:
            crop = _trim_rgba_to_alpha(_apply_artwork_alpha(crop), pad_ratio=0.025)
        else:
            crop = _trim_rgba_to_alpha(_source_crop_with_global_foreground_alpha(source, source_box), pad_ratio=0.035)
        box_w = max(1, crop.width)
        box_h = max(1, crop.height)
    else:
        box_w = max(1, source_box[2] - source_box[0])
        box_h = max(1, source_box[3] - source_box[1])
    scale = min(
        max_scale,
        (target_box[2] - target_box[0]) / box_w,
        (target_box[3] - target_box[1]) / box_h,
    )
    scale = max(0.1, scale)
    paste_w = max(1, int(round(box_w * scale)))
    paste_h = max(1, int(round(box_h * scale)))
    free_w = max(0, target_box[2] - target_box[0] - paste_w)
    free_h = max(0, target_box[3] - target_box[1] - paste_h)
    paste_x = target_box[0] + int(round(free_w * anchor[0]))
    paste_y = target_box[1] + int(round(free_h * anchor[1]))
    if anchor_bottom_if_source_truncated and source_box[3] >= source.height - max(2, int(source.height * 0.012)):
        paste_y = output.height - paste_h
    paste_box = (paste_x, paste_y, paste_x + paste_w, paste_y + paste_h)
    if crop is not None:
        patch = crop.resize((paste_w, paste_h), Image.Resampling.LANCZOS)
        output.alpha_composite(patch, (paste_x, paste_y))
        return {
            "layerId": layer_id,
            "role": role,
            "sourceBox": list(source_box),
            "targetBox": list(target_box),
            "pasteBox": [paste_x, paste_y, paste_x + paste_w, paste_y + paste_h],
            "scale": round(scale, 4),
            "fitMode": "contain_alpha_trimmed",
        }
    feather = max(3, min(18, min(paste_w, paste_h) // 18)) if "marketing_copy" in role else 0
    return _paste_crop_fit(
        output,
        source,
        source_box,
        paste_box,
        layer_id=layer_id,
        role=role,
        mode="contain",
        feather=feather,
        anchor_bottom_if_source_truncated=anchor_bottom_if_source_truncated,
    )


def build_wide_to_portrait_outpaint_layout_seed(
    source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    target_ratio = width / max(1, height)
    visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
    meaningful_visuals = [
        box
        for box in visual_elements
        if _box_area(box) / max(1, source.width * source.height) >= 0.018
    ] or [visual_box]
    partition_visual_box = visual_box
    parts = _partition_resize_layers(source, analysis, partition_visual_box)
    product_label_boxes = parts["product_label"]
    strict_product_label_boxes: list[tuple[int, int, int, int]] = []
    for layer in analysis.marketing_text_layers:
        raw_box = _layer_box(layer, source)
        raw_cx, raw_cy = _box_center(raw_box)
        raw_overlap_visual = _box_overlap(raw_box, visual_box) / max(1, _box_area(raw_box))
        if (visual_box[0] <= raw_cx <= visual_box[2] and visual_box[1] <= raw_cy <= visual_box[3]) or raw_overlap_visual > 0.58:
            strict_product_label_boxes.append(raw_box)
    for layer in analysis.logo_layers:
        raw_box = _layer_box(layer, source)
        raw_overlap_visual = _box_overlap(raw_box, visual_box) / max(1, _box_area(raw_box))
        if raw_overlap_visual > 0.36:
            strict_product_label_boxes.append(raw_box)
    if strict_product_label_boxes:
        product_label_boxes = strict_product_label_boxes
    product_focus = _visual_focus_from_product_label_union(
        source,
        _union_boxes(product_label_boxes),
        visual_box,
        compact_target=width <= 420 or target_ratio < 0.78,
    )
    if product_focus:
        product_focus = _trim_visual_box_to_product_label_side(
            product_focus,
            _union_boxes(product_label_boxes),
            _union_boxes(parts["secondary"]),
            source.width,
            source.height,
        )
        meaningful_visuals = [product_focus]
    background_visual_removal_boxes = [
        visual_box,
        *meaningful_visuals,
        *[_layer_box(layer, source) for layer in analysis.product_layers],
        *parts.get("product_label", []),
    ]
    background_visual_removal_boxes = [
        _pad_box(
            box,
            source.width,
            source.height,
            pad_x=max(10, int((box[2] - box[0]) * 0.16)),
            pad_y=max(8, int((box[3] - box[1]) * 0.12)),
        )
        for box in background_visual_removal_boxes
        if box and box[2] > box[0] and box[3] > box[1]
    ]
    floating_text_boxes: list[tuple[int, int, int, int]] = []
    visual_alpha_cut_boxes: list[tuple[int, int, int, int]] = [
        *parts["brand"],
        *parts.get("trust_badge", []),
        *parts["primary"],
        *parts["secondary"],
    ]
    for layer in analysis.marketing_text_layers:
        raw_box = _layer_box(layer, source)
        box = _expand_text_box_line_region(source, raw_box)
        overlap_with_product_label = max(
            (_box_overlap(box, label_box) / max(1, min(_box_area(box), _box_area(label_box))) for label_box in product_label_boxes),
            default=0.0,
        )
        raw_cx, raw_cy = _box_center(raw_box)
        if overlap_with_product_label < 0.45:
            visual_alpha_cut_boxes.append(_pad_box(raw_box, source.width, source.height, pad_x=4, pad_y=3))
        if overlap_with_product_label < 0.45:
            floating_text_boxes.append(box)
    if visual_alpha_cut_boxes:
        meaningful_visuals = [
            _trim_visual_box_away_from_text_edges(box, visual_alpha_cut_boxes, source.width, source.height)
            for box in meaningful_visuals
        ]
    visual_source_clean = _inpaint_rectangular_overlays(source, floating_text_boxes)
    margin_x = int(width * 0.055)
    portrait_visual_margin_x = int(width * 0.16)
    visual_bounds = (
        portrait_visual_margin_x,
        int(height * 0.37),
        width - portrait_visual_margin_x,
        int(height * 0.86),
    )
    background_scene_source = _remove_foreground_visuals_for_background(
        _inpaint_rectangular_overlays(
            source,
                [
                    *floating_text_boxes,
                    *parts["brand"],
                    *parts.get("trust_badge", []),
                ],
            ),
        meaningful_visuals,
    )
    seed = _build_focus_cover_scene(
        background_scene_source,
        width,
        height,
        source_focus_box=_union_boxes(meaningful_visuals) or visual_box,
        target_focus_box=visual_bounds,
    ).convert("RGBA")
    # Keep the RGB context under transparent pixels so edit providers see the
    # original creative atmosphere, while the alpha mask still only protects
    # the deterministic foreground/product layer.
    seed.putalpha(Image.new("L", (width, height), 0))
    protected: list[dict[str, Any]] = []
    if len(meaningful_visuals) == 1:
        src_box = meaningful_visuals[0]
        overlap_ratio = max(
            (_box_overlap(src_box, text_box) / max(1, _box_area(src_box)) for text_box in floating_text_boxes),
            default=0.0,
        )
        src = visual_source_clean if overlap_ratio > 0.005 else source
        crop = _apply_foreground_alpha(src.crop(_clip_box(src_box, src.width, src.height)).convert("RGBA"))
        left, top, right, bottom, scale = _fit_inside(crop.size, visual_bounds)
        seed.alpha_composite(crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS), (left, top))
        protected.append({"sourceBox": list(src_box), "pasteBox": [left, top, right, bottom], "scale": round(scale, 4)})
    else:
        slot_gap = int(width * 0.035)
        slot_w = max(1, ((visual_bounds[2] - visual_bounds[0]) - slot_gap * (len(meaningful_visuals) - 1)) // len(meaningful_visuals))
        for index, src_box in enumerate(meaningful_visuals[:3]):
            slot = (
                visual_bounds[0] + index * (slot_w + slot_gap),
                visual_bounds[1],
                visual_bounds[0] + index * (slot_w + slot_gap) + slot_w,
                visual_bounds[3],
            )
            crop = _apply_foreground_alpha(source.crop(_clip_box(src_box, source.width, source.height)).convert("RGBA"))
            left, top, right, bottom, scale = _fit_inside(crop.size, slot)
            seed.alpha_composite(crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS), (left, top))
            protected.append({"sourceBox": list(src_box), "pasteBox": [left, top, right, bottom], "scale": round(scale, 4)})
    return seed, {
        "layoutOutpaintSeed": "wide_to_portrait_visual_slot_alpha_seed",
        "layoutOutpaintProtectedVisuals": protected,
        **visual_meta,
    }


def composite_wide_creative_director_relayout(
    background_source: Image.Image,
    source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
    *,
    text_blocks: list[Any] | None = None,
    draw_text: TextRenderer | None = None,
    visual_already_protected: bool = False,
    visual_completion_source: Image.Image | None = None,
    preserve_brand_layers: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    display_placement = preserve_brand_layers
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    provider_background_is_target = background_source.size == (width, height)

    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
    meaningful_visuals = [
        box
        for box in visual_elements
        if _box_area(box) / max(1, source.width * source.height) >= 0.018
    ]
    if not meaningful_visuals:
        meaningful_visuals = [visual_box]
    meaningful_visuals = [
        _pad_box(
            box,
            source.width,
            source.height,
            pad_x=max(6, int((box[2] - box[0]) * 0.10)),
            pad_y=max(6, int((box[3] - box[1]) * 0.06)),
        )
        for box in meaningful_visuals
    ]
    if target_ratio < 0.78 and source_ratio > 1.25 and len(meaningful_visuals) > 1 and analysis.product_layers:
        product_union = _union_boxes([_layer_box(layer, source) for layer in analysis.product_layers])
        if product_union:
            product_ordered = sorted(
                meaningful_visuals,
                key=lambda item: (
                    -_box_overlap(item, product_union) / max(1, min(_box_area(item), _box_area(product_union))),
                    item[0],
                ),
            )
            meaningful_visuals = product_ordered[:1]
    partition_visual_box = visual_box
    parts = _partition_resize_layers(source, analysis, partition_visual_box)
    if target_ratio < 0.78 and not parts["primary"] and parts["secondary"]:
        parts["primary"] = list(parts["secondary"])
        parts["secondary"] = []
    product_label_boxes = parts["product_label"]
    strict_product_label_boxes: list[tuple[int, int, int, int]] = []
    for layer in analysis.marketing_text_layers:
        raw_box = _layer_box(layer, source)
        raw_cx, raw_cy = _box_center(raw_box)
        raw_overlap_visual = _box_overlap(raw_box, visual_box) / max(1, _box_area(raw_box))
        if (visual_box[0] <= raw_cx <= visual_box[2] and visual_box[1] <= raw_cy <= visual_box[3]) or raw_overlap_visual > 0.58:
            strict_product_label_boxes.append(raw_box)
    for layer in analysis.logo_layers:
        raw_box = _layer_box(layer, source)
        raw_overlap_visual = _box_overlap(raw_box, visual_box) / max(1, _box_area(raw_box))
        if raw_overlap_visual > 0.36:
            strict_product_label_boxes.append(raw_box)
    if strict_product_label_boxes:
        product_label_boxes = strict_product_label_boxes
    product_focus = _visual_focus_from_product_label_union(
        source,
        _union_boxes(product_label_boxes),
        visual_box,
        compact_target=width <= 420 or target_ratio < 0.78,
    )
    if product_focus:
        product_focus = _trim_visual_box_to_product_label_side(
            product_focus,
            _union_boxes(product_label_boxes),
            _union_boxes(parts["secondary"]),
            source.width,
            source.height,
        )
        meaningful_visuals = [product_focus]
    background_visual_removal_boxes = [
        visual_box,
        *meaningful_visuals,
        *[_layer_box(layer, source) for layer in analysis.product_layers],
        *parts.get("product_label", []),
    ]
    background_visual_removal_boxes = [
        _pad_box(
            box,
            source.width,
            source.height,
            pad_x=max(10, int((box[2] - box[0]) * 0.16)),
            pad_y=max(8, int((box[3] - box[1]) * 0.12)),
        )
        for box in background_visual_removal_boxes
        if box and box[2] > box[0] and box[3] > box[1]
    ]
    floating_text_boxes: list[tuple[int, int, int, int]] = []
    visual_alpha_cut_boxes: list[tuple[int, int, int, int]] = [
        *parts["brand"],
        *parts.get("trust_badge", []),
        *parts["primary"],
        *parts["secondary"],
    ]
    for layer in analysis.marketing_text_layers:
        raw_box = _layer_box(layer, source)
        box = _expand_text_box_line_region(source, raw_box)
        overlap_with_product_label = max(
            (_box_overlap(box, label_box) / max(1, min(_box_area(box), _box_area(label_box))) for label_box in product_label_boxes),
            default=0.0,
        )
        raw_cx, raw_cy = _box_center(raw_box)
        if overlap_with_product_label < 0.45:
            visual_alpha_cut_boxes.append(_pad_box(raw_box, source.width, source.height, pad_x=4, pad_y=3))
        if overlap_with_product_label < 0.45:
            floating_text_boxes.append(box)
    if visual_alpha_cut_boxes:
        meaningful_visuals = [
            _trim_visual_box_away_from_text_edges(box, visual_alpha_cut_boxes, source.width, source.height)
            for box in meaningful_visuals
        ]
    visual_source_clean = _inpaint_rectangular_overlays(source, floating_text_boxes)
    visual_source = source
    composited: list[dict[str, Any]] = []
    visual_edge_sides = (
        _edge_touch_sides(
            meaningful_visuals[0],
            source.width,
            source.height,
            tolerance=max(2, int(min(source.width, source.height) * 0.008)),
        )
        if meaningful_visuals
        else []
    )

    target_area_smaller = width * height < source.width * source.height
    margin_x = int(width * (0.09 if not display_placement else 0.055))
    margin_y = int(height * (0.075 if not display_placement else 0.045))
    is_social_square = not display_placement and 0.92 <= target_ratio <= 1.08
    if target_ratio < 0.78:
        brand_bounds = (margin_x, margin_y, width - margin_x, int(height * 0.16))
        copy_bounds = (margin_x, int(height * 0.15), width - margin_x, int(height * 0.35))
        cta_bounds = (margin_x, int(height * 0.35), width - margin_x, int(height * 0.43))
        portrait_visual_margin_x = int(width * 0.09)
        visual_bounds = (
            portrait_visual_margin_x,
            int(height * 0.39),
            width - portrait_visual_margin_x,
            int(height * 0.965),
        )
        if display_placement:
            copy_bounds = (margin_x, int(height * 0.14), width - margin_x, int(height * 0.34))
            cta_bounds = (margin_x, int(height * 0.34), width - margin_x, int(height * 0.42))
            visual_bounds = (
                portrait_visual_margin_x,
                int(height * 0.42),
                width - portrait_visual_margin_x,
                int(height * 0.71),
            )
            secondary_bounds = (margin_x, int(height * 0.755), width - margin_x, height - margin_y)
        else:
            visual_bounds = (
                portrait_visual_margin_x,
                int(height * 0.40),
                width - portrait_visual_margin_x,
                int(height * 0.755),
            )
            secondary_bounds = (margin_x, int(height * 0.775), width - margin_x, int(height * 0.94))
    elif target_area_smaller:
        if is_social_square:
            split_x = width // 2
            left_safe = (max(14, int(width * 0.06)), margin_y, split_x - max(14, int(width * 0.06)), height - margin_y)
            right_safe = (split_x + max(12, int(width * 0.045)), margin_y, width - max(12, int(width * 0.045)), height - margin_y)
            brand_bounds = (left_safe[0], left_safe[1], left_safe[2], int(height * 0.16))
            copy_bounds = (left_safe[0], int(height * 0.18), left_safe[2], int(height * 0.57))
            secondary_bounds = (left_safe[0], int(height * 0.61), left_safe[2], int(height * 0.93))
            cta_bounds = (left_safe[0], int(height * 0.76), left_safe[2], int(height * 0.94))
            visual_bounds = (right_safe[0], int(height * 0.08), right_safe[2], int(height * 0.94))
        else:
            brand_bounds = (margin_x, margin_y, int(width * 0.44), int(height * 0.18))
            copy_bounds = (margin_x, int(height * 0.22), int(width * 0.46), int(height * 0.58))
            cta_bounds = (margin_x, int(height * 0.78), int(width * 0.44), int(height * 0.94))
            visual_bounds = (int(width * 0.52), int(height * 0.18), width - margin_x, int(height * 0.90))
        if display_placement:
            secondary_bounds = (int(width * 0.58), int(height * 0.54), width - margin_x, int(height * 0.88))
        elif not is_social_square:
            secondary_bounds = (margin_x, int(height * 0.62), int(width * 0.48), height - margin_y)
    elif source_ratio > 1.25 and target_ratio <= 1.25:
        brand_bounds = (margin_x, margin_y, int(width * 0.42), int(height * 0.17))
        copy_bounds = (margin_x, int(height * 0.30), int(width * 0.43), int(height * 0.67))
        cta_bounds = (margin_x, int(height * 0.68), int(width * 0.43), int(height * 0.80))
        visual_bounds = (int(width * 0.49), int(height * 0.12), width - margin_x, int(height * 0.90))
        if display_placement:
            secondary_bounds = (int(width * 0.55), int(height * 0.54), width - margin_x, int(height * 0.88))
        else:
            copy_bounds = (margin_x, int(height * 0.22), int(width * 0.46), int(height * 0.57))
            secondary_bounds = (margin_x, int(height * 0.62), int(width * 0.48), int(height * 0.92))
    else:
        brand_bounds = (margin_x, margin_y, width - margin_x, int(height * 0.17))
        copy_bounds = (margin_x, int(height * 0.16), int(width * 0.52), int(height * 0.82))
        cta_bounds = (margin_x, int(height * 0.76), int(width * 0.52), int(height * 0.91))
        visual_bounds = (int(width * 0.43), int(height * 0.12), width - margin_x, int(height * 0.88))
        secondary_bounds = (int(width * 0.58), int(height * 0.54), width - margin_x, int(height * 0.88))
    visual_render_bounds = visual_bounds
    if target_ratio < 0.78 and visual_edge_sides:
        visual_w = max(1, visual_bounds[2] - visual_bounds[0])
        visual_h = max(1, visual_bounds[3] - visual_bounds[1])
        inset_x = int(round(visual_w * 0.035))
        inset_y = int(round(visual_h * 0.025))
        visual_render_bounds = (
            visual_bounds[0] + inset_x,
            visual_bounds[1] + inset_y,
            visual_bounds[2] - inset_x,
            visual_bounds[3] - inset_y,
        )

    if provider_background_is_target:
        if target_ratio < 0.78:
            provider_clean_source = _remove_foreground_visuals_for_background(
                _inpaint_rectangular_overlays(
                    source,
                    [
                        *[
                            _expand_text_box_line_region(source, layer.bbox.to_pixel_box(source.width, source.height))
                            for layer in analysis.marketing_text_layers
                        ],
                        *floating_text_boxes,
                        *parts["brand"],
                        *parts.get("trust_badge", []),
                    ],
                ),
                background_visual_removal_boxes or meaningful_visuals,
            )
            output = build_low_artifact_background_canvas(provider_clean_source, width, height).convert("RGBA")
        elif target_area_smaller:
            output = build_deterministic_background_canvas(background_source.convert("RGB"), width, height).convert("RGBA")
        else:
            output = background_source.convert("RGBA")
    else:
        background_scene_source = _remove_foreground_visuals_for_background(
            _inpaint_rectangular_overlays(
                source,
                [
                    *[
                        _expand_text_box_line_region(source, layer.bbox.to_pixel_box(source.width, source.height))
                        for layer in analysis.marketing_text_layers
                    ],
                    *floating_text_boxes,
                    *parts["brand"],
                    *parts.get("trust_badge", []),
                ],
            ),
            background_visual_removal_boxes or meaningful_visuals,
        )
        if target_ratio < 0.78:
            output = build_low_artifact_background_canvas(background_scene_source, width, height).convert("RGBA")
        elif target_area_smaller:
            output = build_deterministic_background_canvas(background_scene_source, width, height).convert("RGBA")
        elif source_ratio > 1.25 and target_ratio <= 1.25:
            output = build_deterministic_background_canvas(background_scene_source, width, height).convert("RGBA")
        else:
            # Landscape/tablet targets can keep a focus-cover underlay because the source
            # aspect-ratio mismatch is smaller and it preserves the original atmosphere.
            output = _build_focus_cover_scene(
                background_scene_source,
                width,
                height,
                source_focus_box=_union_boxes(meaningful_visuals) or visual_box,
                target_focus_box=visual_bounds,
            ).convert("RGBA")

    brand_scale = min(
        1.0,
        max(0.42, min(width / max(1, source.width), height / max(1, source.height)) * (1.15 if target_ratio < 0.78 else 1.0)),
    )
    if preserve_brand_layers:
        for index, box in enumerate(parts["brand"][:4]):
            composited.append(
                _paste_layer_relative(
                    output,
                    source,
                    box,
                    target_bounds=brand_bounds,
                    layer_id=f"brand-{index}",
                    role="brand_or_badge_preserved",
                    scale=brand_scale,
                    preserve_source_position=True,
                )
            )
    badge_bounds = (
        int(width * (0.58 if not display_placement else 0.66)),
        margin_y,
        width - margin_x,
        max(margin_y + 10, int(height * (0.24 if not display_placement else 0.20))),
    )
    for index, box in enumerate(parts.get("trust_badge", [])[:3]):
        badge_scale = brand_scale
        if not display_placement:
            badge_scale = max(brand_scale, min(1.30, max(width, height) / max(1, max(source.width, source.height)) * 1.12))
        composited.append(
            _paste_layer_relative(
                output,
                source,
                box,
                target_bounds=badge_bounds,
                layer_id=f"trust-badge-{index}",
                role="trust_badge_preserved",
                scale=badge_scale,
                preserve_source_position=True,
            )
        )

    redraw_blocks: list[Any] = []
    # Resize is not localization: preserve existing marketing typography as artwork
    # whenever possible. Redraw remains available as an explicit fallback/env override.
    redraw_default = "1"
    redraw_enabled = os.getenv("ADAPTIFAI_RESIZE_ENABLE_TEXT_REDRAW", redraw_default).strip().lower() in {"1", "true", "yes", "on"}
    if redraw_enabled and text_blocks and draw_text:
        layer_boxes = {layer.id: _layer_box(layer, source) for layer in analysis.marketing_text_layers}
        primary_union = _union_boxes(parts["primary"])
        secondary_union = _union_boxes(parts["secondary"])
        fallback_secondary_boxes: list[tuple[int, int, int, int]] = []
        portrait_redraw_source_boxes: list[tuple[int, int, int, int]] = []
        for source_box in layer_boxes.values():
            cx, cy = _box_center(source_box)
            product_label_overlap = max(
                (_box_overlap(source_box, label_box) / max(1, min(_box_area(source_box), _box_area(label_box))) for label_box in product_label_boxes),
                default=0.0,
            )
            if product_label_overlap < 0.45 and cx >= source.width * 0.58 and cy >= source.height * 0.45:
                fallback_secondary_boxes.append(source_box)
        secondary_effective_union = secondary_union or _union_boxes(fallback_secondary_boxes)
        if target_ratio < 0.78:
            for block in text_blocks:
                layer_id = str(getattr(block, "id", "")).replace("v5-resize-", "")
                source_box = layer_boxes.get(layer_id)
                if not source_box:
                    continue
                cx, cy = _box_center(source_box)
                is_top_brand_like = cy <= source.height * 0.24 and (cx <= source.width * 0.34 or cx >= source.width * 0.66)
                product_label_overlap = max(
                    (_box_overlap(source_box, label_box) / max(1, min(_box_area(source_box), _box_area(label_box))) for label_box in product_label_boxes),
                    default=0.0,
                )
                secondary_overlap = _box_overlap(source_box, secondary_effective_union) / max(1, _box_area(source_box)) if secondary_effective_union else 0.0
                if not is_top_brand_like and product_label_overlap < 0.45 and secondary_overlap < 0.35:
                    portrait_redraw_source_boxes.append(source_box)
        portrait_redraw_union = _union_boxes(portrait_redraw_source_boxes)
        portrait_copy_zone = (
            margin_x,
            max(brand_bounds[3] + int(height * 0.018), int(height * 0.15)),
            width - margin_x,
            max(brand_bounds[3] + int(height * 0.08), visual_render_bounds[1] - int(height * 0.025)),
        )
        for block in text_blocks:
            layer_id = str(getattr(block, "id", "")).replace("v5-resize-", "")
            source_box = layer_boxes.get(layer_id)
            if not source_box:
                continue
            cx, cy = _box_center(source_box)
            is_top_brand_like = cy <= source.height * 0.24 and (cx <= source.width * 0.34 or cx >= source.width * 0.66)
            product_label_overlap = max(
                (_box_overlap(source_box, label_box) / max(1, min(_box_area(source_box), _box_area(label_box))) for label_box in product_label_boxes),
                default=0.0,
            )
            secondary_overlap_global = _box_overlap(source_box, secondary_effective_union) / max(1, _box_area(source_box)) if secondary_effective_union else 0.0
            if secondary_overlap_global >= 0.35 and product_label_overlap < 0.45:
                target_box = _map_box_between_unions(source_box, secondary_effective_union, secondary_bounds, width, height) if secondary_effective_union else _clip_box(tuple(getattr(block, "bbox", source_box)), width, height)
            elif target_ratio < 0.78 and product_label_overlap < 0.45:
                if is_top_brand_like:
                    continue
                if portrait_redraw_union:
                    target_box = _map_portrait_text_box_to_safe_copy_zone(
                        source_box,
                        portrait_redraw_union,
                        portrait_copy_zone,
                        width,
                        height,
                    )
                    target_box = (
                        portrait_copy_zone[0],
                        target_box[1],
                        portrait_copy_zone[2],
                        target_box[3],
                    )
                else:
                    target_box = _clip_box(tuple(getattr(block, "bbox", source_box)), width, height)
            elif primary_union and _box_overlap(source_box, primary_union) / max(1, _box_area(source_box)) > 0.35:
                target_box = _map_box_between_unions(source_box, primary_union, copy_bounds, width, height)
            elif secondary_effective_union and _box_overlap(source_box, secondary_effective_union) / max(1, _box_area(source_box)) > 0.35:
                target_box = _map_box_between_unions(source_box, secondary_effective_union, secondary_bounds, width, height)
            else:
                continue
            try:
                cloned = block.model_copy(deep=True)
            except Exception:
                cloned = block
            target_box = _clip_box(target_box, width, height)
            cloned.bbox = target_box
            cloned.clean_box = target_box
            cloned.line_boxes = [target_box]
            cloned.font_size_estimate = max(7, min(int(getattr(cloned, "font_size_estimate", 16) or 16), int((target_box[3] - target_box[1]) * 0.92)))
            if not display_placement:
                social_min = int(width * (0.044 if target_ratio < 0.78 else 0.034))
                cloned.font_size_estimate = max(cloned.font_size_estimate, max(16, social_min))
                max_social_size = int(max(18, (target_box[3] - target_box[1]) * 0.80))
                cloned.font_size_estimate = min(cloned.font_size_estimate, max_social_size)
            if target_area_smaller:
                if display_placement:
                    cloned.font_size_estimate = max(7, min(cloned.font_size_estimate, int(max(8, width * 0.038))))
                else:
                    cloned.font_size_estimate = max(cloned.font_size_estimate, int(max(14, min(width, height) * 0.032)))
            cloned.line_height_estimate = int(round(cloned.font_size_estimate * 1.18))
            for style in getattr(cloned, "source_word_styles", []) or []:
                style["fontSize"] = cloned.font_size_estimate
                style["peerRowFontSize"] = cloned.font_size_estimate
                style["bbox"] = [target_box[0], target_box[1], target_box[2], target_box[3]]
            for span in getattr(cloned, "translated_style_spans", []) or []:
                if isinstance(span, dict):
                    span_style = span.setdefault("style", {})
                    span_style["fontSize"] = cloned.font_size_estimate
                    span_style["lineHeight"] = cloned.line_height_estimate
                    for source_style in span.get("sourceWordStyles", []) or []:
                        source_style["fontSize"] = cloned.font_size_estimate
                        source_style["peerRowFontSize"] = cloned.font_size_estimate
                        source_style["bbox"] = [target_box[0], target_box[1], target_box[2], target_box[3]]
            redraw_blocks.append(cloned)
    else:
        text_artwork_source = source
        primary_candidates = []
        for box in parts["primary"]:
            label_overlap = max(
                (_box_overlap(box, label_box) / max(1, min(_box_area(box), _box_area(label_box))) for label_box in product_label_boxes),
                default=0.0,
            )
            if label_overlap < 0.35:
                primary_candidates.append(box)
        if not primary_candidates and parts["primary"]:
            primary_candidates = list(parts["primary"])
        primary_union = _union_boxes(primary_candidates)
        if primary_union:
            primary_area_ratio = _box_area(primary_union) / max(1, source.width * source.height)
            if primary_area_ratio <= 0.30:
                padded_union = _pad_box(
                    primary_union,
                    source.width,
                    source.height,
                    pad_x=max(1, int((primary_union[2] - primary_union[0]) * (0.012 if source_ratio > 1.25 and target_ratio <= 1.25 else 0.20))),
                    pad_y=max(4, int((primary_union[3] - primary_union[1]) * 0.16)),
                )
                copy_max_scale = 1.0 if target_area_smaller else max(
                    1.0,
                    min(2.35, (width * height / max(1, source.width * source.height)) ** 0.5 * 0.86),
                )
                composited.append(
                    _paste_crop_contain_limited(
                        output,
                        text_artwork_source,
                        padded_union,
                        copy_bounds,
                        layer_id="primary-copy",
                        role="primary_marketing_copy_preserved",
                        max_scale=copy_max_scale,
                        anchor=(0.0, 0.5),
                        artwork_alpha=False,
                    )
                )
            for index, primary_box in enumerate(sorted(primary_candidates if primary_area_ratio > 0.30 else [], key=lambda item: (item[1], item[0]))):
                padded_box = _pad_box(
                    primary_box,
                    source.width,
                    source.height,
                    pad_x=max(3, int((primary_box[2] - primary_box[0]) * 0.10)),
                    pad_y=max(2, int((primary_box[3] - primary_box[1]) * 0.18)),
                )
                target_box = _map_box_between_unions(padded_box, primary_union, copy_bounds, width, height)
                composited.append(
                    _paste_crop_contain_limited(
                        output,
                        text_artwork_source,
                        padded_box,
                        target_box,
                        layer_id=f"primary-copy-{index}",
                        role="primary_marketing_copy_preserved",
                        max_scale=copy_max_scale,
                        anchor=(0.0, 0.5),
                        artwork_alpha=False,
                    )
                )

        secondary_box = _union_boxes(parts["secondary"])
        if secondary_box and target_ratio > 1.25:
            secondary_box = _pad_box(secondary_box, source.width, source.height, pad_x=max(8, source.width // 28), pad_y=max(4, source.height // 48))
            secondary_bounds = (int(width * 0.58), int(height * 0.54), width - margin_x, int(height * 0.88))
            composited.append(
                _paste_crop_contain_limited(
                    output,
                    text_artwork_source,
                    secondary_box,
                    secondary_bounds,
                    layer_id="secondary-copy",
                    role="secondary_marketing_copy_preserved",
                    max_scale=1.05,
                    anchor=(0.0, 0.5),
                    artwork_alpha=True,
                )
            )

    max_visual_text_overlap = max(
        (
            _box_overlap(visual_box_item, text_box) / max(1, _box_area(visual_box_item))
            for visual_box_item in meaningful_visuals
            for text_box in floating_text_boxes
        ),
        default=0.0,
    )
    if max_visual_text_overlap > 0.005:
        visual_source = visual_source_clean
    visual_label_cut_boxes = visual_alpha_cut_boxes
    def source_for_visual_element(box: tuple[int, int, int, int]) -> Image.Image:
        # Product/person visual elements must preserve their original pixels.
        # Text cleanup on these crops can create visible patches on skin, packaging,
        # or product labels, which is worse than preserving the exact source layer.
        return source

    if visual_completion_source is not None:
        composited.append(
            _paste_crop_fit(
                output,
                visual_completion_source,
                visual_bounds,
                visual_render_bounds,
                layer_id="visual-main-provider-completion-foreground",
                role="provider_visual_completion_foreground_only",
                mode="contain",
                foreground_alpha=True,
                alpha_cut_source_boxes=visual_label_cut_boxes,
                anchor_bottom_if_source_truncated=True,
            )
        )
        composited.append(
            _paste_crop_fit(
                output,
                source_for_visual_element(meaningful_visuals[0]),
                meaningful_visuals[0],
                visual_render_bounds,
                layer_id="visual-main-exact-label-overlay",
                role="original_visual_label_readability_overlay",
                mode="contain",
                foreground_alpha=True,
                alpha_cut_source_boxes=visual_label_cut_boxes,
                anchor_bottom_if_source_truncated=True,
            )
        )
    elif visual_already_protected:
        composited.append(
            _paste_crop_fit(
                output,
                source_for_visual_element(meaningful_visuals[0]),
                meaningful_visuals[0],
                visual_render_bounds,
                layer_id="visual-main-exact-foreground",
                role="primary_visual_exact_readable_preserve",
                mode="contain",
                foreground_alpha=True,
                alpha_cut_source_boxes=visual_label_cut_boxes,
                anchor_bottom_if_source_truncated=True,
            )
        )
        composited.append(
            {
                "layerId": "visual-main",
                "role": "primary_visual_preserved_in_layout_outpaint_seed",
                "sourceBox": list(meaningful_visuals[0]) if meaningful_visuals else list(visual_box),
                "targetBox": list(visual_render_bounds),
                "pasteBox": [],
                "scale": None,
                "fitMode": "provider_seed_protected",
            }
        )
    elif len(meaningful_visuals) == 1:
        paste_visual = _paste_crop_contain_limited if target_ratio < 0.78 and visual_edge_sides else _paste_crop_fit
        visual_kwargs: dict[str, Any] = (
            {"max_scale": 3.65}
            if paste_visual is _paste_crop_contain_limited
            else {"alpha_cut_source_boxes": visual_label_cut_boxes}
        )
        visual_kwargs["anchor_bottom_if_source_truncated"] = True
        visual_anchor = (1.0, 1.16) if is_social_square else (0.5, 0.52 if target_area_smaller else (0.68 if visual_edge_sides else 0.74))
        composited.append(
            paste_visual(
                output,
                source_for_visual_element(meaningful_visuals[0]),
                meaningful_visuals[0],
                visual_render_bounds,
                layer_id="visual-main",
                role="primary_visual_readable",
                anchor=visual_anchor,
                foreground_alpha=True,
                **visual_kwargs,
            )
        )
    else:
        slot_gap = int(width * 0.035)
        slot_w = max(1, ((visual_bounds[2] - visual_bounds[0]) - slot_gap * (len(meaningful_visuals) - 1)) // len(meaningful_visuals))
        for index, box in enumerate(meaningful_visuals[:3]):
            slot = (
                visual_bounds[0] + index * (slot_w + slot_gap),
                visual_bounds[1],
                visual_bounds[0] + index * (slot_w + slot_gap) + slot_w,
                visual_bounds[3],
            )
            composited.append(
                _paste_crop_fit(
                    output,
                    source_for_visual_element(box),
                    box,
                    slot,
                    layer_id=f"visual-{index}",
                    role="visual_element_preserved",
                    mode="contain",
                    feather=max(6, min(width, height) // 85),
                    foreground_alpha=True,
                    alpha_cut_source_boxes=visual_label_cut_boxes,
                    anchor_bottom_if_source_truncated=True,
                )
            )

    drawn_redraw_zone_boxes: list[tuple[int, int, int, int]] = []
    if redraw_blocks and draw_text:
        if target_ratio < 0.78 or target_area_smaller or not display_placement:
            secondary_source_union = secondary_effective_union if "secondary_effective_union" in locals() else _union_boxes(parts["secondary"])
            primary_redraw: list[Any] = []
            secondary_redraw: list[Any] = []
            layer_boxes_for_stack = {layer.id: _layer_box(layer, source) for layer in analysis.marketing_text_layers}
            for block in redraw_blocks:
                layer_id = str(getattr(block, "id", "")).replace("v5-resize-", "")
                source_box = layer_boxes_for_stack.get(layer_id, getattr(block, "resize_source_box", None) or getattr(block, "bbox", (0, 0, 0, 0)))
                if secondary_source_union and _box_overlap(source_box, secondary_source_union) / max(1, _box_area(source_box)) >= 0.35:
                    secondary_redraw.append(block)
                else:
                    primary_redraw.append(block)
            blocks_to_draw = [
                *_build_display_copy_stack_blocks(primary_redraw, copy_bounds, stack_role="primary", social=not display_placement),
                *_build_display_copy_stack_blocks(secondary_redraw, secondary_bounds, stack_role="secondary", social=not display_placement),
            ]
        else:
            redraw_blocks = _merge_redraw_blocks_by_inline_rows(redraw_blocks, width)
            blocks_to_draw = redraw_blocks
        drawn_redraw_zone_boxes = [
            tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))
            for block in blocks_to_draw
            if tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))[2]
            > tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))[0]
            and tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))[3]
            > tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))[1]
        ]
        output = draw_text(output.convert("RGB"), blocks_to_draw).convert("RGBA")
        composited.append(
            {
                "layerId": "resize-redrawn-marketing-copy",
                "role": "phase1_typography_redraw",
                "sourceBox": [],
                "targetBox": [],
                "pasteBox": [],
                "scale": None,
                "fitMode": "text_redraw",
                "blockCount": len(redraw_blocks),
            }
        )
        has_secondary_redraw = bool(locals().get("secondary_redraw"))
        if has_secondary_redraw:
            secondary_draw_boxes = [
                tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0)))
                for block in blocks_to_draw
                if getattr(block, "render_strategy", "") == "resize_display_copy_stack"
                and _box_overlap(tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0))), secondary_bounds)
                / max(1, _box_area(tuple(int(value) for value in getattr(block, "bbox", (0, 0, 0, 0))))) > 0.6
            ]
            visual_paste_box = None
            for layer in reversed(composited):
                if str(layer.get("role", "")).startswith("primary_visual") and layer.get("pasteBox"):
                    visual_paste_box = tuple(int(value) for value in layer["pasteBox"])
                    break
            guide = None if is_social_square else _draw_programmatic_rtb_guides(output, visual_paste_box, _union_boxes(secondary_draw_boxes))
            if guide:
                composited.append(guide)

    safe_cta_bounds = (
        _resolve_display_cta_zone(
            cta_bounds,
            drawn_text_boxes=drawn_redraw_zone_boxes,
            width=width,
            height=height,
            margin_x=margin_x,
            margin_y=margin_y,
        )
        if preserve_brand_layers
        else None
    )
    display_cta_layer = _draw_display_cta_button(output, safe_cta_bounds, label=_localized_display_cta_label(analysis)) if safe_cta_bounds else None
    if display_cta_layer:
        composited.append(display_cta_layer)

    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "textRedrawBlocks": len(redraw_blocks),
        "creativeDirectorRelayout": True,
        "sourceRatio": round(source_ratio, 4),
        "targetRatio": round(target_ratio, 4),
        "visualSourceEdgeTouch": visual_edge_sides,
        "visualRenderBounds": list(visual_render_bounds),
        "partitionVisualBox": list(partition_visual_box),
        **visual_meta,
    }


def should_use_horizontal_band_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    if target_ratio <= 1.35:
        return False
    if source_ratio > 1.35:
        return False
    return bool(analysis.marketing_text_layers) and abs(source_ratio - target_ratio) >= 0.42


def should_use_role_aware_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    has_layers = bool(analysis.marketing_text_layers or analysis.product_layers or analysis.logo_layers or analysis.other_layers)
    if should_preserve_full_creative_for_complex_visuals(source, width, height, analysis):
        return False
    return has_layers and abs(target_ratio - source_ratio) >= 0.18


def should_preserve_full_creative_for_complex_visuals(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    if source_ratio <= 1.25:
        multi_product_grid = len(analysis.product_layers) >= 3 and len(analysis.marketing_text_layers) >= 3
        structured_square_creative = bool(analysis.product_layers) and len(analysis.marketing_text_layers) >= 3
        return target_ratio < 0.78 and (multi_product_grid or structured_square_creative)
    if target_ratio < 0.78:
        return False
    if target_ratio < 1.25:
        return True
    visual_box, _ = _detect_role_aware_visual_box(source, analysis)
    visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
    meaningful_elements = [box for box in visual_elements if _box_area(box) / max(1, source.width * source.height) >= 0.025]
    return len(meaningful_elements) >= 2


def _scale_source_box_to_target(
    source_box: tuple[int, int, int, int],
    source: Image.Image,
    width: int,
    height: int,
    *,
    x_scale: float | None = None,
    y_scale: float | None = None,
) -> tuple[int, int, int, int]:
    xs = x_scale if x_scale is not None else width / max(1, source.width)
    ys = y_scale if y_scale is not None else height / max(1, source.height)
    return _clip_box(
        (
            int(round(source_box[0] * xs)),
            int(round(source_box[1] * ys)),
            int(round(source_box[2] * xs)),
            int(round(source_box[3] * ys)),
        ),
        width,
        height,
    )


def _bounded_target_from_source_group(
    group_box: tuple[int, int, int, int],
    source: Image.Image,
    width: int,
    height: int,
    bounds: tuple[int, int, int, int],
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    group_w = max(1, group_box[2] - group_box[0])
    group_h = max(1, group_box[3] - group_box[1])
    target_w = max(1, int(round(group_w * scale)))
    target_h = max(1, int(round(group_h * scale)))
    source_cx = (group_box[0] + group_box[2]) / 2 / max(1, source.width)
    source_cy = (group_box[1] + group_box[3]) / 2 / max(1, source.height)
    cx = int(round(source_cx * width))
    cy = int(round(source_cy * height))
    left = cx - target_w // 2
    top = cy - target_h // 2
    left = max(bounds[0], min(bounds[2] - target_w, left))
    top = max(bounds[1], min(bounds[3] - target_h, top))
    return _clip_box((left, top, left + target_w, top + target_h), width, height)


def _map_box_between_unions(
    box: tuple[int, int, int, int],
    source_union: tuple[int, int, int, int],
    target_union: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    sw = max(1, source_union[2] - source_union[0])
    sh = max(1, source_union[3] - source_union[1])
    tw = max(1, target_union[2] - target_union[0])
    th = max(1, target_union[3] - target_union[1])
    rel = (
        (box[0] - source_union[0]) / sw,
        (box[1] - source_union[1]) / sh,
        (box[2] - source_union[0]) / sw,
        (box[3] - source_union[1]) / sh,
    )
    return _clip_box(
        (
            target_union[0] + int(round(rel[0] * tw)),
            target_union[1] + int(round(rel[1] * th)),
            target_union[0] + int(round(rel[2] * tw)),
            target_union[1] + int(round(rel[3] * th)),
        ),
        width,
        height,
    )


def _copy_text_block_to_box(block: Any, box: tuple[int, int, int, int]) -> Any:
    updates = {"bbox": box, "clean_box": box, "line_boxes": [box]}
    if hasattr(block, "model_copy"):
        return block.model_copy(update=updates, deep=True)
    copied = block.copy(deep=True)
    for key, value in updates.items():
        setattr(copied, key, value)
    return copied


def composite_role_aware_relayout(
    canvas: Image.Image,
    source: Image.Image,
    foreground_source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
    *,
    background_contains_visual: bool = False,
    text_blocks: list[Any] | None = None,
    draw_text: TextRenderer | None = None,
    preserve_brand_layers: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    parts = _partition_resize_layers(source, analysis, visual_box)
    source_ratio = source.width / max(1, source.height)
    target_ratio = width / max(1, height)
    composited: list[dict[str, Any]] = []
    x_scale = width / max(1, source.width)
    y_scale = height / max(1, source.height)
    uniform_scale = min(x_scale, y_scale)
    visual_source = _inpaint_rectangular_overlays(source, [*parts["brand"], *parts["primary"], *parts["secondary"]])

    if target_ratio >= 1.35:
        visual_target = (int(width * 0.43), 0, width, height)
        visual_anchor = (0.52, 0.50)
        primary_bounds = (0, int(height * 0.18), int(width * 0.415), int(height * 0.86))
        secondary_bounds = (int(width * 0.70), int(height * 0.34), width, int(height * 0.88))
    elif target_ratio <= 0.78:
        visual_target = (int(width * 0.08), int(height * 0.33), int(width * 0.92), int(height * 0.78))
        visual_anchor = (0.50, 0.56)
        primary_bounds = (int(width * 0.07), int(height * 0.12), int(width * 0.93), int(height * 0.34))
        secondary_bounds = (int(width * 0.10), int(height * 0.78), int(width * 0.90), int(height * 0.91))
    else:
        visual_target = (int(width * 0.36), int(height * 0.20), int(width * 0.95), int(height * 0.86))
        visual_anchor = (0.52, 0.54)
        primary_bounds = (int(width * 0.06), int(height * 0.16), int(width * 0.43), int(height * 0.80))
        secondary_bounds = (int(width * 0.48), int(height * 0.74), int(width * 0.95), int(height * 0.92))

    if target_ratio >= 1.35:
        output = _build_focus_cover_scene(
            visual_source,
            width,
            height,
            source_focus_box=visual_box,
            target_focus_box=visual_target,
        ).convert("RGBA")
        composited.append(
            {
                "layerId": "role-aware-visual-scene-cover",
                "role": "product_visual_group",
                "sourceBox": list(visual_box),
                "targetBox": list(visual_target),
                "pasteBox": [0, 0, width, height],
                "scale": None,
                "fitMode": "focus_cover_scene",
            }
        )
    else:
        output = build_deterministic_background_canvas(visual_source, width, height).convert("RGBA")
        visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
        elements_union = _union_boxes(visual_elements) or visual_box
        union_w = max(1, elements_union[2] - elements_union[0])
        union_h = max(1, elements_union[3] - elements_union[1])
        target_w = max(1, visual_target[2] - visual_target[0])
        target_h = max(1, visual_target[3] - visual_target[1])
        element_scale = min(target_w / union_w, target_h / union_h)
        scaled_union_w = int(round(union_w * element_scale))
        scaled_union_h = int(round(union_h * element_scale))
        origin_x = visual_target[0] + (target_w - scaled_union_w) // 2
        origin_y = visual_target[1] + (target_h - scaled_union_h) // 2
        for index, element_box in enumerate(visual_elements):
            paste_box = (
                origin_x + int(round((element_box[0] - elements_union[0]) * element_scale)),
                origin_y + int(round((element_box[1] - elements_union[1]) * element_scale)),
                origin_x + int(round((element_box[2] - elements_union[0]) * element_scale)),
                origin_y + int(round((element_box[3] - elements_union[1]) * element_scale)),
            )
            composited.append(
                _paste_crop_exact(
                    output,
                    visual_source,
                    element_box,
                    paste_box,
                    layer_id=f"role-aware-visual-element-{index + 1}",
                    role="product_visual_element",
                    feather=max(4, min(14, width // 120)),
                )
            )

    brand_union = _union_boxes(parts["brand"]) if preserve_brand_layers else None
    if brand_union:
        brand_pad = _pad_box(brand_union, source.width, source.height, pad_x=max(4, source.width // 80), pad_y=max(3, source.height // 60))
        brand_target = _scale_source_box_to_target(brand_pad, source, width, height, x_scale=x_scale, y_scale=uniform_scale)
        # Preserve corner intent: top-left brands stay top-left, top-right badges stay top-right.
        brand_cx, _ = _box_center(brand_pad)
        if brand_cx > source.width * 0.55:
            bw = brand_target[2] - brand_target[0]
            bh = brand_target[3] - brand_target[1]
            brand_target = _clip_box((width - bw - int(width * 0.035), int(height * 0.04), width - int(width * 0.035), int(height * 0.04) + bh), width, height)
        else:
            bw = brand_target[2] - brand_target[0]
            bh = brand_target[3] - brand_target[1]
            brand_target = _clip_box((int(width * 0.035), int(height * 0.04), int(width * 0.035) + bw, int(height * 0.04) + bh), width, height)
        composited.append(
            _paste_crop_fit(
                output,
                source,
                brand_pad,
                brand_target,
                layer_id="role-aware-brand",
                role="brand_logo_or_badge",
                mode="contain",
            )
        )

    primary_union = _union_boxes(parts["primary"])
    source_boxes_by_id = {layer.id: _tighten_marketing_text_box(source, _layer_box(layer, source)) for layer in analysis.marketing_text_layers}
    primary_ids = {layer_id for layer_id, box in source_boxes_by_id.items() if box in parts["primary"]}
    secondary_ids = {layer_id for layer_id, box in source_boxes_by_id.items() if box in parts["secondary"]}
    primary_source_union = _union_boxes([source_boxes_by_id[layer_id] for layer_id in primary_ids])
    secondary_source_union = _union_boxes([source_boxes_by_id[layer_id] for layer_id in secondary_ids])
    selected_text_blocks: list[Any] = []
    if os.getenv("ADAPTIFAI_RESIZE_ENABLE_TEXT_REDRAW", "0").strip().lower() in {"1", "true", "yes", "on"} and text_blocks and draw_text:
        for block in text_blocks:
            block_id = str(getattr(block, "id", "") or "")
            source_id = block_id.replace("v5-resize-", "")
            source_box = source_boxes_by_id.get(source_id)
            if source_id in primary_ids and source_box and primary_source_union:
                target_box = _map_box_between_unions(source_box, primary_source_union, primary_bounds, width, height)
                selected_text_blocks.append(_copy_text_block_to_box(block, target_box))
            elif source_id in secondary_ids and source_box and secondary_source_union and target_ratio >= 1.10:
                target_box = _map_box_between_unions(source_box, secondary_source_union, secondary_bounds, width, height)
                selected_text_blocks.append(_copy_text_block_to_box(block, target_box))

    if primary_union and not selected_text_blocks:
        primary_pad = _pad_box(primary_union, source.width, source.height, pad_x=max(4, source.width // 100), pad_y=max(3, source.height // 70))
        primary_scale = min(width / max(1, source.width), height / max(1, source.height)) if target_ratio < 1.35 else x_scale
        target = _bounded_target_from_source_group(primary_pad, source, width, height, primary_bounds, scale=primary_scale)
        # If the mathematically scaled copy is too large for its zone, contain it inside the zone.
        if target[2] - target[0] > primary_bounds[2] - primary_bounds[0] or target[3] - target[1] > primary_bounds[3] - primary_bounds[1]:
            target = primary_bounds
        composited.append(
            _paste_crop_fit(
                output,
                source,
                primary_pad,
                target,
                layer_id="role-aware-primary-copy",
                role="primary_marketing_copy",
                mode="contain",
            )
        )

    secondary_union = _union_boxes(parts["secondary"])
    if secondary_union and not selected_text_blocks and (target_ratio >= 1.10 or not primary_union):
        secondary_pad = _pad_box(secondary_union, source.width, source.height, pad_x=max(4, source.width // 120), pad_y=max(3, source.height // 80))
        secondary_scale = min(x_scale, y_scale) * (0.92 if target_ratio < 1.35 else 1.0)
        target = _bounded_target_from_source_group(secondary_pad, source, width, height, secondary_bounds, scale=secondary_scale)
        if target[2] - target[0] > secondary_bounds[2] - secondary_bounds[0] or target[3] - target[1] > secondary_bounds[3] - secondary_bounds[1]:
            target = secondary_bounds
        composited.append(
            _paste_crop_fit(
                output,
                source,
                secondary_pad,
                target,
                layer_id="role-aware-secondary-copy",
                role="secondary_callouts",
                mode="contain",
            )
        )
    if selected_text_blocks and draw_text:
        output = draw_text(output.convert("RGB"), selected_text_blocks).convert("RGBA")
        composited.append(
            {
                "layerId": "role-aware-redrawn-marketing-copy",
                "role": "redrawn_marketing_copy",
                "sourceBox": [],
                "targetBox": [],
                "pasteBox": [],
                "scale": None,
                "fitMode": "phase1_text_redraw",
                "blockCount": len(selected_text_blocks),
            }
        )

    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "roleAwareRelayout": True,
        "sourceRatio": round(source_ratio, 4),
        "targetRatio": round(target_ratio, 4),
        "omittedProductLabelTextCount": len(parts["product_label"]),
        **visual_meta,
    }


def _paste_crop_inside(
    output: Image.Image,
    crop_source: Image.Image,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
) -> dict[str, Any]:
    crop = crop_source.crop(source_box).convert("RGBA")
    paste_left, paste_top, paste_right, paste_bottom, scale = _fit_inside(crop.size, target_box)
    resized = crop.resize((paste_right - paste_left, paste_bottom - paste_top), Image.Resampling.LANCZOS)
    output.alpha_composite(resized, (paste_left, paste_top))
    return {
        "layerId": layer_id,
        "role": role,
        "sourceBox": list(source_box),
        "targetBox": list(target_box),
        "pasteBox": [paste_left, paste_top, paste_right, paste_bottom],
        "scale": round(scale, 4),
    }


def composite_vertical_band_relayout(
    canvas: Image.Image,
    source: Image.Image,
    foreground_source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    source_ratio = source.width / max(1, source.height)
    composited: list[dict[str, Any]] = []

    if source_ratio > 1.35:
        text_box = _clip_box((0, 0, int(source.width * 0.48), source.height), source.width, source.height)
        visual_box = _clip_box((int(source.width * 0.52), 0, source.width, source.height), source.width, source.height)
        composited.append(
            _paste_crop_inside(
                output,
                source,
                text_box,
                (int(width * 0.055), int(height * 0.18), int(width * 0.945), int(height * 0.43)),
                layer_id="source-text-zone",
                role="preserved_text_band",
            )
        )
        composited.append(
            _paste_crop_inside(
                output,
                source,
                visual_box,
                (int(width * 0.07), int(height * 0.43), int(width * 0.93), int(height * 0.74)),
                layer_id="source-visual-zone",
                role="relayout_visual_band",
            )
        )
    else:
        text_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
        if text_boxes:
            upper_text_boxes = [box for box in text_boxes if box[1] < int(source.height * 0.46)] or text_boxes
            text_bottom = max(box[3] for box in upper_text_boxes)
            text_bottom = min(source.height, max(text_bottom + int(source.height * 0.045), int(source.height * 0.26)))
        else:
            text_bottom = int(source.height * 0.35)
        text_box = _clip_box((0, 0, source.width, text_bottom), source.width, source.height)
        visual_top = max(0, min(text_bottom - int(source.height * 0.045), int(source.height * 0.42)))
        visual_box = _clip_box((0, visual_top, source.width, source.height), source.width, source.height)
        composited.append(
            _paste_crop_inside(
                output,
                source,
                text_box,
                (int(width * 0.065), int(height * 0.22), int(width * 0.935), int(height * 0.47)),
                layer_id="source-text-zone",
                role="preserved_text_band",
            )
        )
        composited.append(
            _paste_crop_inside(
                output,
                foreground_source,
                visual_box,
                (int(width * 0.065), int(height * 0.47), int(width * 0.935), int(height * 0.81)),
                layer_id="source-visual-zone",
                role="relayout_visual_band",
            )
        )

    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "verticalBandRelayout": True,
    }


def composite_horizontal_band_relayout(
    canvas: Image.Image,
    source: Image.Image,
    foreground_source: Image.Image,
    width: int,
    height: int,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    composited: list[dict[str, Any]] = []
    text_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
    if text_boxes:
        text_bottom = max(box[3] for box in text_boxes if box[1] < int(source.height * 0.55)) if any(box[1] < int(source.height * 0.55) for box in text_boxes) else max(box[3] for box in text_boxes)
        text_bottom = min(source.height, max(text_bottom + int(source.height * 0.04), int(source.height * 0.24)))
    else:
        text_bottom = int(source.height * 0.34)
    text_box = _clip_box((0, 0, source.width, text_bottom), source.width, source.height)
    visual_top = max(0, min(text_bottom - int(source.height * 0.04), int(source.height * 0.42)))
    visual_box = _clip_box((0, visual_top, source.width, source.height), source.width, source.height)
    composited.append(
        _paste_crop_inside(
            output,
            source,
            text_box,
            (int(width * 0.045), int(height * 0.12), int(width * 0.45), int(height * 0.88)),
            layer_id="source-text-zone",
            role="preserved_text_band",
        )
    )
    composited.append(
        _paste_crop_inside(
            output,
            foreground_source,
            visual_box,
            (int(width * 0.43), int(height * 0.08), int(width * 0.97), int(height * 0.92)),
            layer_id="source-visual-zone",
            role="relayout_visual_band",
        )
    )
    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "horizontalBandRelayout": True,
    }


def render_deterministic_compositor(
    source: Image.Image,
    width: int,
    height: int,
    plan: ReframePlan,
    analysis: VisualAnalysis,
    *,
    text_blocks: list[Any],
    draw_text: TextRenderer,
    outpaint_renderer: ImageRenderer | None,
    fallback_renderer: FallbackRenderer,
) -> tuple[Image.Image, dict[str, Any]]:
    placement_id = getattr(plan, "placement_id", None)
    preserve_brand_layers = _placement_preserves_creative_brand_layers(placement_id)
    foreground_source, _foreground_text_mask, foreground_meta = clean_overlay_text_only(source, analysis)
    clean_source, text_mask, cleanup_meta = clean_compositor_background_source(source, analysis)
    provider_safe_seed, provider_safe_seed_meta = build_resize_provider_safe_seed(
        source,
        analysis,
        placement_id=placement_id,
        remove_products=True,
    )
    landscape_anchor = should_use_landscape_width_anchor(source, width, height)
    role_aware_candidate = should_use_role_aware_relayout(source, width, height, analysis)
    preserve_complex_visual = should_preserve_full_creative_for_complex_visuals(source, width, height, analysis)
    should_seed_outpaint_with_foreground = plan.expansion.strategy == ExpansionStrategy.OPENAI_OUTPAINT or plan.expansion.requires_ai
    background_seed = provider_safe_seed if should_seed_outpaint_with_foreground else clean_source
    role_aware_seed_meta: dict[str, Any] = {}
    if role_aware_candidate and should_seed_outpaint_with_foreground:
        visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
        parts = _partition_resize_layers(source, analysis, visual_box)
        # Provider outpaint must never see floating marketing copy or social brand marks.
        # Otherwise it hallucinates corrupted typography in the generated area.
        role_removal_boxes = [*parts["primary"], *parts["secondary"], *parts.get("trust_badge", [])]
        if _resize_provider_must_not_see_brand_layers(placement_id):
            role_removal_boxes.extend(parts["brand"])
        background_seed = _inpaint_rectangular_overlays(source, role_removal_boxes)
        role_aware_seed_meta = {
            "roleAwareOutpaintSeed": "product_label_preserving_text_and_social_brand_removed",
            "roleAwareOutpaintSeedRemovedLayerCount": len(role_removal_boxes),
            "roleAwareOutpaintSeedRemovedBrandLayers": 0 if not _resize_provider_must_not_see_brand_layers(placement_id) else len(parts["brand"]),
            "roleAwareOutpaintSeedRemovedTrustBadges": len(parts.get("trust_badge", [])),
            **{f"seed_{key}": value for key, value in visual_meta.items()},
        }
    if preserve_complex_visual:
        background_source = background_seed if should_seed_outpaint_with_foreground else (foreground_source if _foreground_text_mask.getbbox() else source)
        fallback_background_source = background_source
        provider_background_meta: dict[str, Any] = {}
        provider_ready = False
        provider_visual_completion_source: Image.Image | None = None
        source_ratio = source.width / max(1, source.height)
        target_ratio = width / max(1, height)
        target_area_smaller = width * height < source.width * source.height
        if (
            should_seed_outpaint_with_foreground
            and source_ratio > 1.25
            and target_ratio < 1.25
            and os.getenv("ADAPTIFAI_RESIZE_LAYOUT_SEED_OUTPAINT", "1").strip().lower() in {"1", "true", "yes", "on"}
        ):
            background_source, layout_seed_meta = build_wide_to_portrait_outpaint_layout_seed(source, width, height, analysis)
            provider_background_meta.update(layout_seed_meta)
        if should_seed_outpaint_with_foreground and outpaint_renderer is None:
            # The layout seed contains protected foreground alpha and RGB context
            # for an edit provider. It is not a final background. Without a
            # provider, falling through with this seed clones the product behind
            # the deterministic foreground layer.
            background_source = fallback_background_source
            provider_background_meta = {
                **provider_background_meta,
                "provider": "local",
                "outpaintProviderFailure": "No outpaint provider is enabled for deterministic compositor.",
                "layoutOutpaintSeedUsedAsFinalBackground": False,
                "productionReady": False,
            }
        if should_seed_outpaint_with_foreground and outpaint_renderer is not None:
            try:
                provider_background, generated_provider_meta = outpaint_renderer(background_source, width, height, plan, analysis)
                if provider_background.size != (width, height):
                    provider_background = provider_background.resize((width, height), Image.Resampling.LANCZOS)
                provider_rgb = provider_background.convert("RGB")
                provider_background_meta = {**provider_background_meta, **generated_provider_meta}
                if provider_background_meta.get("layoutOutpaintSeed"):
                    source_edge = _edge_median_rgb(source)
                    provider_edge = _edge_median_rgb(provider_rgb)
                    edge_distance = float(np.linalg.norm(source_edge - provider_edge))
                    provider_background_meta["providerEdgeColorDistance"] = round(edge_distance, 3)
                    if edge_distance > float(os.getenv("ADAPTIFAI_RESIZE_PROVIDER_EDGE_COLOR_REJECT", "70")):
                        background_source = fallback_background_source
                        provider_background_meta["providerRejected"] = "edge_color_drift"
                        provider_background_meta["productionReady"] = False
                        provider_ready = False
                    else:
                        background_source = fallback_background_source
                        provider_background_meta["providerRejected"] = "layout_seed_background_can_hallucinate_product_or_text"
                        provider_background_meta["productionReady"] = False
                        if target_ratio < 0.78:
                            provider_background_meta["providerSalvage"] = "foreground_completion_only"
                            provider_visual_completion_source = provider_rgb
                            provider_ready = True
                        else:
                            provider_ready = False
                else:
                    background_source = provider_rgb
                    provider_ready = True
            except Exception as exc:
                background_source = fallback_background_source
                provider_background_meta = {
                    **provider_background_meta,
                    "provider": "local",
                    "outpaintProviderFailure": str(exc),
                    "productionReady": False,
                }
        if target_area_smaller:
            rendered, fit_meta = composite_priority_layer_resize(
                background_source,
                source,
                width,
                height,
                analysis,
                text_blocks=text_blocks,
                draw_text=draw_text,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_priority_layer_resize"
            pipeline = "resize-deterministic-compositor-v4-priority-layer"
            background_strategy = "deterministic_background_priority_layer_resize"
        elif source_ratio > 1.25 and target_ratio < 1.25:
            rendered, fit_meta = composite_wide_creative_director_relayout(
                background_source,
                source,
                width,
                height,
                analysis,
                text_blocks=text_blocks,
                draw_text=draw_text,
                visual_already_protected=provider_ready and bool(provider_background_meta.get("layoutOutpaintSeed")),
                visual_completion_source=provider_visual_completion_source,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_creative_director_relayout"
            pipeline = "resize-deterministic-compositor-v3-creative-director-relayout"
            background_strategy = "deterministic_focus_cover_underlay_role_safe_layer_relayout"
        elif (
            target_ratio < 0.78
            and not (source_ratio <= 1.25 and len(analysis.marketing_text_layers) >= 3)
        ):
            rendered, fit_meta = composite_priority_layer_resize(
                background_source,
                source,
                width,
                height,
                analysis,
                text_blocks=text_blocks,
                draw_text=draw_text,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_priority_layer_resize"
            pipeline = "resize-deterministic-compositor-v4-priority-layer"
            background_strategy = "deterministic_background_priority_layer_resize"
        else:
            rendered, fit_meta = composite_full_creative_preserve(background_source, source, width, height, analysis)
            strategy = "deterministic_compositor_full_creative_preserve_complex_visual"
            pipeline = "resize-deterministic-compositor-v2-complex-visual-preserve"
            background_strategy = "deterministic_focus_cover_underlay_full_creative_overlay"
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_strategy,
            "backgroundSource": "text_clean_source_focus_cover" if text_mask.getbbox() else "source_focus_cover",
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **provider_background_meta,
            **fit_meta,
            "provider": provider_background_meta.get("provider", "local"),
            "strategy": strategy,
            "pipeline": pipeline,
            # A provider result is production-ready only when the compositor
            # safety gate accepted it into the final background.
            "productionReady": (
                True
                if target_area_smaller or (not preserve_brand_layers and target_ratio >= 0.78)
                else (provider_ready if should_seed_outpaint_with_foreground else True)
            ),
        }
    if landscape_anchor:
        background = build_deterministic_background_canvas(foreground_source, width, height)
        background_meta = {
            "provider": "local",
            "strategy": "deterministic_landscape_background_continuation",
            "backgroundSource": "text_clean_source_background_continuation",
            "productionReady": True,
        }
    elif role_aware_candidate:
        provider_role_meta: dict[str, Any] = {}
        if should_seed_outpaint_with_foreground and outpaint_renderer is not None:
            try:
                provider_background, provider_role_meta = outpaint_renderer(background_seed, width, height, plan, analysis)
                if provider_background.size != (width, height):
                    provider_background = provider_background.resize((width, height), Image.Resampling.LANCZOS)
                background = build_deterministic_background_canvas(background_seed, width, height)
                provider_role_meta = {
                    **provider_role_meta,
                    "providerRejected": "role_provider_background_can_hallucinate_text_or_products",
                    "productionReady": False,
                }
            except Exception as exc:
                background = build_deterministic_background_canvas(background_seed, width, height)
                provider_role_meta = {
                    "provider": "local",
                    "outpaintProviderFailure": str(exc),
                    "productionReady": False,
                }
        else:
            background = build_deterministic_background_canvas(background_seed, width, height)
        background_meta = {
            "provider": provider_role_meta.get("provider", "local"),
            "strategy": provider_role_meta.get("strategy", "role_aware_deterministic_focus_cover_canvas"),
            "backgroundSource": provider_role_meta.get("backgroundSource", "product_label_preserving_visual_seed"),
            "productionReady": bool(provider_role_meta.get("productionReady", not should_seed_outpaint_with_foreground)),
            **role_aware_seed_meta,
            **provider_role_meta,
        }
    else:
        background, background_meta = render_background(
            background_seed,
            width,
            height,
            plan,
            analysis,
            outpaint_renderer=outpaint_renderer,
            fallback_renderer=fallback_renderer,
        )
        background_meta = {**background_meta, **role_aware_seed_meta}
    if role_aware_candidate:
        target_ratio = width / max(1, height)
        target_area_smaller = _target_area_is_smaller(source, width, height)
        if target_area_smaller:
            rendered, relayout_meta = composite_priority_layer_resize(
                background,
                source,
                width,
                height,
                analysis,
                text_blocks=text_blocks,
                draw_text=draw_text,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_priority_layer_resize"
            pipeline = "resize-deterministic-compositor-v4-priority-layer"
            production_ready = True
        elif target_ratio < 0.78:
            rendered, relayout_meta = composite_wide_creative_director_relayout(
                background,
                source,
                width,
                height,
                analysis,
                text_blocks=text_blocks,
                draw_text=draw_text,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_creative_director_relayout"
            pipeline = "resize-deterministic-compositor-v3-creative-director-relayout"
            production_ready = True
        else:
            rendered, relayout_meta = composite_role_aware_relayout(
                background,
                source,
                foreground_source,
                width,
                height,
                analysis,
                background_contains_visual=background_meta.get("provider") not in {None, "local"},
                text_blocks=text_blocks,
                draw_text=draw_text,
                preserve_brand_layers=preserve_brand_layers,
            )
            strategy = "deterministic_compositor_role_aware_relayout"
            pipeline = "resize-deterministic-compositor-v2-role-aware"
            production_ready = background_meta.get("productionReady", False)
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **background_meta,
            **relayout_meta,
            "provider": background_meta.get("provider", "local"),
            "strategy": strategy,
            "pipeline": pipeline,
            "productionReady": production_ready,
        }
    if landscape_anchor:
        rendered, landscape_meta = composite_landscape_width_anchor(
            background,
            source,
            width,
            height,
            preserve_canvas_fill=True,
        )
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **background_meta,
            **landscape_meta,
            "provider": background_meta.get("provider", "local"),
            "strategy": "deterministic_compositor_landscape_width_anchor",
        }
    if should_use_source_fit(source, width, height, analysis):
        rendered, fit_meta = composite_source_fit(background, source, width, height)
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **background_meta,
            **fit_meta,
            "provider": background_meta.get("provider", "local"),
            "strategy": "deterministic_compositor_source_fit",
        }
    if should_use_vertical_band_relayout(source, width, height, analysis):
        rendered, relayout_meta = composite_vertical_band_relayout(background, source, foreground_source, width, height, analysis)
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **background_meta,
            **relayout_meta,
            "provider": background_meta.get("provider", "local"),
            "strategy": "deterministic_compositor_vertical_band_relayout",
        }
    if should_use_horizontal_band_relayout(source, width, height, analysis):
        rendered, relayout_meta = composite_horizontal_band_relayout(background, source, foreground_source, width, height, analysis)
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
            **provider_safe_seed_meta,
            **background_meta,
            **relayout_meta,
            "provider": background_meta.get("provider", "local"),
            "strategy": "deterministic_compositor_horizontal_band_relayout",
        }
    relayout, relayout_meta = composite_relayout_layers(background, source, foreground_source, plan, analysis)
    rendered = draw_text(relayout, text_blocks) if text_blocks else relayout
    return rendered.convert("RGB"), {
        "backgroundStrategy": background_meta.get("strategy"),
        "backgroundSource": background_meta.get("backgroundSource"),
        "textRedrawBlocks": len(text_blocks),
        "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
        **cleanup_meta,
        **foreground_meta,
        **provider_safe_seed_meta,
        **background_meta,
        **relayout_meta,
        "provider": background_meta.get("provider", "local"),
        "strategy": "deterministic_compositor",
        "pipeline": "resize-deterministic-compositor-v1",
    }
