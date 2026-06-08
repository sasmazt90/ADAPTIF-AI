from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    from app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer
except ImportError:
    from backend.app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer


ImageRenderer = Callable[[Image.Image, int, int, ReframePlan, VisualAnalysis], tuple[Image.Image, dict[str, Any]]]
FallbackRenderer = Callable[[Image.Image, int, int], Image.Image]
TextRenderer = Callable[[Image.Image, list[Any]], Image.Image]


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(left)))
    top = max(0, min(height - 1, int(top)))
    right = max(left + 1, min(width, int(right)))
    bottom = max(top + 1, min(height, int(bottom)))
    return left, top, right, bottom


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


def composite_relayout_layers(
    canvas: Image.Image,
    source: Image.Image,
    plan: ReframePlan,
    analysis: VisualAnalysis,
) -> tuple[Image.Image, dict[str, Any]]:
    output = canvas.convert("RGBA")
    layers = _layer_by_id(analysis)
    composited = []
    for placement in sorted(plan.placements, key=lambda item: item.z_index):
        if placement.role not in {LayerRole.PRODUCT, LayerRole.PERSON, LayerRole.LOGO}:
            continue
        layer = layers.get(placement.layer_id)
        if layer is None:
            continue
        crop, source_box = _extract_layer_crop(source, layer)
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
    clean_source, text_mask, cleanup_meta = clean_compositor_background_source(source, analysis)
    background, background_meta = render_background(
        clean_source,
        width,
        height,
        plan,
        analysis,
        outpaint_renderer=outpaint_renderer,
        fallback_renderer=fallback_renderer,
    )
    relayout, relayout_meta = composite_relayout_layers(background, source, plan, analysis)
    rendered = draw_text(relayout, text_blocks) if text_blocks else relayout
    return rendered.convert("RGB"), {
        "backgroundStrategy": background_meta.get("strategy"),
        "backgroundSource": background_meta.get("backgroundSource"),
        "textRedrawBlocks": len(text_blocks),
        "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
        **cleanup_meta,
        **background_meta,
        **relayout_meta,
        "provider": background_meta.get("provider", "local"),
        "strategy": "deterministic_compositor",
        "pipeline": "resize-deterministic-compositor-v1",
    }
