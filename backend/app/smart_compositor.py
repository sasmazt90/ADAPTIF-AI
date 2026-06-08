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
    if texture < float(__import__("os").getenv("ADAPTIFAI_COMPOSITOR_AI_TEXTURE_THRESHOLD", "0.46")):
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
    return canvas.filter(ImageFilter.GaussianBlur(radius=0.35))


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


def should_use_horizontal_band_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> bool:
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    if target_ratio <= 1.35:
        return False
    if source_ratio > 1.35:
        return False
    return bool(analysis.marketing_text_layers) and abs(source_ratio - target_ratio) >= 0.42


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
    foreground_source, _foreground_text_mask, foreground_meta = clean_overlay_text_only(source, analysis)
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
    if should_use_source_fit(source, width, height, analysis):
        rendered, fit_meta = composite_source_fit(background, source, width, height)
        return rendered.convert("RGB"), {
            "backgroundStrategy": background_meta.get("strategy"),
            "backgroundSource": background_meta.get("backgroundSource"),
            "textRedrawBlocks": 0,
            "textMaskNonZero": int(np.count_nonzero(np.array(text_mask, dtype=np.uint8))),
            **cleanup_meta,
            **foreground_meta,
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
        **background_meta,
        **relayout_meta,
        "provider": background_meta.get("provider", "local"),
        "strategy": "deterministic_compositor",
        "pipeline": "resize-deterministic-compositor-v1",
    }
