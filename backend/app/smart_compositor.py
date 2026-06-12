from __future__ import annotations

import os
import base64
import io
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

try:
    from app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer
except ImportError:
    from backend.app.smart_reframe import ExpansionStrategy, LayerRole, ReframePlan, VisualAnalysis, VisualLayer


ImageRenderer = Callable[[Image.Image, int, int, ReframePlan, VisualAnalysis], tuple[Image.Image, dict[str, Any]]]
ProductCompletionRenderer = Callable[[Image.Image, dict[str, Any]], tuple[Image.Image, dict[str, Any]]]
FallbackRenderer = Callable[[Image.Image, int, int], Image.Image]
TextRenderer = Callable[[Image.Image, list[Any]], Image.Image]
_REMBG_SESSION: Any | None = None
_PRODUCT_ALPHA_BASE_CACHE: dict[tuple[int, tuple[int, int, int, int], str], Image.Image] = {}


def _placement_preserves_creative_brand_layers(placement_id: str | None) -> bool:
    pid = (placement_id or "").strip().lower()
    return pid.startswith("gdn-") or pid.startswith("google-responsive") or pid == "custom-display"


def _placement_is_display_ad(placement_id: str | None) -> bool:
    return _placement_preserves_creative_brand_layers(placement_id)


def _rgba_alpha_edge_touch(rgba: Image.Image, *, threshold: int = 24) -> list[str]:
    try:
        alpha = np.array(rgba.convert("RGBA").getchannel("A"), dtype=np.uint8)
        if alpha.size == 0:
            return []
        band = max(1, min(alpha.shape[:2]) // 80)
        touches: list[str] = []
        if float(np.count_nonzero(alpha[:band, :] > threshold)) / max(1, alpha[:band, :].size) > 0.025:
            touches.append("top")
        if float(np.count_nonzero(alpha[-band:, :] > threshold)) / max(1, alpha[-band:, :].size) > 0.025:
            touches.append("bottom")
        if float(np.count_nonzero(alpha[:, :band] > threshold)) / max(1, alpha[:, :band].size) > 0.025:
            touches.append("left")
        if float(np.count_nonzero(alpha[:, -band:] > threshold)) / max(1, alpha[:, -band:].size) > 0.025:
            touches.append("right")
        return touches
    except Exception:
        return []


def _resize_provider_must_not_see_brand_layers(placement_id: str | None) -> bool:
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


def _image_data_uri(image: Image.Image, *, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _replicate_product_cutout(crop: Image.Image) -> tuple[Image.Image | None, dict[str, Any]]:
    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    model = os.getenv("REPLICATE_PRODUCT_CUTOUT_MODEL", "851-labs/background-remover").strip()
    if not token or "/" not in model:
        return None, {
            "productCutoutProvider": "replicate",
            "productCutoutSkipped": "missing_token_or_model",
        }
    try:
        import requests
        owner, name = model.split("/", 1)
        endpoint = f"https://api.replicate.com/v1/models/{owner}/{name}/predictions"
        payload = {"input": {"image": _image_data_uri(crop.convert("RGBA"))}}
        timeout = int(os.getenv("REPLICATE_PRODUCT_CUTOUT_TIMEOUT", "90"))
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json",
                "Prefer": "wait=45",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        prediction = response.json()
        deadline = time.time() + int(os.getenv("REPLICATE_PRODUCT_CUTOUT_POLL_TIMEOUT", "120"))
        while prediction.get("status") not in {"succeeded", "failed", "canceled"} and time.time() < deadline:
            poll_url = (prediction.get("urls") or {}).get("get")
            if not poll_url:
                break
            time.sleep(1.5)
            poll = requests.get(poll_url, headers={"Authorization": f"Token {token}"}, timeout=timeout)
            poll.raise_for_status()
            prediction = poll.json()
        if prediction.get("status") != "succeeded":
            return None, {
                "productCutoutProvider": "replicate",
                "productCutoutModel": model,
                "productCutoutError": f"status={prediction.get('status')}; error={prediction.get('error')}",
            }
        output = prediction.get("output")
        if isinstance(output, list):
            output = output[0] if output else None
        if isinstance(output, dict):
            output = output.get("image") or output.get("url") or output.get("output")
        if not isinstance(output, str) or not output:
            return None, {
                "productCutoutProvider": "replicate",
                "productCutoutModel": model,
                "productCutoutError": "empty_output",
            }
        if output.startswith("data:image"):
            encoded = output.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded)
        else:
            download = requests.get(output, timeout=timeout)
            download.raise_for_status()
            image_bytes = download.content
        extracted = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        if extracted.size != crop.size:
            extracted = extracted.resize(crop.size, Image.Resampling.LANCZOS)
        alpha = np.array(extracted.getchannel("A"), dtype=np.uint8)
        alpha_ratio = float(np.count_nonzero(alpha > 8)) / max(1, alpha.size)
        if not (0.025 <= alpha_ratio <= 0.82):
            return None, {
                "productCutoutProvider": "replicate",
                "productCutoutModel": model,
                "productCutoutError": "unsafe_alpha_ratio",
                "productCutoutAlphaRatio": round(alpha_ratio, 4),
            }
        return _prepare_foreground_rgba_crop(extracted), {
            "productCutoutProvider": "replicate",
            "productCutoutModel": model,
            "productCutoutAlphaRatio": round(alpha_ratio, 4),
        }
    except Exception as exc:
        return None, {
            "productCutoutProvider": "replicate",
            "productCutoutModel": model,
            "productCutoutError": str(exc)[:500],
        }


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
    return Image.blend(texture, canvas.filter(ImageFilter.GaussianBlur(radius=0.45)), 0.16)


def build_low_artifact_background_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
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
    mixed = np.clip((texture_arr * 0.34) + (gradient_arr * 0.66), 0, 255).astype(np.uint8)
    return Image.fromarray(mixed, "RGB").filter(ImageFilter.GaussianBlur(radius=0.28))


def build_edge_gradient_background_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
    source = clean_source.convert("RGB")
    arr = np.array(source, dtype=np.uint8)
    top_strip = arr[: max(1, source.height // 10), :, :].reshape(-1, 3)
    bottom_strip = arr[-max(1, source.height // 10) :, :, :].reshape(-1, 3)
    side_samples = np.concatenate(
        [
            arr[:, : max(1, source.width // 12), :].reshape(-1, 3),
            arr[:, -max(1, source.width // 12) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    neutral = np.median(side_samples, axis=0)
    top_color = np.percentile(top_strip, 72, axis=0) * 0.70 + neutral * 0.30
    bottom_color = np.percentile(bottom_strip, 72, axis=0) * 0.70 + neutral * 0.30
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(1, height - 1)
        color = top_color * (1 - t) + bottom_color * t
        gradient[y, :, :] = np.clip(color, 0, 255)
    return Image.fromarray(gradient, "RGB").filter(ImageFilter.GaussianBlur(radius=0.35))


def build_safe_background_texture_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
    source = clean_source.convert("RGB")
    source_ratio = source.width / max(1, source.height)
    if source_ratio > 1.25:
        gradient = build_edge_gradient_background_canvas(source, width, height)
        arr = np.array(gradient, dtype=np.int16)
        yy, xx = np.mgrid[0:height, 0:width]
        wave = (
            np.sin((xx / max(1, width)) * np.pi * 2.0 + (yy / max(1, height)) * np.pi * 0.65) * 2.6
            + np.cos((yy / max(1, height)) * np.pi * 2.4) * 1.8
        )
        arr = np.clip(arr + wave[:, :, None], 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.25))
    else:
        margin_x = max(1, source.width // 12)
        margin_y = max(1, source.height // 12)
        crop_box = (margin_x, margin_y, max(margin_x + 1, source.width - margin_x), max(margin_y + 1, source.height - margin_y))
    safe_region = source.crop(crop_box)
    cover_scale = max(width / max(1, safe_region.width), height / max(1, safe_region.height))
    cover_w = max(1, int(round(safe_region.width * cover_scale)))
    cover_h = max(1, int(round(safe_region.height * cover_scale)))
    cover = safe_region.resize((cover_w, cover_h), Image.Resampling.LANCZOS)
    left = max(0, (cover_w - width) // 2)
    top = max(0, (cover_h - height) // 2)
    texture = cover.crop((left, top, left + width, top + height)).filter(
        ImageFilter.GaussianBlur(radius=max(0.65, min(width, height) * 0.004))
    )
    gradient = build_edge_gradient_background_canvas(source, width, height)
    return Image.blend(texture, gradient, 0.22)


def build_resize_texture_background_canvas(clean_source: Image.Image, width: int, height: int) -> Image.Image:
    source_ratio = clean_source.width / max(1, clean_source.height)
    target_ratio = width / max(1, height)
    if abs(source_ratio - target_ratio) >= 0.35 or width * height <= clean_source.width * clean_source.height:
        return build_safe_background_texture_canvas(clean_source, width, height)
    return build_low_artifact_background_canvas(clean_source, width, height)


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
    
    # ADAPTİFAİ KURALI: Analizden ziyade deterministik foreground maskeyi kullan.
    # Logonun, metinlerin ve rozetlerin maskeden TAMAMEN dışlandığından _foreground_component_visual_boxes içinde eminiz.
    foreground_boxes = _foreground_component_visual_boxes(source, analysis)
    if not boxes and foreground_boxes:
        boxes = foreground_boxes
    if not boxes:
        return [visual_box]
    if foreground_boxes and w / max(1, h) > 1.35:
        avg_detected_height = sum(max(1, box[3] - box[1]) for box in boxes) / max(1, len(boxes))
        avg_foreground_height = sum(max(1, box[3] - box[1]) for box in foreground_boxes) / max(1, len(foreground_boxes))
        if avg_foreground_height >= avg_detected_height * 1.35:
            boxes = foreground_boxes
    merged = _merge_overlapping_boxes(boxes, pad=max(8, min(w, h) // 60))
    if len(merged) == 1 and _box_area(merged[0]) / source_area > 0.72:
        return [visual_box]
    return merged or [visual_box]


def _foreground_component_visual_boxes(source: Image.Image, analysis: VisualAnalysis | None = None) -> list[tuple[int, int, int, int]]:
    """Return deterministic product/person foreground boxes from alpha segmentation, strictly excluding marketing/logo fragments."""
    try:
        import cv2

        rgba = _cv_foreground_alpha_crop(source.convert("RGBA"))
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        h, w = alpha.shape[:2]
        
        # YENİ KURAL: Logo ve Metinleri maskeden KESİN DIŞLA (Gerçek Dekupe Zırhı)
        if analysis:
            exclusion_boxes = [layer.bbox.to_pixel_box(w, h) for layer in getattr(analysis, "marketing_text_layers", []) + getattr(analysis, "logo_layers", []) + getattr(analysis, "text_layers", [])]
            for ex_box in exclusion_boxes:
                left = max(0, int(ex_box[0]))
                top = max(0, int(ex_box[1]))
                right = min(w, int(ex_box[2]))
                bottom = min(h, int(ex_box[3]))
                alpha[top:bottom, left:right] = 0 # Maskeyi sil, sadece ana ürün (şişe) kalsın

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
    if not label_union or not compact_target:
        return None
    if _box_overlap(label_union, visual_box) <= 0:
        return None
    label_w = max(1, label_union[2] - label_union[0])
    label_h = max(1, label_union[3] - label_union[1])
    if _box_area(label_union) < max(16, source.width * source.height * 0.002):
        return None

    pad_x_ratio = 0.34 if compact_target else 0.42
    pad_x = max(int(round(label_w * pad_x_ratio)), source.width // (38 if compact_target else 32), 8)
    if compact_target:
        pad_top = max(int(round(label_h * 1.35)), int(round(source.height * 0.46)), 8)
        pad_bottom = max(int(round(label_h * 1.10)), int(round(source.height * 0.34)), 8)
    else:
        pad_top = max(int(round(label_h * 0.85)), source.height // 15, 8)
        pad_bottom = max(int(round(label_h * 0.75)), source.height // 16, 8)
    candidate = _clip_box(
        (
            label_union[0] - pad_x,
            label_union[1] - pad_top,
            label_union[2] + pad_x,
            label_union[3] + pad_bottom,
        ),
        source.width,
        source.height,
    )

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
    def package_label_expanded_box() -> tuple[tuple[int, int, int, int] | None, int]:
        package_label_candidates: list[tuple[int, int, int, int]] = []
        candidate_layers = [
            *getattr(analysis, "text_layers", []),
            *getattr(analysis, "marketing_text_layers", []),
        ]
        seen_candidate_ids: set[str] = set()
        for layer in candidate_layers:
            layer_id = str(getattr(layer, "id", ""))
            if layer_id and layer_id in seen_candidate_ids:
                continue
            if layer_id:
                seen_candidate_ids.add(layer_id)
            box = _layer_box(layer, source)
            cx, cy = _box_center(box)
            rel_x = cx / max(1, source.width)
            rel_y = cy / max(1, source.height)
            area_ratio = _box_area(box) / max(1, source.width * source.height)
            role = str(getattr(layer, "role", "") or "").lower()
            label_like_role = "label" in role or "package" in role or "product" in role
            max_area_ratio = 0.09 if label_like_role else 0.045
            if 0.42 <= rel_x <= 0.78 and rel_y >= 0.26 and area_ratio <= max_area_ratio:
                package_label_candidates.append(box)
        label_union = _union_boxes(package_label_candidates)
        if not label_union:
            return None, 0
        label_w = max(1, label_union[2] - label_union[0])
        label_h = max(1, label_union[3] - label_union[1])
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
        ), len(package_label_candidates)

    product_boxes = []
    source_ratio = source.width / max(1, source.height)
    text_boxes_for_filter = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
    label_expanded_box, label_candidate_count = package_label_expanded_box()
    for layer in analysis.product_layers:
        box = _layer_box(layer, source)
        _, cy = _box_center(box)
        box_h_ratio = (box[3] - box[1]) / max(1, source.height)
        if source_ratio > 1.35 and cy < source.height * 0.24 and box_h_ratio < 0.30:
            continue
        touches_vertical_edge = box[1] <= source.height * 0.035 or box[3] >= source.height * 0.965
        if source_ratio > 1.35 and (box_h_ratio < 0.45 or touches_vertical_edge):
            box = _expand_product_seed_box(source, box)
        text_overlap_ratio = sum(_box_overlap(box, text_box) for text_box in text_boxes_for_filter) / max(1, _box_area(box))
        if source_ratio > 1.35 and text_overlap_ratio > 0.18:
            continue
        if _box_area(box) / max(1, source.width * source.height) <= 0.62:
            product_boxes.append(box)
    product_union = _union_boxes(product_boxes)
    if product_union:
        if source_ratio > 1.35 and label_expanded_box:
            product_w = max(1, product_union[2] - product_union[0])
            label_w = max(1, label_expanded_box[2] - label_expanded_box[0])
            misses_likely_body = product_union[0] > label_expanded_box[0] + source.width * 0.07
            overreaches_right_artwork = product_union[2] > source.width * 0.82 and product_w > label_w * 0.82
            too_narrow_for_package = product_w < label_w * 0.72
            if misses_likely_body or overreaches_right_artwork or too_narrow_for_package:
                return label_expanded_box, {
                    "visualBoxMethod": "package_label_expanded_product_override",
                    "packageLabelCandidateCount": label_candidate_count,
                    "rejectedProductUnion": list(product_union),
                }
        return _pad_box(product_union, source.width, source.height, pad_x=max(6, source.width // 36), pad_y=max(6, source.height // 28)), {
            "visualBoxMethod": "product_layer_union"
        }
    
    # Kural 1 entegrasyonu: Sadece şişeyi al.
    foreground_boxes = _foreground_component_visual_boxes(source, analysis)
    if foreground_boxes:
        foreground_union = _union_boxes(foreground_boxes)
        if foreground_union:
            return _pad_box(foreground_union, source.width, source.height, pad_x=max(6, source.width // 44), pad_y=max(6, source.height // 34)), {
                "visualBoxMethod": "foreground_component_union"
            }
            
    # OpenCV fallback
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
                if abs(cx - seed_cx) < w * 0.25:
                    keep.append(box)
            fallback = _union_boxes(keep) or seed
            return _pad_box(fallback, w, h, pad_x=max(8, w // 26), pad_y=max(8, h // 20)), {"visualBoxMethod": "opencv_fallback"}
    except Exception:
        pass

    return (0, 0, source.width, source.height), {"visualBoxMethod": "failed_full_source"}


def _sort_redraw_blocks_by_source_yx(blocks: list[Any]) -> list[Any]:
    def source_order_box(item: Any) -> tuple[int, int, int, int]:
        box = getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0))
        try:
            if len(box) == 4:
                return tuple(int(value) for value in box)
        except Exception:
            pass
        return (0, 0, 0, 0)
    return sorted(blocks, key=lambda item: (source_order_box(item)[1], source_order_box(item)[0]))


def _merge_redraw_blocks_by_inline_rows(blocks: list[Any], copy_zone_width: int) -> list[Any]:
    if not blocks:
        return []
    ordered = sorted(blocks, key=lambda item: (
        (getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0)))[1],
        (getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0)))[0]
    ))
    merged: list[Any] = []
    index = 0
    while index < len(ordered):
        current = ordered[index]
        group = [current]
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            current_box = getattr(current, "resize_source_box", None) or getattr(current, "bbox", (0, 0, 0, 0))
            candidate_box = getattr(candidate, "resize_source_box", None) or getattr(candidate, "bbox", (0, 0, 0, 0))
            if current_box[2] < candidate_box[0] and abs(current_box[1] - candidate_box[1]) < (current_box[3] - current_box[1]) * 0.4:
                group.append(candidate)
                next_index += 1
            else:
                break
        if len(group) == 1:
            merged.append(current)
        else:
            try:
                base = current.model_copy(deep=True)
            except Exception:
                base = current.copy(deep=True)
            source_word_styles: list[dict[str, Any]] = []
            translated_style_spans: list[dict[str, Any]] = []
            item_words = []
            for item_index, item in enumerate(group):
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
        index = next_index
    return sorted(merged, key=lambda item: (
        (getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0)))[1],
        (getattr(item, "resize_source_box", None) or getattr(item, "bbox", (0, 0, 0, 0)))[0]
    ))


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
        base = ordered[0].copy(deep=True)

    copy_w = max(1, copy_zone[2] - copy_zone[0])
    copy_h = max(1, copy_zone[3] - copy_zone[1])
    pad_x = max(0 if social else 4, int(round(copy_w * (0.02 if social else 0.10))))
    stack_zone = (
        min(copy_zone[2] - 1, copy_zone[0] + pad_x),
        copy_zone[1],
        max(copy_zone[0] + pad_x + 1, copy_zone[2] - pad_x),
        copy_zone[3],
    )
    stack_w = max(1, stack_zone[2] - stack_zone[0])
    stack_h = max(1, stack_zone[3] - stack_zone[1])

    def source_lines_for_block(block: Any) -> list[str]:
        line_texts = [str(line).strip() for line in (getattr(block, "line_texts", []) or []) if str(line).strip()]
        if line_texts:
            return line_texts
        raw = str(getattr(block, "translated_text", None) or getattr(block, "text", "") or "")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines if lines else ([raw.strip()] if raw.strip() else [])

    is_secondary = stack_role != "primary"
    min_readable_font_size = 14 if is_secondary else 18
    max_font_size = 64 if is_secondary else 86

    spans: list[dict[str, Any]] = []
    source_word_styles: list[dict[str, Any]] = []
    texts: list[str] = []
    
    all_lines = []
    for block in ordered:
        all_lines.extend(source_lines_for_block(block))
        
    if not all_lines:
        return []

    # ADAPTİF TİPOGRAFİ MOTORU: Satır sayısı önceliği
    best_fit_font_size = max_font_size
    for fs in range(max_font_size, min_readable_font_size - 1, -1):
        fits = True
        for line in all_lines:
            estimated_w = len(line) * fs * 0.58
            if estimated_w > stack_w:
                fits = False
                break
        if fits:
            best_fit_font_size = fs
            break
    else:
        best_fit_font_size = min_readable_font_size

    stack_font_size = best_fit_font_size
    stack_line_height = max(stack_font_size + 2, int(round(stack_font_size * 1.22)))

    for block_index, block in enumerate(ordered):
        block_lines = source_lines_for_block(block)
        texts.extend(block_lines)
        block_spans = [dict(span) for span in (getattr(block, "translated_style_spans", []) or []) if isinstance(span, dict)]
        if not block_spans:
            block_spans = [{"text": line, "translatedText": line, "style": {"fontWeight": 700}} for line in block_lines]
        for span in block_spans:
            style = span.setdefault("style", {})
            style["fontSize"] = stack_font_size
            style["lineHeight"] = stack_line_height
        spans.extend(block_spans)
        
        item_words = [dict(word) for word in (getattr(block, "source_word_styles", []) or []) if isinstance(word, dict)]
        source_word_styles.extend(item_words)

    total_h = len(all_lines) * stack_line_height
    start_y = stack_zone[1] + max(0, (stack_h - total_h) // 2)

    updates = {
        "text": "\n".join(texts),
        "translated_text": "\n".join(texts),
        "translated_style_spans": spans,
        "source_word_styles": source_word_styles,
        "bbox": (stack_zone[0], start_y, stack_zone[2], start_y + total_h),
        "clean_box": (stack_zone[0], start_y, stack_zone[2], start_y + total_h),
        "line_boxes": [],
        "line_texts": texts,
        "font_size_estimate": stack_font_size,
        "line_height_estimate": stack_line_height,
        "render_strategy": "resize_display_copy_stack",
        "resize_stack_role": stack_role,
        "align": "left" if not social else "center",
    }
    for key, value in updates.items():
        setattr(base, key, value)

    return [base]


def _inpaint_rectangular_overlays(source: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    if not boxes:
        return source.copy()
    try:
        import cv2
        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        for box in boxes:
            box = _clip_box(box, source.width, source.height)
            mask[box[1] : box[3], box[0] : box[2]] = 255
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        cleaned = cv2.inpaint(rgb, mask, 7, cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB")
    except Exception:
        return source.copy()


def _remove_foreground_visuals_for_background(source: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    if not boxes:
        return source.copy()
    try:
        import cv2
        rgba = _cv_foreground_alpha_crop(source.convert("RGBA"))
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        mask = np.zeros((source.height, source.width), dtype=np.uint8)
        for box in boxes:
            box = _clip_box(box, source.width, source.height)
            local = alpha[box[1] : box[3], box[0] : box[2]]
            if int(np.count_nonzero(local > 32)) > 0:
                mask[box[1] : box[3], box[0] : box[2]][local > 32] = 255
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        cleaned = cv2.inpaint(rgb, mask_dilated, 9, cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB")
    except Exception:
        return source.copy()


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
    
    # Alfa kirliliğini engellemek için doğrudan global segmentation üzerinden gelindiğini varsayıyoruz.
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
        "scale": None,
        "fitMode": "exact_stretch",
    }


def _paste_layer_relative(
    output: Image.Image,
    source: Image.Image,
    layer_box: tuple[int, int, int, int],
    target_bounds: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    scale: float,
    preserve_source_position: bool,
) -> dict[str, Any]:
    layer_box = _clip_box(layer_box, source.width, source.height)
    source_w = max(1, layer_box[2] - layer_box[0])
    source_h = max(1, layer_box[3] - layer_box[1])
    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))
    target_bounds = _clip_box(target_bounds, output.width, output.height)
    if preserve_source_position:
        rel_x = (layer_box[0] + layer_box[2]) / 2 / max(1, source.width)
        rel_y = (layer_box[1] + layer_box[3]) / 2 / max(1, source.height)
        cx = target_bounds[0] + int(round(rel_x * (target_bounds[2] - target_bounds[0])))
        cy = target_bounds[1] + int(round(rel_y * (target_bounds[3] - target_bounds[1])))
    else:
        cx = (target_bounds[0] + target_bounds[2]) // 2
        cy = (target_bounds[1] + target_bounds[3]) // 2
    paste_x = cx - new_w // 2
    paste_y = cy - new_h // 2
    paste_x = max(target_bounds[0], min(target_bounds[2] - new_w, paste_x))
    paste_y = max(target_bounds[1], min(target_bounds[3] - new_h, paste_y))
    return _paste_crop_exact(output, source, layer_box, (paste_x, paste_y, paste_x + new_w, paste_y + new_h), layer_id=layer_id, role=role)


def _paste_crop_contain_limited(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    target_bounds: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    max_scale: float,
    anchor: tuple[float, float] = (0.5, 0.5),
    artwork_alpha: bool = False,
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    target_bounds = _clip_box(target_bounds, output.width, output.height)
    sw = max(1, source_box[2] - source_box[0])
    sh = max(1, source_box[3] - source_box[1])
    tw = max(1, target_bounds[2] - target_bounds[0])
    th = max(1, target_bounds[3] - target_bounds[1])
    scale = min(tw / sw, th / sh, max_scale)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    px = target_bounds[0] + int(round((tw - nw) * anchor[0]))
    py = target_bounds[1] + int(round((th - nh) * anchor[1]))
    if artwork_alpha:
        try:
            crop = source.crop(source_box).convert("RGBA")
            hsv = np.array(crop.convert("HSV"), dtype=np.uint8)
            alpha = crop.getchannel("A")
            alpha_np = np.array(alpha, dtype=np.uint8)
            alpha_np[hsv[:, :, 1] < 12] = 0
            crop.putalpha(Image.fromarray(alpha_np, "L").filter(ImageFilter.GaussianBlur(radius=0.75)))
            output.alpha_composite(crop.resize((nw, nh), Image.Resampling.LANCZOS), (px, py))
            return {"layerId": layer_id, "role": role, "sourceBox": list(source_box), "targetBox": list(target_bounds), "pasteBox": [px, py, px + nw, py + nh], "scale": round(scale, 4), "fitMode": "contain_limited_artwork_alpha"}
        except Exception:
            pass
    return _paste_crop_exact(output, source, source_box, (px, py, px + nw, py + nh), layer_id=layer_id, role=role)


def _paste_crop_fit(
    output: Image.Image,
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    target_bounds: tuple[int, int, int, int],
    *,
    layer_id: str,
    role: str,
    mode: str = "contain",
) -> dict[str, Any]:
    source_box = _clip_box(source_box, source.width, source.height)
    target_bounds = _clip_box(target_bounds, output.width, output.height)
    sw = max(1, source_box[2] - source_box[0])
    sh = max(1, source_box[3] - source_box[1])
    tw = max(1, target_bounds[2] - target_bounds[0])
    th = max(1, target_bounds[3] - target_bounds[1])
    if mode == "cover":
        scale = max(tw / sw, th / sh)
    else:
        scale = min(tw / sw, th / sh)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    px = target_bounds[0] + (tw - nw) // 2
    py = target_bounds[1] + (th - nh) // 2
    return _paste_crop_exact(output, source, source_box, (px, py, px + nw, py + nh), layer_id=layer_id, role=role)


def _partition_resize_layers(
    source: Image.Image,
    analysis: VisualAnalysis,
    visual_box: tuple[int, int, int, int],
) -> dict[str, list[tuple[int, int, int, int]]]:
    parts: dict[str, list[tuple[int, int, int, int]]] = {
        "brand": [],
        "primary": [],
        "rtb": [],
        "trust_badge": [],
    }
    visual_cy = (visual_box[1] + visual_box[3]) / 2
    brand_texts = [layer for layer in analysis.marketing_text_layers if classify_text_role(layer.text) == "product_label" and layer.bbox.to_pixel_box(source.width, source.height)[3] < source.height * 0.22]
    brand_union = _union_boxes([_layer_box(layer, source) for layer in [*analysis.logo_layers, *brand_texts]])
    for layer in analysis.marketing_text_layers:
        box = _layer_box(layer, source)
        if brand_union and _box_overlap(box, brand_union) / max(1, _box_area(box)) > 0.45:
            continue
        cx, cy = _box_center(box)
        if cy < visual_cy and cy < source.height * 0.42:
            parts["primary"].append(box)
        else:
            parts["secondary"].append(box)
    for layer in analysis.logo_layers:
        box = _layer_box(layer, source)
        parts["brand"].append(box)
    if brand_union:
        parts["brand"] = [brand_union]
    for layer in analysis.other_layers:
        box = _layer_box(layer, source)
        cx, cy = _box_center(box)
        top_badge_like = cy < source.height * 0.22 and (box[2] - box[0]) < source.width * 0.32 and (box[3] - box[1]) < source.height * 0.25
        if top_badge_like:
            parts["trust_badge"].append(box)
    return parts


def _load_cta_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _localized_display_cta_label(analysis: VisualAnalysis) -> str:
    cta_layers = [layer for layer in analysis.marketing_text_layers if classify_text_role(layer.text) == "cta"]
    if cta_layers:
        return cta_layers[0].text.strip()
    return "İncele"


def _draw_display_cta_button(
    output: Image.Image,
    zone: tuple[int, int, int, int],
    *,
    label: str,
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
    
    try:
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
    except Exception:
        text_w = len(label) * font_size * 0.58
        text_h = font_size
        
    while (text_w > button_w * 0.78 or text_h > button_h * 0.72) and font_size > 6:
        font_size -= 1
        font = _load_cta_font(font_size)
        try:
            text_box = draw.textbbox((0, 0), label, font=font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
        except Exception:
            text_w = len(label) * font_size * 0.58
            text_h = font_size

    text_x = left + (button_w - text_w) // 2
    text_y = top + (button_h - text_h) // 2 - (text_box[1] if 'text_box' in locals() else 0)
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
        
    safe_gap = max(20, int(round(height * 0.045)))
    min_cta_h = max(18, min(zone_h, int(round(height * 0.16))))

    max_text_bottom = max(box[3] for box in blocking_boxes)
    # YENİ KURAL: CTA Collision Engeli (Y-Margin ZORUNLULUĞU)
    # Metinlerin ve ürünlerin bittiği Y koordinatının en az güvenli margin kadar altında olmalı.
    below_top = max(zone[1], max_text_bottom + safe_gap)
    below_bottom = min(height - margin_y, below_top + zone_h)
    
    if below_bottom - below_top >= min_cta_h:
        return _clip_box((zone[0], below_top, zone[2], below_bottom), width, height)

    # Eğer sığmıyorsa, buton en alt sınırda kalır, metinleri yukarı iteriz. 
    # Bu fonksiyon sadece bölge (zone) döndüğü için butonu çizmez veya yukarıyı itme garantisini burda veremez.
    # Bu yüzden collision varsa None döndürerek çizimini engeller veya küçülterek aşağı sığdırır.
    if height - margin_y - max_text_bottom >= 24:
        return _clip_box((zone[0], max_text_bottom + 4, zone[2], height - margin_y), width, height)
        
    return None


def _draw_programmatic_rtb_guides(
    output: Image.Image,
    product_paste_box: tuple[int, int, int, int] | None,
    rtb_paste_box: tuple[int, int, int, int] | None,
) -> dict[str, Any] | None:
    if not product_paste_box or not rtb_paste_box:
        return None
    px1, py1, px2, py2 = product_paste_box
    rx1, ry1, rx2, ry2 = rtb_paste_box
    rtb_cy = (ry1 + ry2) // 2
    if px2 < rx1:
        line_start = (px2 + 6, rtb_cy)
        line_end = (rx1 - 6, rtb_cy)
    elif rx2 < px1:
        line_start = (rx2 + 6, rtb_cy)
        line_end = (px1 - 6, rtb_cy)
    else:
        return None
    if line_end[0] - line_start[0] < 12:
        return None
    draw = ImageDraw.Draw(output)
    draw.line([line_start, line_end], fill=(2, 59, 110, 180), width=1)
    return {
        "layerId": "programmatic-rtb-guide",
        "role": "deterministic_guide_line",
        "sourceBox": [],
        "targetBox": [line_start[0], line_start[1], line_end[0], line_end[1]],
        "pasteBox": [line_start[0], line_start[1], line_end[0], line_end[1]],
        "scale": None,
        "fitMode": "programmatic_line",
    }


def _expand_text_box_line_region(source: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return _clip_box(
        (
            box[0] - max(4, int((box[2] - box[0]) * 0.05)),
            box[1] - max(4, int((box[3] - box[1]) * 0.15)),
            box[2] + max(4, int((box[2] - box[0]) * 0.05)),
            box[3] + max(4, int((box[3] - box[1]) * 0.15)),
        ),
        source.width,
        source.height,
    )


def _trim_visual_box_away_from_text_edges(
    box: tuple[int, int, int, int],
    text_boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    for tbox in text_boxes:
        if _box_overlap(box, tbox) <= 0:
            continue
        tw = tbox[2] - tbox[0]
        th = tbox[3] - tbox[1]
        cx = (box[0] + box[2]) / 2
        tcx = (tbox[0] + tbox[2]) / 2
        cy = (box[1] + box[3]) / 2
        tcy = (tbox[1] + tbox[3]) / 2
        if tcx < cx and tbox[2] <= left + max(10, int((right - left) * 0.22)):
            left = max(left, tbox[2] + 2)
        elif tcx > cx and tbox[0] >= right - max(10, int((right - left) * 0.22)):
            right = min(right, tbox[0] - 2)
        if tcy < cy and tbox[3] <= top + max(10, int((bottom - top) * 0.22)):
            top = max(top, tbox[3] + 2)
        elif tcy > cy and tbox[1] >= bottom - max(10, int((bottom - top) * 0.22)):
            bottom = min(bottom, tbox[1] - 2)
    return _clip_box((left, top, right, bottom), width, height)


def render_deterministic_compositor(
    source: Image.Image,
    width: int,
    height: int,
    plan: ReframePlan,
    analysis: VisualAnalysis,
    *,
    text_blocks: list[Any] | None = None,
    outpaint_renderer: ImageRenderer | None,
    fallback_renderer: FallbackRenderer,
    draw_text: TextRenderer | None = None,
    product_completion_renderer: Any | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    # ADAPTİFAİ KURAL 3: KATMAN DİZİLİMİ Z-INDEX SIRASI (Background -> Product -> Slogan -> RTB/Badges -> Lines)
    
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    is_social_square = abs(target_ratio - 1.0) < 0.1
    display_placement = _placement_is_display_ad(plan.expansion.placement_id)
    preserve_brand_layers = display_placement
    
    margin_x = max(12, int(width * 0.055))
    margin_y = max(12, int(height * 0.055))

    ai_layout_plan = getattr(plan, "ai_layout_plan", None)
    layout_plan = ai_layout_plan if isinstance(ai_layout_plan, dict) and ai_layout_plan.get("source") else _fallback_creative_director_plan(width, height, display_placement=display_placement)
    
    # Extract zones directly from the plan
    brand_bounds = tuple(layout_plan.get("boxes", {}).get("brand", (margin_x, margin_y, width - margin_x, int(height * 0.16))))
    copy_bounds = tuple(layout_plan.get("boxes", {}).get("copy", (margin_x, int(height * 0.16), int(width * 0.50), int(height * 0.88))))
    secondary_bounds = tuple(layout_plan.get("boxes", {}).get("rtb", (int(width * 0.58), int(height * 0.52), width - margin_x, int(height * 0.90))))
    visual_bounds = tuple(layout_plan.get("boxes", {}).get("visual", (int(width * 0.47), int(height * 0.10), width - margin_x, int(height * 0.93))))
    badge_bounds = tuple(layout_plan.get("boxes", {}).get("badge", (int(width * 0.68), margin_y, width - margin_x, int(height * 0.18))))
    cta_bounds = tuple(layout_plan.get("boxes", {}).get("cta", (margin_x, int(height * 0.76), int(width * 0.52), int(height * 0.92))))

    # KURAL 1 & 2: Background Color & Texture (Katman 1 ve 2)
    visual_box, visual_meta = _detect_role_aware_visual_box(source, analysis)
    parts = _partition_resize_layers(source, analysis, visual_box)
    
    floating_text_boxes: list[tuple[int, int, int, int]] = []
    for layer in analysis.marketing_text_layers:
        box = _expand_text_box_line_region(source, _layer_box(layer, source))
        floating_text_boxes.append(_pad_box(box, source.width, source.height, pad_x=8, pad_y=6))
        
    visual_source_clean = _inpaint_rectangular_overlays(source, floating_text_boxes)
    output = build_resize_texture_background_canvas(visual_source_clean, width, height).convert("RGBA")
    
    composited: list[dict[str, Any]] = []

    # KURAL 3: Ürün (Katman 3) - Kesik ürünü alta yaslamadan, mantıklı yerine yerleştir.
    visual_elements = _collect_visual_element_boxes(source, analysis, visual_box)
    elements_union = _union_boxes(visual_elements) or visual_box
    union_w = max(1, elements_union[2] - elements_union[0])
    union_h = max(1, elements_union[3] - elements_union[1])
    target_w = max(1, visual_bounds[2] - visual_bounds[0])
    target_h = max(1, visual_bounds[3] - visual_bounds[1])
    element_scale = min(target_w / union_w, target_h / union_h)
    
    scaled_union_w = int(round(union_w * element_scale))
    scaled_union_h = int(round(union_h * element_scale))
    origin_x = visual_bounds[0] + (target_w - scaled_union_w) // 2
    origin_y = visual_bounds[1] + (target_h - scaled_union_h) // 2
    
    # Sadece Foreground product piksellerini izole et
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
                visual_source_clean,
                element_box,
                paste_box,
                layer_id=f"role-aware-visual-element-{index + 1}",
                role="product_visual_element",
                feather=max(2, min(8, width // 150)),
            )
        )

    # KURAL 4 & 5: Tipografi (Katman 4 ve 5) - Adaptif Satır Korumalı Slogan ve RTB
    redraw_blocks: list[Any] = []
    drawn_redraw_zone_boxes: list[tuple[int, int, int, int]] = []
    
    source_text_blocks = list(text_blocks or getattr(plan, "extracted_blocks", []) or [])

    primary_union = _union_boxes(parts["primary"])
    if primary_union:
        primary_source_blocks = [
            block for block in source_text_blocks
            if _box_overlap(getattr(block, "bbox", (0, 0, 0, 0)), primary_union) / max(1, _box_area(getattr(block, "bbox", (1, 1, 1, 1)))) >= 0.35
        ]
        if primary_source_blocks:
            redrawn = _build_display_copy_stack_blocks(
                primary_source_blocks,
                copy_bounds,
                stack_role="primary",
                social=is_social_square,
            )
            redraw_blocks.extend(redrawn)
            for rb in redrawn:
                drawn_redraw_zone_boxes.append(tuple(int(v) for v in getattr(rb, "bbox", (0, 0, 0, 0))))

    secondary_union_for_ids = _union_boxes(parts["secondary"])
    if secondary_union_for_ids:
        secondary_source_blocks = [
            block for block in source_text_blocks
            if _box_overlap(getattr(block, "bbox", (0, 0, 0, 0)), secondary_union_for_ids) / max(1, _box_area(getattr(block, "bbox", (1, 1, 1, 1)))) >= 0.35
        ]
        if secondary_source_blocks:
            redrawn = _build_display_copy_stack_blocks(
                secondary_source_blocks,
                secondary_bounds,
                stack_role="secondary",
                social=is_social_square,
            )
            redraw_blocks.extend(redrawn)
            for rb in redrawn:
                drawn_redraw_zone_boxes.append(tuple(int(v) for v in getattr(rb, "bbox", (0, 0, 0, 0))))

    if redraw_blocks and draw_text:
        # Drawing is done via the injected draw_text (which links to draw_resize_display_copy_stack in main.py)
        output = draw_text(output.convert("RGB"), redraw_blocks).convert("RGBA")
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

    brand_union = _union_boxes(parts["brand"])
    if preserve_brand_layers and brand_union:
        brand_pad = _pad_box(brand_union, source.width, source.height, pad_x=max(4, source.width // 80), pad_y=max(3, source.height // 60))
        composited.append(
            _paste_crop_fit(
                output, source, brand_pad, brand_bounds, layer_id="role-aware-brand", role="brand_logo_or_badge", mode="contain",
            )
        )
        
    badge_union = _union_boxes(parts.get("trust_badge", []))
    if preserve_brand_layers and badge_union:
        badge_pad = _pad_box(badge_union, source.width, source.height, pad_x=4, pad_y=4)
        composited.append(
            _paste_crop_fit(
                output, source, badge_pad, badge_bounds, layer_id="role-aware-trust-badge", role="brand_logo_or_badge", mode="contain",
            )
        )

    # KURAL 6: Diğer Ögeler (Programmatic RTB Lines & CTA)
    primary_visual_paste_box = next(
        (tuple(int(value) for value in layer["pasteBox"]) for layer in reversed(composited) if str(layer.get("role", "")).startswith("product_visual")),
        None
    )
    
    guide = _draw_programmatic_rtb_guides(output, primary_visual_paste_box, _union_boxes(drawn_redraw_zone_boxes))
    if guide:
        composited.append(guide)

    if preserve_brand_layers:
        safe_cta_bounds = _resolve_display_cta_zone(
            cta_bounds,
            drawn_text_boxes=[
                *drawn_redraw_zone_boxes,
                *([primary_visual_paste_box] if primary_visual_paste_box else []),
            ],
            width=width,
            height=height,
            margin_x=margin_x,
            margin_y=margin_y,
        )
        display_cta_layer = _draw_display_cta_button(output, safe_cta_bounds, label=_localized_display_cta_label(analysis)) if safe_cta_bounds else None
        if display_cta_layer:
            composited.append(display_cta_layer)

    return output.convert("RGB"), {
        "compositedLayers": composited,
        "compositedLayerCount": len(composited),
        "textRedrawBlocks": len(redraw_blocks),
        "creativeDirectorRelayout": True,
        "layoutPlanBoxes": {
            "brand": list(brand_bounds),
            "copy": list(copy_bounds),
            "rtb": list(secondary_bounds),
            "visual": list(visual_bounds),
            "badge": list(badge_bounds),
            "cta": list(cta_bounds),
        },
    }

def _fallback_creative_director_plan(width: int, height: int, *, display_placement: bool) -> dict[str, Any]:
    margin_x = max(16, int(width * 0.055))
    margin_y = max(16, int(height * 0.055))
    target_ratio = width / max(1, height)
    if target_ratio >= 1.35:
        return {
            "boxes": {
                "brand": (margin_x, margin_y, width - margin_x, int(height * 0.17)),
                "badge": (int(width * 0.68), margin_y, width - margin_x, int(height * 0.19)),
                "copy": (margin_x, int(height * 0.18), int(width * 0.48), int(height * 0.86)),
                "cta": (margin_x, int(height * 0.76), int(width * 0.52), int(height * 0.92)),
                "visual": (int(width * 0.50), int(height * 0.10), width - margin_x, int(height * 0.93)),
                "rtb": (int(width * 0.58), int(height * 0.52), width - margin_x, int(height * 0.90)),
            }
        }
    if target_ratio <= 1.25:
        return {
            "boxes": {
                "brand": (margin_x, margin_y, width - margin_x, int(height * 0.16)),
                "badge": (int(width * 0.64), margin_y, width - margin_x, int(height * 0.18)),
                "copy": (margin_x, int(height * 0.16), width - margin_x, int(height * 0.38)),
                "cta": (margin_x, int(height * 0.40), int(width * 0.58), int(height * 0.53)),
                "visual": (int(width * 0.16), int(height * 0.38), width - int(width * 0.16), height - margin_y),
                "rtb": (margin_x, int(height * 0.70), width - margin_x, height - margin_y),
            }
        }
    return {
        "boxes": {
            "brand": (margin_x, margin_y, width - margin_x, int(height * 0.16)),
            "badge": (int(width * 0.68), margin_y, width - margin_x, int(height * 0.18)),
            "copy": (margin_x, int(height * 0.16), int(width * 0.50), int(height * 0.88)),
            "cta": (margin_x, int(height * 0.76), int(width * 0.50), int(height * 0.92)),
            "visual": (int(width * 0.47), int(height * 0.12), width - margin_x, height - margin_y),
            "rtb": (int(width * 0.62), int(height * 0.52), width - margin_x, height - margin_y),
        }
    }
