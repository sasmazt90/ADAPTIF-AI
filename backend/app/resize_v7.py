"""AdaptifAI resize pipeline v7: decompose once, re-lay out per placement.

A flat ad creative is decomposed into design layers, then every placement is
composed from those layers with aspect-family layout rules - the way a
designer adapts a master layout, instead of cropping/pasting rectangles.

Layers
------
- scene: the background with copy, logo and buttons removed. Badges stay
  on it exactly where the designer placed them relative to the hero, so the
  hero and the offer scale and move together.
- panel: a flat colour band the copy sits on (e.g. the orange block of a
  split layout), detected on the clean background
- scene: the rest of the background, with the hero visual box
- logo / badge sprites: pixel-exact RGBA mattes (never re-typed)
- buttons: measured colour + corner radius, redrawn sharp at any size
- copy: re-set with the measured font size/weight/width/colour so it stays
  sharp and can be re-flowed (e.g. a two-line headline on one line for
  728x90)

Nothing that is not in the source creative is ever added.
"""

from __future__ import annotations

import contextvars
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app import localize_v7 as v7

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Sprite:
    image: Image.Image  # RGBA
    box: Box  # position in the source creative


@dataclass
class Button:
    box: Box
    color: tuple[int, int, int]
    radius: float  # relative to button height


@dataclass
class TextLayer:
    element: v7.CopyElement
    calibrations: list[dict[str, float]]
    color: tuple[int, int, int]
    priority: int
    button: Button | None = None
    short_lines: list[str] = field(default_factory=list)  # shorter wording for tiny formats

    @property
    def role(self) -> str:
        return self.element.role

    @property
    def box(self) -> Box:
        return self.button.box if self.button else self.element.box


@dataclass
class Panel:
    side: str  # top | bottom | left | right | full
    box: Box
    color: tuple[int, int, int]


@dataclass
class CreativeModel:
    size: tuple[int, int]
    clean: Image.Image  # copy, logo and badges removed
    scene_clean: Image.Image  # copy and logo removed; badges stay where the designer put them
    scene_box: Box
    hero_box: Box
    panel: Panel | None
    logo: Sprite | None
    badges: list[Sprite]
    texts: list[TextLayer]
    copy_backdrop: tuple[int, int, int]
    logo_home: str = "scene"  # where the designer put the logo: scene | panel
    attached_badges: list[bool] = field(default_factory=list)  # badge glued onto the hero (sticker on a pack)
    report: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Analysis (vision LLM, same set-of-marks image as localization)
# ---------------------------------------------------------------------------

RESIZE_SYSTEM_PROMPT = """You are a senior ad designer preparing a flat ad creative for multi-format adaptation.
You receive the creative twice: clean, and with numbered magenta boxes from a text detector. Return compact JSON only."""


def _resize_payload(lines: list[v7.OcrLine], size: tuple[int, int]) -> dict[str, Any]:
    return {
        "task": "Decompose this ad creative into design layers so it can be re-laid out for other sizes.",
        "image_size": {"width": size[0], "height": size[1]},
        "detected_boxes": [{"id": line.id, "box_xyxy": list(line.box), "detector_guess": line.text} for line in lines],
        "instructions": [
            "Group the numbered boxes into ELEMENTS (one typographic unit each: logo lockup, headline, subheadline, body, offer badge, price, CTA button label, disclaimer...). Every box belongs to exactly one element.",
            "source_lines: the exact visible text per visual line, top to bottom, read from the pixels.",
            "role: logo | headline | subheadline | body | badge | price | cta | disclaimer | scene_text. The small tagline printed with a logo is part of the logo element.",
            "scene_text = text that is part of the photo or product itself (packaging and labels, signs, menus, screens, licence plates, clothing). It is never moved or removed.",
            "priority: 1 = must appear in every size (logo, main headline, main offer/badge, CTA); 2 = keep when there is room (subheadline, secondary claims); 3 = drop first (disclaimer, legal, fine print).",
            "container_box_xyxy: for a CTA button, badge frame, sticker or label shape that belongs to the element, the pixel box of that whole shape; otherwise null.",
            "hero_box_xyxy: pixel box of the main visual subject (product, vehicle, person, packshot) - the thing that must never be cropped. Exclude copy and logos. null if the creative has no hero visual.",
            "align, font_style (sans, condensed_sans, serif, slab, script, display), font_weight (100-900), uppercase, uniform_size as seen in the creative.",
            "short_lines: ONLY for the main headline - a shorter wording of the same message in the same language and casing (at most ~60% of the characters) for tiny banners like 320x50. Never invent claims. Omit for other elements.",
        ],
        "response_schema": {
            "hero_box_xyxy": [0, 0, 10, 10],
            "elements": [
                {
                    "id": "e1",
                    "box_ids": [1],
                    "role": "headline",
                    "priority": 1,
                    "source_lines": ["..."],
                    "container_box_xyxy": None,
                    "align": "center",
                    "font_style": "sans",
                    "font_weight": 800,
                    "uppercase": True,
                    "uniform_size": True,
                    "short_lines": ["..."],
                }
            ],
        },
    }


def analyze_for_resize(
    image: Image.Image,
    lines: list[v7.OcrLine],
    vision_caller: v7.VisionCaller | None = None,
) -> dict[str, Any]:
    payload = _resize_payload(lines, image.size)
    images = [image, v7.build_set_of_marks(image, lines)]
    errors: list[str] = []
    callers: list[tuple[str, v7.VisionCaller]] = []
    if vision_caller is not None:
        callers.append(("custom", vision_caller))
    else:
        if os.getenv("OPENAI_API_KEY", "").strip():
            callers.append(("openai", v7._openai_vision_json))
        if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip():
            callers.append(("gemini", v7._gemini_vision_json))
    for name, caller in callers:
        for attempt in range(2):
            try:
                parsed = caller(RESIZE_SYSTEM_PROMPT, payload, images)
                if isinstance(parsed.get("elements"), list):
                    parsed["_provider"] = name
                    return parsed
                errors.append(f"{name}: missing elements")
            except Exception as exc:  # pragma: no cover - provider failure
                errors.append(f"{name}#{attempt + 1}: {str(exc)[:200]}")
    raise v7.LocalizeError("Resize layer analysis failed: " + (" | ".join(errors) or "no provider configured"))


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _clip_box(box: Any, size: tuple[int, int]) -> Box | None:
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    except Exception:
        return None
    x0, x1 = sorted((max(0, min(size[0], x0)), max(0, min(size[0], x1))))
    y0, y1 = sorted((max(0, min(size[1], y0)), max(0, min(size[1], y1))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return (x0, y0, x1, y1)


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _expand(box: Box, pad: float, size: tuple[int, int]) -> Box:
    x0, y0, x1, y1 = box
    p = int(round(pad))
    return (max(0, x0 - p), max(0, y0 - p), min(size[0], x1 + p), min(size[1], y1 + p))


def _surface_fill(rgb: np.ndarray, box: Box, ring: int, exclude: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Fit a smooth background surface for `box` from a ring around it."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = max(0, x0 - ring), max(0, y0 - ring), min(w, x1 + ring), min(h, y1 + ring)
    region = rgb[ry0:ry1, rx0:rx1].astype(np.float64)
    gy, gx = np.mgrid[ry0:ry1, rx0:rx1]
    inner = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
    ring_mask = ~inner
    if exclude is not None:
        free = ring_mask & ~exclude[ry0:ry1, rx0:rx1]
        if free.sum() >= 40:
            ring_mask = free
    nx = (gx - rx0) / max(1, rx1 - rx0)
    ny = (gy - ry0) / max(1, ry1 - ry0)
    pixels = region[ring_mask]
    best = None
    for degree in (0, 1, 2):
        if pixels.shape[0] < 40 * (degree + 1):
            break
        coef, _keep, std = v7._robust_surface_fit(pixels, nx[ring_mask], ny[ring_mask], degree)
        if best is None or std < best[2] * 0.85:
            best = (coef, degree, std)
    if best is None:
        median = np.median(region.reshape(-1, 3), axis=0)
        return np.broadcast_to(median, (y1 - y0, x1 - x0, 3)).copy(), 999.0
    coef, degree, std = best
    fill = (v7._poly_design(nx.ravel(), ny.ravel(), degree) @ coef).reshape(region.shape)
    low = np.percentile(pixels, 1, axis=0) - 2
    high = np.percentile(pixels, 99, axis=0) + 2
    fill = np.clip(np.clip(fill, low, high), 0, 255)
    return fill[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0], float(std)


def _matte(rgb: np.ndarray, box: Box, background: np.ndarray, threshold: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    """Soft alpha + un-premultiplied colour of everything that differs from
    the known background inside `box`."""
    x0, y0, x1, y1 = box
    region = rgb[y0:y1, x0:x1].astype(np.float64)
    distance = np.linalg.norm(region - background, axis=2)
    strong = distance[distance > threshold]
    reference = float(np.percentile(strong, 90)) if strong.size else 255.0
    alpha = np.clip((distance - 0.35 * threshold) / max(1.0, reference - 0.35 * threshold), 0.0, 1.0)
    safe = np.maximum(alpha, 1e-3)[..., None]
    colour = np.clip(background + (region - background) / safe, 0, 255)
    colour = np.where(alpha[..., None] > 0.02, colour, region)
    return alpha, colour


def _palette(pixels: np.ndarray, k: int = 3) -> list[np.ndarray]:
    """Dominant colours of a set of pixels (clusters >= 8% of the pixels)."""
    if pixels.shape[0] < 12:
        return [np.median(pixels, axis=0)] if pixels.size else []
    import cv2  # type: ignore

    data = pixels.astype(np.float32)
    k = min(k, max(1, data.shape[0] // 12))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compact, labels, centres = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    return [centres[i].astype(np.float64) for i in range(k) if counts[i] >= 0.08 * data.shape[0]]


def _palette_sprite(
    rgb: np.ndarray,
    box: Box,
    background: np.ndarray,
    palette: list[np.ndarray],
    near: np.ndarray | None = None,
    tolerance: float = 62.0,
) -> tuple[Sprite | None, np.ndarray]:
    """Matte of the pixels whose (un-premultiplied) colour belongs to the
    element's own palette - background texture and overlapping objects
    (e.g. a car roof under a badge frame) stay out."""
    alpha, colour = _matte(rgb, box, background)
    if palette:
        distance = np.min(np.stack([np.linalg.norm(colour - p, axis=2) for p in palette]), axis=0)
        alpha = alpha * (distance < tolerance)
    if near is not None:
        alpha = alpha * near
    # Drop specks.
    try:
        import cv2  # type: ignore

        solid = (alpha > 0.25).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
        keep = np.zeros_like(solid, dtype=bool)
        biggest = stats[1:, cv2.CC_STAT_AREA].max() if count > 1 else 0
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= max(3, 0.002 * biggest):
                keep |= labels == label
        alpha = alpha * v7._dilate(keep, 1)
    except Exception:
        pass
    mask = alpha > 0.04
    if alpha.max() < 0.2:
        return None, mask
    ys, xs = np.nonzero(mask)
    cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    rgba = np.dstack([colour, alpha * 255.0])[cy0:cy1, cx0:cx1]
    image = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
    return Sprite(image=image, box=(box[0] + int(cx0), box[1] + int(cy0), box[0] + int(cx1), box[1] + int(cy1))), mask


def _glyph_pixels(rgb: np.ndarray, element: v7.CopyElement) -> np.ndarray:
    chunks = []
    for line in element.lines:
        if line.glyph_mask is None or line.region is None:
            continue
        x0, y0, x1, y1 = line.region
        chunks.append(rgb[y0:y1, x0:x1][line.glyph_mask].reshape(-1, 3))
    return np.concatenate(chunks).astype(np.float64) if chunks else np.zeros((0, 3))


def _near_glyphs(element: v7.CopyElement, box: Box, radius_ratio: float) -> np.ndarray:
    """Pixels within reach of the element's detected glyphs (logo lockups:
    the slash/mark next to the letters is close to them)."""
    x0, y0, x1, y1 = box
    near = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for line in element.lines:
        if line.glyph_mask is None or line.region is None:
            continue
        rx0, ry0, rx1, ry1 = line.region
        ix0, iy0, ix1, iy1 = max(rx0, x0), max(ry0, y0), min(rx1, x1), min(ry1, y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        near[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] |= line.glyph_mask[iy0 - ry0 : iy1 - ry0, ix0 - rx0 : ix1 - rx0]
    height = max(1, max((line.box[3] - line.box[1]) for line in element.lines) if element.lines else 10)
    return v7._dilate(near, max(2, int(radius_ratio * height)))


def _detect_button(
    rgb: np.ndarray, layer_box: Box, inside_color: np.ndarray, size: tuple[int, int], hint: Box | None = None
) -> Button | None:
    """A filled shape around the label: grow from the label over pixels of
    the label's own background colour."""
    import cv2  # type: ignore

    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = layer_box
    lh = max(4, y1 - y0)
    base = _union([layer_box, hint]) if hint is not None else layer_box
    search = _expand(base, max(lh * 3.0, 0.6 * (x1 - x0)), size)
    sx0, sy0, sx1, sy1 = search
    region = rgb[sy0:sy1, sx0:sx1].astype(np.float64)
    close = np.linalg.norm(region - inside_color, axis=2) < 30
    label = np.zeros_like(close)
    label[y0 - sy0 : y1 - sy0, x0 - sx0 : x1 - sx0] = True
    candidate = (close | label).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=4)
    seed_label = labels[(y0 + y1) // 2 - sy0, max(0, x0 - 2 - sx0)]
    if seed_label == 0:
        seed_label = labels[(y0 + y1) // 2 - sy0, (x0 + x1) // 2 - sx0]
    if seed_label == 0:
        return None
    bx, by, bw, bh, area = stats[seed_label]
    touches = bx <= 0 or by <= 0 or bx + bw >= sx1 - sx0 or by + bh >= sy1 - sy0
    if touches or bw < (x1 - x0) * 1.05 or bh < lh * 1.15 or area < 0.6 * bw * bh:
        return None
    mask = labels == seed_label
    # Corner radius: how far the first fully-filled row is from the top-left.
    top_row = mask[by, bx : bx + bw]
    inset = int(np.argmax(top_row)) if top_row.any() else 0
    radius = min(0.5, (inset * 2.4) / max(1, bh))
    return Button(
        box=(int(sx0 + bx), int(sy0 + by), int(sx0 + bx + bw), int(sy0 + by + bh)),
        color=tuple(int(v) for v in inside_color),  # type: ignore[arg-type]
        radius=float(radius),
    )


def detect_panel(clean: Image.Image) -> Panel | None:
    rgb = np.asarray(clean.convert("RGB")).astype(np.float64)
    h, w = rgb.shape[:2]

    def flat_runs(axis: int) -> list[tuple[int, int, np.ndarray]]:
        data = rgb if axis == 0 else rgb.transpose(1, 0, 2)
        medians = np.median(data, axis=1)
        frac = (np.abs(data - medians[:, None, :]).max(axis=2) < 16).mean(axis=1)
        flat = frac > 0.97
        # Tolerate short interruptions (an element we could not remove).
        gap_limit = int(0.08 * len(flat))
        index = 0
        while index < len(flat):
            if not flat[index]:
                end = index
                while end < len(flat) and not flat[end]:
                    end += 1
                if 0 < index and end < len(flat) and end - index <= gap_limit:
                    before, after = medians[index - 1], medians[end]
                    if np.abs(before - after).max() < 16:
                        flat[index:end] = True
                index = end
            else:
                index += 1
        runs = []
        start = None
        for i, value in enumerate(list(flat) + [False]):
            if value and start is None:
                start = i
            elif not value and start is not None:
                runs.append((start, i, medians[start:i]))
                start = None
        return runs

    candidates: list[Panel] = []
    for axis, length, other in ((0, h, w), (1, w, h)):
        for start, end, medians in flat_runs(axis):
            if end - start < 0.12 * length:
                continue
            centre = np.median(medians, axis=0)
            deviation = np.abs(medians - centre).max(axis=1)
            # Anti-aliased boundary rows belong to neither side: trim them.
            while end - start > 2 and deviation[0] > 22:
                start, medians, deviation = start + 1, medians[1:], deviation[1:]
            while end - start > 2 and deviation[-1] > 22:
                end, medians, deviation = end - 1, medians[:-1], deviation[:-1]
            if end - start < 0.12 * length or deviation.max() > 22:
                continue
            colour = tuple(int(v) for v in np.median(medians, axis=0))
            touches_start, touches_end = start <= 4, end >= length - 4
            if not (touches_start or touches_end):
                continue
            if touches_start and touches_end:
                candidates.append(Panel(side="full", box=(0, 0, w, h), color=colour))  # type: ignore[arg-type]
                continue
            if axis == 0:
                side = "top" if touches_start else "bottom"
                box = (0, start, w, end)
            else:
                side = "left" if touches_start else "right"
                box = (start, 0, end, h)
            candidates.append(Panel(side=side, box=box, color=colour))  # type: ignore[arg-type]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))


def _scene_edge(clean: Image.Image, boundary: int, panel_side: str) -> int:
    """Step away from the anti-aliased seam between scene and panel."""
    rgb = np.asarray(clean.convert("RGB")).astype(np.float64)
    vertical = panel_side in {"top", "bottom"}
    profile = rgb.mean(axis=1) if vertical else rgb.mean(axis=0)
    direction = -1 if panel_side in {"bottom", "right"} else 1
    edge = boundary
    for step in range(1, 8):
        index = boundary + direction * step
        nxt = index + direction
        if not (0 <= nxt < len(profile)):
            break
        edge = index
        if np.abs(profile[index] - profile[nxt]).max() < 6:
            break
    return int(edge) if direction == 1 else int(edge)


SCENE_ROLES = {"scene_text", "product_text", "product", "packaging", "ui", "sign"}


def decompose(
    image_path: Path,
    *,
    vision_caller: v7.VisionCaller | None = None,
    ocr_lines: list[v7.OcrLine] | None = None,
    inpaint: Callable[[Image.Image, Image.Image], Image.Image | None] | None = None,
) -> CreativeModel:
    started = time.time()
    source = Image.open(image_path).convert("RGB")
    size = source.size
    rgb = np.asarray(source)
    if ocr_lines is None:
        ocr_lines, detector = v7.detect_text_lines(image_path, source)
    else:
        detector = "provided"
    analysis = analyze_for_resize(source, ocr_lines, vision_caller) if ocr_lines else {"elements": []}

    raw_elements = [raw for raw in analysis.get("elements") or [] if isinstance(raw, dict)]
    for raw in raw_elements:
        raw["translate"] = True
        raw["target_lines"] = raw.get("source_lines") or []
    elements = v7.build_elements({"elements": raw_elements}, ocr_lines, "EN")
    raw_by_id = {str(raw.get("id")): raw for raw in raw_elements}
    scene_ids = {element.id for element in elements if element.role in SCENE_ROLES}
    working_elements = [element for element in elements if element.id not in scene_ids]
    for element in working_elements:
        element.changed = True
    # Scene text is never erased, but it must not be sampled as background.
    v7._analyze_changed_lines(rgb, working_elements, ocr_lines)

    base = rgb.astype(np.float64).copy()  # copy + logo removed
    textured = np.zeros(rgb.shape[:2], dtype=bool)
    badge_jobs: list[tuple[Box, np.ndarray, np.ndarray, float]] = []
    logo: Sprite | None = None
    badges: list[Sprite] = []
    texts: list[TextLayer] = []
    occupied = v7._occupancy(rgb.shape[:2], [line.box for line in ocr_lines])
    texture_limit = float(os.getenv("ADAPTIFAI_V7_TEXTURE_STD", "14"))

    def erase(target: np.ndarray, box: Box, mask: np.ndarray, fill: np.ndarray, std: float) -> None:
        x0, y0, x1, y1 = box
        grown = v7._dilate(mask, 2)
        if std > texture_limit:
            textured[y0:y1, x0:x1] |= grown
        region = target[y0:y1, x0:x1]
        region[grown] = fill[grown]

    copy_backdrops: list[np.ndarray] = []
    for element in working_elements:
        raw = raw_by_id.get(element.id, {})
        priority = int(v7._clamp(v7._safe_float(raw.get("priority"), 2), 1, 3))
        container = _clip_box(raw.get("container_box_xyxy"), size) if raw.get("container_box_xyxy") else None
        element_box = element.box
        glyphs = _glyph_pixels(rgb, element)
        if element.role in {"logo", "brand"}:
            box = _expand(element_box, max(3, 0.25 * (element_box[3] - element_box[1])), size)
            fill, std = _surface_fill(rgb, box, max(4, round(0.12 * (box[3] - box[1]))), exclude=None)
            sprite, mask = _palette_sprite(rgb, box, fill, _palette(glyphs), near=_near_glyphs(element, box, 0.6))
            if sprite is not None:
                # Logos often sit on photos: anything but a calm backdrop is inpainted.
                erase(base, box, mask, fill, std * 1.75)
                logo = logo or sprite
            continue
        if element.role in {"badge", "price", "sticker"}:
            box = container or element_box
            box = _expand(box, max(3, 0.06 * (box[3] - box[1])), size)
            ring = max(5, round(0.08 * (box[3] - box[1])))
            fill, std = _surface_fill(rgb, box, ring, exclude=None)
            palette = _palette(glyphs)
            # A filled sticker: the colour right around the copy differs from
            # the scene around the badge - it belongs to the badge too.
            inner = element.lines[0].fill if element.lines and element.lines[0].fill is not None else None
            if inner is not None:
                inside = np.median(inner.reshape(-1, 3), axis=0)
                outside = np.median(fill.reshape(-1, 3), axis=0)
                if np.linalg.norm(inside - outside) > 35:
                    palette.append(inside)
            sprite, mask = _palette_sprite(rgb, box, fill, palette)
            if sprite is not None:
                badges.append(sprite)
                badge_jobs.append((box, mask, fill, std))
            continue
        if element.role in {"cta", "button"} and not element.lines:
            continue

        sources = list(element.source_lines)
        if len(sources) != len(element.lines):
            sources = [line.text for line in element.lines]
        calibrations = [v7.calibrate_line(element, line, text) for line, text in zip(element.lines, sources)]
        colour = v7._element_color(element)
        button = None
        if (element.role in {"cta", "button"} or container) and element.lines and element.lines[0].fill is not None:
            fill_arr = element.lines[0].fill
            inside = np.array(fill_arr[fill_arr.shape[0] // 2, 2])
            button = _detect_button(rgb, element_box, inside, size, container)
        for line in element.lines:
            if line.text_mask is None or line.region is None or line.fill is None:
                continue
            line_std = 999.0 if line.background == "textured" else 0.0
            erase(base, line.region, line.text_mask, line.fill, line_std)
            if priority == 1 and button is None:
                copy_backdrops.append(np.median(line.fill.reshape(-1, 3), axis=0))
        if button is not None:
            bbox = _expand(button.box, 2, size)
            fill, std = _surface_fill(rgb, bbox, max(4, round(0.25 * (bbox[3] - bbox[1]))), exclude=occupied)
            erase(base, bbox, np.ones((bbox[3] - bbox[1], bbox[2] - bbox[0]), dtype=bool), fill, std)
        short = [str(x).strip() for x in raw.get("short_lines") or [] if str(x).strip()]
        if element.uppercase:
            short = [x.upper() for x in short]
        texts.append(TextLayer(element=element, calibrations=calibrations, color=colour, priority=priority, button=button, short_lines=short))

    inpaint_provider = "analytic-surface"
    scene_clean_arr = np.clip(base, 0, 255).astype(np.uint8)
    if textured.any():
        repaired = None
        mask_image = Image.fromarray((textured * 255).astype(np.uint8))
        if inpaint is not None:
            try:
                repaired = inpaint(Image.fromarray(scene_clean_arr), mask_image)
                inpaint_provider = "lama"
            except Exception as exc:  # pragma: no cover
                print(f"[resize-v7] inpaint failed: {exc}", flush=True)
        if repaired is None:
            repaired = v7.openai_inpaint(Image.fromarray(scene_clean_arr), mask_image)
            inpaint_provider = "openai-inpaint"
        if repaired is None:
            import cv2  # type: ignore

            bgr = cv2.inpaint(cv2.cvtColor(scene_clean_arr, cv2.COLOR_RGB2BGR), textured.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
            repaired = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inpaint_provider = "opencv"
        repaired_arr = np.asarray(repaired.convert("RGB").resize(size))
        scene_clean_arr = np.where(textured[..., None], repaired_arr, scene_clean_arr)
    scene_clean = Image.fromarray(scene_clean_arr)

    # Banner backgrounds need the scene without badges too.
    clean_arr = scene_clean_arr.astype(np.float64).copy()
    badge_textured = np.zeros(rgb.shape[:2], dtype=bool)
    for box, mask, fill, std in badge_jobs:
        x0, y0, x1, y1 = box
        grown = v7._dilate(mask, 2)
        clean_arr[y0:y1, x0:x1][grown] = fill[grown]
        if std > texture_limit:
            badge_textured[y0:y1, x0:x1] |= grown
    clean = Image.fromarray(np.clip(clean_arr, 0, 255).astype(np.uint8))
    if badge_textured.any():
        import cv2  # type: ignore

        bgr = cv2.inpaint(cv2.cvtColor(np.asarray(clean), cv2.COLOR_RGB2BGR), badge_textured.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        clean = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    panel = detect_panel(clean)
    width, height = size
    if panel is None or panel.side == "full":
        scene_box: Box = (0, 0, width, height)
    elif panel.side == "bottom":
        scene_box = (0, 0, width, _scene_edge(clean, panel.box[1], "bottom"))
    elif panel.side == "top":
        scene_box = (0, _scene_edge(clean, panel.box[3], "top"), width, height)
    elif panel.side == "left":
        scene_box = (_scene_edge(clean, panel.box[2], "left"), 0, width, height)
    else:
        scene_box = (0, 0, _scene_edge(clean, panel.box[0], "right"), height)
    hero = _clip_box(analysis.get("hero_box_xyxy"), size) if analysis.get("hero_box_xyxy") else None
    if hero is not None:
        hero = (max(hero[0], scene_box[0]), max(hero[1], scene_box[1]), min(hero[2], scene_box[2]), min(hero[3], scene_box[3]))
        if hero[2] - hero[0] < 8 or hero[3] - hero[1] < 8:
            hero = None
    hero_box = hero or scene_box
    # Badges outside the scene (e.g. on the copy panel) are not part of the
    # scene group; keep only those that overlap the scene.
    texts.sort(key=lambda layer: (layer.box[1], layer.box[0]))
    backdrop = tuple(int(v) for v in (np.median(np.stack(copy_backdrops), axis=0) if copy_backdrops else np.median(np.asarray(clean).reshape(-1, 3), axis=0)))
    report = {
        "pipeline": "resize-v7-layers",
        "detector": detector,
        "analysis_provider": analysis.get("_provider"),
        "analysis_model": analysis.get("_model"),
        "inpaint_provider": inpaint_provider,
        "panel": None if panel is None else {"side": panel.side, "box": list(panel.box), "color": list(panel.color)},
        "scene_box": list(scene_box),
        "hero_box": list(hero_box),
        "logo": None if logo is None else list(logo.box),
        "badges": [list(sprite.box) for sprite in badges],
        "scene_text": [element.source_lines for element in elements if element.id in scene_ids],
        "texts": [
            {
                "id": layer.element.id,
                "role": layer.role,
                "priority": layer.priority,
                "lines": layer.element.source_lines,
                "button": None if layer.button is None else {"box": list(layer.button.box), "color": list(layer.button.color)},
            }
            for layer in texts
        ],
        "seconds": round(time.time() - started, 2),
    }
    logo_home = "scene"
    if logo is not None and panel is not None and panel.side != "full":
        lb = logo.box
        in_panel = max(0, min(lb[2], panel.box[2]) - max(lb[0], panel.box[0])) * max(0, min(lb[3], panel.box[3]) - max(lb[1], panel.box[1]))
        if in_panel > 0.5 * (lb[2] - lb[0]) * (lb[3] - lb[1]):
            logo_home = "panel"
    attached = []
    for sprite in badges:
        b = sprite.box
        overlap = max(0, min(b[2], hero_box[2]) - max(b[0], hero_box[0])) * max(0, min(b[3], hero_box[3]) - max(b[1], hero_box[1]))
        attached.append(overlap > 0.25 * (b[2] - b[0]) * (b[3] - b[1]))
    report["logo_home"] = logo_home
    report["attached_badges"] = attached
    return CreativeModel(
        logo_home=logo_home,
        attached_badges=attached,
        size=size,
        clean=clean,
        scene_clean=scene_clean,
        scene_box=scene_box,
        hero_box=hero_box,
        panel=panel,
        logo=logo,
        badges=badges,
        texts=texts,
        copy_backdrop=backdrop,  # type: ignore[arg-type]
        report=report,
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _style(layer: TextLayer) -> str:
    return "serif" if layer.element.font_style in {"serif", "slab"} else "sans"


def _line_texts(layer: TextLayer) -> list[str]:
    texts = list(layer.element.source_lines)
    if len(texts) != len(layer.calibrations):
        texts = [" ".join(texts)] if texts else [line.text for line in layer.element.lines]
    return texts


def render_text_block(
    layer: TextLayer,
    scale: float,
    max_width: float,
    *,
    single_line: bool = False,
    align: str | None = None,
) -> Image.Image:
    """Re-set the copy at `scale` x source size, re-flowing when needed."""
    style = _style(layer)
    cals = layer.calibrations or [{"size": 16, "weight": 600, "width": 100, "cap_height": 12, "ink_top": 0, "ink_bottom": 16}]
    texts = _line_texts(layer)
    if len(texts) > len(cals):
        cals = cals + [cals[-1]] * (len(texts) - len(cals))
    lines: list[tuple[str, dict[str, float]]] = list(zip(texts, cals))
    script = v7._script_of(" ".join(texts))
    if single_line and len(lines) > 1:
        main = max(cals, key=lambda c: c["size"])
        lines = [(" ".join(texts), main)]

    def font_for(cal: dict[str, float], s: float) -> ImageFont.FreeTypeFont:
        return v7.load_font(style, int(round(cal["weight"])), int(round(cal["width"])), max(4, round(cal["size"] * s)), script)

    def widths(items: list[tuple[str, dict[str, float]]], s: float) -> list[float]:
        out = []
        for text, cal in items:
            left, _t, right, _b = v7._ink_box(text, font_for(cal, s))
            out.append(right - left)
        return out

    # Re-flow lines that do not fit: wrap words greedily with that line's style.
    if not single_line and max(widths(lines, scale)) > max_width:
        reflowed: list[tuple[str, dict[str, float]]] = []
        for text, cal in lines:
            font = font_for(cal, scale)
            words = text.split()
            current = ""
            for word in words:
                trial = f"{current} {word}".strip()
                left, _t, right, _b = v7._ink_box(trial, font)
                if right - left <= max_width or not current:
                    current = trial
                else:
                    reflowed.append((current, cal))
                    current = word
            if current:
                reflowed.append((current, cal))
        lines = reflowed
    # Still too wide (a single long word): scale down.
    widest = max(widths(lines, scale)) if lines else 1
    if widest > max_width:
        scale *= max_width / widest

    gaps = []
    for i in range(len(cals) - 1):
        gaps.append(max(0.0, cals[i + 1]["ink_top"] - cals[i]["ink_bottom"]) / max(1.0, cals[i]["cap_height"]))
    gap_ratio = float(np.median(gaps)) if gaps else 0.28
    gap_ratio = v7._clamp(gap_ratio, 0.15, 0.9)

    rendered = []
    for text, cal in lines:
        font = font_for(cal, scale)
        left, top, right, bottom = v7._ink_box(text, font)
        rendered.append((text, font, left, top, right, bottom, cal["cap_height"] * scale))
    block_w = max(r[4] - r[2] for r in rendered)
    heights = [r[5] - r[3] for r in rendered]
    gap_px = [gap_ratio * rendered[i][6] for i in range(len(rendered) - 1)]
    block_h = sum(heights) + sum(gap_px)
    pad = 2
    canvas = Image.new("RGBA", (int(math.ceil(block_w)) + 2 * pad, int(math.ceil(block_h)) + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    y = float(pad)
    alignment = align or layer.element.align
    for index, (text, font, left, top, right, bottom, _cap) in enumerate(rendered):
        ink_w = right - left
        if alignment == "left":
            x = pad - left
        elif alignment == "right":
            x = pad + block_w - right
        else:
            x = pad + (block_w - ink_w) / 2 - left
        draw.text((x, y - top), text, font=font, fill=layer.color + (255,), anchor="ls")
        y += (bottom - top) + (gap_px[index] if index < len(gap_px) else 0)
    return canvas


def render_layer(layer: TextLayer, scale: float, max_width: float, *, single_line: bool = False, align: str | None = None) -> Image.Image:
    """Copy block, or a redrawn button around it."""
    if layer.button is None:
        return render_text_block(layer, scale, max_width, single_line=single_line, align=align)
    bx0, by0, bx1, by1 = layer.button.box
    src_w, src_h = bx1 - bx0, by1 - by0
    label_box = layer.element.box
    pad_x = min(max(4.0, ((label_box[0] - bx0) + (bx1 - label_box[2])) / 2), 0.9 * src_h)
    for _ in range(6):
        label = render_text_block(layer, scale, 10**6, single_line=True, align="center")
        natural = max(label.width + 2 * pad_x * scale, min(src_w * scale, max_width))
        if natural <= max_width + 0.5:
            break
        scale *= max_width / natural
    width = min(natural, max_width)
    height = src_h * scale
    size = (max(1, int(round(width))), max(1, int(round(height))))
    ss = 4  # supersampled shape -> anti-aliased edges at small sizes
    shape = Image.new("RGBA", (size[0] * ss, size[1] * ss), (0, 0, 0, 0))
    ImageDraw.Draw(shape).rounded_rectangle(
        (0, 0, shape.width - 1, shape.height - 1), radius=layer.button.radius * shape.height, fill=layer.button.color + (255,)
    )
    canvas = shape.resize(size, Image.Resampling.LANCZOS)
    canvas.alpha_composite(label, (int(round((canvas.width - label.width) / 2)), int(round((canvas.height - label.height) / 2))))
    return canvas


def _scale_sprite(sprite: Sprite, target_h: float, max_w: float | None = None) -> Image.Image:
    image = sprite.image
    scale = target_h / max(1, image.height)
    if max_w is not None and image.width * scale > max_w:
        scale = max_w / max(1, image.width)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def _paste(canvas: Image.Image, layer: Image.Image, x: float, y: float) -> None:
    canvas.alpha_composite(layer, (int(round(x)), int(round(y))))


# ---------------------------------------------------------------------------
# Scene: background continuation + placement of the scene group
# ---------------------------------------------------------------------------


def _border_texture(gray: np.ndarray) -> float:
    """High-frequency energy along the image border (0 = smooth sweep)."""
    import cv2  # type: ignore

    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    b = max(2, int(0.04 * min(gray.shape)))
    strips = [lap[:b], lap[-b:], lap[:, :b], lap[:, -b:]]
    return float(np.mean([s.mean() for s in strips]))


def _harmonic_fill(canvas: np.ndarray, known: np.ndarray, far_color: np.ndarray | None = None, grain: bool = True) -> np.ndarray:
    """Continue a smooth backdrop (studio sweep, gradient, flat colour) into
    the unknown area as a membrane: the fill matches the known border
    exactly (no seam, no tone jump) and stays smooth further out.  Solved on
    a coarse grid, then relaxed near the seam at full resolution."""
    import cv2  # type: ignore

    h, w = known.shape
    factor = max(1.0, max(h, w) / 180.0)
    sh, sw = max(4, int(round(h / factor))), max(4, int(round(w / factor)))
    weights = cv2.resize(known.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    colour = cv2.resize((canvas * known[..., None]).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    small_known = weights > 0.98
    small = np.where(small_known[..., None], colour / np.maximum(weights, 1e-6)[..., None], 0.0)
    border = small[small_known]
    small[~small_known] = border.mean(axis=0) if border.size else 128.0

    def relax(img: np.ndarray, fixed: np.ndarray, iterations: int, outer: np.ndarray | None = None) -> np.ndarray:
        img = img.copy()
        for _ in range(iterations):
            if outer is None:
                padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")  # Neumann at the canvas edge
            else:
                # The creative's shared backdrop colour lies beyond the edge.
                padded = np.empty((img.shape[0] + 2, img.shape[1] + 2, 3))
                padded[:] = outer
                padded[1:-1, 1:-1] = img
            avg = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
            img = np.where(fixed[..., None], img, avg)
        return img

    small = relax(small, small_known, int(2.5 * max(sw, sh)), far_color)
    up = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float64)
    result = np.where(known[..., None], canvas, up)
    # Seam relaxation at full resolution.
    dist = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_L2, 3)
    band = (dist > 0) & (dist < max(6.0, factor * 2.5))
    if band.any():
        ys, xs = np.nonzero(band)
        y0, y1 = max(0, ys.min() - 2), min(h, ys.max() + 3)
        x0, x1 = max(0, xs.min() - 2), min(w, xs.max() + 3)
        sub = result[y0:y1, x0:x1]
        fixed = ~band[y0:y1, x0:x1]
        result[y0:y1, x0:x1] = relax(sub, fixed, int(3 * max(6.0, factor * 2.5)))
    # Re-apply the source's grain so the continuation is not plastic-smooth.
    known_u8 = known.astype(np.uint8)
    edge_band = (cv2.distanceTransform(known_u8, cv2.DIST_L2, 3) < 12) & known
    if grain and edge_band.sum() > 50:
        gray = canvas.mean(axis=2)
        detail = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
        values = detail[edge_band]
        grain = float(np.clip(1.4826 * np.median(np.abs(values - np.median(values))), 0.0, 3.0))
        if grain > 0.3:
            noise = np.random.default_rng(5).normal(0, grain, (h, w))
            noise = cv2.GaussianBlur(noise, (0, 0), 0.6)
            ramp = np.clip(dist / 6.0, 0, 1)
            result = result + (noise * ramp)[..., None] * (~known)[..., None]
    return result


def _provider_outpaint(canvas: Image.Image, known: np.ndarray) -> Image.Image | None:
    """Generative continuation for photographic / textured scenes."""
    if os.getenv("ADAPTIFAI_RESIZE_V7_OUTPAINT", "1") == "0" or not os.getenv("OPENAI_API_KEY", "").strip():
        return None
    try:
        import base64
        import io

        from openai import OpenAI  # type: ignore

        w, h = canvas.size
        ratio = w / h
        target = (1536, 1024) if ratio > 1.2 else (1024, 1536) if ratio < 0.83 else (1024, 1024)
        work = canvas.convert("RGB").resize(target, Image.Resampling.LANCZOS)
        hole = Image.fromarray(((~known) * 255).astype(np.uint8)).resize(target, Image.Resampling.NEAREST)
        edit_mask = Image.new("RGBA", target, (0, 0, 0, 255))
        edit_mask.putalpha(Image.eval(hole, lambda v: 0 if v > 127 else 255))
        image_buffer, mask_buffer = io.BytesIO(), io.BytesIO()
        work.save(image_buffer, "PNG")
        edit_mask.save(mask_buffer, "PNG")
        image_buffer.seek(0)
        mask_buffer.seek(0)
        image_buffer.name, mask_buffer.name = "scene.png", "mask.png"
        client = OpenAI(timeout=float(os.getenv("ADAPTIFAI_RESIZE_V7_OUTPAINT_TIMEOUT", "120")))
        result = client.images.edit(
            model=os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-2"),
            image=image_buffer,
            mask=mask_buffer,
            prompt=(
                "Extend this photograph into the transparent masked areas so it continues seamlessly with the same "
                "lighting, perspective, colours and grain. Do not add any text, letters, logos, people, products or "
                "new objects."
            ),
            size=f"{target[0]}x{target[1]}",
            quality=os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium"),
        )
        generated = Image.open(io.BytesIO(base64.b64decode(result.data[0].b64_json))).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        return _match_generated(np.asarray(canvas.convert("RGB")).astype(np.float64), np.asarray(generated).astype(np.float64), known)
    except Exception as exc:  # pragma: no cover - provider failures
        print(f"[resize-v7] outpaint provider failed, using surface extension: {exc}", flush=True)
        return None


def _match_generated(original: np.ndarray, generated: np.ndarray, known: np.ndarray) -> Image.Image:
    """Generative fills drift in tone.  The provider re-renders the known
    area too, so the drift is measurable there: carry the smooth
    (original - generated) difference into the new area as a membrane, so
    the continuation meets the original exactly at the seam."""
    import cv2  # type: ignore

    diff = np.where(known[..., None], original - generated, 0.0)
    # Only the low-frequency part is a tone drift; details are real content.
    smooth = cv2.GaussianBlur(diff, (0, 0), 6.0)
    weight = cv2.GaussianBlur(known.astype(np.float64), (0, 0), 6.0)
    low = np.where(known[..., None], smooth / np.maximum(weight, 1e-3)[..., None], 0.0)
    inner = known & (cv2.distanceTransform(known.astype(np.uint8), cv2.DIST_L2, 3) > 3)
    correction = _harmonic_fill(np.where(inner[..., None], low, 0.0), inner, grain=False)
    corrected = np.clip(generated + correction, 0, 255)
    # Feather only inside the known area: outside it there is no original.
    alpha = cv2.GaussianBlur(known.astype(np.float64), (0, 0), 2.0)
    alpha = np.where(known, alpha, 0.0)[..., None]
    out = np.where(known[..., None], original * alpha + corrected * (1 - alpha), corrected)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


@dataclass
class ScenePlacement:
    image: Image.Image
    scale: float
    offset: tuple[float, float]  # area position of the scene crop's origin
    meta: dict[str, Any]

    def to_area(self, box: Box, scene_origin: tuple[int, int]) -> tuple[float, float, float, float]:
        sx, sy = scene_origin
        ox, oy = self.offset
        return (
            ox + (box[0] - sx) * self.scale,
            oy + (box[1] - sy) * self.scale,
            ox + (box[2] - sx) * self.scale,
            oy + (box[3] - sy) * self.scale,
        )


def _overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


# Manual editor adjustments for the scene (zoom %, shift % of the area).
SCENE_ADJUST: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar("scene_adjust", default={})


def _touching_edges(model: CreativeModel) -> dict[str, bool]:
    """Sides where the hero is cut by the frame in the source.  Those sides
    must stay cut by the frame: we never invent the missing part of a
    product, a car or a person."""
    W, H = model.size
    hx0, hy0, hx1, hy1 = model.hero_box
    tol_x, tol_y = 0.015 * W, 0.015 * H
    # Only the real image frame counts - a seam with the copy panel does not.
    return {
        "left": hx0 <= tol_x,
        "right": W - hx1 <= tol_x,
        "top": hy0 <= tol_y,
        "bottom": H - hy1 <= tol_y,
    }


def _scene_boxes(model: CreativeModel, with_badges: bool) -> list[Box]:
    boxes = [model.hero_box]
    if with_badges:
        boxes += [badge.box for badge in model.badges if _overlaps(badge.box, model.scene_box)]
    return boxes


def scene_scale_for(model: CreativeModel, safe: tuple[float, float, float, float], with_badges: bool, fill: float = 0.94) -> float:
    ix0, iy0, ix1, iy1 = _union(_scene_boxes(model, with_badges))
    return min((safe[2] - safe[0]) * fill / max(1, ix1 - ix0), (safe[3] - safe[1]) * fill / max(1, iy1 - iy0))


def place_scene(
    model: CreativeModel,
    area_w: int,
    area_h: int,
    safe: tuple[float, float, float, float] | None = None,
    *,
    with_badges: bool = True,
    fill: float = 0.94,
) -> ScenePlacement:
    """Scale and position the scene so the important content is as large as
    the safe area allows.  The area is covered by cropping where possible;
    only what cannot be covered is continued (flat/smooth backdrops
    analytically, photographs generatively)."""
    source = (model.scene_clean if with_badges else model.clean).crop(model.scene_box)
    sx0, sy0 = model.scene_box[0], model.scene_box[1]
    sw, sh = source.size
    ix0, iy0, ix1, iy1 = _union(_scene_boxes(model, with_badges))
    ix0, iy0, ix1, iy1 = ix0 - sx0, iy0 - sy0, ix1 - sx0, iy1 - sy0
    bx0, by0, bx1, by1 = safe or (0, 0, area_w, area_h)
    scale = scene_scale_for(model, (bx0, by0, bx1, by1), with_badges, fill)
    cover = max(area_w / sw, area_h / sh)
    # If the important content almost fills the area anyway, prefer covering
    # (a slightly bigger hero) over inventing background.
    if scale < cover <= scale * 1.12 and (ix1 - ix0) * cover <= area_w and (iy1 - iy0) * cover <= area_h:
        scale = cover
    adjust = SCENE_ADJUST.get()
    zoom = float(adjust.get("zoom", 1.0) or 1.0)
    scale *= zoom
    sw_s, sh_s = sw * scale, sh * scale

    def axis_offset(lo: float, hi: float, i0: float, i1: float, size_scaled: float, area: float) -> float:
        centred = (lo + hi) / 2 - (i0 + i1) / 2 * scale
        keep_a, keep_b = sorted((lo - i0 * scale, hi - i1 * scale))  # important stays in [lo, hi]
        if size_scaled >= area:
            low, high = max(keep_a, area - size_scaled), min(keep_b, 0.0)
            if low <= high:
                return float(min(max(centred, low), high))
            return float(min(max(centred, area - size_scaled), 0.0))
        return float(min(max(centred, keep_a), keep_b))

    ox = axis_offset(bx0, bx1, ix0, ix1, sw_s, area_w)
    oy = axis_offset(by0, by1, iy0, iy1, sh_s, area_h)
    ox += float(adjust.get("dx", 0.0)) / 100.0 * area_w
    oy += float(adjust.get("dy", 0.0)) / 100.0 * area_h
    # A hero cut by the source frame stays cut by the new frame.
    edges = _touching_edges(model)
    if edges["left"]:
        ox = min(ox, 0.0)
    if edges["right"]:
        ox = max(ox, area_w - sw_s)
    if edges["top"]:
        oy = min(oy, 0.0)
    if edges["bottom"]:
        oy = max(oy, area_h - sh_s)
    resample = Image.Resampling.LANCZOS
    scaled = source.resize((max(1, round(sw_s)), max(1, round(sh_s))), resample)
    meta: dict[str, Any] = {"sceneScale": round(scale, 3), "scene": "cover-crop", "badgesInScene": with_badges}
    if scale > 1.3:
        # Upscaled raster: recover some crispness (copy/logo are vector/sprite anyway).
        scaled = scaled.filter(ImageFilter.UnsharpMask(radius=max(1.0, scale * 0.6), percent=60, threshold=2))
        meta["upscaled"] = round(scale, 2)
        if scale > 2.2:
            meta["warning"] = "source resolution is low for this size"
    canvas = np.zeros((area_h, area_w, 3), dtype=np.float64)
    known = np.zeros((area_h, area_w), dtype=bool)
    px, py = int(round(ox)), int(round(oy))
    cx0, cy0 = max(0, px), max(0, py)
    cx1, cy1 = min(area_w, px + scaled.width), min(area_h, py + scaled.height)
    arr = np.asarray(scaled).astype(np.float64)
    canvas[cy0:cy1, cx0:cx1] = arr[cy0 - py : cy1 - py, cx0 - px : cx1 - px]
    known[cy0:cy1, cx0:cx1] = True
    if not known.all():
        missing = 1.0 - known.mean()
        texture = _border_texture(np.asarray(scaled.convert("L"), dtype=np.float64))
        meta.update({"scene": "extend-membrane", "missingRatio": round(missing, 3), "borderTexture": round(texture, 2)})
        result = None
        photographic = texture > float(os.getenv("ADAPTIFAI_RESIZE_V7_TEXTURE_FOR_PROVIDER", "6"))
        large = missing > float(os.getenv("ADAPTIFAI_RESIZE_V7_PROVIDER_MISSING", "0.15"))
        shared = _shared_backdrop(model, scaled) is not None  # flat packshot backdrop: membrane is exact
        if (photographic and missing > 0.03) or (large and not shared):
            result = _provider_outpaint(Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)), known)
            if result is not None:
                meta["scene"] = "extend-provider"
        if result is None:
            result = Image.fromarray(np.clip(_harmonic_fill(canvas, known, _shared_backdrop(model, scaled)), 0, 255).astype(np.uint8))
        image = result
    else:
        image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return ScenePlacement(image=image, scale=scale, offset=(ox, oy), meta=meta)


def _shared_backdrop(model: CreativeModel, scaled: Image.Image) -> np.ndarray | None:
    """Packshot-style creatives: the scene sits on the same flat colour as
    the copy panel.  Continuations must fade into that colour, not into a
    shadow or vignette at the scene edge."""
    if model.panel is None:
        return None
    arr = np.asarray(scaled.convert("RGB")).astype(np.float64)
    b = max(2, int(0.04 * min(arr.shape[:2])))
    edges = {"left": arr[:, :b], "right": arr[:, -b:], "top": arr[:b], "bottom": arr[-b:]}
    seam = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}.get(model.panel.side)
    # The scene edge facing the panel.
    side = {"left": "left", "right": "right", "top": "top", "bottom": "bottom"}.get(model.panel.side)
    strip = edges.get(side or "", None)
    if strip is None:
        return None
    colour = np.array(model.panel.color, dtype=np.float64)
    if np.linalg.norm(np.median(strip.reshape(-1, 3), axis=0) - colour) < 18:
        return colour
    del seam
    return None


def _free_badge_layer(
    model: CreativeModel,
    canvas: Image.Image,
    placement: ScenePlacement,
    area: tuple[float, float, float, float],
    avoid: list[tuple[float, float, float, float]],
    meta: dict[str, Any],
) -> None:
    """Badges that were taken off the scene: keep their direction relative to
    the hero (e.g. upper right of the car) and their size relative to it,
    inside free space."""
    ax0, ay0, ax1, ay1 = area
    origin = (model.scene_box[0], model.scene_box[1])
    hero = placement.to_area(model.hero_box, origin)
    hero = (hero[0] + ax0, hero[1] + ay0, hero[2] + ax0, hero[3] + ay0)
    margin = 0.04 * min(canvas.size)
    shx = (model.hero_box[0] + model.hero_box[2]) / 2
    shy = (model.hero_box[1] + model.hero_box[3]) / 2
    for badge, attached in zip(model.badges, model.attached_badges):
        if attached:
            continue
        bcx, bcy = (badge.box[0] + badge.box[2]) / 2, (badge.box[1] + badge.box[3]) / 2
        dx, dy = bcx - shx, bcy - shy
        best = None
        # 1) Exactly where the designer put it relative to the hero, at the
        #    hero's scale - if that lands inside the area.
        for shrink in (1.0, 0.9, 0.8):
            image = _scale_sprite(badge, badge.image.height * placement.scale * shrink)
            bx0 = hero[0] + (badge.box[0] - model.hero_box[0]) * placement.scale
            by0 = hero[1] + (badge.box[1] - model.hero_box[1]) * placement.scale
            bx0 += (badge.image.width * placement.scale - image.width) / 2
            by0 += (badge.image.height * placement.scale - image.height) / 2
            rect = (bx0, by0, bx0 + image.width, by0 + image.height)
            inside = rect[0] >= ax0 + margin * 0.5 and rect[1] >= ay0 + margin * 0.5 and rect[2] <= ax1 - margin * 0.5 and rect[3] <= ay1 - margin * 0.5
            if inside and not any(_overlap_area(rect, other) > 0 for other in avoid):
                best = (image, rect)
                break
        for shrink in (1.0, 0.9, 0.8, 0.7, 0.6):
            if best is not None:
                break
            image = _scale_sprite(badge, badge.image.height * placement.scale * shrink)
            bw, bh = image.size
            hx0, hy0, hx1, hy1 = hero
            hcx, hcy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
            candidates = []
            # Same direction as in the source, then the other free corners.
            for fx, fy in ((dx, dy), (abs(dx) + 1, -abs(dy) - 1), (-(abs(dx) + 1), -abs(dy) - 1), (abs(dx) + 1, 0), (0, -abs(dy) - 1)):
                norm = math.hypot(fx, fy) or 1.0
                ux, uy = fx / norm, fy / norm
                for reach in np.linspace(0.3, 1.6, 14):
                    cx = hcx + ux * reach * (hx1 - hx0) / 2 + ux * bw / 2
                    cy = hcy + uy * reach * (hy1 - hy0) / 2 + uy * bh / 2
                    x, y = cx - bw / 2, cy - bh / 2
                    x = min(max(x, ax0 + margin), ax1 - margin - bw)
                    y = min(max(y, ay0 + margin), ay1 - margin - bh)
                    rect = (x, y, x + bw, y + bh)
                    hero_overlap = _overlap_area(rect, hero) / max(1.0, bw * bh)
                    clash = any(_overlap_area(rect, other) > 0 for other in avoid)
                    if hero_overlap <= 0.18 and not clash:
                        candidates.append((reach, rect))
                        break
                if candidates:
                    break
            if candidates:
                best = (image, candidates[0][1])
                break
        if best is not None:
            image, rect = best
            _paste(canvas, image, rect[0], rect[1])
            avoid.append(rect)
            meta.setdefault("badges", []).append([int(v) for v in rect])


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _scene_with_badges(
    model: CreativeModel,
    canvas: Image.Image,
    area: tuple[int, int, int, int],
    safe: tuple[float, float, float, float],
    avoid: list[tuple[float, float, float, float]],
    meta: dict[str, Any],
) -> ScenePlacement:
    """Rigid (badges stay on the scene) unless that costs the hero a lot of
    size, or the badge would become unreadable - then the badges are taken
    off the scene and re-placed next to the hero."""
    ax0, ay0, ax1, ay1 = area
    aw, ah = ax1 - ax0, ay1 - ay0
    free_badges = [b for b, a in zip(model.badges, model.attached_badges) if not a and _overlaps(b.box, model.scene_box)]
    rigid_scale = scene_scale_for(model, safe, True)
    free_scale = scene_scale_for(model, safe, False)
    min_badge = max(30.0, 0.13 * min(canvas.size))
    rigid_readable = all(b.image.height * rigid_scale >= min_badge for b in free_badges)
    use_free = bool(free_badges) and (free_scale > rigid_scale * 1.25 or not rigid_readable)
    placement = place_scene(model, aw, ah, safe, with_badges=not use_free)
    canvas.paste(placement.image, (ax0, ay0))
    meta["badgeMode"] = "free" if use_free else "on-scene"
    if use_free:
        _free_badge_layer(model, canvas, placement, (ax0, ay0, ax1, ay1), avoid, meta)
    return placement


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _source_reference(model: CreativeModel) -> float:
    return float(min(model.size))


def _text_layers(model: CreativeModel, max_priority: int) -> list[TextLayer]:
    return [layer for layer in model.texts if layer.priority <= max_priority]


def _stack_height(layers: list[Image.Image], gaps: list[float]) -> float:
    return sum(layer.height for layer in layers) + sum(gaps)


def _layer_cap(layer: TextLayer) -> float:
    return max((c["cap_height"] for c in layer.calibrations), default=10.0)


def _layer_scale(layer: TextLayer, scale: float, min_cap: float) -> float:
    """Buttons never shrink below a readable label."""
    if layer.button is not None:
        return min(max(scale, min_cap / max(1.0, _layer_cap(layer))), scale * 1.5)
    return scale


def _fit_stack(
    model: CreativeModel, layers: list[TextLayer], bw: float, bh: float, min_cap: float = 0.0
) -> tuple[float, list[Image.Image], list[float]] | None:
    widest = max((l.box[2] - l.box[0]) for l in layers)
    low, high = 0.03, min(4.0, bw / max(1, widest) * 1.3)
    best = None
    for _ in range(16):
        mid = (low + high) / 2
        images = [render_layer(layer, _layer_scale(layer, mid, min_cap), bw) for layer in layers]
        gaps = _source_gaps(layers, mid)
        if _stack_height(images, gaps) <= bh and max(im.width for im in images) <= bw + 1:
            best = (mid, images, gaps)
            low = mid
        else:
            high = mid
    return best


def _headline_cap(layers: list[TextLayer], scale: float) -> float:
    headline = next((l for l in layers if l.priority == 1 and l.button is None), None) or next((l for l in layers if l.button is None), layers[0])
    return max((c["cap_height"] for c in headline.calibrations), default=0.0) * scale


def _compose_text_stack(
    model: CreativeModel,
    box: tuple[float, float, float, float],
    *,
    align: str = "center",
    target_cap: float | None = None,
) -> tuple[Image.Image | None, dict[str, Any]]:
    """Fit the copy into `box`. Keep as much copy as possible while the
    headline stays at a readable size; drop fine print first."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    W_ref = min(bw, bh) if target_cap is None else 0
    target = target_cap if target_cap is not None else 0.08 * W_ref
    min_legible = max(7.0, 0.025 * min_side) if (min_side := (target / 0.05 if target else min(bw, bh))) else 6.0
    options = []
    for max_priority in (3, 2, 1):
        layers = _text_layers(model, max_priority)
        if not layers or (options and len(layers) == len(options[-1][1])):
            continue
        fitted = _fit_stack(model, layers, bw, bh, min_legible)
        # Secondary copy that would be unreadable is dropped, not shrunk.
        while fitted is not None:
            tiny = [l for l in layers if l.priority > 1 and _layer_cap(l) * fitted[0] < min_legible]
            if not tiny:
                break
            layers = [l for l in layers if l not in tiny]
            fitted = _fit_stack(model, layers, bw, bh, min_legible) if layers else None
        if fitted is None:
            continue
        if options and [l.element.id for l in layers] == [l.element.id for l in options[-1][1]]:
            continue
        options.append((max_priority, layers, fitted, _headline_cap(layers, fitted[0])))
    if not options:
        return None, {"textScale": 0, "keptPriority": 0}
    chosen = next((option for option in options if option[3] >= target), max(options, key=lambda o: o[3]))
    max_priority, layers, (scale, images, gaps), cap = chosen
    total = _stack_height(images, gaps)
    stack = Image.new("RGBA", (max(1, int(bw)), int(math.ceil(total)) + 1), (0, 0, 0, 0))
    y = 0.0
    for index, image in enumerate(images):
        x = 0 if align == "left" else (bw - image.width) if align == "right" else (bw - image.width) / 2
        _paste(stack, image, x, y)
        y += image.height + (gaps[index] if index < len(gaps) else 0)
    return stack, {
        "textScale": round(scale, 3),
        "keptPriority": max_priority,
        "headlineCap": round(cap, 1),
        "elements": [l.element.id for l in layers],
    }


def _source_gaps(layers: list[TextLayer], scale: float) -> list[float]:
    gaps = []
    for a, b in zip(layers, layers[1:]):
        gap = max(0.0, b.box[1] - a.box[3])
        ref = max(8.0, (a.box[3] - a.box[1]) * 0.35)
        gaps.append(v7._clamp(gap, ref * 0.6, ref * 1.8) * scale)
    return gaps



def _target_cap(model: CreativeModel, W: int, H: int) -> float:
    """Headline cap height to aim for: the source's proportion, a little
    smaller (the hero gets the room), never below a legible floor."""
    headline = next((l for l in model.texts if l.priority == 1 and l.button is None), None)
    if headline is None or not headline.calibrations:
        return 0.04 * min(W, H)
    ratio = max(c["cap_height"] for c in headline.calibrations) / _source_reference(model)
    return max(9.0, 0.66 * ratio * min(W, H))


def _logo_image(model: CreativeModel, W: int, H: int, max_w: float) -> Image.Image | None:
    """Logo keeps its share of the canvas (by the shorter side), within
    sensible bounds; sprites are never upscaled beyond 2.2x."""
    if model.logo is None:
        return None
    m = min(W, H)
    share = model.logo.image.height / _source_reference(model)
    target = v7._clamp(share * m, 0.04 * m, 0.13 * m)
    target = min(target, model.logo.image.height * 2.2)
    return _scale_sprite(model.logo, target, max_w=max_w)


def _logo_corner(model: CreativeModel) -> tuple[str, str]:
    if model.logo is None:
        return ("left", "top")
    cx = (model.logo.box[0] + model.logo.box[2]) / 2 / model.size[0]
    cy = (model.logo.box[1] + model.logo.box[3]) / 2 / model.size[1]
    return ("left" if cx < 0.4 else "right" if cx > 0.6 else "center", "top" if cy < 0.5 else "bottom")


def _logo_rect(model: CreativeModel, canvas_size: tuple[int, int], area: tuple[float, float, float, float]) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    W, H = canvas_size
    ax0, ay0, ax1, ay1 = area
    logo = _logo_image(model, W, H, (ax1 - ax0) * 0.42)
    if logo is None:
        return None
    margin = 0.05 * min(W, H)
    horiz, vert = _logo_corner(model)
    x = ax0 + margin if horiz == "left" else (ax1 - margin - logo.width if horiz == "right" else ax0 + (ax1 - ax0 - logo.width) / 2)
    y = ay0 + margin if vert == "top" else ay1 - margin - logo.height
    return logo, (x, y, x + logo.width, y + logo.height)


def compose(model: CreativeModel, W: int, H: int) -> tuple[Image.Image, dict[str, Any]]:
    ratio = W / H
    source_ratio = model.size[0] / model.size[1]
    meta: dict[str, Any] = {"width": W, "height": H, "ratio": round(ratio, 3), "scaleFactor": round(min(W / model.size[0], H / model.size[1]), 3)}
    if abs(math.log(ratio / source_ratio)) < 0.06:
        family = "same-aspect"
    elif ratio >= 3.0:
        family = "banner"
    elif model.panel is not None and model.panel.side != "full":
        family = "stack" if ratio < 1.3 else "side"
    else:
        family = "overlay"
    meta["family"] = family
    if family == "same-aspect":
        canvas = _compose_same_aspect(model, W, H, meta)
    elif family == "banner":
        canvas = _compose_banner(model, W, H, meta)
    elif family == "stack":
        canvas = _compose_stack(model, W, H, meta)
    elif family == "side":
        canvas = _compose_side(model, W, H, meta)
    else:
        canvas = _compose_overlay(model, W, H, meta)
    return canvas.convert("RGB"), meta


def _compose_same_aspect(model: CreativeModel, W: int, H: int, meta: dict[str, Any]) -> Image.Image:
    """Same proportions: keep the designer's layout exactly; only scale.
    Copy, buttons and logo are re-set/redrawn so they stay sharp when the
    target is larger than the upload."""
    sw, sh = model.size
    k = max(W / sw, H / sh)
    background = model.scene_clean.resize((round(sw * k), round(sh * k)), Image.Resampling.LANCZOS)
    if k > 1.3:
        background = background.filter(ImageFilter.UnsharpMask(radius=max(1.0, k * 0.6), percent=60, threshold=2))
    ox, oy = (background.width - W) / 2, (background.height - H) / 2
    canvas = background.crop((int(ox), int(oy), int(ox) + W, int(oy) + H)).convert("RGBA")
    m = min(W, H)
    min_legible = max(7.0, 0.022 * m)
    if model.logo is not None:
        logo = _scale_sprite(model.logo, model.logo.image.height * k)
        _paste(canvas, logo, model.logo.box[0] * k - ox, model.logo.box[1] * k - oy)
    for layer in model.texts:
        if layer.priority > 1 and _layer_cap(layer) * k < min_legible:
            continue  # unreadable at this size: fine print is dropped, not shrunk to dust
        image = render_layer(layer, _layer_scale(layer, k, min_legible), (layer.box[2] - layer.box[0]) * k * 1.15 + 4)
        bx0, by0, bx1, by1 = layer.box
        cx = (bx0 + bx1) / 2 * k - ox
        if layer.element.align == "left" and layer.button is None:
            x = bx0 * k - ox
        elif layer.element.align == "right" and layer.button is None:
            x = bx1 * k - ox - image.width
        else:
            x = cx - image.width / 2
        y = (by0 + by1) / 2 * k - oy - image.height / 2
        _paste(canvas, image, x, y)
    meta.update({"scale": round(k, 3)})
    if k > 2.2:
        meta["warning"] = "source resolution is low for this size"
    return canvas


def _smallest_panel(model: CreativeModel, W: int, H: int, fractions: list[float], horizontal: bool, reserve: float = 0.0) -> tuple[float, Image.Image | None, dict[str, Any]]:
    """The thinnest copy panel at which the headline is comfortably readable;
    the scene (hero + offer) gets everything else.

    1. Within the source's own panel share: the first size that reaches the
       target headline size (secondary copy that no longer fits is dropped).
    2. Otherwise the panel stays at the source share as long as the
       headline is still clearly readable (>= 55% of the target).
    3. Only then the panel grows.
    """
    target_cap = _target_cap(model, W, H)
    margin = 0.06 * min(W, H)
    panel = model.panel
    source_share = 0.4
    if panel is not None:
        source_share = (panel.box[2] - panel.box[0]) / model.size[0] if horizontal else (panel.box[3] - panel.box[1]) / model.size[1]
    share_cap = max(0.28, min(0.5, source_share + 0.01))

    def attempt(fraction: float) -> tuple[float, Image.Image, dict[str, Any]] | None:
        if horizontal:
            box = (0, 0, W * fraction - 2 * margin, H - 2 * margin - reserve)
        else:
            box = (0, 0, W - 2 * margin, H * fraction - 1.6 * margin - reserve)
        if box[2] <= 10 or box[3] <= 10:
            return None
        stack, smeta = _compose_text_stack(model, box, target_cap=target_cap)
        if stack is None:
            return None
        return fraction, stack, {**smeta, "targetCap": round(target_cap, 1), "panelShareCap": round(share_cap, 3)}

    within = [f for f in fractions if f <= share_cap + 1e-6] or fractions[:1]
    beyond = [f for f in fractions if f > share_cap + 1e-6]
    last = None
    for fraction in within:
        result = attempt(fraction)
        if result is None:
            continue
        last = result
        if result[2]["headlineCap"] >= target_cap:
            return result
    if last is not None and last[2]["headlineCap"] >= max(9.0, 0.55 * target_cap):
        return last
    best = last
    for fraction in beyond:
        result = attempt(fraction)
        if result is None:
            continue
        if best is None or result[2]["headlineCap"] > best[2]["headlineCap"]:
            best = result
        if result[2]["headlineCap"] >= target_cap:
            return result
    if best is None:
        return fractions[-1], None, {}
    return best


def _panel_logo(model: CreativeModel, canvas_size: tuple[int, int]) -> Image.Image | None:
    if model.logo is None or model.logo_home != "panel":
        return None
    W, H = canvas_size
    return _logo_image(model, W, H, W * 0.4)


def _compose_stack(model: CreativeModel, W: int, H: int, meta: dict[str, Any]) -> Image.Image:
    panel = model.panel
    assert panel is not None
    canvas = Image.new("RGBA", (W, H), panel.color + (255,))
    margin = 0.06 * min(W, H)
    panel_on_top = panel.side == "top"
    panel_logo = _panel_logo(model, (W, H))
    reserve = panel_logo.height + margin * 0.6 if panel_logo is not None else 0.0
    fraction, stack, smeta = _smallest_panel(model, W, H, [0.2, 0.23, 0.26, 0.3, 0.34, 0.38, 0.43, 0.5], horizontal=False, reserve=reserve)
    panel_h = int(round(H * fraction))
    scene_h = H - panel_h
    scene_y = panel_h if panel_on_top else 0
    scene_area = (0, scene_y, W, scene_y + scene_h)
    avoid: list[tuple[float, float, float, float]] = []
    logo_rect = _logo_rect(model, (W, H), scene_area) if model.logo_home == "scene" else None
    top_band = margin * 0.5
    if logo_rect is not None:
        avoid.append(logo_rect[1])
        if _logo_corner(model)[1] == "top":
            top_band = min(logo_rect[1][3] - scene_y + margin * 0.4, scene_h * 0.3)
    safe = (margin * 0.5, top_band, W - margin * 0.5, scene_h - margin * 0.5)
    placement = _scene_with_badges(model, canvas, scene_area, safe, avoid, meta)
    py0 = 0 if panel_on_top else scene_h
    content_h = (stack.height if stack is not None else 0) + reserve
    y = py0 + (panel_h - content_h) / 2
    if panel_logo is not None:
        _paste(canvas, panel_logo, (W - panel_logo.width) / 2, y)
        y += reserve
    if stack is not None:
        _paste(canvas, stack, margin, y)
    if logo_rect is not None:
        _paste(canvas, logo_rect[0], logo_rect[1][0], logo_rect[1][1])
    meta.update({"panelHeight": panel_h, "sceneHeight": scene_h, **placement.meta, **smeta})
    return canvas


def _compose_side(model: CreativeModel, W: int, H: int, meta: dict[str, Any]) -> Image.Image:
    panel = model.panel
    assert panel is not None
    canvas = Image.new("RGBA", (W, H), panel.color + (255,))
    margin = 0.06 * min(W, H)
    panel_left = panel.side == "left" or (panel.side in {"top", "bottom"} and _logo_corner(model)[0] == "right")
    panel_logo = _panel_logo(model, (W, H))
    reserve = panel_logo.height + margin * 0.6 if panel_logo is not None else 0.0
    fraction, stack, smeta = _smallest_panel(model, W, H, [0.3, 0.34, 0.38, 0.42, 0.47, 0.52], horizontal=True, reserve=reserve)
    panel_w = int(round(W * fraction))
    scene_w = W - panel_w
    scene_x = panel_w if panel_left else 0
    scene_area = (scene_x, 0, scene_x + scene_w, H)
    avoid: list[tuple[float, float, float, float]] = []
    logo_rect = _logo_rect(model, (W, H), scene_area) if model.logo_home == "scene" else None
    top_band = margin * 0.5
    if logo_rect is not None:
        avoid.append(logo_rect[1])
        if _logo_corner(model)[1] == "top":
            top_band = min(logo_rect[1][3] + margin * 0.4, H * 0.3)
    safe = (margin * 0.5, top_band, scene_w - margin * 0.5, H - margin * 0.5)
    placement = _scene_with_badges(model, canvas, scene_area, safe, avoid, meta)
    px0 = 0 if panel_left else scene_w
    content_h = (stack.height if stack is not None else 0) + reserve
    y = (H - content_h) / 2
    if panel_logo is not None:
        _paste(canvas, panel_logo, px0 + margin, y)
        y += reserve
    if stack is not None:
        _paste(canvas, stack, px0 + margin, y)
    if logo_rect is not None:
        _paste(canvas, logo_rect[0], logo_rect[1][0], logo_rect[1][1])
    meta.update({"panelWidth": panel_w, "sceneWidth": scene_w, **placement.meta, **smeta})
    return canvas


def _compose_overlay(model: CreativeModel, W: int, H: int, meta: dict[str, Any]) -> Image.Image:
    """Full-bleed scene, copy where the designer put it (same vertical zone
    in portrait/square, same side in landscape)."""
    margin = 0.06 * min(W, H)
    sw, sh = model.size
    if model.texts:
        ux0, uy0, ux1, uy1 = _union([l.box for l in model.texts])
    else:
        ux0, uy0, ux1, uy1 = 0, int(sh * 0.7), sw, sh
    landscape = W / H >= 1.3
    if landscape:
        left_side = (ux0 + ux1) / 2 / sw < 0.5
        copy_box = (margin, margin * 2, W * 0.48 - margin, H - margin * 2) if left_side else (W * 0.52 + margin, margin * 2, W - margin, H - margin * 2)
    else:
        top = (uy0 + uy1) / 2 / sh < 0.45
        band = H * v7._clamp((uy1 - uy0) / sh * 1.1, 0.22, 0.5)
        copy_box = (margin, margin, W - margin, margin + band) if top else (margin, H - margin - band, W - margin, H - margin)
    stack, smeta = _compose_text_stack(model, copy_box, target_cap=_target_cap(model, W, H))
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    avoid: list[tuple[float, float, float, float]] = []
    logo_rect = _logo_rect(model, (W, H), (0, 0, W, H))
    top_band = margin * 0.5
    if logo_rect is not None:
        avoid.append(logo_rect[1])
        if _logo_corner(model)[1] == "top":
            top_band = min(logo_rect[1][3] + margin * 0.4, H * 0.25)
    if stack is not None:
        avoid.append(copy_box)
    # The hero may sit behind the copy (as in the source) but must not be
    # pushed out of frame.
    safe = (margin * 0.5, top_band, W - margin * 0.5, H - margin * 0.5)
    placement = _scene_with_badges(model, canvas, (0, 0, W, H), safe, avoid, meta)
    if stack is not None:
        if landscape:
            y = copy_box[1] + (copy_box[3] - copy_box[1] - stack.height) / 2
        else:
            y = copy_box[3] - stack.height if copy_box[1] > H / 2 else copy_box[1]
        _paste(canvas, stack, copy_box[0], y)
    if logo_rect is not None:
        _paste(canvas, logo_rect[0], logo_rect[1][0], logo_rect[1][1])
    meta.update({**placement.meta, **smeta})
    return canvas


def _headline_for_banner(layer: TextLayer, avail_w: float, avail_h: float, min_cap: float) -> tuple[Image.Image | None, bool]:
    """Largest headline that fits; falls back to the model's short wording
    when the full one would be unreadable."""

    def fit(target: TextLayer) -> Image.Image | None:
        best = None
        for single in (True, False):
            low, high = 0.02, 4.0
            for _ in range(16):
                mid = (low + high) / 2
                image = render_text_block(target, mid, avail_w * 4, single_line=single)
                if image.width <= avail_w and image.height <= avail_h:
                    low = mid
                else:
                    high = mid
            if low > 0.021:
                image = render_text_block(target, low, avail_w * 4, single_line=single)
                if best is None or image.height > best[0].height * 1.05:
                    best = (image, low)
        if best is None:
            return None
        cap = _layer_cap(target) * best[1]
        return best[0] if cap >= min_cap else None

    def fit_any(target: TextLayer) -> Image.Image | None:
        nonlocal min_cap
        saved, min_cap = min_cap, 0.0
        try:
            return fit(target)
        finally:
            min_cap = saved

    full = fit(layer)
    if full is not None:
        return full, False
    if not layer.short_lines:
        return fit_any(layer), False
    short = TextLayer(
        element=v7.CopyElement(
            id=layer.element.id, role=layer.role, translate=False, source_lines=list(layer.short_lines),
            target_lines=list(layer.short_lines), ocr_ids=[], align=layer.element.align, uppercase=layer.element.uppercase,
            font_style=layer.element.font_style, font_weight=layer.element.font_weight,
        ),
        calibrations=[max(layer.calibrations, key=lambda c: c["size"])] if layer.calibrations else [],
        color=layer.color,
        priority=layer.priority,
    )
    short.element.lines = layer.element.lines[:1]
    image = fit(short) if layer.short_lines else None
    if image is not None:
        return image, True
    # Nothing reaches the comfort size: use the largest full headline that fits.
    return fit_any(layer), False


def _compose_banner(model: CreativeModel, W: int, H: int, meta: dict[str, Any]) -> Image.Image:
    """Leaderboards and mobile banners - one row, in priority order:
    logo, headline and CTA always; hero thumbnail when there is room for a
    recognisable crop; the offer badge when it stays readable; sub copy and
    fine print never."""
    background = model.panel.color if model.panel is not None else model.copy_backdrop
    canvas = Image.new("RGBA", (W, H), tuple(background) + (255,))
    pad = max(3.0, 0.12 * H)
    x = 0.0
    hero_w = model.hero_box[2] - model.hero_box[0]
    hero_h = model.hero_box[3] - model.hero_box[1]
    show_scene = (W / H >= 4.5 or W >= 600) and H >= 40
    attached_any = any(model.attached_badges)
    scene_w = int(round(v7._clamp(H * hero_w / max(1, hero_h) * 1.15, H * 1.2, W * 0.3))) if show_scene else 0
    logo_on_scene = show_scene and model.logo_home == "scene"
    if show_scene:
        placement = place_scene(model, scene_w, H, (2, 2, scene_w - 2, H - 2), with_badges=attached_any, fill=0.96)
        canvas.paste(placement.image, (0, 0))
        meta.update(placement.meta)
        x = scene_w + pad
    if model.logo is not None:
        if logo_on_scene:
            logo = _scale_sprite(model.logo, H * 0.3, max_w=scene_w * 0.55)
            _paste(canvas, logo, pad * 0.5, pad * 0.5)
        else:
            logo = _scale_sprite(model.logo, H * 0.42, max_w=W * 0.22)
            if _contrast(model.logo, tuple(background)) < 2.0:
                chip = Image.new("RGBA", (int(logo.width + 2 * pad), H), _logo_backdrop(model) + (255,))
                _paste(chip, logo, pad, (H - logo.height) / 2)
                _paste(canvas, chip, x, 0)
                x += chip.width + pad
            else:
                _paste(canvas, logo, x + pad * (0 if x else 1), (H - logo.height) / 2)
                x += logo.width + pad * (1 if x else 2)
    cta_layer = next((l for l in model.texts if l.role in {"cta", "button"} or l.button is not None), None)
    cta_img = None
    if cta_layer is not None:
        scale = (H * 0.56) / max(1, (cta_layer.box[3] - cta_layer.box[1]))
        cta_img = render_layer(cta_layer, scale, W * 0.26)
    right = W - pad - (cta_img.width + pad if cta_img is not None else 0)
    badge_img = None
    free_badges = [b for b, a in zip(model.badges, model.attached_badges) if not a]
    if free_badges and H >= 60:
        badge_img = _scale_sprite(free_badges[0], H * 0.84, max_w=W * 0.16)
        if right - x - badge_img.width - pad < W * 0.3 or badge_img.height < 30:
            badge_img = None
        else:
            right -= badge_img.width + pad
    headline = next((l for l in model.texts if l.priority == 1 and l is not cta_layer), None) or next((l for l in model.texts if l is not cta_layer), None)
    if headline is not None:
        avail_w, avail_h = right - x, H - 2 * pad
        image, used_short = _headline_for_banner(headline, avail_w, avail_h, max(7.0, 0.11 * H))
        if image is not None:
            _paste(canvas, image, x + (avail_w - image.width) / 2, (H - image.height) / 2)
            meta["headlineShortVersion"] = used_short
    if badge_img is not None:
        _paste(canvas, badge_img, right + pad, (H - badge_img.height) / 2)
        meta["bannerBadge"] = True
    if cta_img is not None:
        _paste(canvas, cta_img, W - pad - cta_img.width, (H - cta_img.height) / 2)
    meta["bannerSceneShown"] = show_scene
    return canvas


def _contrast(sprite: Sprite, background: tuple[int, int, int]) -> float:
    arr = np.asarray(sprite.image).astype(np.float64)
    alpha = arr[..., 3] / 255.0
    if alpha.sum() < 1:
        return 21.0
    colour = (arr[..., :3] * alpha[..., None]).sum(axis=(0, 1)) / alpha.sum()

    def lum(c: np.ndarray) -> float:
        c = c / 255.0
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])

    a, b = lum(colour), lum(np.array(background, dtype=np.float64))
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _logo_backdrop(model: CreativeModel) -> tuple[int, int, int]:
    assert model.logo is not None
    x0, y0, x1, y1 = model.logo.box
    region = np.asarray(model.clean)[y0:y1, x0:x1].reshape(-1, 3)
    return tuple(int(v) for v in np.median(region, axis=0))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compose_adjusted(model: CreativeModel, W: int, H: int, zoom: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> tuple[Image.Image, dict[str, Any]]:
    token = SCENE_ADJUST.set({"zoom": zoom, "dx": dx, "dy": dy})
    try:
        image, meta = compose(model, W, H)
    finally:
        SCENE_ADJUST.reset(token)
    meta["manualAdjust"] = {"zoom": zoom, "dx": dx, "dy": dy}
    return image, meta


def resize_creative(
    image_path: Path,
    targets: list[tuple[str, int, int]],
    *,
    vision_caller: v7.VisionCaller | None = None,
    ocr_lines: list[v7.OcrLine] | None = None,
    model: CreativeModel | None = None,
    inpaint: Callable[[Image.Image, Image.Image], Image.Image | None] | None = None,
) -> tuple[dict[str, tuple[Image.Image, dict[str, Any]]], CreativeModel]:
    model = model or decompose(image_path, vision_caller=vision_caller, ocr_lines=ocr_lines, inpaint=inpaint)

    def render(target: tuple[str, int, int]) -> tuple[str, tuple[Image.Image, dict[str, Any]]]:
        placement_id, width, height = target
        started = time.time()
        try:
            image, meta = compose(model, width, height)
            meta.update({"provider": "local", "strategy": f"resize_v7_{meta.get('family')}", "productionReady": True})
        except Exception as exc:  # pragma: no cover - defensive
            import traceback

            traceback.print_exc()
            print(f"[resize-v7] compose failed for {placement_id}: {exc}", flush=True)
            image = model.clean.resize((width, height))
            meta = {"provider": "local", "strategy": "resize_v7_failed", "productionReady": False, "error": str(exc)[:300]}
        meta["seconds"] = round(time.time() - started, 2)
        return placement_id, (image, meta)

    # Placements are independent; generative background calls are I/O bound,
    # so run them side by side to keep large requests inside gateway limits.
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(os.getenv("ADAPTIFAI_RESIZE_V7_WORKERS", "4")))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(pool.map(render, targets))
    return results, model
