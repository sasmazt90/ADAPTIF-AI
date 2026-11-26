"""AdaptifAI localization pipeline v7.

A deliberately small, deterministic pipeline that replaces the heuristic
"v5/v6 polygon" localize path.  The stages are:

1. Geometry   - a text detector (Google Vision when configured, otherwise
                EasyOCR) returns *line boxes*.  Its recognised text is only a
                hint; geometry is the only thing we trust it for.
2. Semantics  - a vision LLM sees the creative twice (clean + numbered boxes,
                "set-of-marks") and groups the boxes into copy elements, reads
                the real text from the pixels, decides what must NOT be touched
                (logos, wordmarks, logo taglines, packaging, product names) and
                writes the target-language copy line by line.
3. Removal    - per text line we fit the surrounding background (robust
                polynomial fit on a ring around the line), segment the glyph
                pixels against it and erase only those.  Flat panels, buttons
                and gradients are rebuilt analytically; textured backgrounds go
                to LaMa (Replicate) with an OpenCV fallback.
4. Rendering  - glyph size, weight and width are *measured* from the source
                pixels and matched with variable fonts (Archivo wdth/wght, Noto
                Serif), then the target copy is laid out on the source
                baselines and fitted into the original line widths.

Elements whose copy does not change (brand names, numbers that stay the same)
are never repainted, so the original pixels survive untouched.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"

LANGUAGE_NAMES = {
    "EN": "English",
    "DE": "German",
    "FR": "French",
    "IT": "Italian",
    "ES": "Spanish",
    "PT": "Portuguese",
    "TR": "Turkish",
    "AR": "Arabic",
    "ZH": "Simplified Chinese",
    "JA": "Japanese",
}

RTL_LANGUAGES = {"AR", "HE", "FA", "UR"}


class LocalizeError(RuntimeError):
    """Raised when the creative cannot be localized reliably."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


Box = tuple[int, int, int, int]


@dataclass
class OcrLine:
    id: int
    text: str
    box: Box
    confidence: float = 1.0


@dataclass
class SourceLine:
    box: Box  # union of detector boxes for this visual line
    text: str = ""
    ink_box: Box | None = None
    text_mask: np.ndarray | None = None  # bool removal mask (glyphs + halo), aligned to `region`
    glyph_mask: np.ndarray | None = None  # bool strict glyph mask, aligned to `region`
    contrast_threshold: float = 0.33
    ink_alpha_sum: float = 0.0
    grain: float = 0.0
    region: Box | None = None  # padded region the mask lives in
    color: tuple[int, int, int] = (17, 17, 17)
    background: str = "flat"  # flat | textured
    fill: np.ndarray | None = None  # RGB float background reconstruction for `region`


@dataclass
class CopyElement:
    id: str
    role: str
    translate: bool
    source_lines: list[str]
    target_lines: list[str]
    ocr_ids: list[int]
    align: str = "center"
    uppercase: bool = False
    font_style: str = "sans"
    font_weight: int = 700
    uniform_size: bool = True
    target_styles: list[int] = field(default_factory=list)  # source line index each target line takes its size from
    lines: list[SourceLine] = field(default_factory=list)
    style: dict[str, Any] = field(default_factory=dict)
    changed: bool = True

    @property
    def box(self) -> Box:
        boxes = [line.box for line in self.lines]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


# ---------------------------------------------------------------------------
# Stage 1: geometry
# ---------------------------------------------------------------------------

_EASYOCR_LOCK = threading.Lock()
_EASYOCR_READER: Any = None


def _easyocr_reader() -> Any:
    global _EASYOCR_READER
    with _EASYOCR_LOCK:
        if _EASYOCR_READER is None:
            import easyocr  # type: ignore

            kwargs: dict[str, Any] = {"gpu": False, "verbose": False}
            model_dir = os.getenv("EASYOCR_MODULE_PATH", "").strip()
            if model_dir:
                kwargs["model_storage_directory"] = model_dir
            try:
                import torch  # type: ignore

                kwargs["gpu"] = bool(torch.cuda.is_available())
            except Exception:
                pass
            _EASYOCR_READER = easyocr.Reader(["en"], **kwargs)
    return _EASYOCR_READER


def _google_vision_lines(image_path: Path) -> list[OcrLine]:
    if not (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    ):
        return []
    try:
        from google.cloud import vision  # type: ignore
        from google.oauth2 import service_account  # type: ignore

        inline_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        credentials = service_account.Credentials.from_service_account_info(json.loads(inline_json)) if inline_json else None
        client = vision.ImageAnnotatorClient(credentials=credentials) if credentials else vision.ImageAnnotatorClient()
        response = client.document_text_detection(image=vision.Image(content=image_path.read_bytes()))
        if response.error.message:
            print(f"[localize-v7] vision error: {response.error.message}", flush=True)
            return []
    except Exception as exc:  # pragma: no cover - network/provider failures
        print(f"[localize-v7] vision unavailable: {exc}", flush=True)
        return []

    words: list[tuple[str, Box]] = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(symbol.text for symbol in word.symbols).strip()
                    xs = [int(v.x or 0) for v in word.bounding_box.vertices]
                    ys = [int(v.y or 0) for v in word.bounding_box.vertices]
                    if text and xs and ys and max(xs) > min(xs) and max(ys) > min(ys):
                        words.append((text, (min(xs), min(ys), max(xs), max(ys))))
    return _merge_words_into_lines(words)


def _merge_words_into_lines(words: list[tuple[str, Box]]) -> list[OcrLine]:
    """Join word boxes that sit on the same baseline and are close together."""
    words = sorted(words, key=lambda item: (item[1][1], item[1][0]))
    lines: list[list[tuple[str, Box]]] = []
    for text, box in words:
        height = box[3] - box[1]
        placed = False
        for line in lines:
            last_text, last_box = line[-1]
            line_h = max(b[3] - b[1] for _, b in line)
            overlap = min(box[3], last_box[3]) - max(box[1], last_box[1])
            gap = box[0] - last_box[2]
            similar_size = 0.6 <= height / max(1, line_h) <= 1.6
            if overlap >= 0.55 * min(height, line_h) and -0.3 * line_h <= gap <= 1.2 * line_h and similar_size:
                line.append((text, box))
                placed = True
                break
        if not placed:
            lines.append([(text, box)])
    result: list[OcrLine] = []
    for index, line in enumerate(lines):
        boxes = [b for _, b in line]
        result.append(
            OcrLine(
                id=index + 1,
                text=" ".join(t for t, _ in line),
                box=(min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)),
            )
        )
    return result


def _easyocr_lines(image: Image.Image) -> list[OcrLine]:
    reader = _easyocr_reader()
    # Upscale small creatives: CRAFT misses small disclaimers below ~12px.
    long_side = max(image.size)
    scale = 1.0
    target_long = int(os.getenv("ADAPTIFAI_V7_OCR_LONG_SIDE", "1600"))
    if long_side < target_long:
        scale = target_long / long_side
    elif long_side > 2400:
        scale = 2400 / long_side
    work = image if scale == 1.0 else image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    detections = reader.readtext(
        np.array(work.convert("RGB")),
        detail=1,
        paragraph=False,
        width_ths=float(os.getenv("ADAPTIFAI_V7_OCR_WIDTH_THS", "0.9")),
        text_threshold=float(os.getenv("ADAPTIFAI_V7_OCR_TEXT_THRESHOLD", "0.6")),
        low_text=float(os.getenv("ADAPTIFAI_V7_OCR_LOW_TEXT", "0.35")),
        decoder="greedy",
    )
    words: list[tuple[str, Box]] = []
    for points, text, confidence in detections:
        xs = [p[0] / scale for p in points]
        ys = [p[1] / scale for p in points]
        box = (
            max(0, int(np.floor(min(xs)))),
            max(0, int(np.floor(min(ys)))),
            min(image.width, int(np.ceil(max(xs)))),
            min(image.height, int(np.ceil(max(ys)))),
        )
        if box[2] - box[0] < 3 or box[3] - box[1] < 3:
            continue
        words.append((str(text or "").strip() or "?", box))
    return _merge_words_into_lines(words)


def detect_text_lines(image_path: Path, image: Image.Image) -> tuple[list[OcrLine], str]:
    lines = _google_vision_lines(image_path)
    if lines:
        return lines, "google-vision"
    return _easyocr_lines(image), "easyocr"


# ---------------------------------------------------------------------------
# Stage 2: semantics (vision LLM, set-of-marks prompting)
# ---------------------------------------------------------------------------


def _image_data_url(image: Image.Image, max_side: int = 1536) -> str:
    work = image.convert("RGB")
    if max(work.size) > max_side:
        ratio = max_side / max(work.size)
        work = work.resize((round(work.width * ratio), round(work.height * ratio)), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    work.save(buffer, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_set_of_marks(image: Image.Image, lines: list[OcrLine]) -> Image.Image:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    label_size = max(12, round(max(image.size) / 55))
    font = load_font("sans", 700, 100, label_size)
    for line in lines:
        x0, y0, x1, y1 = line.box
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(255, 0, 170), width=2)
        tag = str(line.id)
        tw = draw.textlength(tag, font=font)
        ty = y0 - label_size - 4 if y0 - label_size - 4 >= 0 else y1 + 2
        draw.rectangle((x0 - 2, ty, x0 + tw + 6, ty + label_size + 4), fill=(255, 0, 170))
        draw.text((x0 + 2, ty + 1), tag, fill=(255, 255, 255), font=font)
    return marked


SEMANTIC_SYSTEM_PROMPT = """You are a senior advertising localization lead and typographer.
You receive an ad creative twice: (1) the clean creative and (2) the same creative with numbered magenta boxes from a text detector.
The detector's own text guesses are unreliable - always read the real text from the pixels.
Return compact JSON only."""


def _semantic_payload(lines: list[OcrLine], image_size: tuple[int, int], target_language: str) -> dict[str, Any]:
    language_name = LANGUAGE_NAMES.get(target_language.upper(), target_language)
    return {
        "task": f"Localize this ad creative into {language_name}.",
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "detected_boxes": [
            {"id": line.id, "box_xyxy": list(line.box), "detector_guess": line.text}
            for line in lines
        ],
        "instructions": [
            "Group the numbered boxes into copy ELEMENTS. An element is one typographic unit: a headline (all its lines), a subheadline, a CTA button label, a badge, a price, a disclaimer, a logo lockup, etc. Every numbered box must belong to exactly one element.",
            "For each element give source_lines: the exact visible text, one entry per visual line, top to bottom, with correct spelling, case and punctuation (including * and ® marks) as printed.",
            "Set translate=false for: logos and wordmarks, the tagline/descriptor that is part of a logo lockup (e.g. small text printed under/next to a logo), brand and model names, text printed on products or packaging, licence plates, UI chrome of a screenshot, legal entity names, URLs, hashtags and handles. These pixels will be left untouched.",
            "Set translate=true for marketing overlay copy: headlines, subheadlines, body copy, offers, badges, CTAs, disclaimers and footnotes.",
            f"For translate=true elements write target_lines in {language_name}: natural, idiomatic, persuasive ad copy as a native copywriter would write it for this market - not a literal translation. Keep the claim, the offer and every number exactly (localize number/percent formatting to the market, e.g. Turkish writes %15). Keep footnote markers such as * and keep end punctuation style.",
            "Slogans and taglines must be transcreated, not translated: carry the feeling/benefit with a phrase the target market's ads actually use. Word-by-word renderings of emotional verbs (love, enjoy, discover, experience...) are a red flag.",
            "LENGTH: the target must fit the same space. Keep each target line about the same character count as its source line (ideally within +15%). Prefer the same number of lines as the source. Use a shorter synonym before you ever make a line longer.",
            "Before answering, read every element's target_lines aloud in order as a native speaker and fix anything unnatural, literal or ungrammatical.",
            "If the source is ALL CAPS, write target_lines in ALL CAPS using the target language's casing rules (Turkish: i -> İ, ı -> I).",
            "If the copy would be identical in the target language (e.g. a product name), still return it as target_lines; it will simply not be repainted.",
            "align: how the lines of the element are aligned relative to each other and to their container (left, center or right).",
            "font_style: sans, condensed_sans, serif, slab, script or display. font_weight: 100-900 visual weight estimate.",
            "uniform_size: true when all lines of the element share one font size; false when lines differ in size (typical for badges like 'UP TO / 15% / OFF').",
            "target_line_styles: for every target line, the 0-based index of the source line whose size and weight it should take. Target-language word order often differs, so lines may move: put the emphasised information (numbers, the offer) on the emphasised (largest) style. Example: source 'UP TO / 15% / OFF*' (index 1 is the big line) -> Turkish target_lines ['%15\'E VARAN', 'İNDİRİM*'] with target_line_styles [1, 2]. Never use word-for-word structures that a native speaker would not say (Turkish says '%15\'e varan indirim', never 'en fazla %15 indirim').",
            "Never merge or split elements across different visual groups (a badge, a headline and a button are always separate elements).",
            "Badges, stickers and buttons are fixed shapes: keep every line about as long as the source line in the same style; never make a line much longer than the shape.",
            "HARD RULE: target_lines are read top to bottom. Joined with spaces they must form one grammatical, natural sentence/phrase in the target language. Never reorder words to make a style mapping work - change target_line_styles instead (e.g. Turkish ['%15\'E', 'VARAN', 'İNDİRİM*'] with styles [1, 0, 2] is fine; ['VARAN', '%15\'E', 'İNDİRİM*'] is wrong).",
        ],
        "response_schema": {
            "source_language": "ISO 639-1 code of the creative's language, upper case",
            "elements": [
                {
                    "id": "e1",
                    "box_ids": [1, 2],
                    "role": "headline|subheadline|body|cta|badge|price|disclaimer|logo|brand|product|ui|other",
                    "translate": True,
                    "source_lines": ["..."],
                    "target_lines": ["..."],
                    "align": "left|center|right",
                    "font_style": "sans",
                    "font_weight": 800,
                    "uppercase": True,
                    "uniform_size": True,
                    "target_line_styles": [0],
                }
            ],
        },
    }


def _openai_vision_json(
    system_prompt: str,
    payload: dict[str, Any],
    images: list[Image.Image],
    model: str | None = None,
) -> dict[str, Any]:
    from openai import OpenAI  # type: ignore

    model = model or (
        os.getenv("ADAPTIFAI_LOCALIZE_VISION_MODEL", "").strip()
        or os.getenv("OPENAI_TRANSLATION_MODEL", "").strip()
        or "gpt-5.6-terra"
    )
    client = OpenAI(timeout=float(os.getenv("ADAPTIFAI_LOCALIZE_VISION_TIMEOUT", "90")))
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image), "detail": "high"}})
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }
    effort = os.getenv("ADAPTIFAI_LOCALIZE_VISION_REASONING", "medium").strip()
    if model.startswith(("gpt-5", "o")) and effort:
        request["reasoning_effort"] = effort
    elif not model.startswith(("gpt-5", "o")):
        request["temperature"] = 0
    response = client.chat.completions.create(**request)
    raw = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0)) if match else {}
    if not isinstance(parsed, dict):
        raise LocalizeError("Vision model returned a non-object response.")
    parsed["_model"] = model
    return parsed


def _gemini_vision_json(system_prompt: str, payload: dict[str, Any], images: list[Image.Image]) -> dict[str, Any]:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise LocalizeError("No Gemini API key configured.")
    import requests  # type: ignore

    model = os.getenv("ADAPTIFAI_LOCALIZE_GEMINI_MODEL", os.getenv("VERTEX_GEMINI_MODEL", "gemini-3.6-flash"))
    parts: list[dict[str, Any]] = [{"text": system_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)}]
    for image in images:
        data_url = _image_data_url(image)
        parts.append({"inline_data": {"mime_type": "image/png", "data": data_url.split(",", 1)[1]}})
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={"contents": [{"role": "user", "parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}},
        timeout=float(os.getenv("ADAPTIFAI_LOCALIZE_VISION_TIMEOUT", "90")),
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    parsed["_model"] = model
    return parsed


VisionCaller = Callable[[str, dict[str, Any], list[Image.Image]], dict[str, Any]]


def analyze_copy(
    image: Image.Image,
    lines: list[OcrLine],
    target_language: str,
    vision_caller: VisionCaller | None = None,
) -> dict[str, Any]:
    payload = _semantic_payload(lines, image.size, target_language)
    images = [image, build_set_of_marks(image, lines)]
    callers: list[tuple[str, VisionCaller]] = []
    if vision_caller is not None:
        callers.append(("custom", vision_caller))
    else:
        if os.getenv("OPENAI_API_KEY", "").strip():
            callers.append(("openai", _openai_vision_json))
        if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip():
            callers.append(("gemini", _gemini_vision_json))
    if not callers:
        raise LocalizeError("Localization needs OPENAI_API_KEY (or GEMINI_API_KEY) for the vision copy analysis.")
    errors: list[str] = []
    for name, caller in callers:
        for attempt in range(2):
            try:
                started = time.time()
                parsed = caller(SEMANTIC_SYSTEM_PROMPT, payload, images)
                if isinstance(parsed.get("elements"), list):
                    parsed["_provider"] = name
                    parsed["_seconds"] = round(time.time() - started, 2)
                    return parsed
                errors.append(f"{name}: missing elements")
            except Exception as exc:  # pragma: no cover - provider failures
                errors.append(f"{name}#{attempt + 1}: {str(exc)[:300]}")
                print(f"[localize-v7] {name} vision analysis failed: {exc}", flush=True)
    raise LocalizeError("Vision copy analysis failed: " + " | ".join(errors))


REVIEW_SYSTEM_PROMPT = """You are the native-speaker copy chief who signs off localized advertising.
You never accept literal translations, calques or word orders a native speaker would not use. Return compact JSON only."""


def fit_budgets(render_report: list[dict[str, Any]], elements: list[CopyElement]) -> dict[str, list[int | None]]:
    """Per target line: max characters that still fit at ~the source size
    (None when the line already fits)."""
    threshold = float(os.getenv("ADAPTIFAI_V7_REFINE_BELOW", "0.88"))
    roles = {element.id: element.role for element in elements}
    budgets: dict[str, list[int | None]] = {}
    for item in render_report:
        if roles.get(item["id"]) in {"disclaimer", "legal"}:
            continue
        lines = item.get("lines") or []
        ratios = [line["font_size"] / max(1e-6, line["source_font_size"]) for line in lines]
        if ratios and min(ratios) < threshold:
            budgets[item["id"]] = [
                max(3, int(len(line["text"]) * ratio / 0.97)) if ratio < threshold else None
                for line, ratio in zip(lines, ratios)
            ]
    return budgets


def review_copy(
    analysis: dict[str, Any],
    target_language: str,
    image: Image.Image | None = None,
    vision_caller: VisionCaller | None = None,
    budgets: dict[str, list[int | None]] | None = None,
) -> dict[str, Any]:
    """Second pair of eyes on the translated copy (fixes in place)."""
    if os.getenv("ADAPTIFAI_V7_COPY_REVIEW", "1") == "0":
        return {"reviewed": False, "skipped": "disabled"}
    language_name = LANGUAGE_NAMES.get(target_language.upper(), target_language)
    items = [
        {
            "id": raw.get("id"),
            "role": raw.get("role"),
            "source_lines": raw.get("source_lines"),
            "target_lines": raw.get("target_lines"),
            "target_line_styles": raw.get("target_line_styles"),
            **({"max_characters_per_line": (budgets or {}).get(str(raw.get("id")))} if (budgets or {}).get(str(raw.get("id"))) else {}),
        }
        for raw in analysis.get("elements") or []
        if isinstance(raw, dict) and raw.get("translate") and raw.get("target_lines")
    ]
    if not items:
        return {"reviewed": False, "skipped": "nothing-to-review"}
    payload = {
        "task": f"Review this {language_name} localization of an ad creative (source copy and target lines below).",
        "checks": [
            f"Read each element's target_lines top to bottom, joined with spaces. Is it exactly what a professional {language_name} copywriter would write for this ad in this market? Natural word order, idiomatic, correct grammar, suffixes and casing, no calques.",
            "Slogans must be transcreated: a word-by-word rendering of emotional verbs (love, enjoy, discover...) that a local copywriter would never write must be replaced.",
            "Is the meaning, the offer and every number preserved? Are footnote markers (*) and the source's end punctuation pattern (e.g. a period after each headline line) preserved?",
            "Is it about as short as the source (it must fit the same space)? Prefer the shorter of two equally natural options.",
            "Elements with max_characters_per_line do not fit the layout at the original type size: find a natural phrasing within that budget per line (null = no limit). Naturalness always wins: if no natural phrasing fits, keep the natural one and the layout will set it smaller.",
            "Use one consistent form of address across all elements (e.g. Turkish: either all 'sen' or all 'siz' imperatives in headline and CTA).",
            "Logos/brand names are not in this list and must not be added.",
        ],
        "output_rules": [
            "Return only elements you changed. Keep the same id. Return the full corrected target_lines and matching target_line_styles (0-based source line index each target line takes its size from; put the number/offer on the emphasised style).",
            "Keep ALL CAPS if the current lines are ALL CAPS.",
            "If everything is already right, return {\"elements\": []}.",
        ],
        "elements": items,
        "response_schema": {"elements": [{"id": "e2", "target_lines": ["..."], "target_line_styles": [0], "reason": "short reason"}]},
    }
    images = [image] if image is not None else []
    try:
        if vision_caller is not None:
            response = vision_caller(REVIEW_SYSTEM_PROMPT, payload, images)
        elif os.getenv("OPENAI_API_KEY", "").strip():
            model = (
                os.getenv("ADAPTIFAI_LOCALIZE_REVIEW_MODEL", "").strip()
                or os.getenv("ADAPTIFAI_LOCALIZE_VISION_MODEL", "").strip()
                or os.getenv("OPENAI_TRANSLATION_MODEL", "").strip()
                or "gpt-5.6-terra"
            )
            response = _openai_vision_json(REVIEW_SYSTEM_PROMPT, payload, images, model=model)
        else:
            return {"reviewed": False, "skipped": "no-provider"}
    except Exception as exc:  # pragma: no cover
        print(f"[localize-v7] copy review failed: {exc}", flush=True)
        return {"reviewed": False, "error": str(exc)[:300]}
    by_id = {str(raw.get("id")): raw for raw in analysis.get("elements") or [] if isinstance(raw, dict)}
    changes: dict[str, Any] = {}
    for fix in response.get("elements") or []:
        if not isinstance(fix, dict):
            continue
        raw = by_id.get(str(fix.get("id")))
        lines = [str(line).strip() for line in fix.get("target_lines") or [] if str(line).strip()]
        if raw is None or not lines or not raw.get("translate"):
            continue
        changes[str(fix.get("id"))] = {"from": raw.get("target_lines"), "to": lines, "reason": fix.get("reason")}
        raw["target_lines"] = lines
        styles = fix.get("target_line_styles")
        raw["target_line_styles"] = styles if isinstance(styles, list) and len(styles) == len(lines) else None
    return {"reviewed": True, "changes": changes, "model": response.get("_model")}


def _normalize_copy(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"[\s ]+", " ", text).strip()


def locale_upper(text: str, language: str) -> str:
    if language.upper() in {"TR", "AZ"}:
        text = text.replace("i", "İ").replace("ı", "I")
    return text.upper()


def build_elements(analysis: dict[str, Any], lines: list[OcrLine], target_language: str) -> list[CopyElement]:
    by_id = {line.id: line for line in lines}
    used: set[int] = set()
    elements: list[CopyElement] = []
    for index, raw in enumerate(analysis.get("elements") or []):
        if not isinstance(raw, dict):
            continue
        ids = []
        for value in raw.get("box_ids") or raw.get("ocr_ids") or []:
            try:
                box_id = int(value)
            except (TypeError, ValueError):
                continue
            if box_id in by_id and box_id not in used:
                ids.append(box_id)
        if not ids:
            continue
        used.update(ids)
        source_lines = [str(item).strip() for item in raw.get("source_lines") or [] if str(item).strip()]
        target_lines = [str(item).strip() for item in raw.get("target_lines") or [] if str(item).strip()]
        uppercase = bool(raw.get("uppercase"))
        if uppercase:
            target_lines = [locale_upper(item, target_language) for item in target_lines]
        translate = bool(raw.get("translate")) and bool(target_lines)
        element = CopyElement(
            id=str(raw.get("id") or f"e{index + 1}"),
            role=str(raw.get("role") or "other").lower(),
            translate=translate,
            source_lines=source_lines or [by_id[i].text for i in ids],
            target_lines=target_lines,
            ocr_ids=ids,
            align=str(raw.get("align") or "center").lower() if str(raw.get("align") or "").lower() in {"left", "center", "right"} else "center",
            uppercase=uppercase,
            font_style=str(raw.get("font_style") or "sans").lower(),
            font_weight=int(_clamp(_safe_float(raw.get("font_weight"), 700), 100, 900)),
            uniform_size=bool(raw.get("uniform_size", True)),
        )
        element.lines = _visual_lines([by_id[i] for i in ids])
        styles: list[int] = []
        for value in raw.get("target_line_styles") or []:
            try:
                styles.append(int(_clamp(int(value), 0, len(element.lines) - 1)))
            except (TypeError, ValueError):
                styles = []
                break
        element.target_styles = styles if len(styles) == len(element.target_lines) else []
        _guard_numeric_emphasis(element)
        element.changed = translate and _normalize_copy(" ".join(element.source_lines)) != _normalize_copy(" ".join(element.target_lines))
        elements.append(element)
    return elements


def _guard_numeric_emphasis(element: CopyElement) -> None:
    """In mixed-size elements the line carrying the number (the offer) must
    keep the style of the source line that carried it."""
    if element.uniform_size or len(element.lines) < 2 or not element.target_lines:
        return
    has_digit = lambda text: any(ch.isdigit() for ch in text)  # noqa: E731
    source_rows = element.source_lines if len(element.source_lines) == len(element.lines) else [line.text for line in element.lines]
    source_num = next((i for i, text in enumerate(source_rows) if has_digit(text)), None)
    target_num = next((i for i, text in enumerate(element.target_lines) if has_digit(text)), None)
    if source_num is None or target_num is None:
        return
    styles = list(element.target_styles) if element.target_styles else []
    if len(styles) == len(element.target_lines) and styles[target_num] == source_num:
        return
    remaining = [i for i in range(len(element.lines)) if i != source_num]
    styles = []
    for index in range(len(element.target_lines)):
        if index == target_num:
            styles.append(source_num)
            continue
        # Lines before the number take the styles above it, lines after it
        # the styles below it.
        above = [i for i in remaining if i < source_num]
        below = [i for i in remaining if i > source_num]
        pool = (above[::-1] if index < target_num else below) or remaining
        offset = (target_num - index - 1) if index < target_num else (index - target_num - 1)
        styles.append(pool[min(offset, len(pool) - 1)])
    element.target_styles = styles


def _visual_lines(ocr_lines: list[OcrLine]) -> list[SourceLine]:
    """Cluster detector boxes of one element into visual rows."""
    rows: list[list[OcrLine]] = []
    for item in sorted(ocr_lines, key=lambda line: (line.box[1] + line.box[3]) / 2):
        cy = (item.box[1] + item.box[3]) / 2
        h = item.box[3] - item.box[1]
        for row in rows:
            ry0 = min(r.box[1] for r in row)
            ry1 = max(r.box[3] for r in row)
            overlap = min(ry1, item.box[3]) - max(ry0, item.box[1])
            if overlap >= 0.5 * min(h, ry1 - ry0) and ry0 <= cy <= ry1:
                row.append(item)
                break
        else:
            rows.append([item])
    result = []
    for row in rows:
        row.sort(key=lambda line: line.box[0])
        result.append(
            SourceLine(
                box=(min(r.box[0] for r in row), min(r.box[1] for r in row), max(r.box[2] for r in row), max(r.box[3] for r in row)),
                text=" ".join(r.text for r in row),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Stage 3: text removal
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _poly_design(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(xs)]
    if degree >= 1:
        cols += [xs, ys]
    if degree >= 2:
        cols += [xs * xs, xs * ys, ys * ys]
    return np.stack(cols, axis=1)


def _robust_surface_fit(
    pixels: np.ndarray, xs: np.ndarray, ys: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Iteratively reweighted least squares fit of an RGB surface.

    Returns (coefficients[k,3], inlier mask, residual std of inliers).
    """
    design = _poly_design(xs, ys, degree)
    keep = np.ones(len(pixels), dtype=bool)
    coef = np.zeros((design.shape[1], 3))
    for _ in range(4):
        if keep.sum() < design.shape[1] * 4:
            break
        coef, *_ = np.linalg.lstsq(design[keep], pixels[keep], rcond=None)
        residual = np.linalg.norm(pixels - design @ coef, axis=1)
        mad = np.median(residual[keep]) + 1e-6
        keep = residual <= max(3.0, 2.5 * 1.4826 * mad)
    residual = np.linalg.norm(pixels - design @ coef, axis=1)
    std = float(np.sqrt(np.mean(residual[keep] ** 2))) if keep.any() else 999.0
    return coef, keep, std


def analyze_source_line(rgb: np.ndarray, line: SourceLine, others: list[Box] | None = None) -> None:
    """Segment glyph pixels and model the background of one source line.

    `others` are the boxes of all *other* detected text.  Their pixels are
    never used as background samples, and glyph components whose centre falls
    in a smaller (tighter) foreign box belong to that line, not to us.
    """
    others = others or []
    occupied = _occupancy(rgb.shape[:2], others) if others else None
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = line.box
    lh = max(4, y1 - y0)
    pad = max(2, round(lh * 0.18))
    ring = max(3, round(lh * 0.22))
    rx0, ry0 = max(0, x0 - pad - ring), max(0, y0 - pad - ring)
    rx1, ry1 = min(width, x1 + pad + ring), min(height, y1 + pad + ring)
    region = rgb[ry0:ry1, rx0:rx1].astype(np.float64)
    gy, gx = np.mgrid[ry0:ry1, rx0:rx1]
    inner = (gx >= x0 - pad) & (gx < x1 + pad) & (gy >= y0 - pad) & (gy < y1 + pad)
    ring_mask = ~inner
    if occupied is not None:
        free = ~occupied[ry0:ry1, rx0:rx1]
        if (ring_mask & free).sum() >= 30:
            ring_mask = ring_mask & free
    # Normalised coordinates keep the polynomial well conditioned.
    nx = (gx - rx0) / max(1, rx1 - rx0)
    ny = (gy - ry0) / max(1, ry1 - ry0)

    ring_pixels = region[ring_mask]
    best: tuple[np.ndarray, int, float] | None = None
    for degree in (0, 1, 2):
        if ring_pixels.shape[0] < 30 * (degree + 1):
            break
        coef, _keep, std = _robust_surface_fit(ring_pixels, nx[ring_mask], ny[ring_mask], degree)
        if best is None or std < best[2] * 0.85:
            best = (coef, degree, std)
    if best is None:
        coef, degree, std = np.median(region.reshape(-1, 3), axis=0)[None, :], 0, 999.0
    else:
        coef, degree, std = best
    fill = (_poly_design(nx.ravel(), ny.ravel(), degree) @ coef).reshape(region.shape)
    # Never extrapolate outside the colours actually seen around the line:
    # a sparse ring (e.g. text squeezed between frame lines) can make the
    # quadratic overshoot into a visible colour cast.
    if ring_pixels.shape[0] >= 30:
        residual = np.linalg.norm(ring_pixels - (_poly_design(nx[ring_mask], ny[ring_mask], degree) @ coef), axis=1)
        inliers = ring_pixels[residual <= max(6.0, 2.5 * std)]
        if inliers.shape[0] >= 10:
            low = np.percentile(inliers, 1, axis=0) - 2
            high = np.percentile(inliers, 99, axis=0) + 2
            fill = np.clip(fill, low, high)
    fill = np.clip(fill, 0, 255)

    distance = np.linalg.norm(region - fill, axis=2)
    inner_distance = distance[inner]
    contrast = float(np.percentile(inner_distance, 97)) if inner_distance.size else 0.0
    threshold = max(22.0, 0.33 * contrast, 2.8 * std)
    glyph = (distance > threshold) & inner

    try:
        import cv2  # type: ignore

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(glyph.astype(np.uint8), connectivity=8)
        own_area = max(1, (x1 - x0) * (y1 - y0))
        core = np.zeros_like(glyph)
        foreign_mask = np.zeros_like(glyph)
        cx0, cy0, cx1, cy1 = x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0
        px0, py0 = x0 - pad - rx0, y0 - pad - ry0
        px1, py1 = x1 + pad - rx0, y1 + pad - ry0
        for label in range(1, count):
            sx, sy, sw, sh, area = stats[label]
            if area < 2:
                continue
            touches_edge = sx <= px0 or sy <= py0 or sx + sw >= px1 or sy + sh >= py1
            ix = max(0, min(sx + sw, cx1) - max(sx, cx0))
            iy = max(0, min(sy + sh, cy1) - max(sy, cy0))
            # Glyphs sit inside the padded line box.  Components that run into
            # the padding edge are frame lines / neighbouring shapes unless
            # they are mostly inside the detector box.
            component = labels == label
            # Glyphs of neighbouring copy belong to the tightest detector box
            # that contains their centre, even when our loose box overlaps.
            gcx, gcy = centroids[label][0] + rx0, centroids[label][1] + ry0
            claimed = any(
                b[0] <= gcx <= b[2] and b[1] <= gcy <= b[3] and (b[2] - b[0]) * (b[3] - b[1]) < own_area
                for b in others
            )
            if claimed:
                foreign_mask |= component
                continue
            if not touches_edge or ix * iy >= 0.35 * sw * sh:
                core |= component
        rejected = glyph & ~core & ~foreign_mask
        foreign = foreign_mask
        glyph = core
    except Exception:
        rejected = np.zeros_like(glyph)
        foreign = None

    # Anti-aliasing / compression halo around the glyphs: weaker contrast
    # pixels that hug the strict glyph mask - never pixels of shapes we
    # rejected above (frame lines, neighbouring copy).
    halo_radius = max(2, round(lh * 0.2))
    halo = (distance > max(6.0, 2.0 * std)) & _dilate(glyph, halo_radius) & inner & ~_dilate(rejected, 1)
    if foreign is not None and foreign.any():
        halo &= ~_dilate(foreign, halo_radius)
    glyph_strict = glyph
    glyph = glyph | halo

    textured = std > float(os.getenv("ADAPTIFAI_V7_TEXTURE_STD", "14"))
    line.region = (rx0, ry0, rx1, ry1)
    line.background = "textured" if textured else "flat"
    line.fill = fill
    if glyph_strict.sum() < 4:
        line.text_mask = inner.copy()
        line.glyph_mask = inner.copy()
        line.ink_box = line.box
    else:
        line.text_mask = glyph
        line.glyph_mask = glyph_strict
        ys_idx, xs_idx = np.nonzero(glyph_strict)
        line.ink_box = (int(xs_idx.min() + rx0), int(ys_idx.min() + ry0), int(xs_idx.max() + rx0 + 1), int(ys_idx.max() + ry0 + 1))
    # Text colour = the most saturated stroke pixels (thin strokes never reach
    # full colour at their anti-aliased edges).
    strict_distance = distance[glyph_strict]
    if strict_distance.size:
        cutoff = np.percentile(strict_distance, 85)
        core_pixels = region[glyph_strict & (distance >= cutoff)]
    else:
        core_pixels = region[inner]
    line.color = tuple(int(v) for v in np.median(core_pixels, axis=0))  # type: ignore[assignment]
    line.contrast_threshold = float(threshold / max(1.0, contrast))
    # Soft ink coverage: sum of per-pixel text opacity.  Blur moves ink around
    # but preserves this sum, which makes it a robust weight measurement.
    reference = float(np.median(distance[glyph_strict & (distance >= np.percentile(strict_distance, 85))])) if strict_distance.size else 0.0
    if reference > 1:
        alpha = np.clip(distance / reference, 0.0, 1.0)
        alpha[~glyph] = 0.0
        alpha[alpha < 0.12] = 0.0
        line.ink_alpha_sum = float(alpha.sum())
    # Film grain of the background, re-applied to the reconstruction.
    line.grain = float(min(6.0, std / np.sqrt(3.0)))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    try:
        import cv2  # type: ignore

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0
    except Exception:
        out = mask.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return out


def remove_text(
    image: Image.Image,
    elements: list[CopyElement],
    lama: Callable[[Image.Image, Image.Image], Image.Image | None] | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    rgb = np.array(image.convert("RGB"))
    out = rgb.astype(np.float64).copy()
    full_mask = np.zeros(rgb.shape[:2], dtype=bool)
    textured_mask = np.zeros(rgb.shape[:2], dtype=bool)
    stats = {"flatLines": 0, "texturedLines": 0, "provider": "analytic-surface"}
    for element in elements:
        if not element.changed:
            continue
        for line in element.lines:
            if line.text_mask is None or line.region is None or line.fill is None:
                continue
            rx0, ry0, rx1, ry1 = line.region
            lh = max(4, line.box[3] - line.box[1])
            grow = max(2, round(lh * 0.07))
            mask = _dilate(line.text_mask, grow)
            if line.background == "flat":
                region = out[ry0:ry1, rx0:rx1]
                # Feather the outer edge of the mask so anti-aliasing blends.
                soft = _dilate(mask, 1) & ~mask
                fill = line.fill
                if line.grain > 0.4:
                    rng = np.random.default_rng(abs(hash(line.box)) % (2**32))
                    fill = fill + rng.normal(0.0, line.grain, size=fill.shape[:2])[..., None]
                region[mask] = fill[mask]
                region[soft] = 0.5 * region[soft] + 0.5 * line.fill[soft]
                stats["flatLines"] += 1
            else:
                textured_mask[ry0:ry1, rx0:rx1] |= mask
                stats["texturedLines"] += 1
            full_mask[ry0:ry1, rx0:rx1] |= mask
    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if textured_mask.any():
        mask_image = Image.fromarray((textured_mask * 255).astype(np.uint8))
        repaired = None
        if lama is not None:
            try:
                repaired = lama(result, mask_image)
            except Exception as exc:  # pragma: no cover
                print(f"[localize-v7] LaMa failed: {exc}", flush=True)
        provider_name = "lama"
        if repaired is None:
            repaired = openai_inpaint(result, mask_image)
            provider_name = "openai-inpaint"
        if repaired is not None:
            stats["provider"] = f"analytic-surface+{provider_name}"
            repaired_np = np.array(repaired.convert("RGB").resize(result.size))
            base = np.array(result)
            base[textured_mask] = repaired_np[textured_mask]
            result = Image.fromarray(base)
        else:
            import cv2  # type: ignore

            stats["provider"] = "analytic-surface+opencv"
            bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            bgr = cv2.inpaint(bgr, textured_mask.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
            result = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return result, Image.fromarray((full_mask * 255).astype(np.uint8)), stats


def openai_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image | None:
    """Generative fill of `mask` (white = fill) with the OpenAI image edit
    endpoint.  Only masked pixels are taken from the result."""
    if os.getenv("ADAPTIFAI_V7_OPENAI_INPAINT", "1") == "0" or not os.getenv("OPENAI_API_KEY", "").strip():
        return None
    try:
        from openai import OpenAI  # type: ignore

        w, h = image.size
        ratio = w / h
        target = (1536, 1024) if ratio > 1.2 else (1024, 1536) if ratio < 0.83 else (1024, 1024)
        work = image.convert("RGB").resize(target, Image.Resampling.LANCZOS)
        hole = mask.convert("L").resize(target, Image.Resampling.NEAREST)
        edit_mask = Image.new("RGBA", target, (0, 0, 0, 255))
        edit_mask.putalpha(Image.eval(hole, lambda v: 0 if v > 127 else 255))
        image_buffer, mask_buffer = io.BytesIO(), io.BytesIO()
        work.save(image_buffer, "PNG")
        edit_mask.save(mask_buffer, "PNG")
        image_buffer.seek(0)
        mask_buffer.seek(0)
        image_buffer.name, mask_buffer.name = "image.png", "mask.png"
        client = OpenAI(timeout=float(os.getenv("ADAPTIFAI_V7_INPAINT_TIMEOUT", "120")))
        response = client.images.edit(
            model=os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-2"),
            image=image_buffer,
            mask=mask_buffer,
            prompt=(
                "Remove the text in the transparent masked areas and rebuild exactly what is behind it, continuing the "
                "surrounding photo seamlessly. Do not add any text, letters, logos or new objects."
            ),
            size=f"{target[0]}x{target[1]}",
            quality=os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium"),
        )
        generated = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        soft = mask.convert("L").filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
        return Image.composite(generated, image.convert("RGB"), soft)
    except Exception as exc:  # pragma: no cover - provider failures
        print(f"[localize-v7] OpenAI inpaint failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Stage 4: typography
# ---------------------------------------------------------------------------

_FONT_LOCK = threading.Lock()

SYSTEM_FONT_CANDIDATES = {
    "arabic": [
        "/usr/share/fonts/truetype/noto/NotoSansArabic-{w}.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-{w}.ttf",
    ],
    "cjk": [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-{w}.ttc",
    ],
}


def _script_of(text: str) -> str:
    for char in text:
        code = ord(char)
        if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0xFB50 <= code <= 0xFEFF:
            return "arabic"
        if 0x3040 <= code <= 0x30FF or 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xAC00 <= code <= 0xD7AF:
            return "cjk"
    return "latin"


@lru_cache(maxsize=512)
def load_font(style: str, weight: int, width: int, size: int, script: str = "latin") -> ImageFont.FreeTypeFont:
    size = max(4, int(size))
    if script in SYSTEM_FONT_CANDIDATES:
        names = ["Black", "Bold"] if weight >= 750 else ["Bold"] if weight >= 600 else ["Medium", "Regular"] if weight >= 450 else ["Regular"]
        for template in SYSTEM_FONT_CANDIDATES[script]:
            for name in names + ["Bold", "Regular"]:
                path = template.format(w=name)
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
    if style in {"serif", "slab"}:
        path, axes = FONT_DIR / "NotoSerif-Variable.ttf", {"wght": _clamp(weight, 100, 900), "wdth": _clamp(width, 62.5, 100)}
    else:
        path, axes = FONT_DIR / "Archivo-Variable.ttf", {"wght": _clamp(weight, 100, 900), "wdth": _clamp(width, 62, 125)}
    with _FONT_LOCK:
        font = ImageFont.truetype(str(path), size)
        try:
            ordered = []
            for axis in font.get_variation_axes():
                tag = axis.get("name", "")
                if isinstance(tag, bytes):
                    tag = tag.decode("utf-8", "ignore")
                tag_key = {"weight": "wght", "width": "wdth"}.get(str(tag).strip().lower(), str(tag).lower()[:4])
                ordered.append(axes.get(tag_key, axis.get("default", axis.get("minimum", 0))))
            font.set_variation_by_axes(ordered)
        except Exception:
            pass
    return font


def _ink_box(text: str, font: ImageFont.FreeTypeFont, direction: str | None = None) -> tuple[int, int, int, int]:
    """Exact ink bbox relative to the left-baseline anchor."""
    kwargs: dict[str, Any] = {"anchor": "ls"}
    if direction:
        kwargs["direction"] = direction
    try:
        left, top, right, bottom = font.getbbox(text, **kwargs)
    except Exception:
        kwargs.pop("direction", None)
        left, top, right, bottom = font.getbbox(text, **kwargs)
    return int(left), int(top), int(right), int(bottom)


def _unused_measure_stroke_ratio(mask: np.ndarray) -> float | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    if mask.sum() < 20:
        return None
    ys, xs = np.nonzero(mask)
    ink_h = ys.max() - ys.min() + 1
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[dist >= np.percentile(dist[mask], 75)]
    if ridge.size == 0:
        return None
    return float(2.0 * np.median(ridge) / max(1, ink_h))


def _render_mask(text: str, font: ImageFont.FreeTypeFont, threshold: float = 0.5) -> np.ndarray:
    left, top, right, bottom = _ink_box(text, font)
    w, h = max(1, right - left + 4), max(1, bottom - top + 4)
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).text((2 - left, 2 - top), text, font=font, fill=255, anchor="ls")
    return np.array(canvas) > 255 * _clamp(threshold, 0.05, 0.95)


def _soft_ink(text: str, font: ImageFont.FreeTypeFont) -> float:
    left, top, right, bottom = _ink_box(text, font)
    w, h = max(1, right - left + 4), max(1, bottom - top + 4)
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).text((2 - left, 2 - top), text, font=font, fill=255, anchor="ls")
    return float(np.asarray(canvas, dtype=np.float64).sum() / 255.0)


def calibrate_line(element: CopyElement, line: SourceLine, source_text: str) -> dict[str, float]:
    """Find font size, weight and width that reproduce the source line's ink.

    Height fixes the size, ink width fixes the width axis and the total soft
    ink coverage (blur-invariant) fixes the weight.
    """
    style = "serif" if element.font_style in {"serif", "slab"} else "sans"
    ink = line.ink_box or line.box
    ink_w, ink_h = max(1, ink[2] - ink[0]), max(1, ink[3] - ink[1])
    source_text = source_text.strip() or line.text
    probe = 120
    low_axis, high_axis = (62.5, 100.0) if style == "serif" else (62.0, 125.0)

    def size_for(weight: float, width_axis: float) -> float:
        font = load_font(style, int(weight), int(width_axis), probe)
        _, top, _, bottom = _ink_box(source_text, font)
        return probe * ink_h / max(1, bottom - top)

    def width_at(weight: float, width_axis: float) -> float:
        font = load_font(style, int(weight), int(width_axis), max(4, round(size_for(weight, width_axis))))
        left, _, right, _ = _ink_box(source_text, font)
        return float(right - left)

    def solve_width(weight: float) -> float:
        low, high = low_axis, high_axis
        if width_at(weight, low) >= ink_w:
            return low
        if width_at(weight, high) <= ink_w:
            return high
        for _ in range(10):
            mid = (low + high) / 2
            if width_at(weight, mid) < ink_w:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    weight = float(element.font_weight)
    if line.ink_alpha_sum > 0 and ink_h >= 9:
        best: tuple[float, float, float] | None = None
        for candidate in range(300, 901, 50):
            width_axis = solve_width(candidate)
            font = load_font(style, candidate, int(width_axis), max(4, round(size_for(candidate, width_axis))))
            coverage = _soft_ink(source_text, font)
            err = abs(coverage - line.ink_alpha_sum) / line.ink_alpha_sum
            if best is None or err < best[0]:
                best = (err, float(candidate), width_axis)
        if best is not None:
            trust = _clamp((ink_h - 8) / 24, 0.35, 1.0)
            weight = trust * best[1] + (1 - trust) * weight
    width_axis = solve_width(weight)
    size = size_for(weight, width_axis)
    font = load_font(style, int(weight), int(width_axis), max(4, round(size)))
    _, _top, _, bottom = _ink_box(source_text, font)
    baseline = ink[3] - bottom
    _, cap_top, _, _ = _ink_box("H", font)
    return {
        "size": size,
        "weight": weight,
        "width": width_axis,
        "baseline": float(baseline),
        "cap_height": float(-cap_top),
        "ink_left": float(ink[0]),
        "ink_right": float(ink[2]),
        "ink_center": float((ink[0] + ink[2]) / 2),
        "ink_top": float(ink[1]),
        "ink_bottom": float(ink[3]),
    }


def _fit_line(
    text: str,
    style: str,
    weight: float,
    width_axis: float,
    size: float,
    max_width: float,
    script: str,
    direction: str | None,
    min_scale: float,
) -> tuple[ImageFont.FreeTypeFont, float, float]:
    """Shrink a line into max_width: condense first (max 12%), then scale."""
    min_width_axis = max(62.0 if style != "serif" else 62.5, width_axis * 0.88)
    current_width = width_axis
    current_size = size
    for _ in range(40):
        font = load_font(style, int(round(weight)), int(round(current_width)), max(4, round(current_size)), script)
        left, _, right, _ = _ink_box(text, font, direction)
        if right - left <= max_width:
            return font, current_size, current_width
        if script == "latin" and current_width > min_width_axis + 0.5:
            current_width = max(min_width_axis, current_width - 3)
            continue
        if current_size <= size * min_scale:
            return font, current_size, current_width
        current_size = max(size * min_scale, current_size * max(0.9, (max_width / max(1, right - left)) ** 0.9))
    return font, current_size, current_width


def _wrap_to_lines(text: str, count: int) -> list[str]:
    words = text.split()
    if count <= 1 or len(words) <= 1:
        return [text]
    best: list[str] | None = None
    best_score = 1e9
    # Balanced split by character count (exhaustive for small inputs).
    def splits(items: list[str], parts: int):
        if parts == 1:
            yield [" ".join(items)]
            return
        for cut in range(1, len(items) - parts + 2):
            for rest in splits(items[cut:], parts - 1):
                yield [" ".join(items[:cut])] + rest

    for candidate in splits(words, min(count, len(words))):
        lengths = [len(item) for item in candidate]
        score = max(lengths) - min(lengths)
        if score < best_score:
            best, best_score = candidate, score
    return best or [text]


def render_elements(
    base: Image.Image,
    elements: list[CopyElement],
    target_language: str,
    grid_margin: float | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    canvas = base.convert("RGBA")
    width, height = canvas.size
    report: list[dict[str, Any]] = []
    direction = "rtl" if target_language.upper() in RTL_LANGUAGES else None
    others = [e for e in elements]
    for element in elements:
        if not element.changed:
            continue
        style = "serif" if element.font_style in {"serif", "slab"} else "sans"
        src_lines = element.lines
        sources = list(element.source_lines)
        if len(sources) != len(src_lines):
            sources = _wrap_to_lines(" ".join(sources), len(src_lines)) if len(src_lines) > 1 else [" ".join(sources)]
            if len(sources) != len(src_lines):
                sources = [line.text for line in src_lines]
        calibrations = [calibrate_line(element, line, text) for line, text in zip(src_lines, sources)]
        sizes = [c["size"] for c in calibrations]
        uniform = element.uniform_size and (max(sizes) / max(1e-6, min(sizes)) < 1.25)
        if uniform:
            shared = {
                "size": float(np.median(sizes)),
                "weight": float(np.median([c["weight"] for c in calibrations])),
                "width": float(np.median([c["width"] for c in calibrations])),
            }
            for c in calibrations:
                c.update(shared)

        targets = list(element.target_lines)
        has_styles = len(element.target_styles) == len(targets) and bool(targets)
        if len(targets) != len(src_lines) and uniform and not has_styles:
            if len(src_lines) > 1 and len(targets) < len(src_lines):
                rewrapped = _wrap_to_lines(" ".join(targets), len(src_lines))
                if len(rewrapped) == len(src_lines):
                    targets = rewrapped
        script = _script_of(" ".join(targets))

        # Horizontal room: the element's own column, extended to its margins.
        ex0, ey0, ex1, ey1 = element.box
        margin = grid_margin if grid_margin is not None else max(4.0, min(ex0, width - ex1))
        if element.align == "left":
            container_w = max(ex1 - ex0, width - margin - ex0)
        elif element.align == "right":
            container_w = max(ex1 - ex0, ex1 - margin)
        else:
            centre = (ex0 + ex1) / 2
            container_w = max(ex1 - ex0, 2 * min(centre - margin, width - margin - centre))
        if element.role in {"cta", "button"}:
            container_w = (ex1 - ex0) * 1.06 + 0.6 * (ey1 - ey0)  # stay inside the button
        elif element.role in {"badge", "price"}:
            container_w = (ex1 - ex0) * 1.02
        min_scale = float(os.getenv("ADAPTIFAI_V7_MIN_SCALE", "0.62"))
        if element.role in {"badge", "price", "cta", "button"}:
            # Contained shapes: the copy must stay inside, smaller is better
            # than overflowing the frame/button.
            min_scale = min(min_scale, 0.45)

        placements: list[dict[str, Any]] = []
        styles = element.target_styles if len(element.target_styles) == len(targets) else []
        identity = styles == list(range(len(targets))) or (not styles and len(targets) == len(src_lines))
        if not uniform and len(src_lines) > 1 and not identity:
            if not styles:
                n_t, n_s = len(targets), len(src_lines)
                styles = [round(i * (n_s - 1) / max(1, n_t - 1)) for i in range(n_t)] if n_t > 1 else [int(np.argmax(sizes))]
            placements = _stack_layout(targets, styles, calibrations, style, container_w, script, direction, min_scale)
        elif len(targets) == len(src_lines):
            for text, calibration in zip(targets, calibrations):
                font, size, width_axis = _fit_line(
                    text, style, calibration["weight"], calibration["width"], calibration["size"], container_w, script, direction, min_scale
                )
                placements.append({"text": text, "font": font, "size": size, "width": width_axis, "cal": calibration})
            if uniform and placements:
                # Keep one size across the element after fitting.
                min_size = min(p["size"] for p in placements)
                min_width_axis = min(p["width"] for p in placements)
                for p in placements:
                    p["font"] = load_font(style, int(round(p["cal"]["weight"])), int(round(min_width_axis)), max(4, round(min_size)), script)
                    p["size"], p["width"] = min_size, min_width_axis
            for p in placements:
                cal = p["cal"]
                shrink = p["size"] / max(1e-6, cal["size"])
                p["baseline"] = cal["baseline"] - cal["cap_height"] * (1 - shrink) / 2
            _resolve_line_collisions(placements, style, script, direction)
        else:
            # Different line count: lay out a block centred on the source block.
            base_cal = calibrations[0]
            pitch = (
                float(np.median(np.diff([c["baseline"] for c in calibrations])))
                if len(calibrations) > 1
                else base_cal["cap_height"] * 1.45
            )
            n = max(1, len(targets))
            available_h = (ey1 - ey0) + 0.6 * pitch
            scale = min(1.0, available_h / max(1.0, pitch * (n - 1) + base_cal["cap_height"]))
            fitted = []
            for text in targets:
                font, size, width_axis = _fit_line(
                    text, style, base_cal["weight"], base_cal["width"], base_cal["size"] * scale, container_w, script, direction, min_scale
                )
                fitted.append((text, font, size, width_axis))
            min_size = min(item[2] for item in fitted)
            min_width_axis = min(item[3] for item in fitted)
            pitch_used = pitch * (min_size / base_cal["size"])
            cap = base_cal["cap_height"] * (min_size / base_cal["size"])
            block_h = pitch_used * (n - 1) + cap
            center_y = (calibrations[0]["baseline"] - calibrations[0]["cap_height"] + calibrations[-1]["baseline"]) / 2
            first_baseline = center_y - block_h / 2 + cap
            for index, (text, _font, _size, _w) in enumerate(fitted):
                placements.append(
                    {
                        "text": text,
                        "font": load_font(style, int(round(base_cal["weight"])), int(round(min_width_axis)), max(4, round(min_size)), script),
                        "size": min_size,
                        "width": min_width_axis,
                        "cal": base_cal,
                        "baseline": first_baseline + index * pitch_used,
                    }
                )

        element_left = min(c["ink_left"] for c in calibrations)
        element_right = max(c["ink_right"] for c in calibrations)
        element_center = (element_left + element_right) / 2
        color = _element_color(element)
        draw = ImageDraw.Draw(canvas)
        for p in placements:
            left, _top, right, _bottom = _ink_box(p["text"], p["font"], direction)
            ink_w = right - left
            align = element.align if direction is None else ({"left": "right", "right": "left"}.get(element.align, element.align))
            if align == "left":
                x = element_left - left
            elif align == "right":
                x = element_right - right
            else:
                center = element_center if uniform or len(placements) != len(calibrations) or not identity else p["cal"]["ink_center"]
                x = center - (left + ink_w / 2)
            x = _clamp(x, 2 - left, width - 2 - right)
            kwargs: dict[str, Any] = {"font": p["font"], "fill": color + (255,), "anchor": "ls"}
            if direction:
                kwargs["direction"] = direction
            try:
                draw.text((x, p["baseline"]), p["text"], **kwargs)
            except Exception:
                kwargs.pop("direction", None)
                draw.text((x, p["baseline"]), p["text"], **kwargs)
        report.append(
            {
                "id": element.id,
                "role": element.role,
                "source_lines": element.source_lines,
                "target_lines": [p["text"] for p in placements],
                "color": "#%02x%02x%02x" % color,
                "lines": [
                    {
                        "text": p["text"],
                        "font_size": round(p["size"], 1),
                        "source_font_size": round(p["cal"]["size"], 1),
                        "weight": round(p["cal"]["weight"]),
                        "width_axis": round(p["width"], 1),
                    }
                    for p in placements
                ],
            }
        )
    del others
    return canvas.convert("RGB"), report


def _resolve_line_collisions(placements: list[dict[str, Any]], style: str, script: str, direction: str | None) -> None:
    """Accented capitals (İ, Ü, Ş, Ç, Â...) need room the source never had.
    Spread the lines a little around the block centre, then shrink."""
    if len(placements) < 2:
        return
    for _ in range(8):
        extents = []
        for p in placements:
            _l, top, _r, bottom = _ink_box(p["text"], p["font"], direction)
            extents.append((p["baseline"] + top, p["baseline"] + bottom))
        cap = float(np.median([p["cal"]["cap_height"] * p["size"] / max(1e-6, p["cal"]["size"]) for p in placements]))
        min_gap = max(2.0, 0.07 * cap)
        needed = max(min_gap - (extents[i + 1][0] - extents[i][1]) for i in range(len(extents) - 1))
        if needed <= 0.5:
            return
        spread = min(needed, 0.18 * cap)
        centre = (len(placements) - 1) / 2
        for index, p in enumerate(placements):
            p["baseline"] += (index - centre) * spread
        if spread >= needed:
            return
        for p in placements:
            p["size"] *= 0.96
            p["font"] = load_font(style, int(round(p["cal"]["weight"])), int(round(p["width"])), max(4, round(p["size"])), script)


def _stack_layout(
    targets: list[str],
    styles: list[int],
    calibrations: list[dict[str, float]],
    style: str,
    container_w: float,
    script: str,
    direction: str | None,
    min_scale: float,
) -> list[dict[str, Any]]:
    """Mixed-size lines (badges): each target line takes the size of the
    source line named in `styles`; lines are re-stacked with the source's
    vertical rhythm and centred on the source block."""
    src_top = calibrations[0]["ink_top"]
    src_bottom = calibrations[-1]["ink_bottom"]
    gaps = [calibrations[i + 1]["ink_top"] - calibrations[i]["ink_bottom"] for i in range(len(calibrations) - 1)]
    gap_ratio = float(np.median([g / max(1.0, calibrations[i]["cap_height"]) for i, g in enumerate(gaps)])) if gaps else 0.25
    gap_ratio = _clamp(gap_ratio, 0.08, 0.8)
    scale = 1.0
    placements: list[dict[str, Any]] = []
    for _ in range(3):
        placements = []
        for text, index in zip(targets, styles):
            cal = calibrations[index]
            font, size, width_axis = _fit_line(
                text, style, cal["weight"], cal["width"], cal["size"] * scale, container_w, script, direction, min_scale * scale
            )
            _l, top, _r, bottom = _ink_box(text, font, direction)
            placements.append({"text": text, "font": font, "size": size, "width": width_axis, "cal": cal, "top": top, "bottom": bottom})
        heights = [p["bottom"] - p["top"] for p in placements]
        gap_px = [gap_ratio * p["cal"]["cap_height"] * (p["size"] / p["cal"]["size"]) for p in placements[:-1]]
        block = sum(heights) + sum(gap_px)
        available = (src_bottom - src_top) * 1.06
        if block <= available or scale < 0.7:
            break
        scale *= max(0.7, available / block)
    y = (src_top + src_bottom) / 2 - block / 2
    for index, p in enumerate(placements):
        p["baseline"] = y - p["top"]
        y += (p["bottom"] - p["top"]) + (gap_px[index] if index < len(gap_px) else 0)
    return placements


def _element_color(element: CopyElement) -> tuple[int, int, int]:
    colors = np.array([line.color for line in element.lines], dtype=np.float64)
    return tuple(int(v) for v in np.median(colors, axis=0))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _occupancy(shape: tuple[int, int], boxes: list[Box], grow: float = 0.15) -> np.ndarray:
    occupied = np.zeros(shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        g = max(2, round((y1 - y0) * grow))
        occupied[max(0, y0 - g): y1 + g, max(0, x0 - g): x1 + g] = True
    return occupied


def _analyze_changed_lines(rgb: np.ndarray, elements: list[CopyElement], ocr_lines: list[OcrLine] | None = None) -> None:
    all_boxes = [line.box for line in ocr_lines] if ocr_lines else [line.box for element in elements for line in element.lines]
    for element in elements:
        if not element.changed:
            continue
        for line in element.lines:
            others = [box for box in all_boxes if box != line.box and not _contains(line.box, box)]
            analyze_source_line(rgb, line, others)


def _contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def layout_margin(size: tuple[int, int], lines: list[OcrLine] | list[Box]) -> float:
    """Smallest horizontal gap between any text and the canvas edge.

    Designers keep all copy on one grid, so this is the safe column for
    re-flowed copy.
    """
    width = size[0]
    boxes = [item.box if isinstance(item, OcrLine) else item for item in lines]
    gaps = [min(box[0], width - box[2]) for box in boxes if box[2] - box[0] > 0]
    floor = width * 0.04
    return max(floor, min(gaps)) if gaps else width * 0.08


def _apply_reviewed_copy(elements: list[CopyElement], analysis: dict[str, Any], target_language: str) -> None:
    raw_by_id = {str(raw.get("id")): raw for raw in analysis.get("elements") or [] if isinstance(raw, dict)}
    for element in elements:
        raw = raw_by_id.get(element.id)
        if raw is None or not element.translate:
            continue
        lines = [str(item).strip() for item in raw.get("target_lines") or [] if str(item).strip()]
        if not lines or lines == element.target_lines:
            continue
        element.target_lines = [locale_upper(line, target_language) for line in lines] if element.uppercase else lines
        styles = raw.get("target_line_styles")
        element.target_styles = (
            [int(_clamp(int(v), 0, len(element.lines) - 1)) for v in styles]
            if isinstance(styles, list) and len(styles) == len(element.target_lines)
            else []
        )
        _guard_numeric_emphasis(element)


@dataclass
class LocalizeResult:
    image: Image.Image
    clean_base: Image.Image
    mask: Image.Image
    marks: Image.Image
    elements: list[CopyElement]
    analysis: dict[str, Any]
    report: dict[str, Any]


def localize_image(
    image_path: Path,
    target_language: str,
    *,
    vision_caller: VisionCaller | None = None,
    lama: Callable[[Image.Image, Image.Image], Image.Image | None] | None = None,
    ocr_lines: list[OcrLine] | None = None,
) -> LocalizeResult:
    source = Image.open(image_path).convert("RGB")
    started = time.time()
    if ocr_lines is None:
        ocr_lines, detector = detect_text_lines(image_path, source)
    else:
        detector = "provided"
    if not ocr_lines:
        raise LocalizeError("No text was detected on the creative.")
    marks = build_set_of_marks(source, ocr_lines)
    analysis = analyze_copy(source, ocr_lines, target_language, vision_caller)
    elements = build_elements(analysis, ocr_lines, target_language)

    rgb = np.array(source)
    _analyze_changed_lines(rgb, elements, ocr_lines)
    clean, mask, removal_stats = remove_text(source, elements, lama)
    margin = layout_margin(source.size, ocr_lines)
    # Dry run to learn which lines do not fit, then one native-speaker review
    # that fixes naturalness and length together.
    _dry, dry_report = render_elements(clean, elements, target_language, margin)
    budgets = fit_budgets(dry_report, elements)
    review = review_copy(analysis, target_language, source, vision_caller, budgets)
    if review.get("changes"):
        _apply_reviewed_copy(elements, analysis, target_language)
    rendered, render_report = render_elements(clean, elements, target_language, margin)
    report = {
        "pipeline": "v7-vision-semantic",
        "detector": detector,
        "detected_boxes": [{"id": line.id, "box": list(line.box), "guess": line.text} for line in ocr_lines],
        "analysis_provider": analysis.get("_provider"),
        "analysis_model": analysis.get("_model"),
        "source_language": str(analysis.get("source_language") or "").upper() or None,
        "removal": removal_stats,
        "copy_review": review,
        "fit_budgets": budgets,
        "elements": [
            {
                "id": element.id,
                "role": element.role,
                "translate": element.translate,
                "changed": element.changed,
                "box_ids": element.ocr_ids,
                "source_lines": element.source_lines,
                "target_lines": element.target_lines,
                "target_line_styles": element.target_styles,
                "uniform_size": element.uniform_size,
                "background": [line.background for line in element.lines],
            }
            for element in elements
        ],
        "render": render_report,
        "seconds": round(time.time() - started, 2),
    }
    return LocalizeResult(
        image=rendered,
        clean_base=clean,
        mask=mask,
        marks=marks,
        elements=elements,
        analysis=analysis,
        report=report,
    )


def rerender_with_copy(
    source: Image.Image,
    clean_base: Image.Image,
    elements_state: list[dict[str, Any]],
    replacement_copy: list[str] | None,
    target_language: str,
) -> tuple[Image.Image, list[CopyElement]]:
    """Re-render a stored localization with edited copy (editor round-trip)."""
    elements = [element_from_state(state) for state in elements_state]
    editable = [element for element in elements if element.translate]
    if replacement_copy:
        for element, text in zip(editable, replacement_copy):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                element.target_lines = [locale_upper(line, target_language) if element.uppercase else line for line in lines]
                if len(element.target_styles) != len(element.target_lines):
                    element.target_styles = []
                element.changed = _normalize_copy(" ".join(element.source_lines)) != _normalize_copy(" ".join(element.target_lines))
    for element in elements:
        # Text already erased from the clean base must always be painted back.
        if element.style.get("was_changed"):
            element.changed = True
    rgb = np.array(source.convert("RGB"))
    _analyze_changed_lines(rgb, elements)
    # Elements that were unchanged originally but are edited now still need
    # their source text removed from the stored clean base.
    clean, _mask, _ = remove_text(clean_base, [e for e in elements if e.changed and not e.style.get("was_changed")], None)
    all_boxes = [line.box for element in elements for line in element.lines]
    rendered, _ = render_elements(clean, elements, target_language, layout_margin(source.size, all_boxes))
    return rendered, elements


def element_state(element: CopyElement) -> dict[str, Any]:
    return {
        "id": element.id,
        "role": element.role,
        "translate": element.translate,
        "changed": element.changed,
        "source_lines": element.source_lines,
        "target_lines": element.target_lines,
        "ocr_ids": element.ocr_ids,
        "align": element.align,
        "uppercase": element.uppercase,
        "font_style": element.font_style,
        "font_weight": element.font_weight,
        "uniform_size": element.uniform_size,
        "target_styles": element.target_styles,
        "line_boxes": [list(line.box) for line in element.lines],
        "line_texts": [line.text for line in element.lines],
    }


def element_from_state(state: dict[str, Any]) -> CopyElement:
    element = CopyElement(
        id=str(state.get("id")),
        role=str(state.get("role") or "other"),
        translate=bool(state.get("translate")),
        source_lines=list(state.get("source_lines") or []),
        target_lines=list(state.get("target_lines") or []),
        ocr_ids=list(state.get("ocr_ids") or []),
        align=str(state.get("align") or "center"),
        uppercase=bool(state.get("uppercase")),
        font_style=str(state.get("font_style") or "sans"),
        font_weight=int(state.get("font_weight") or 700),
        uniform_size=bool(state.get("uniform_size", True)),
        changed=bool(state.get("changed")),
        target_styles=[int(v) for v in state.get("target_styles") or []],
    )
    element.style["was_changed"] = bool(state.get("changed"))
    element.lines = [
        SourceLine(box=tuple(int(v) for v in box), text=str(text))  # type: ignore[arg-type]
        for box, text in zip(state.get("line_boxes") or [], state.get("line_texts") or [])
    ]
    return element
