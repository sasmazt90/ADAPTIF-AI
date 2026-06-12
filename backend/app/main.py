from __future__ import annotations

import base64
import hashlib
import io
import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import unicodedata
import threading
import zipfile
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

try:
    from app.localization_protocol import (
        LocalizationStatus,
        RiskLevel,
        classify_creative_risk,
        build_segmentation_masks,
        analyze_depth_layering,
        route_cleanup_providers,
        determine_inpaint_strategy,
        run_quality_gate,
        run_provider_bakeoff,
        compute_candidate_score,
        iterative_copy_fit,
        decide_render_strategy,
        should_proceed_to_preview,
        propagate_status,
        run_localization_protocol,
        CreativeRiskReport,
        SegmentationMasks,
        DepthLayeringReport,
        QualityGateReport,
        ProviderBakeoffReport,
        CleanupCandidate,
        PipelineResult,
        CopyFitResult,
        InpaintStrategy,
        ProviderRoute,
    )
    _PROTOCOL_AVAILABLE = True
except ImportError:
    _PROTOCOL_AVAILABLE = False
    # Stub fallbacks so the rest of main.py compiles without the protocol module
    class RiskLevel:  # type: ignore[no-redef]
        LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"
        UNSUPPORTED_AUTO_CLEANUP = "UNSUPPORTED_AUTO_CLEANUP"
        REJECT_LOW_CONFIDENCE = "REJECT_LOW_CONFIDENCE"
        PACKAGING_PROTECTION_RISK = "PACKAGING_PROTECTION_RISK"
    def route_cleanup_providers(risk_level, block=None, image_size=(0, 0)):  # type: ignore[no-redef]
        return []
    def run_localization_protocol(*a, **kw):  # type: ignore[no-redef]
        return None
    def run_quality_gate(*a, **kw):  # type: ignore[no-redef]
        return None

try:
    from app.smart_reframe import (
        BackgroundStyle,
        BackgroundType,
        BBox1000,
        ExpansionStrategy,
        LayerRole,
        LogicBucket,
        ProductLayer,
        RGBColor,
        SmartReframe,
        TextLayer,
        TextStyle,
        VisualLayer,
        VisualAnalysis,
        build_visual_analysis_prompt,
        parse_visual_analysis_payload,
    )
    from app.smart_compositor import _prepare_foreground_rgba_crop, render_deterministic_compositor
except ImportError:
    from backend.app.smart_reframe import (  # type: ignore[no-redef]
        BackgroundStyle,
        BackgroundType,
        BBox1000,
        ExpansionStrategy,
        LayerRole,
        LogicBucket,
        ProductLayer,
        RGBColor,
        SmartReframe,
        TextLayer,
        TextStyle,
        VisualLayer,
        VisualAnalysis,
        build_visual_analysis_prompt,
        parse_visual_analysis_payload,
    )
    from backend.app.smart_compositor import _prepare_foreground_rgba_crop, render_deterministic_compositor  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(REPO_ROOT / ".env")
_RESIZE_PRODUCT_COMPLETION_CACHE: dict[str, tuple[Image.Image, dict[str, Any]]] = {}

SUPPORTED_UPLOAD_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".pdf", ".zip"}
IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
MARKETING_HINTS = (
    "buy",
    "shop",
    "sale",
    "save",
    "now",
    "free",
    "new",
    "limited",
    "launch",
    "discover",
    "join",
    "try",
    "start",
    "learn",
    "get",
    "new",
    "old",
    "before",
    "after",
    "neu",
    "alt",
    "technology",
    "technologie",
    "defensive",
    "fresh",
    "frische",
    "anhaltender",
    "versorgt",
    "schuhe",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PREVIEW_TEMPLATE_PATH = REPO_ROOT / "src" / "shared" / "preview-templates.json"


def load_shared_preview_template_schema() -> dict[str, Any]:
    # In Docker: __file__ = /app/app/main.py â†’ parents[2] = / (wrong)
    # Fall back to parents[1] = /app which matches WORKDIR
    candidates = [
        SHARED_PREVIEW_TEMPLATE_PATH,
        Path(__file__).resolve().parent.parent / "src" / "shared" / "preview-templates.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


SHARED_PREVIEW_TEMPLATE_SCHEMA = load_shared_preview_template_schema()
SHARED_TEMPLATE_SCHEMA_VERSION = str(SHARED_PREVIEW_TEMPLATE_SCHEMA.get("schemaVersion", "unknown"))
SHARED_PREVIEW_PLACEMENTS = SHARED_PREVIEW_TEMPLATE_SCHEMA.get("placements", [])
SHARED_PREVIEW_PLACEMENT_MAP: dict[str, dict[str, Any]] = {
    str(item["placementId"]): item
    for item in SHARED_PREVIEW_PLACEMENTS
    if isinstance(item, dict) and item.get("placementId")
}

PLACEMENT_DIMENSIONS = {
    placement_id: (
        int(item.get("dimensions", {}).get("width", 1200)),
        int(item.get("dimensions", {}).get("height", 800)),
    )
    for placement_id, item in SHARED_PREVIEW_PLACEMENT_MAP.items()
}
MANIFEST_NAME = "manifest.json"

OCR_DETECTOR = None
TROCR_PROCESSOR = None
TROCR_MODEL = None
OCR_MODEL_LOCK = threading.Lock()
ADAPT_PROCESSING_SEMAPHORE = asyncio.Semaphore(int(os.getenv("ADAPTIFAI_MAX_ACTIVE_JOBS", "1")))
HF_CLEANUP_SESSION = None
HF_CLEANUP_SESSION_LOCK = threading.Lock()
LAST_REPLICATE_LAMA_ERROR = ""
HF_INPAINT_MODELS = [
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
]


class TextBlock(BaseModel):
    id: str | None = None
    text: str
    role: str
    translate: bool
    bbox: tuple[int, int, int, int]
    clean_box: tuple[int, int, int, int] | None = None
    font_family: str = "DejaVu Sans"
    font_weight: int = 700
    font_size_estimate: int = 16
    line_height_estimate: int = 18
    color: str = "#111111"
    translated_text: str | None = None
    align: str = "center"
    surface: str = "overlay"
    line_boxes: list[tuple[int, int, int, int]] = Field(default_factory=list)
    line_texts: list[str] = Field(default_factory=list)
    polygon: list[tuple[int, int]] = Field(default_factory=list)
    line_polygons: list[list[tuple[int, int]]] = Field(default_factory=list)
    symbol_polygons: list[list[tuple[int, int]]] = Field(default_factory=list)
    overflow_warning: bool = False
    source_style_spans: list[dict[str, Any]] = Field(default_factory=list)
    translated_style_spans: list[dict[str, Any]] = Field(default_factory=list)
    source_word_styles: list[dict[str, Any]] = Field(default_factory=list)
    translation_candidates: list[dict[str, str]] = Field(default_factory=list)
    target_language: str | None = None
    resize_source_box: tuple[int, int, int, int] | None = None
    resize_target_fill: float = 0.72
    resize_min_font_size: int = 8
    resize_max_font_size: int = 42
    resize_preferred_line_count: int = 0
    resize_stack_role: str = "primary"
    cleanup_confidence: float = 1.0
    cleanup_strategy: str = "clean_replace"
    render_strategy: str = "clean_replace"


LOCALIZE_V2_RICH_TEXT_PROMPT_RULES = [
    "Translate each OCR block semantically as one complete copy unit before deciding target line breaks; never translate line-by-line or word-by-word.",
    "Analyze the original overlay marketing text at word/phrase level, including font weight, color, typographic casing, and punctuation.",
    "Return translated copy as rich text segments, not only as one flat string.",
    "Map style by meaning, not by source position or source line. If a styled source phrase moves to another target line because of target-language syntax, apply that style to the translated phrase that carries the same meaning on its new line.",
    "Support N:M style inheritance: one source word may map to multiple target words, and multiple source words may map to one target word.",
    "Preserve punctuation intent exactly. Exclamation marks, question marks, slashes, and other expressive punctuation must not be silently converted to periods or removed unless the target language requires equivalent punctuation.",
    "Preserve emphasis from bold, heavier weight, italic, underline, strikethrough, distinctive color, all-caps, numeric claims, discount percentages, and key benefit phrases.",
    "Do not apply a source style to the wrong translated word just because it is in the same visual position or same line.",
    "Return font_category for each segment from the visual source style: sans-serif, serif, slab-serif, display, handwriting, or monospace.",
    "Product or brand names that are part of overlay marketing/instruction copy are movable semantic tokens; do not lock them to their original coordinate or source line.",
    "Never split a semantic instruction into literal line translations. Build the target sentence first, then lay it out.",
    "Keep translated_text as the plain concatenation of segments for fallback rendering.",
]


LOCALIZE_V212_TRANSLATION_SCHEMA = {
    "blocks": [
        {
            "id": "ocr-block-id from input",
            "translate": True,
            "translated_text": "plain fallback localized copy, equal to concatenated segment text",
            "lines": [
                {
                    "segments": [
                        {
                            "text": "translated word or phrase",
                            "source_word_id": "id from input_blocks[].source_words[].id whose meaning/style this target segment inherited",
                            "source_word_ids": ["one or more source word ids for N:M style inheritance"],
                            "semantic_role": "discount | modifier | product_condition | benefit | instruction | cta | other",
                            "font_category": "sans-serif | serif | slab-serif | display | handwriting | monospace",
                            "is_italic": False,
                            "is_underlined": False,
                            "is_strikethrough": False,
                        }
                    ]
                }
            ],
            "segments": [
                {
                    "text": "translated word or phrase",
                    "is_uppercase": False,
                    "source_word_id": "id from input_blocks[].source_words[].id whose meaning/style this target segment inherited",
                    "source_word_ids": ["one or more source word ids for N:M style inheritance"],
                    "source_segment_hint": "source word or phrase whose meaning/style this segment inherited",
                    "semantic_role": "discount | modifier | product_condition | benefit | instruction | cta | other",
                    "font_category": "sans-serif | serif | slab-serif | display | handwriting | monospace",
                    "is_italic": False,
                    "is_underlined": False,
                    "is_strikethrough": False,
                }
            ],
        }
    ]
}


class OutputAsset(BaseModel):
    placement_id: str | None = None
    filename: str
    width: int
    height: int
    safe_zone_warnings: list[str] = []
    download_url: str
    source_name: str
    language: str | None = None
    source_language: str | None = None
    translated_text: str = ""
    extracted_blocks: list[TextBlock] = []
    debug: dict[str, Any] | None = None


class AdaptResponse(BaseModel):
    job_id: str
    stateless: bool
    expires_at: datetime
    credits_estimated: int
    extracted_blocks: list[TextBlock]
    translations: dict[str, list[str]]
    outputs: list[OutputAsset]


class EditRequest(BaseModel):
    job_id: str
    filename: str
    mode: str
    copy_text: str = Field(default="", alias="copy")
    x: int = 0
    y: int = 0
    opacity: int = 18
    scale: int = 100
    fit: str = "cover"
    preserve_bold: bool = True
    text_color: str = ""
    font_size_scale: int = 100
    text_italic: bool = False
    text_underline: bool = False
    text_strike: bool = False
    mask_cleanup: bool = True
    fit_bounds: bool = True

    model_config = {"populate_by_name": True}


app = FastAPI(title="AdaptifAI Backend", version="0.2.0")
cors_origins = [
    origin.strip()
    for origin in os.getenv("ADAPTIFAI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["authorization", "content-type", "accept", "x-requested-with"],
)


def torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def is_cpu_runtime() -> bool:
    return torch_device() == "cpu"


def runtime_device_label() -> str:
    return os.getenv("ADAPTIFAI_RUNTIME_DEVICE", "cpu").strip() or "cpu"


def temp_root() -> Path:
    root = Path(os.getenv("ADAPTIFAI_TMP_DIR", Path(tempfile.gettempdir()) / "adaptifai"))
    root.mkdir(parents=True, exist_ok=True)
    return root


GOOGLE_FONTS_CATALOG_LOCK = threading.Lock()
GOOGLE_FONTS_CATALOG_CACHE: dict[str, dict[str, Any]] | None = None
GOOGLE_FONT_FILE_CACHE: dict[tuple[str, str], Path | None] = {}
LOCAL_FONT_FILE_CACHE: dict[tuple[str, bool], Any] = {}
DEFAULT_GOOGLE_FONT_STACK = ("Inter", "Roboto", "Montserrat", "Poppins", "Open Sans", "Lato")
GOOGLE_FONT_CATEGORY_STACKS = {
    "sans-serif": ("Inter", "Roboto", "Montserrat", "Poppins", "Open Sans", "Lato"),
    "serif": ("Libre Baskerville", "Merriweather", "Lora", "Playfair Display", "Roboto Serif"),
    "slab-serif": ("Roboto Slab", "Arvo", "Bitter", "Zilla Slab"),
    "display": ("Montserrat", "Poppins", "Oswald", "Bebas Neue", "Anton"),
    "handwriting": ("Caveat", "Patrick Hand", "Kalam"),
    "monospace": ("Roboto Mono", "Source Code Pro", "IBM Plex Mono"),
}


def google_fonts_enabled() -> bool:
    return bool(os.getenv("GOOGLE_FONTS_API_KEY", "").strip())


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def localize_generative_cleanup_enabled() -> bool:
    default = "1" if os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") else "0"
    return env_flag("ADAPTIFAI_ENABLE_LOCALIZE_GENERATIVE_CLEANUP", default)


def localize_cleanup_gate_enabled() -> bool:
    default = "0" if is_cpu_runtime() else "1"
    return env_flag("ADAPTIFAI_ENABLE_LOCALIZE_CLEANUP_GATE", default)


def localize_fast_cleanup_enabled() -> bool:
    default = "1" if is_cpu_runtime() else "0"
    return env_flag("ADAPTIFAI_LOCALIZE_FAST_CLEANUP", default)


def font_cache_dir() -> Path:
    root = Path(os.getenv("ADAPTIFAI_FONT_CACHE_DIR", temp_root() / "fonts"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def sanitize_font_cache_name(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return sanitized[:80] or "font"


def load_google_fonts_catalog() -> dict[str, dict[str, Any]]:
    global GOOGLE_FONTS_CATALOG_CACHE
    if GOOGLE_FONTS_CATALOG_CACHE is not None:
        return GOOGLE_FONTS_CATALOG_CACHE

    key = os.getenv("GOOGLE_FONTS_API_KEY", "").strip()
    if not key:
        GOOGLE_FONTS_CATALOG_CACHE = {}
        return GOOGLE_FONTS_CATALOG_CACHE

    with GOOGLE_FONTS_CATALOG_LOCK:
        if GOOGLE_FONTS_CATALOG_CACHE is not None:
            return GOOGLE_FONTS_CATALOG_CACHE

        cache_path = font_cache_dir() / "google-webfonts-catalog.json"
        payload: dict[str, Any] | None = None
        try:
            if cache_path.exists() and time.time() - cache_path.stat().st_mtime < 24 * 60 * 60:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None

        if payload is None:
            try:
                response = requests.get(
                    "https://www.googleapis.com/webfonts/v1/webfonts",
                    params={"key": key, "sort": "popularity"},
                    timeout=8,
                )
                response.raise_for_status()
                payload = response.json()
                cache_path.write_text(json.dumps(payload), encoding="utf-8")
            except Exception as exc:
                print(f"[fonts] Google Fonts catalog unavailable, using local fonts: {exc}", flush=True)
                payload = {}

        items = payload.get("items", []) if isinstance(payload, dict) else []
        GOOGLE_FONTS_CATALOG_CACHE = {
            str(item.get("family", "")).strip().lower(): item
            for item in items
            if isinstance(item, dict) and str(item.get("family", "")).strip()
        }
        return GOOGLE_FONTS_CATALOG_CACHE


def normalize_font_category(value: str | None) -> str:
    normalized = normalize_ocr_text(str(value or ""))
    if "mono" in normalized:
        return "monospace"
    if "hand" in normalized or "script" in normalized:
        return "handwriting"
    if "slab" in normalized:
        return "slab-serif"
    if "serif" in normalized and "sans" not in normalized:
        return "serif"
    if "display" in normalized or "condensed" in normalized:
        return "display"
    return "sans-serif"


def pick_google_font_family(requested_family: str | None, bold: bool, category: str | None = None) -> str | None:
    catalog = load_google_fonts_catalog()
    if not catalog:
        return None

    requested = (requested_family or "").strip().strip("'\"")
    if requested:
        for candidate in [part.strip().strip("'\"") for part in requested.split(",") if part.strip()]:
            key = candidate.lower()
            if key in catalog:
                return str(catalog[key].get("family") or candidate)

    preferred = os.getenv("ADAPTIFAI_DEFAULT_GOOGLE_FONT", "").strip()
    category_stack = GOOGLE_FONT_CATEGORY_STACKS.get(normalize_font_category(category), DEFAULT_GOOGLE_FONT_STACK)
    fallback_stack = (preferred,) + category_stack if preferred else category_stack
    if bold:
        bold_stack = GOOGLE_FONT_CATEGORY_STACKS.get(normalize_font_category(category), ("Montserrat", "Poppins", "Inter", "Roboto", "Open Sans", "Lato"))
        if normalize_font_category(category) == "sans-serif":
            bold_stack = ("Montserrat", "Poppins", "Inter", "Roboto", "Open Sans", "Lato")
        fallback_stack = (preferred,) + bold_stack if preferred else bold_stack

    for family in fallback_stack:
        if family and family.lower() in catalog:
            return str(catalog[family.lower()].get("family") or family)
    return None


def pick_google_font_variant(item: dict[str, Any], bold: bool, weight: int | None = None) -> tuple[str, str] | None:
    files = item.get("files", {})
    if not isinstance(files, dict) or not files:
        return None
    variants = [str(variant) for variant in item.get("variants", [])]
    requested_weight = int(weight or (700 if bold else 400))
    if requested_weight >= 700:
        preferred = [str(requested_weight), "700", "800", "600", "regular"]
    else:
        preferred = [str(requested_weight), "regular", "400", "500", "300"]
    for variant in preferred + variants:
        url = files.get(variant)
        if isinstance(url, str) and url:
            return variant, url.replace("http://", "https://")
    first_variant, first_url = next(iter(files.items()))
    return str(first_variant), str(first_url).replace("http://", "https://")


def get_google_font_file(requested_family: str | None, bold: bool, category: str | None = None, weight: int | None = None) -> Path | None:
    if not google_fonts_enabled():
        return None
    family = pick_google_font_family(requested_family, bold, category)
    if not family:
        return None
    requested_weight = int(weight or (700 if bold else 400))
    cache_key = (family.lower(), str(requested_weight), "bold" if bold else "regular")
    if cache_key in GOOGLE_FONT_FILE_CACHE:
        return GOOGLE_FONT_FILE_CACHE[cache_key]

    item = load_google_fonts_catalog().get(family.lower())
    if not item:
        GOOGLE_FONT_FILE_CACHE[cache_key] = None
        return None
    selected = pick_google_font_variant(item, bold, requested_weight)
    if not selected:
        GOOGLE_FONT_FILE_CACHE[cache_key] = None
        return None

    variant, url = selected
    suffix = Path(url.split("?", 1)[0]).suffix or ".ttf"
    target = font_cache_dir() / f"{sanitize_font_cache_name(family)}-{sanitize_font_cache_name(variant)}{suffix}"
    if not target.exists() or target.stat().st_size == 0:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            target.write_bytes(response.content)
        except Exception as exc:
            print(f"[fonts] Google font download failed for {family}/{variant}: {exc}", flush=True)
            GOOGLE_FONT_FILE_CACHE[cache_key] = None
            return None

    GOOGLE_FONT_FILE_CACHE[cache_key] = target
    return target


def cleanup_old_temp_files() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for child in temp_root().iterdir():
        try:
            modified = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        except OSError:
            continue


def manifest_path(job_dir: Path) -> Path:
    return job_dir / MANIFEST_NAME


def read_manifest(job_dir: Path) -> dict[str, Any]:
    return json.loads(manifest_path(job_dir).read_text(encoding="utf-8"))


def write_manifest(job_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path(job_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def image_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS]


def normalize_output_format(value: str, source_path: Path | None = None) -> str:
    normalized = value.strip().lower()
    if normalized == "original":
        if source_path and source_path.suffix.lower() in IMAGE_EXTENSIONS:
            return source_path.suffix.lower().lstrip(".")
        return "png"
    if normalized == "jpg":
        return "jpeg"
    if normalized not in {"png", "jpeg", "webp", "pdf"}:
        return "png"
    return normalized


def fit_for_ocr(image: Image.Image) -> tuple[Image.Image, float]:
    max_side = int(os.getenv("ADAPTIFAI_OCR_MAX_SIDE", "1280" if is_cpu_runtime() else "2200"))
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    if scale >= 1.0:
        return image, 1.0
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS), scale


def load_ocr_models():
    global OCR_DETECTOR, TROCR_MODEL, TROCR_PROCESSOR

    with OCR_MODEL_LOCK:
        if OCR_DETECTOR is None:
            import easyocr

            OCR_DETECTOR = easyocr.Reader(["en"], gpu=not is_cpu_runtime(), verbose=False)

        if TROCR_PROCESSOR is None or TROCR_MODEL is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            model_id = os.getenv("ADAPTIFAI_TROCR_MODEL", "microsoft/trocr-base-printed")
            TROCR_PROCESSOR = TrOCRProcessor.from_pretrained(model_id)
            TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained(model_id)
            TROCR_MODEL.to(torch_device())
            TROCR_MODEL.eval()

    return OCR_DETECTOR, TROCR_PROCESSOR, TROCR_MODEL


def load_ocr_detector():
    global OCR_DETECTOR

    with OCR_MODEL_LOCK:
        if OCR_DETECTOR is None:
            import easyocr

            OCR_DETECTOR = easyocr.Reader(["en"], gpu=not is_cpu_runtime(), verbose=False)

    return OCR_DETECTOR


def get_huggingface_cleanup_token() -> str:
    return (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )


def get_huggingface_cleanup_session() -> requests.Session:
    global HF_CLEANUP_SESSION

    with HF_CLEANUP_SESSION_LOCK:
        if HF_CLEANUP_SESSION is None:
            HF_CLEANUP_SESSION = requests.Session()
    return HF_CLEANUP_SESSION


async def persist_uploads(files: list[UploadFile], job_dir: Path) -> list[Path]:
    saved: list[Path] = []
    seen_names: set[str] = set()
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename}")

        # Preserve the original filename (sanitised) so source_name matches the
        # frontend file.name and output filenames are human-readable.
        original_stem = Path(upload.filename or "upload").stem
        safe_stem = re.sub(r"[^a-zA-Z0-9._-]", "-", original_stem).strip("-")[:80] or "upload"
        candidate = f"{safe_stem}{suffix}"
        # Avoid collisions when multiple files share the same name
        counter = 1
        while candidate in seen_names or (job_dir / candidate).exists():
            candidate = f"{safe_stem}-{counter}{suffix}"
            counter += 1
        seen_names.add(candidate)
        target = job_dir / candidate
        with target.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                handle.write(chunk)
        saved.append(target)
    return expand_uploads(saved, job_dir)


def expand_uploads(paths: list[Path], job_dir: Path) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.suffix.lower() == ".zip":
            expanded.extend(extract_zip_images(path, job_dir / "zip"))
        elif path.suffix.lower() == ".pdf":
            expanded.extend(render_pdf_pages(path, job_dir / "pdf"))
        else:
            expanded.append(path)
    return expanded


def extract_zip_images(zip_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for index, member in enumerate(archive.infolist()):
            suffix = Path(member.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS or member.is_dir():
                continue
            target = output_dir / f"{zip_path.stem}-{index}{suffix}"
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(target)
    return extracted


def render_pdf_pages(pdf_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return []

    rendered: list[Path] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    max_pages = int(os.getenv("ADAPTIFAI_PDF_MAX_PAGES", "8"))
    for index in range(min(len(pdf), max_pages)):
        page = pdf[index]
        bitmap = page.render(scale=2).to_pil().convert("RGB")
        target = output_dir / f"{pdf_path.stem}-page-{index + 1}.png"
        bitmap.save(target)
        rendered.append(target)
    return rendered


def classify_text_role(text: str) -> str:
    normalized = text.lower()
    if any(hint in normalized for hint in MARKETING_HINTS):
        return "cta" if len(text.split()) <= 6 else "headline"
    if text.isupper() and len(text.split()) <= 3:
        return "product_label"
    if any(char.isdigit() for char in text) and len(text.split()) <= 4 and "%" not in text:
        return "product_label"
    return "headline" if len(text) > 10 else "product_label"


def overlap_fraction(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    inter_left = max(left[0], right[0])
    inter_top = max(left[1], right[1])
    inter_right = min(left[2], right[2])
    inter_bottom = min(left[3], right[3])
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
    return intersection / left_area


def text_tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", normalize_ocr_text(text)) if len(token) >= 3}


def loose_word_tokens(text: str) -> set[str]:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_text)}


PACKAGING_CUE_TERMS = (
    "dr scholl",
    "scholl",
    "deodorant",
    "deo",
    "chauss",
    "shoe",
    "schuh",
    "geruch",
    "odeur",
    "protection",
    "technolog",
    "48h",
    "24h",
    "instant",
    "bioderma",
    "photoderm",
    "sensibio",
    "atoderm",
    "sebium",
    "sÃ©bium",
    "serum",
    "bioder",
    "boder",
    "toder",
    "naos",
    "laboratoire",
    "dermatologique",
    "spf",
    "uva",
    "uvb",
    "uvr",
    "fluid",
    "fluide",
    "ultra fluid",
    "ultra-fluid",
    "xdefense",
    "invisible",
    "sun active",
    "active defense",
    "ecobiology",
    "eco biology",
    "ml",
    "fl oz",
    "fl.oz",
)


def has_packaging_cues(text: str) -> bool:
    normalized = normalize_match_text(text)
    return any(normalize_match_text(cue) in normalized for cue in PACKAGING_CUE_TERMS)


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def normalize_match_text(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def infer_alignment(bbox: tuple[int, int, int, int], image_width: int) -> str:
    left, _, right, _ = bbox
    center = (left + right) / 2
    width = right - left
    if abs(center - image_width / 2) <= image_width * 0.12 and width <= image_width * 0.72:
        return "center"
    return "left"


def estimate_font_size_from_bbox(bbox: tuple[int, int, int, int]) -> int:
    return max(10, int((bbox[3] - bbox[1]) * 0.72))


def estimate_line_height_from_bbox(bbox: tuple[int, int, int, int]) -> int:
    return max(12, int((bbox[3] - bbox[1]) * 0.96))


def color_distance(left: str, right: str) -> float:
    def parse_hex(value: str) -> tuple[int, int, int]:
        cleaned = value.lstrip("#")
        if len(cleaned) != 6:
            return (17, 17, 17)
        return tuple(int(cleaned[index:index + 2], 16) for index in range(0, 6, 2))

    lr, lg, lb = parse_hex(left)
    rr, rg, rb = parse_hex(right)
    return float(((lr - rr) ** 2 + (lg - rg) ** 2 + (lb - rb) ** 2) ** 0.5)


def compute_clean_box(bbox: tuple[int, int, int, int], image_size: tuple[int, int], *, large_block: bool = False) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    pad_x = min(max(12, int(width * (0.16 if large_block else 0.1))), 42 if large_block else 28)
    pad_y = min(max(10, int(height * (0.32 if large_block else 0.22))), 30 if large_block else 18)
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_size[0], right + pad_x),
        min(image_size[1], bottom + pad_y),
    )


def merge_blocks_into_lines(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not blocks:
        return []

    image_width, _ = image_size
    ordered = sorted(blocks, key=lambda block: ((block.bbox[1] + block.bbox[3]) / 2, block.bbox[0]))
    groups: list[list[TextBlock]] = []

    for block in ordered:
        left, top, right, bottom = block.bbox
        center_y = (top + bottom) / 2
        height = bottom - top
        matched_group: list[TextBlock] | None = None
        for group in groups:
            group_top = min(item.bbox[1] for item in group)
            group_bottom = max(item.bbox[3] for item in group)
            group_left = min(item.bbox[0] for item in group)
            group_right = max(item.bbox[2] for item in group)
            group_center_y = (group_top + group_bottom) / 2
            group_height = max(1, group_bottom - group_top)
            vertical_close = abs(center_y - group_center_y) <= max(height, group_height) * 0.6
            horizontal_gap = max(0, left - group_right, group_left - right)
            horizontal_close = horizontal_gap <= image_width * 0.1
            if vertical_close and horizontal_close:
                matched_group = group
                break
        if matched_group is None:
            groups.append([block])
        else:
            matched_group.append(block)

    merged: list[TextBlock] = []
    for group in groups:
        ordered_group = sorted(group, key=lambda item: item.bbox[0])
        left = min(item.bbox[0] for item in ordered_group)
        top = min(item.bbox[1] for item in ordered_group)
        right = max(item.bbox[2] for item in ordered_group)
        bottom = max(item.bbox[3] for item in ordered_group)
        text = " ".join(item.text.strip() for item in ordered_group if item.text.strip())
        if not text:
            continue
        align = infer_alignment((left, top, right, bottom), image_width)
        merged.append(
            TextBlock(
                text=text,
                role=classify_text_role(text),
                translate=True,
                bbox=(left, top, right, bottom),
                color=ordered_group[0].color,
                font_weight=max(item.font_weight for item in ordered_group),
                align=align,
                surface=ordered_group[0].surface,
                line_boxes=[item.bbox for item in ordered_group if item.surface == "overlay"],
                line_texts=[item.text for item in ordered_group if item.text.strip()],
            )
        )
    return merged


def semantic_group_blocks(tokens: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not tokens:
        return []

    image_width, _ = image_size
    ordered = sorted(tokens, key=lambda token: (token.bbox[1], token.bbox[0]))
    groups: list[list[TextBlock]] = []

    for token in ordered:
        token_height = max(1, token.bbox[3] - token.bbox[1])
        token_size = estimate_font_size_from_bbox(token.bbox)
        token_center_x = (token.bbox[0] + token.bbox[2]) / 2
        matched: list[TextBlock] | None = None
        for group in groups:
            last = group[-1]
            group_bbox = union_bbox([item.bbox for item in group]) or token.bbox
            group_center_x = (group_bbox[0] + group_bbox[2]) / 2
            vertical_gap = token.bbox[1] - last.bbox[3]
            last_height = max(1, last.bbox[3] - last.bbox[1])
            avg_line_height = (last_height + token_height) / 2
            avg_font_size = (
                sum(estimate_font_size_from_bbox(item.bbox) for item in group) / max(1, len(group))
            )
            same_surface = token.surface == group[0].surface
            same_translate = token.translate == group[0].translate
            align_match = token.align == group[0].align
            if token.align == "center":
                anchor_close = abs(token_center_x - group_center_x) <= max(20, image_width * 0.055)
            else:
                anchor_close = abs(token.bbox[0] - group_bbox[0]) <= max(22, int(avg_font_size * 1.7), int(image_width * 0.032))
            line_spacing_continuity = -max(6, avg_line_height * 0.15) <= vertical_gap <= max(28, avg_line_height * 1.05)
            font_similarity = abs(token_size - estimate_font_size_from_bbox(last.bbox)) <= max(8, int(token_size * 0.35))
            weight_similarity = abs(token.font_weight - last.font_weight) <= 200
            color_similarity = color_distance(token.color, last.color) <= 64
            reading_order = token.bbox[1] <= group_bbox[3] + max(28, int(avg_line_height * 1.25))
            if same_surface and same_translate and align_match and anchor_close and line_spacing_continuity and font_similarity and weight_similarity and color_similarity and reading_order:
                matched = group
                break
        if matched is None:
            groups.append([token])
        else:
            matched.append(token)

    semantic_blocks: list[TextBlock] = []
    for index, group in enumerate(groups):
        bbox = union_bbox([item.bbox for item in group]) or group[0].bbox
        ordered_group = sorted(group, key=lambda item: (item.bbox[1], item.bbox[0]))
        line_texts = [item.text.strip() for item in ordered_group if item.text.strip()]
        text = "\n".join(line_texts)
        font_sizes = [estimate_font_size_from_bbox(item.bbox) for item in ordered_group]
        line_heights = [estimate_line_height_from_bbox(item.bbox) for item in ordered_group]
        semantic_blocks.append(
            TextBlock(
                id=f"block-{index + 1}",
                text=text,
                role=group[0].role if len(group) == 1 else "headline",
                translate=group[0].translate,
                bbox=bbox,
                clean_box=compute_clean_box(bbox, image_size, large_block=(bbox[2] - bbox[0]) > image_size[0] * 0.45),
                font_family=group[0].font_family,
                font_weight=max(item.font_weight for item in group),
                font_size_estimate=int(round(sum(font_sizes) / max(1, len(font_sizes)))),
                line_height_estimate=int(round(sum(line_heights) / max(1, len(line_heights)))),
                color=group[0].color,
                align=group[0].align,
                surface=group[0].surface,
                line_boxes=[item.bbox for item in ordered_group],
                line_texts=line_texts,
            )
        )
    return semantic_blocks


def text_similarity(left: str, right: str) -> float:
    left_normalized = normalize_ocr_text(left)
    right_normalized = normalize_ocr_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    return SequenceMatcher(a=left_normalized, b=right_normalized).ratio()


def merge_centered_stacks(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not blocks:
        return []

    image_width, _ = image_size
    ordered = sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
    merged: list[TextBlock] = []
    index = 0

    while index < len(ordered):
        current = ordered[index]
        group = [current]
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            current_left, current_top, current_right, current_bottom = group[-1].bbox
            candidate_left, candidate_top, candidate_right, candidate_bottom = candidate.bbox
            group_center = ((current_left + current_right) / 2 + (candidate_left + candidate_right) / 2) / 2
            center_close = abs(((candidate_left + candidate_right) / 2) - group_center) <= image_width * 0.08
            gap = candidate_top - current_bottom
            width_overlap = min(current_right, candidate_right) - max(current_left, candidate_left)
            min_width = max(1, min(current_right - current_left, candidate_right - candidate_left))
            overlap_ratio = width_overlap / min_width
            can_merge = (
                current.translate
                and candidate.translate
                and current.align == "center"
                and candidate.align == "center"
                and center_close
                and gap >= -6
                and gap <= max(current_bottom - current_top, candidate_bottom - candidate_top) * 1.35
                and overlap_ratio >= 0.25
                and len(group) < 3
            )
            if not can_merge:
                break
            group.append(candidate)
            next_index += 1

        if len(group) == 1:
            merged.append(current)
        else:
            left = min(item.bbox[0] for item in group)
            top = min(item.bbox[1] for item in group)
            right = max(item.bbox[2] for item in group)
            bottom = max(item.bbox[3] for item in group)
            merged.append(
                current.model_copy(
                    update={
                        "text": "\n".join(item.text for item in group),
                        "bbox": (left, top, right, bottom),
                        "font_weight": max(item.font_weight for item in group),
                        "translated_text": None,
                        "line_boxes": [box for item in group for box in (item.line_boxes or [item.bbox] if item.surface == "overlay" else [])],
                        "line_texts": [text for item in group for text in (item.line_texts or [item.text]) if text.strip()],
                    }
                )
            )
        index = next_index

    return merged


def merge_translate_runs(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not blocks:
        return []

    image_width, _ = image_size
    ordered = sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
    merged: list[TextBlock] = []
    index = 0

    while index < len(ordered):
        current = ordered[index]
        if not current.translate:
            merged.append(current)
            index += 1
            continue

        group = [current]
        next_index = index + 1
        while next_index < len(ordered):
            candidate = ordered[next_index]
            if not candidate.translate:
                break
            if candidate.align != current.align:
                break

            prev_left, prev_top, prev_right, prev_bottom = group[-1].bbox
            cand_left, cand_top, cand_right, cand_bottom = candidate.bbox
            prev_width = max(1, prev_right - prev_left)
            cand_width = max(1, cand_right - cand_left)
            avg_height = ((prev_bottom - prev_top) + (cand_bottom - cand_top)) / 2
            gap = cand_top - prev_bottom

            if current.align == "left":
                anchor_close = abs(cand_left - group[0].bbox[0]) <= image_width * 0.12
            else:
                prev_center = (prev_left + prev_right) / 2
                cand_center = (cand_left + cand_right) / 2
                anchor_close = abs(cand_center - prev_center) <= image_width * 0.1

            width_overlap = min(prev_right, cand_right) - max(prev_left, cand_left)
            overlap_ratio = width_overlap / max(1, min(prev_width, cand_width))
            can_merge = (
                anchor_close
                and gap >= -max(14, avg_height * 0.5)
                and gap <= max(24, avg_height * 0.42)
                and (overlap_ratio >= 0.1 or min(prev_width, cand_width) >= image_width * 0.2)
                and len(group) < 3
            )
            if not can_merge:
                break
            group.append(candidate)
            next_index += 1

        if len(group) == 1:
            merged.append(current)
        else:
            left = min(item.bbox[0] for item in group)
            top = min(item.bbox[1] for item in group)
            right = max(item.bbox[2] for item in group)
            bottom = max(item.bbox[3] for item in group)
            merged.append(
                current.model_copy(
                    update={
                        "text": "\n".join(item.text for item in group),
                        "bbox": (left, top, right, bottom),
                        "font_weight": max(item.font_weight for item in group),
                        "translated_text": None,
                        "role": "headline",
                        "line_boxes": [box for item in group for box in (item.line_boxes or [item.bbox] if item.surface == "overlay" else [])],
                        "line_texts": [text for item in group for text in (item.line_texts or [item.text]) if text.strip()],
                    }
                )
            )
        index = next_index

    return merged


def block_area_ratio(block: TextBlock, image_size: tuple[int, int]) -> float:
    width, height = image_size
    area = max(1, (block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]))
    return area / max(1, width * height)


def should_split_oversized_overlay_block(block: TextBlock, image_size: tuple[int, int]) -> bool:
    if block.surface != "overlay" or not block.translate:
        return False
    line_count = max(
        len([line for line in (block.text or "").splitlines() if line.strip()]),
        len(block.line_boxes),
        len(block.line_texts),
    )
    if line_count < 5:
        return False
    if block_area_ratio(block, image_size) >= 0.16:
        return True
    block_height = max(1, block.bbox[3] - block.bbox[1])
    return block_height / max(1, image_size[1]) >= 0.34


def should_translate_split_overlay_line(
    text: str,
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None,
) -> bool:
    cleaned = normalize_ocr_text(text)
    if not cleaned:
        return False
    image_width, image_height = image_size
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    area_ratio = (width * height) / max(1, image_width * image_height)
    overlap_with_foreground = overlap_fraction(bbox, foreground_bbox) if foreground_bbox else 0.0
    if has_packaging_cues(text) and not is_instructional_context_text(text):
        return False
    if overlap_with_foreground >= 0.18 and area_ratio <= 0.045:
        return False
    if overlap_with_foreground >= 0.34:
        return False
    if height < max(8, image_height * 0.012) and area_ratio < 0.006:
        return False
    return True


def split_oversized_overlay_blocks(
    blocks: list[TextBlock],
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> list[TextBlock]:
    split_blocks: list[TextBlock] = []
    for block in blocks:
        if not should_split_oversized_overlay_block(block, image_size):
            split_blocks.append(block)
            continue

        line_boxes = list(block.line_boxes or [])
        line_texts = [line.strip() for line in (block.line_texts or []) if line.strip()]
        if not line_texts:
            line_texts = [line.strip() for line in (block.text or "").splitlines() if line.strip()]
        if not line_boxes or len(line_boxes) != len(line_texts):
            split_blocks.append(block)
            continue

        for line_index, (line_text, line_box) in enumerate(zip(line_texts, line_boxes), start=1):
            translate_line = should_translate_split_overlay_line(line_text, line_box, image_size, foreground_bbox)
            role = classify_text_role(line_text)
            split_blocks.append(
                block.model_copy(
                    update={
                        "id": f"{block.id or 'overlay'}-{line_index}",
                        "text": line_text,
                        "role": role,
                        "translate": translate_line,
                        "surface": "overlay" if translate_line else "packaging",
                        "bbox": line_box,
                        "clean_box": compute_clean_box(line_box, image_size),
                        "font_size_estimate": estimate_font_size_from_bbox(line_box),
                        "line_height_estimate": estimate_line_height_from_bbox(line_box),
                        "translated_text": None,
                        "align": infer_alignment(line_box, image_size[0]),
                        "line_boxes": [line_box] if translate_line else [],
                        "line_texts": [line_text] if translate_line else [],
                    }
                )
            )
    return sorted(split_blocks, key=lambda item: (item.bbox[1], item.bbox[0]))


def dedupe_and_filter_translate_blocks(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    translate_blocks = [block for block in blocks if block.translate and block.surface == "overlay"]
    survivors: list[TextBlock] = []
    for block in blocks:
        normalized = normalize_ocr_text(block.text)
        if block.translate and block.surface == "overlay":
            if len(normalized) <= 2 and not re.search(r"\d|%", block.text):
                survivors.append(block.model_copy(update={"translate": False, "surface": "packaging", "role": "product_label", "line_boxes": [], "line_texts": []}))
                continue
            is_duplicate_fragment = False
            for other in translate_blocks:
                if other is block:
                    continue
                other_normalized = normalize_ocr_text(other.text)
                if not other_normalized or len(other_normalized) <= len(normalized):
                    continue
                text_contained = normalized in other_normalized or text_similarity(block.text, other.text) >= 0.76
                layout_close = (
                    overlap_fraction(block.bbox, other.bbox) >= 0.12
                    or overlap_fraction(other.bbox, block.bbox) >= 0.35
                    or (
                        abs(block.bbox[0] - other.bbox[0]) <= image_size[0] * 0.16
                        and abs(block.bbox[2] - other.bbox[2]) <= image_size[0] * 0.2
                        and abs(block.bbox[1] - other.bbox[3]) <= image_size[1] * 0.12
                    )
                )
                if text_contained and layout_close:
                    is_duplicate_fragment = True
                    break
            if is_duplicate_fragment:
                continue
        survivors.append(block)
    return sorted(survivors, key=lambda item: (item.bbox[1], item.bbox[0]))


def merge_overlapping_translate_blocks(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    ordered = sorted(blocks, key=lambda item: (item.bbox[1], item.bbox[0]))
    merged: list[TextBlock] = []
    consumed: set[int] = set()
    for index, block in enumerate(ordered):
        if index in consumed:
            continue
        if not (block.translate and block.surface == "overlay"):
            merged.append(block)
            continue
        group = [block]
        consumed.add(index)
        for other_index in range(index + 1, len(ordered)):
            other = ordered[other_index]
            if other_index in consumed or not (other.translate and other.surface == "overlay"):
                continue
            bbox_overlap = max(overlap_fraction(block.bbox, other.bbox), overlap_fraction(other.bbox, block.bbox))
            line_overlap = 0.0
            for left_box in block.line_boxes or [block.bbox]:
                for right_box in other.line_boxes or [other.bbox]:
                    line_overlap = max(line_overlap, overlap_fraction(left_box, right_box), overlap_fraction(right_box, left_box))
            same_region = bbox_overlap >= 0.28 or line_overlap >= 0.55
            vertical_neighbor = (
                abs(block.bbox[0] - other.bbox[0]) <= image_size[0] * 0.12
                and max(0, other.bbox[1] - block.bbox[3], block.bbox[1] - other.bbox[3]) <= image_size[1] * 0.12
            )
            if same_region or vertical_neighbor:
                group.append(other)
                consumed.add(other_index)
        if len(group) == 1:
            merged.append(block)
            continue
        bbox = union_bbox([item.bbox for item in group]) or block.bbox
        texts: list[str] = []
        for item in sorted(group, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0])):
            for line in str(item.text).splitlines():
                cleaned_line = line.strip()
                if cleaned_line and not any(text_similarity(cleaned_line, existing) >= 0.78 for existing in texts):
                    texts.append(cleaned_line)
        line_pairs: list[tuple[tuple[int, int, int, int], str]] = []
        for item in group:
            for line_box, line_text in zip(item.line_boxes or [item.bbox], item.line_texts or [item.text]):
                if not any(overlap_fraction(line_box, existing_box) >= 0.75 for existing_box, _ in line_pairs):
                    line_pairs.append((line_box, line_text))
        line_pairs = sorted(line_pairs, key=lambda pair: (pair[0][1], pair[0][0]))
        merged.append(
            block.model_copy(
                update={
                    "text": "\n".join(texts) or block.text,
                    "bbox": bbox,
                    "clean_box": compute_clean_box(bbox, image_size, large_block=(bbox[2] - bbox[0]) > image_size[0] * 0.45),
                    "translated_text": None,
                    "line_boxes": [pair[0] for pair in line_pairs],
                    "line_texts": [pair[1] for pair in line_pairs],
                }
            )
        )
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def explode_ocr_blocks_to_lines(blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    exploded: list[TextBlock] = []
    for block in blocks:
        if block.line_boxes and block.line_texts and len(block.line_boxes) == len(block.line_texts):
            for index, (line_box, line_text) in enumerate(zip(block.line_boxes, block.line_texts), start=1):
                line_text = line_text.strip()
                if not line_text:
                    continue
                exploded.append(
                    block.model_copy(
                        update={
                            "id": f"{block.id or 'ocr'}-line-{index}",
                            "text": line_text,
                            "bbox": line_box,
                            "clean_box": compute_clean_box(line_box, image_size),
                            "role": classify_text_role(line_text),
                            "font_size_estimate": estimate_font_size_from_bbox(line_box),
                            "line_height_estimate": estimate_line_height_from_bbox(line_box),
                            "align": infer_alignment(line_box, image_size[0]),
                            "line_boxes": [line_box],
                            "line_texts": [line_text],
                        }
                    )
                )
        else:
            exploded.append(block)
    return sorted(exploded, key=lambda item: (item.bbox[1], item.bbox[0]))


def append_missing_translate_ocr_blocks(refined_blocks: list[TextBlock], source_blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not source_blocks:
        return refined_blocks

    image_width, image_height = image_size
    combined = list(refined_blocks)
    packaging_blocks = [block for block in combined if block.surface in {"packaging", "product"} or not block.translate]
    for source in source_blocks:
        if not source.translate:
            continue
        normalized_source = normalize_ocr_text(source.text)
        if not normalized_source:
            continue
        source_width = max(1, source.bbox[2] - source.bbox[0])
        source_height = max(1, source.bbox[3] - source.bbox[1])
        source_area_ratio = (source_width * source_height) / max(1, image_width * image_height)
        if should_split_oversized_overlay_block(source, image_size) or (
            source_area_ratio >= 0.16 and len(source.line_boxes) >= 3
        ):
            for line_index, (line_box, line_text) in enumerate(zip(source.line_boxes, source.line_texts), start=1):
                line_text = line_text.strip()
                if not line_text:
                    continue
                already_line_present = any(
                    text_similarity(line_text, block.text) >= 0.72
                    or (
                        len(normalize_ocr_text(line_text)) <= 10
                        and text_similarity(line_text, block.text) >= 0.54
                    )
                    or any(text_similarity(line_text, existing_line) >= 0.72 for existing_line in block.line_texts)
                    or any(
                        len(normalize_ocr_text(line_text)) <= 10
                        and text_similarity(line_text, existing_line) >= 0.54
                        for existing_line in block.line_texts
                    )
                    or any(
                        len(normalize_ocr_text(line_text)) <= 10
                        and text_similarity(line_text, existing_part) >= 0.54
                        for existing_line in ([block.text] + list(block.line_texts))
                        for existing_part in str(existing_line).splitlines()
                        if existing_part.strip()
                    )
                    or (
                        block.translate
                        and block.surface == "overlay"
                        and (
                            bool(text_tokens(line_text) & text_tokens(block.text))
                            or text_similarity(line_text, block.text) >= 0.35
                        )
                        and (
                            overlap_fraction(line_box, block.bbox) >= 0.10
                            or overlap_fraction(block.bbox, line_box) >= 0.10
                        )
                    )
                    for block in combined
                )
                if already_line_present:
                    continue
                line_overlaps_packaging = any(
                    overlap_fraction(line_box, block.bbox) >= 0.35
                    or overlap_fraction(block.bbox, line_box) >= 0.55
                    for block in packaging_blocks
                )
                if line_overlaps_packaging or has_packaging_cues(line_text):
                    continue
                line_width = max(1, line_box[2] - line_box[0])
                line_height = max(1, line_box[3] - line_box[1])
                if line_width < image_width * 0.08 and "%" not in line_text:
                    continue
                combined.append(
                    source.model_copy(
                        update={
                            "id": f"{source.id or 'missing-ocr'}-{line_index}",
                            "text": line_text,
                            "bbox": line_box,
                            "clean_box": compute_clean_box(line_box, image_size, large_block=line_width > image_width * 0.45),
                            "role": classify_text_role(line_text),
                            "surface": "overlay",
                            "translate": True,
                            "translated_text": None,
                            "align": infer_alignment(line_box, image_width),
                            "font_size_estimate": estimate_font_size_from_bbox(line_box),
                            "line_height_estimate": estimate_line_height_from_bbox(line_box),
                            "line_boxes": [line_box],
                            "line_texts": [line_text],
                        }
                    )
                )
            packaging_blocks = [block for block in combined if block.surface in {"packaging", "product"} or not block.translate]
            continue
        already_present = any(
            text_similarity(source.text, block.text) >= 0.72
            or any(text_similarity(source.text, line_text) >= 0.72 for line_text in block.line_texts)
            for block in combined
        )
        block_width = source.bbox[2] - source.bbox[0]
        block_height = source.bbox[3] - source.bbox[1]
        source_tokens = text_tokens(source.text)
        existing_translate_tokens = set().union(*(text_tokens(block.text) for block in combined if block.translate)) if combined else set()
        existing_token_coverage = len(source_tokens & existing_translate_tokens) / max(1, len(source_tokens))
        overlaps_existing_layout = any(
            overlap_fraction(source.bbox, block.bbox) >= 0.18
            or overlap_fraction(block.bbox, source.bbox) >= 0.42
            for block in combined
        )
        if (
            source_area_ratio >= 0.18
            and (existing_token_coverage >= 0.25 or overlaps_existing_layout)
        ):
            continue
        looks_salient = (
            block_width >= image_width * 0.18
            or "%" in source.text
            or source.text.isupper()
            or block_height >= 28
        )
        overlaps_packaging = any(
            overlap_fraction(source.bbox, block.bbox) >= 0.35
            or overlap_fraction(block.bbox, source.bbox) >= 0.55
            for block in packaging_blocks
        )
        if already_present or not looks_salient or overlaps_packaging:
            continue
        combined.append(source)
    return sorted(combined, key=lambda block: (block.bbox[1], block.bbox[0]))


def annotate_ocr_tokens_with_vision(source_blocks: list[TextBlock], vision_blocks: list[TextBlock]) -> list[TextBlock]:
    if not vision_blocks:
        return source_blocks

    annotated: list[TextBlock] = []
    for source in source_blocks:
        best_match: TextBlock | None = None
        best_score = 0.0
        for vision in vision_blocks:
            score = text_similarity(source.text, vision.text)
            if score > best_score:
                best_score = score
                best_match = vision
        if best_match and best_score >= 0.42:
            annotated.append(
                source.model_copy(
                    update={
                        "translate": best_match.translate,
                        "surface": best_match.surface,
                        "role": best_match.role,
                        "align": best_match.align if best_match.align in {"left", "center"} else source.align,
                    }
                )
            )
        else:
            annotated.append(source)
    return annotated


def suppress_packaging_translation(blocks: list[TextBlock], image_size: tuple[int, int], foreground_bbox: tuple[int, int, int, int] | None) -> list[TextBlock]:
    image_width, image_height = image_size
    updated: list[TextBlock] = []
    for block in blocks:
        block_width = max(1, block.bbox[2] - block.bbox[0])
        block_height = max(1, block.bbox[3] - block.bbox[1])
        overlap_with_foreground = overlap_fraction(block.bbox, foreground_bbox) if foreground_bbox else 0.0
        normalized = block.text.lower()
        word_count = len([part for part in re.split(r"\s+", normalized) if part])
        packaging_cues = has_packaging_cues(block.text)
        likely_packaging = (
            block.surface in {"packaging", "product"}
            or (
                block.surface != "overlay"
                and overlap_with_foreground >= 0.78
                and (
                    (
                        block.role == "product_label"
                        and block_width <= image_width * 0.38
                        and block_height <= image_height * 0.1
                    )
                    or (
                        packaging_cues
                        and word_count >= 4
                        and block_width <= image_width * 0.5
                        and block_height <= image_height * 0.2
                    )
                )
            )
            or (
                overlap_with_foreground >= 0.55
                and packaging_cues
                and word_count >= 1
                and block_width <= image_width * 0.5
                and block_height <= image_height * 0.2
            )
        )
        updated.append(block.model_copy(update={"translate": False if likely_packaging else block.translate}))
    return updated


def snap_blocks_to_ocr_bboxes(refined_blocks: list[TextBlock], source_blocks: list[TextBlock]) -> list[TextBlock]:
    if not refined_blocks or not source_blocks:
        return refined_blocks

    snapped: list[TextBlock] = []
    used_indices: set[int] = set()
    for refined in refined_blocks:
        if refined.align == "center":
            snapped.append(refined)
            continue
        refined_lines = [line.strip() for line in refined.text.splitlines() if line.strip()]
        if not refined_lines:
            snapped.append(refined)
            continue

        matched_indices: list[int] = []
        for line in refined_lines:
            best_index = None
            best_score = 0.0
            for index, source in enumerate(source_blocks):
                if index in used_indices:
                    continue
                score = text_similarity(line, source.text)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None and best_score >= 0.38:
                matched_indices.append(best_index)
                used_indices.add(best_index)

        if matched_indices:
            boxes = [source_blocks[index].bbox for index in matched_indices]
            left = min(box[0] for box in boxes)
            top = min(box[1] for box in boxes)
            right = max(box[2] for box in boxes)
            bottom = max(box[3] for box in boxes)
            color = source_blocks[matched_indices[0]].color
            align = source_blocks[matched_indices[0]].align
            snapped.append(
                refined.model_copy(
                    update={
                        "bbox": (left, top, right, bottom),
                        "color": color,
                        "align": align,
                        "line_boxes": [source_blocks[index].bbox for index in matched_indices],
                        "line_texts": [source_blocks[index].text for index in matched_indices if source_blocks[index].text.strip()],
                    }
                )
            )
        else:
            snapped.append(refined)

    return snapped


def estimate_text_color(image: Image.Image) -> str:
    pixels = np.array(image.convert("RGB")).reshape(-1, 3)
    if len(pixels) == 0:
        return "#111111"
    brightness = pixels.mean(axis=1).astype(np.float32)
    saturation = (pixels.max(axis=1) - pixels.min(axis=1)).astype(np.float32)
    dark_pixels = pixels[brightness <= np.quantile(brightness, 0.1)]
    bright_pixels = pixels[brightness >= np.quantile(brightness, 0.9)]
    saturated_pixels = pixels[saturation >= np.quantile(saturation, 0.9)]
    background_level = float(np.median(brightness))

    if len(saturated_pixels) and float(saturation.max()) >= 28 and background_level >= 170:
        color = saturated_pixels.mean(axis=0)
    elif float(np.quantile(brightness, 0.95)) >= 242 and background_level >= 150:
        return "#ffffff"
    elif background_level >= 170 and len(dark_pixels):
        color = dark_pixels.mean(axis=0)
    elif background_level <= 100 and len(bright_pixels):
        color = bright_pixels.mean(axis=0)
    else:
        dark_distance = abs(float(dark_pixels.mean()) - background_level) if len(dark_pixels) else 0.0
        bright_distance = abs(float(bright_pixels.mean()) - background_level) if len(bright_pixels) else 0.0
        color = dark_pixels.mean(axis=0) if dark_distance >= bright_distance and len(dark_pixels) else (bright_pixels.mean(axis=0) if len(bright_pixels) else pixels.mean(axis=0))

    if color.mean() > 220:
        return "#ffffff"
    if color.mean() < 50:
        return "#111111"
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in color))


def sample_deterministic_text_color(image: Image.Image, bbox: tuple[int, int, int, int], fallback: str = "#111111") -> str:
    left, top, right, bottom = bbox
    left = max(0, min(image.width, left))
    right = max(0, min(image.width, right))
    top = max(0, min(image.height, top))
    bottom = max(0, min(image.height, bottom))
    if right <= left or bottom <= top:
        return fallback if re.fullmatch(r"#[0-9a-fA-F]{6}", fallback or "") else "#111111"
    crop = np.array(image.crop((left, top, right, bottom)).convert("RGB"), dtype=np.uint8)
    pixels = crop.reshape(-1, 3)
    if len(pixels) == 0:
        return fallback if re.fullmatch(r"#[0-9a-fA-F]{6}", fallback or "") else "#111111"
    brightness = pixels.mean(axis=1).astype(np.float32)
    chroma = (pixels.max(axis=1) - pixels.min(axis=1)).astype(np.float32)
    bg_brightness = float(np.median(brightness))
    bg_chroma = float(np.median(chroma))
    non_background = pixels[~((brightness >= 225) & (chroma <= 22))]
    if len(non_background) >= 8:
        pixels = non_background
        brightness = pixels.mean(axis=1).astype(np.float32)
        chroma = (pixels.max(axis=1) - pixels.min(axis=1)).astype(np.float32)
    saturated = pixels[(chroma >= 28) & (brightness <= 230)]
    dark = pixels[brightness <= min(165, bg_brightness - 18 if bg_brightness >= 180 else 120)]
    if len(saturated) >= 8:
        candidates = saturated
    elif len(dark) >= 8:
        candidates = dark
    elif bg_brightness >= 180:
        candidates = pixels[(brightness <= bg_brightness - 12) | (chroma >= max(16, bg_chroma + 8))]
    elif bg_brightness <= 90:
        candidates = pixels[(brightness >= bg_brightness + 12) | (chroma >= max(16, bg_chroma + 8))]
    else:
        candidates = pixels[np.abs(brightness - bg_brightness) >= 10]
    if len(candidates) < 8:
        candidates = pixels[np.argsort(chroma + np.abs(brightness - bg_brightness))[-max(8, min(len(pixels), len(pixels) // 8)) :]]
    rounded = (candidates.astype(np.uint16) // 8) * 8
    colors, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    return "#{:02x}{:02x}{:02x}".format(*(int(min(255, value)) for value in dominant))


def source_word_id(block_id: str | None, line_index: int, word_index: int, text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", normalize_ocr_text(text)).strip("-") or "word"
    return f"{block_id or 'block'}-l{line_index + 1}-w{word_index + 1}-{normalized[:24]}"


def split_line_text_to_word_boxes(text: str, box: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    words = [match for match in re.finditer(r"\S+", text or "")]
    if not words:
        return []
    left, top, right, bottom = box
    width = max(1, right - left)
    text_len = max(1, len(text))
    output: list[dict[str, Any]] = []
    for index, match in enumerate(words):
        start_ratio = match.start() / text_len
        end_ratio = max(start_ratio, match.end() / text_len)
        word_left = int(round(left + width * start_ratio))
        word_right = int(round(left + width * end_ratio))
        if word_right <= word_left:
            word_right = min(right, word_left + max(2, width // max(1, len(words))))
        output.append({"text": match.group(0), "wordIndex": index, "bbox": (word_left, top, min(right, word_right), bottom)})
    return output


def choose_raw_foreground_mask(crop_rgb: np.ndarray) -> np.ndarray:
    import cv2

    if crop_rgb.size == 0:
        return np.zeros(crop_rgb.shape[:2], dtype=bool)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        otsu = np.where(gray >= np.median(gray), 255, 0).astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        max(3, min(31, (min(gray.shape[:2]) // 2) * 2 + 1)),
        3,
    )
    h, w = gray.shape[:2]
    border = np.concatenate([crop_rgb[0, :, :], crop_rgb[-1, :, :], crop_rgb[:, 0, :], crop_rgb[:, -1, :]], axis=0)
    bg_median = np.median(border, axis=0).astype(np.float32) if len(border) else np.median(crop_rgb.reshape(-1, 3), axis=0).astype(np.float32)
    candidates = [
        otsu == 0,
        otsu == 255,
        adaptive == 0,
        adaptive == 255,
    ]
    best_mask = np.zeros((h, w), dtype=bool)
    best_score = -1.0
    crop_area = max(1, h * w)
    for mask in candidates:
        count = int(mask.sum())
        ratio = count / crop_area
        if count < 2 or ratio <= 0.004 or ratio >= 0.88:
            continue
        pixels = crop_rgb[mask].astype(np.float32)
        fg_median = np.median(pixels, axis=0)
        distance = float(np.linalg.norm(fg_median - bg_median))
        chroma = float(np.median(pixels.max(axis=1) - pixels.min(axis=1)))
        compactness_penalty = abs(ratio - 0.18) * 18
        score = distance + chroma * 0.35 - compactness_penalty
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def split_character_and_graphic_contours(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if mask.size == 0:
        empty = mask.astype(bool)
        return empty, empty
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = np.zeros_like(binary)
    graphics = np.zeros_like(binary)
    crop_h, crop_w = binary.shape[:2]
    crop_area = max(1, crop_h * crop_w)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 1.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        rect_area = max(1, w * h)
        aspect = w / max(1, h)
        fill_ratio = area / rect_area
        extent_ratio = rect_area / crop_area
        hull = cv2.convexHull(contour)
        hull_area = max(1.0, float(cv2.contourArea(hull)))
        solidity = area / hull_area

        long_thin_horizontal = aspect >= 5.5 and h <= max(3, crop_h * 0.22)
        long_thin_vertical = aspect <= 0.16 and w <= max(3, crop_w * 0.08) and h >= crop_h * 0.55
        large_frame_or_bar = extent_ratio >= 0.32 and (aspect >= 4.0 or aspect <= 0.22)
        solid_geometric_bar = fill_ratio >= 0.72 and solidity >= 0.82 and (aspect >= 4.0 or aspect <= 0.25)
        spans_most_width = w >= crop_w * 0.78 and h <= crop_h * 0.34

        if long_thin_horizontal or long_thin_vertical or large_frame_or_bar or solid_geometric_bar or spans_most_width:
            cv2.drawContours(graphics, [contour], -1, 255, thickness=-1)
            continue
        if area / crop_area > 0.46:
            cv2.drawContours(graphics, [contour], -1, 255, thickness=-1)
            continue
        cv2.drawContours(characters, [contour], -1, 255, thickness=-1)
    return characters > 0, graphics > 0


def filter_character_like_contours(mask: np.ndarray) -> np.ndarray:
    characters, _graphics = split_character_and_graphic_contours(mask)
    return characters


def choose_foreground_text_mask(crop_rgb: np.ndarray) -> np.ndarray:
    return filter_character_like_contours(choose_raw_foreground_mask(crop_rgb))


def anti_ghost_text_dilation_px(image_size: tuple[int, int]) -> int:
    configured = os.getenv("ADAPTIFAI_TEXT_ANTIGHOST_DILATION_PX", "").strip()
    if configured:
        try:
            return max(3, min(8, int(configured)))
        except ValueError:
            pass
    return max(6, min(8, int(round(max(image_size) * 0.014))))


def analyze_text_decoration_traits(raw_mask: np.ndarray, text_mask: np.ndarray) -> dict[str, bool]:
    """Detect simple word-level typography traits from foreground geometry."""
    try:
        import cv2
    except Exception:
        return {"isItalic": False, "isUnderlined": False, "isStrikethrough": False}

    if raw_mask.size == 0 or not raw_mask.any():
        return {"isItalic": False, "isUnderlined": False, "isStrikethrough": False}

    raw = raw_mask.astype(np.uint8) * 255
    text = text_mask.astype(np.uint8) * 255 if text_mask.size == raw_mask.size else raw
    height, width = raw.shape[:2]
    min_line_width = max(6, int(width * 0.45))
    max_line_height = max(2, int(height * 0.16))
    underline = False
    strike = False
    contours, _hierarchy = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_line_width or h > max_line_height:
            continue
        aspect = w / max(1, h)
        if aspect < 5.0:
            continue
        center_y = y + h / 2.0
        if center_y >= height * 0.72:
            underline = True
        elif height * 0.38 <= center_y <= height * 0.66:
            strike = True

    italic = False
    if text.any():
        text_contours, _hierarchy = cv2.findContours(text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        slopes: list[float] = []
        for contour in text_contours:
            if cv2.contourArea(contour) < 4:
                continue
            points = contour.reshape(-1, 2)
            if len(points) < 5:
                continue
            ys = points[:, 1].astype(np.float32)
            xs = points[:, 0].astype(np.float32)
            if float(ys.max() - ys.min()) < max(5.0, height * 0.35):
                continue
            try:
                slope = float(np.polyfit(ys, xs, 1)[0])
            except Exception:
                continue
            slopes.append(slope)
        if slopes:
            median_slope = float(np.median(slopes))
            same_direction = sum(1 for slope in slopes if (slope < 0) == (median_slope < 0)) / max(1, len(slopes))
            italic = len(slopes) >= 2 and same_direction >= 0.72 and abs(median_slope) >= 0.32

    return {"isItalic": bool(italic), "isUnderlined": bool(underline), "isStrikethrough": bool(strike)}


def sample_word_foreground_style(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    fallback_color: str = "#111111",
    fallback_weight: int = 700,
) -> dict[str, Any]:
    left, top, right, bottom = expand_bbox(bbox, image.size, 1)
    if right <= left or bottom <= top:
        return {"color": fallback_color, "fontWeight": fallback_weight, "isBold": fallback_weight >= 700}
    crop_rgb = np.array(image.crop((left, top, right, bottom)).convert("RGB"), dtype=np.uint8)
    raw_mask = choose_raw_foreground_mask(crop_rgb)
    mask = filter_character_like_contours(raw_mask)
    traits = analyze_text_decoration_traits(raw_mask, mask)
    if mask.any():
        import cv2

        row_has_ink = mask.any(axis=1)
        clusters: list[tuple[int, int]] = []
        start: int | None = None
        for row_index, has_ink in enumerate(row_has_ink):
            if has_ink and start is None:
                start = row_index
            elif not has_ink and start is not None:
                clusters.append((start, row_index))
                start = None
        if start is not None:
            clusters.append((start, len(row_has_ink)))
        if len(clusters) > 1:
            fallback_rgb = np.array(parse_hex_color(fallback_color), dtype=np.float32)

            def cluster_score(item: tuple[int, int]) -> float:
                cluster_mask = mask[item[0] : item[1], :]
                pixels = crop_rgb[item[0] : item[1], :, :][cluster_mask]
                if len(pixels) == 0:
                    return -9999.0
                median_rgb = np.median(pixels.astype(np.float32), axis=0)
                color_distance = float(np.linalg.norm(median_rgb - fallback_rgb))
                area_score = min(240.0, float(cluster_mask.sum()))
                upper_bias = max(0.0, 40.0 * (1.0 - item[0] / max(1, mask.shape[0])))
                return area_score + upper_bias - color_distance * 1.6

            best_start, best_end = max(clusters, key=cluster_score)
            clustered = np.zeros_like(mask, dtype=bool)
            clustered[best_start:best_end, :] = mask[best_start:best_end, :]
            mask = clustered
        mask_u8 = mask.astype(np.uint8) * 255
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        core_mask = cv2.erode(mask_u8, erosion_kernel, iterations=1) > 0
        if int(core_mask.sum()) < max(3, int(mask.sum() * 0.18)):
            core_mask = mask
        pixels = crop_rgb[core_mask].reshape(-1, 3)
        bg_color = np.median(crop_rgb.reshape(-1, 3), axis=0)
        distances = np.linalg.norm(pixels.astype(np.float32) - bg_color.astype(np.float32), axis=1)
        if len(distances) >= 8:
            pixels = pixels[distances >= np.percentile(distances, 35)]
        median = np.median(pixels, axis=0)
        color = "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(channel)))) for channel in median))
        density = float(mask.sum()) / max(1, mask.shape[0] * mask.shape[1])
        core_density = float(core_mask.sum()) / max(1, mask.shape[0] * mask.shape[1])
        ys, xs = np.where(mask)
        foreground_bbox = (
            left + int(xs.min()),
            top + int(ys.min()),
            left + int(xs.max()) + 1,
            top + int(ys.max()) + 1,
        )
    else:
        color = sample_deterministic_text_color(image, bbox, fallback_color)
        density = 0.0
        core_density = 0.0
        foreground_bbox = bbox
    is_bold = fallback_weight >= 700 or density >= 0.24
    return {
        "color": color,
        "fontWeight": 700 if is_bold else 400,
        "isBold": is_bold,
        "fontCategory": "sans-serif",
        "foregroundDensity": round(density, 4),
        "coreForegroundDensity": round(core_density, 4),
        "foregroundBbox": foreground_bbox,
        **traits,
    }


def build_source_word_styles(block: TextBlock, image: Image.Image) -> list[dict[str, Any]]:
    line_boxes = list(block.line_boxes or []) or [block.bbox]
    line_texts = list(block.line_texts or []) or [block.text]
    styles: list[dict[str, Any]] = []
    for line_index, line_box in enumerate(line_boxes):
        line_text = line_texts[line_index] if line_index < len(line_texts) else block.text
        word_boxes = split_line_text_to_word_boxes(line_text, tuple(line_box))
        for word_index, word in enumerate(word_boxes):
            text = str(word["text"]).strip()
            if not text:
                continue
            style = sample_word_foreground_style(
                image,
                tuple(word["bbox"]),
                fallback_color=block.color,
                fallback_weight=block.font_weight,
            )
            styles.append(
                {
                    "id": source_word_id(block.id, line_index, word_index, text),
                    "text": text,
                    "lineIndex": line_index,
                    "wordIndex": word_index,
                    "bbox": list(style.get("foregroundBbox") or word["bbox"]),
                    "estimatedBbox": list(word["bbox"]),
                    "color": style["color"],
                    "fontWeight": style["fontWeight"],
                    "fontCategory": style.get("fontCategory", "sans-serif"),
                    "isBold": style["isBold"],
                    "isUppercase": text.upper() == text and any(char.isalpha() for char in text),
                    "isItalic": bool(style.get("isItalic")),
                    "isUnderlined": bool(style.get("isUnderlined")),
                    "isStrikethrough": bool(style.get("isStrikethrough")),
                    "semanticRole": classify_semantic_role(text),
                    "foregroundDensity": style.get("foregroundDensity", 0.0),
                    "coreForegroundDensity": style.get("coreForegroundDensity", 0.0),
                }
            )
    return styles


def sample_polygon_foreground_style(
    image: Image.Image,
    polygon: list[tuple[int, int]],
    *,
    fallback_color: str = "#111111",
    fallback_weight: int = 700,
) -> dict[str, Any]:
    import cv2

    if len(polygon) < 3:
        return {"color": fallback_color, "fontWeight": fallback_weight, "isBold": fallback_weight >= 700}
    envelope = expand_bbox(bbox_from_polygon(polygon), image.size, 3)
    left, top, right, bottom = envelope
    if right <= left or bottom <= top:
        return {"color": fallback_color, "fontWeight": fallback_weight, "isBold": fallback_weight >= 700}
    crop_rgb = np.array(image.crop(envelope).convert("RGB"), dtype=np.uint8)
    local_polygon = [(max(0, x - left), max(0, y - top)) for x, y in polygon]
    polygon_region = np.array(polygon_mask((right - left, bottom - top), [local_polygon]).convert("L")) > 0
    if not polygon_region.any():
        return sample_word_foreground_style(image, envelope, fallback_color=fallback_color, fallback_weight=fallback_weight)
    raw_foreground = choose_raw_foreground_mask(crop_rgb)
    foreground = raw_foreground & polygon_region
    if int(foreground.sum()) < 3:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        local = gray[polygon_region]
        if local.size:
            delta = np.abs(gray.astype(np.float32) - float(np.median(local)))
            foreground = (delta >= max(4.0, float(np.percentile(delta[polygon_region], 70)))) & polygon_region
    if int(foreground.sum()) < 3:
        foreground = polygon_region
    foreground_u8 = foreground.astype(np.uint8) * 255
    core = cv2.erode(foreground_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
    if int(core.sum()) < max(3, int(foreground.sum() * 0.12)):
        core = foreground
    pixels = crop_rgb[core].reshape(-1, 3)
    if pixels.size == 0:
        pixels = crop_rgb[foreground].reshape(-1, 3)
    bg_pixels = crop_rgb[polygon_region & ~foreground]
    bg_color = np.median(bg_pixels.reshape(-1, 3), axis=0) if bg_pixels.size else np.median(crop_rgb.reshape(-1, 3), axis=0)
    bg_hex = "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(channel)))) for channel in bg_color))
    distances = np.linalg.norm(pixels.astype(np.float32) - bg_color.astype(np.float32), axis=1)
    if len(distances) >= 8:
        pixels = pixels[distances >= np.percentile(distances, 35)]
    median = np.median(pixels, axis=0)
    color = "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(channel)))) for channel in median))
    region_pixels = crop_rgb[polygon_region].reshape(-1, 3)
    region_luma = region_pixels.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    region_chroma = region_pixels.max(axis=1).astype(np.int16) - region_pixels.min(axis=1).astype(np.int16)
    region_dist = np.linalg.norm(region_pixels.astype(np.float32) - bg_color.astype(np.float32), axis=1)
    density = float(foreground.sum()) / max(1, int(polygon_region.sum()))
    core_density = float(core.sum()) / max(1, int(polygon_region.sum()))
    fg_ys, fg_xs = np.where(foreground)
    if fg_xs.size and fg_ys.size:
        foreground_bbox = (
            left + int(fg_xs.min()),
            top + int(fg_ys.min()),
            left + int(fg_xs.max()) + 1,
            top + int(fg_ys.max()) + 1,
        )
    else:
        foreground_bbox = bbox_from_polygon(polygon)
    height = max(1, foreground_bbox[3] - foreground_bbox[1])
    font_size = max(8, int(height))
    bg_hex = "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(channel)))) for channel in bg_color))
    bg_luma = color_luminance(bg_hex)
    word_box = bbox_from_polygon(polygon)
    ring_box = expand_bbox(word_box, image.size, max(8, int((word_box[3] - word_box[1]) * 0.75)))
    ring_crop = np.array(image.crop(ring_box).convert("RGB"), dtype=np.uint8)
    ring_local_box = (
        max(0, word_box[0] - ring_box[0]),
        max(0, word_box[1] - ring_box[1]),
        min(ring_crop.shape[1], word_box[2] - ring_box[0]),
        min(ring_crop.shape[0], word_box[3] - ring_box[1]),
    )
    ring_mask = np.ones(ring_crop.shape[:2], dtype=bool)
    if ring_local_box[2] > ring_local_box[0] and ring_local_box[3] > ring_local_box[1]:
        ring_mask[ring_local_box[1]:ring_local_box[3], ring_local_box[0]:ring_local_box[2]] = False
    ring_pixels = ring_crop[ring_mask]
    ring_color = np.median(ring_pixels.reshape(-1, 3), axis=0) if ring_pixels.size else np.median(ring_crop.reshape(-1, 3), axis=0)
    background_contrast = float(np.linalg.norm(bg_color.astype(np.float32) - ring_color.astype(np.float32)))
    contrast_mask = (
        (region_luma <= max(230.0, bg_luma - 18.0))
        & (region_chroma >= 18)
        & (region_dist >= 24)
    )
    contrast_pixels = region_pixels[contrast_mask]
    contrast_fraction = float(len(contrast_pixels)) / max(1, len(region_pixels))
    contrast_color = color
    if len(contrast_pixels) >= 8:
        contrast_median = np.median(contrast_pixels, axis=0)
        contrast_color = "#{:02x}{:02x}{:02x}".format(*(int(max(0, min(255, round(channel)))) for channel in contrast_median))
    contrast_rgb = parse_hex_color(contrast_color)
    green_dominant_solid = contrast_rgb[1] >= max(contrast_rgb[0], contrast_rgb[2]) and (
        contrast_rgb[1] - contrast_rgb[0] >= 18 or contrast_rgb[1] - contrast_rgb[2] >= 18
    )
    outline_like = (
        height >= 24
        and bg_luma >= 235
        and 0.02 <= contrast_fraction <= 0.42
        and color_chroma(contrast_color) >= 18
        and not green_dominant_solid
    )
    traits = analyze_text_decoration_traits(raw_foreground & polygon_region, foreground)
    stroke_width = max(2, min(9, int(round(height * 0.09)))) if outline_like else 0
    if outline_like:
        color = contrast_color
    is_bold = fallback_weight >= 700 or density >= 0.24 or outline_like
    text_luma = color_luminance(color)
    has_text_background = (
        text_luma >= 205
        and bg_luma <= 235
        and color_chroma(bg_hex) >= 26
        and background_contrast >= 32.0
        and np.linalg.norm(np.array(parse_hex_color(color), dtype=np.float32) - np.array(parse_hex_color(bg_hex), dtype=np.float32)) >= 42
    )
    if has_text_background:
        crop_f = crop_rgb.astype(np.float32)
        bg_rgb = np.array(parse_hex_color(bg_hex), dtype=np.float32)
        text_rgb = np.array(parse_hex_color(color), dtype=np.float32)
        local_luma = crop_f[:, :, 0] * 0.299 + crop_f[:, :, 1] * 0.587 + crop_f[:, :, 2] * 0.114
        local_bg_distance = np.linalg.norm(crop_f - bg_rgb, axis=2)
        if text_luma >= 205:
            refined_foreground = (local_luma >= text_luma - 52.0) & (local_bg_distance >= 20.0) & polygon_region
        elif text_luma <= 80:
            refined_foreground = (local_luma <= text_luma + 52.0) & (local_bg_distance >= 20.0) & polygon_region
        else:
            refined_foreground = (np.linalg.norm(crop_f - text_rgb, axis=2) <= 72.0) & (local_bg_distance >= 20.0) & polygon_region
        refined_ys, refined_xs = np.where(refined_foreground)
        if refined_xs.size and refined_ys.size and int(refined_foreground.sum()) >= 3:
            foreground_bbox = (
                left + int(refined_xs.min()),
                top + int(refined_ys.min()),
                left + int(refined_xs.max()) + 1,
                top + int(refined_ys.max()) + 1,
            )
            height = max(1, foreground_bbox[3] - foreground_bbox[1])
            font_size = max(8, int(height))
    return {
        "color": color,
        "backgroundColor": bg_hex if has_text_background else None,
        "hasTextBackground": bool(has_text_background),
        "backgroundContrast": round(background_contrast, 3),
        "fontWeight": 700 if is_bold else 400,
        "fontSize": font_size,
        "lineHeight": max(font_size + 2, int(round(font_size * 1.12))),
        "isBold": is_bold,
        "fontCategory": "sans-serif",
        "foregroundDensity": round(density, 4),
        "coreForegroundDensity": round(core_density, 4),
        "foregroundBbox": foreground_bbox,
        "outline": bool(outline_like),
        "strokeWidth": stroke_width,
        "strokeFill": color if outline_like else None,
        **traits,
    }


def build_v5_polygon_source_word_styles(group: list[TextBlock], block: TextBlock, image: Image.Image) -> list[dict[str, Any]]:
    styles: list[dict[str, Any]] = []
    ordered = sorted(group, key=lambda item: (item.bbox[1], item.bbox[0]))
    for word_index, word in enumerate(ordered):
        text = word.text.strip()
        if not text or not word.polygon:
            continue
        line_index = 0
        for index, line_box in enumerate(block.line_boxes or []):
            word_center_y = (word.bbox[1] + word.bbox[3]) / 2
            if line_box[1] - 2 <= word_center_y <= line_box[3] + 2:
                line_index = index
                break
        style = sample_polygon_foreground_style(
            image,
            word.polygon,
            fallback_color=block.color,
            fallback_weight=word.font_weight or block.font_weight,
        )
        if (
            text.upper() == text
            and any(char.isalpha() for char in text)
            and not style.get("outline")
            and int(style.get("fontWeight") or 700) >= 700
            and int(style.get("fontSize") or 0) >= 18
        ):
            style["fontWeight"] = 800
        styles.append(
            {
                "id": source_word_id(block.id, line_index, word_index, text),
                "text": text,
                "lineIndex": line_index,
                "wordIndex": word_index,
                "bbox": list(style.get("foregroundBbox") or word.bbox),
                "estimatedBbox": list(word.bbox),
                "polygon": [list(point) for point in word.polygon],
                "symbolPolygons": [[list(point) for point in polygon] for polygon in (word.symbol_polygons or [word.polygon])],
                "color": style["color"],
                "backgroundColor": style.get("backgroundColor"),
                "hasTextBackground": bool(style.get("hasTextBackground")),
                "backgroundContrast": style.get("backgroundContrast", 0.0),
                "fontWeight": style["fontWeight"],
                "fontSize": style.get("fontSize", max(8, word.bbox[3] - word.bbox[1])),
                "lineHeight": style.get("lineHeight", max(10, int((word.bbox[3] - word.bbox[1]) * 1.12))),
                "fontCategory": style.get("fontCategory", "sans-serif"),
                "isBold": style["isBold"],
                "isUppercase": text.upper() == text and any(char.isalpha() for char in text),
                "isItalic": bool(style.get("isItalic")),
                "isUnderlined": bool(style.get("isUnderlined")),
                "isStrikethrough": bool(style.get("isStrikethrough")),
                "semanticRole": classify_semantic_role(text),
                "foregroundDensity": style.get("foregroundDensity", 0.0),
                "coreForegroundDensity": style.get("coreForegroundDensity", 0.0),
                "outline": bool(style.get("outline")),
                "strokeWidth": int(style.get("strokeWidth") or 0),
                "strokeFill": style.get("strokeFill"),
            }
        )
    return styles


def source_word_style_lookup(block: TextBlock) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    source_words = [word for word in (block.source_word_styles or []) if isinstance(word, dict)]
    for word in block.source_word_styles or []:
        if not isinstance(word, dict):
            continue
        if is_v5_isolated_step_marker_word(word, source_words):
            continue
        word_id = str(word.get("id") or "").strip()
        if word_id:
            lookup[word_id] = word
        text_key = normalize_ocr_text(str(word.get("text") or ""))
        if text_key:
            lookup.setdefault(text_key, word)
    return lookup


def color_dominance_score(color: str) -> float:
    rgb = np.array(parse_hex_color(color), dtype=np.float32)
    brightness = float(np.mean(rgb))
    chroma = float(np.max(rgb) - np.min(rgb))
    neutral_penalty = 40.0 if chroma < 18 and (brightness < 48 or brightness > 215) else 0.0
    return chroma * 2.0 + abs(brightness - 128.0) * 0.15 - neutral_penalty


def color_chroma(color: str) -> float:
    rgb = np.array(parse_hex_color(color), dtype=np.float32)
    return float(np.max(rgb) - np.min(rgb))


def color_luminance(color: str) -> float:
    rgb = np.array(parse_hex_color(color), dtype=np.float32)
    return float(np.dot(rgb, [0.299, 0.587, 0.114]))


def majority_source_style(source_styles: list[dict[str, Any]], base_style: dict[str, Any]) -> dict[str, Any]:
    if not source_styles:
        return dict(base_style)
    neutral_styles = [
        item
        for item in source_styles
        if color_chroma(str(item.get("color") or base_style.get("color") or "#111111")) < 32
    ]
    if len(neutral_styles) > len(source_styles) / 2:
        max_weight = max(int(item.get("fontWeight") or base_style.get("fontWeight") or 700) for item in neutral_styles)
        neutral = min(
            neutral_styles,
            key=lambda item: abs(float(np.mean(parse_hex_color(str(item.get("color") or base_style.get("color") or "#111111")))) - 96.0),
        )
        style = style_from_source_word_style(neutral, base_style)
        style["fontWeight"] = 700 if max_weight >= 700 else 400
        return style
    return dominant_source_style(source_styles, base_style)


def dominant_source_style(source_styles: list[dict[str, Any]], base_style: dict[str, Any]) -> dict[str, Any]:
    if not source_styles:
        return dict(base_style)
    dominant_color_style = max(source_styles, key=lambda item: color_dominance_score(str(item.get("color") or base_style["color"])))
    background_style = next(
        (
            item
            for item in source_styles
            if item.get("hasTextBackground") and item.get("backgroundColor")
        ),
        None,
    )
    max_weight = max(int(item.get("fontWeight") or base_style["fontWeight"]) for item in source_styles)
    uppercase_votes = sum(1 for item in source_styles if item.get("isUppercase"))
    dominant = dict(base_style)
    dominant.update(
        {
            "color": str(dominant_color_style.get("color") or base_style["color"]),
            "fontWeight": 700 if max_weight >= 700 else 400,
            "fontSize": max(
                8,
                int(
                    round(
                        max(
                            float(item.get("fontSize") or item.get("font_size") or base_style.get("fontSize") or 16)
                            for item in source_styles
                        )
                    )
                ),
            ),
            "lineHeight": max(
                10,
                int(
                    round(
                        max(
                            float(item.get("lineHeight") or item.get("line_height") or base_style.get("lineHeight") or 18)
                            for item in source_styles
                        )
                    )
                ),
            ),
            "casing": "uppercase" if uppercase_votes > len(source_styles) / 2 else "mixed",
            "fontCategory": str(dominant_color_style.get("fontCategory") or base_style.get("fontCategory") or "sans-serif"),
            "backgroundColor": background_style.get("backgroundColor") if background_style else None,
            "hasTextBackground": bool(background_style),
            "backgroundContrast": max(float(item.get("backgroundContrast") or 0.0) for item in source_styles),
            "strokeWidth": max(int(item.get("strokeWidth") or 0) for item in source_styles),
            "strokeFill": next((item.get("strokeFill") for item in source_styles if item.get("strokeFill")), None),
            "fillTransparent": any(bool(item.get("outline") or item.get("fillTransparent")) for item in source_styles),
            "isItalic": any(bool(item.get("isItalic")) for item in source_styles),
            "isUnderlined": any(bool(item.get("isUnderlined")) for item in source_styles),
            "isStrikethrough": any(bool(item.get("isStrikethrough")) for item in source_styles),
        }
    )
    return dominant


def source_styles_for_segment(raw: dict[str, Any], lookup: dict[str, dict[str, Any]], text: str, hint: str) -> list[dict[str, Any]]:
    raw_ids = raw.get("source_word_ids") or raw.get("sourceWordIds") or raw.get("source_word_id") or raw.get("sourceWordId") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    source_styles: list[dict[str, Any]] = []
    for source_id in raw_ids if isinstance(raw_ids, list) else []:
        source_id = str(source_id or "").strip()
        if source_id and source_id in lookup:
            source_styles.append(lookup[source_id])
    if not source_styles and hint:
        hinted = lookup.get(normalize_ocr_text(hint))
        if hinted:
            source_styles.append(hinted)
    if not source_styles:
        for token in text.split():
            style = lookup.get(normalize_ocr_text(token))
            if style:
                source_styles.append(style)
    return source_styles


def source_style_line_index(style: dict[str, Any]) -> int:
    try:
        return int(style.get("lineIndex") or 0)
    except Exception:
        return 0


def is_visual_heading_brand_block(block: TextBlock) -> bool:
    source_lines = [line.strip() for line in (block.line_texts or []) if str(line).strip()]
    if len(source_lines) < 2:
        return False
    first_line = source_lines[0]
    first_has_letters = any(char.isalpha() for char in first_line)
    first_is_heading = first_has_letters and first_line.upper() == first_line
    later_text = " ".join(source_lines[1:])
    return first_is_heading and has_packaging_cues(later_text)


def infer_v5_polygon_alignment(line_boxes: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], image_width: int) -> str:
    valid_boxes = [box for box in line_boxes if len(box) == 4 and box[2] > box[0] and box[3] > box[1]]
    if len(valid_boxes) >= 2:
        lefts = [box[0] for box in valid_boxes]
        if max(lefts) - min(lefts) <= max(8, int(image_width * 0.015)):
            return "left"
    return infer_alignment(bbox, image_width)


def preserve_visual_heading_span_order(block: TextBlock, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not is_visual_heading_brand_block(block) or len(spans) < 2:
        return spans

    def span_line_index(span: dict[str, Any]) -> int:
        styles = [style for style in (span.get("sourceWordStyles") or []) if isinstance(style, dict)]
        if not styles:
            return 0
        return min(source_style_line_index(style) for style in styles)

    ordered = sorted(enumerate(spans), key=lambda item: (span_line_index(item[1]), item[0]))
    result: list[dict[str, Any]] = []
    for index, (_original_index, span) in enumerate(ordered):
        current_line = span_line_index(span)
        next_line = span_line_index(ordered[index + 1][1]) if index < len(ordered) - 1 else current_line
        result.append({**span, "forceBreakAfter": index < len(ordered) - 1 and current_line != next_line})
    return result


def style_from_source_word_style(source_style: dict[str, Any], base_style: dict[str, Any]) -> dict[str, Any]:
    style = dict(base_style)
    color = str(source_style.get("color") or base_style.get("color") or "#111111")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = str(base_style.get("color") or "#111111")
    font_weight = int(source_style.get("fontWeight") or base_style.get("fontWeight") or 700)
    style.update(
        {
            "color": color,
            "fontWeight": 700 if font_weight >= 700 else 400,
            "fontSize": max(8, int(round(float(source_style.get("fontSize") or base_style.get("fontSize") or 16)))),
            "lineHeight": max(10, int(round(float(source_style.get("lineHeight") or base_style.get("lineHeight") or 18)))),
            "casing": "uppercase" if source_style.get("isUppercase") else "mixed",
            "fontCategory": normalize_font_category(source_style.get("fontCategory") or base_style.get("fontCategory")),
            "backgroundColor": source_style.get("backgroundColor"),
            "hasTextBackground": bool(source_style.get("hasTextBackground")),
            "backgroundContrast": float(source_style.get("backgroundContrast") or 0.0),
            "strokeWidth": int(source_style.get("strokeWidth") or 0),
            "strokeFill": source_style.get("strokeFill"),
            "fillTransparent": bool(source_style.get("outline") or source_style.get("fillTransparent")),
            "isItalic": bool(source_style.get("isItalic")),
            "isUnderlined": bool(source_style.get("isUnderlined")),
            "isStrikethrough": bool(source_style.get("isStrikethrough")),
        }
    )
    return style


def run_trocr_ocr_on_image(image_path: Path) -> list[TextBlock]:
    detector = load_ocr_detector()
    use_trocr = os.getenv("ADAPTIFAI_OCR_ENGINE", "easyocr").lower() == "trocr"
    if use_trocr and is_cpu_runtime() and os.getenv("ADAPTIFAI_ALLOW_CPU_TROCR") != "1":
        use_trocr = False

    processor = model = None
    if use_trocr:
        import torch

        _, processor, model = load_ocr_models()

    image = Image.open(image_path).convert("RGB")
    ocr_image, scale = fit_for_ocr(image)
    detections = detector.readtext(
        np.array(ocr_image),
        detail=1,
        paragraph=False,
        batch_size=int(os.getenv("ADAPTIFAI_OCR_BATCH_SIZE", "1")),
        width_ths=float(os.getenv("ADAPTIFAI_OCR_WIDTH_THS", "0.7")),
        decoder=os.getenv("ADAPTIFAI_EASYOCR_DECODER", "greedy"),
    )

    blocks: list[TextBlock] = []
    for points, detected_text, confidence in detections:
        if confidence < float(os.getenv("ADAPTIFAI_OCR_MIN_CONFIDENCE", "0.22")):
            continue

        xs = [int(point[0] / scale) for point in points]
        ys = [int(point[1] / scale) for point in points]
        padding = int(os.getenv("ADAPTIFAI_OCR_BOX_PADDING", "8"))
        left = max(0, min(xs) - padding)
        top = max(0, min(ys) - padding)
        right = min(image.width, max(xs) + padding)
        bottom = min(image.height, max(ys) + padding)
        if right <= left or bottom <= top:
            continue

        recognized = ""
        if use_trocr and processor is not None and model is not None:
            import torch

            crop = image.crop((left, top, right, bottom))
            pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(torch_device())
            with torch.inference_mode():
                generated_ids = model.generate(pixel_values, max_new_tokens=48)
            recognized = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = (recognized or str(detected_text)).strip()
        if not text:
            continue

        crop = image.crop((left, top, right, bottom))
        blocks.append(
            TextBlock(
                text=text,
                role=classify_text_role(text),
                translate=True,
                bbox=(left, top, right, bottom),
                color=estimate_text_color(crop),
                font_weight=800 if text.isupper() else 700,
                align=infer_alignment((left, top, right, bottom), image.width),
                line_boxes=[(left, top, right, bottom)],
                line_texts=[text],
            )
        )

    return merge_blocks_into_lines(blocks, image.size)


def run_resize_raw_ocr_on_image(image_path: Path) -> list[TextBlock]:
    image = Image.open(image_path).convert("RGB")
    vision_blocks = google_vision_word_blocks(image_path, image.size)
    if vision_blocks:
        return merge_blocks_into_lines(vision_blocks, image.size)

    allow_local_resize_ocr = os.getenv("ADAPTIFAI_ALLOW_LOCAL_RESIZE_OCR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not allow_local_resize_ocr:
        return []

    detector = load_ocr_detector()
    ocr_image, scale = fit_for_ocr(image)
    detections = detector.readtext(
        np.array(ocr_image),
        detail=1,
        paragraph=False,
        batch_size=int(os.getenv("ADAPTIFAI_OCR_BATCH_SIZE", "1")),
        width_ths=float(os.getenv("ADAPTIFAI_RESIZE_OCR_WIDTH_THS", "0.35")),
        decoder=os.getenv("ADAPTIFAI_EASYOCR_DECODER", "greedy"),
    )
    blocks: list[TextBlock] = []
    min_confidence = float(os.getenv("ADAPTIFAI_RESIZE_OCR_MIN_CONFIDENCE", "0.36"))
    for points, detected_text, confidence in detections:
        if confidence < min_confidence:
            continue
        text = str(detected_text or "").strip()
        if not text:
            continue
        xs = [int(point[0] / scale) for point in points]
        ys = [int(point[1] / scale) for point in points]
        padding = int(os.getenv("ADAPTIFAI_RESIZE_OCR_BOX_PADDING", "4"))
        left = max(0, min(xs) - padding)
        top = max(0, min(ys) - padding)
        right = min(image.width, max(xs) + padding)
        bottom = min(image.height, max(ys) + padding)
        if right <= left or bottom <= top:
            continue
        crop = image.crop((left, top, right, bottom))
        blocks.append(
            TextBlock(
                text=text,
                role=classify_text_role(text),
                translate=True,
                bbox=(left, top, right, bottom),
                color=estimate_text_color(crop),
                font_weight=800 if text.isupper() else 700,
                align=infer_alignment((left, top, right, bottom), image.width),
                line_boxes=[(left, top, right, bottom)],
                line_texts=[text],
            )
        )
    return blocks


def marketing_filter(blocks: list[TextBlock]) -> list[TextBlock]:
    filtered: list[TextBlock] = []
    for block in blocks:
        normalized = block.text.lower()
        bbox_width = max(1, block.bbox[2] - block.bbox[0])
        bbox_height = max(1, block.bbox[3] - block.bbox[1])
        is_tiny_dense = bbox_height <= 24 and len(block.text) >= 18
        looks_like_url = any(token in normalized for token in ("www.", ".com", ".de", ".fr", ".co", "http", "ask.naos"))
        looks_like_legal = any(token in normalized for token in ("subject", "days", "satisfaction", "monitoring", "study", "tested"))
        instructional_copy = normalized.startswith(("with ", "apply ", "cleanse ", "decode ", "share ", "connect "))
        if block.surface in {"packaging", "product"}:
            filtered.append(block.model_copy(update={"translate": False}))
            continue
        protected_label = any(
            token in normalized
            for token in (
                "ingredients",
                "ingredient",
                "niacinamide",
                "retinol",
                "spf",
                "ml",
                "oz",
                "sku",
                "item",
                "bioderma",
                "sensibio",
                "defensive serum",
                "dr.scholl",
                "scholl",
                "android tv",
                "ipad pro",
                "our projector",
            )
        )
        is_marketing = block.role in {"headline", "cta"} or any(hint in normalized for hint in MARKETING_HINTS) or "%" in block.text or any(
            token in normalized for token in ("skin", "moist", "luminous", "hydrat", "brighter", "glow", "wrinkle", "fresh", "technology", "audio", "bluetooth", "tragbar", "anwendung", "cleanse", "apply", "neu", "alt", "new", "old")
        )
        should_translate = ((is_marketing and not protected_label) or instructional_copy) and not looks_like_url and not looks_like_legal and not is_tiny_dense
        filtered.append(block.model_copy(update={"translate": should_translate}))
    return filtered


def refine_blocks_with_openai_vision(image_path: Path, blocks: list[TextBlock]) -> list[TextBlock]:
    if not blocks or not os.getenv("OPENAI_API_KEY"):
        return blocks

    try:
        source_image = Image.open(image_path).convert("RGB")
        image_width, image_height = source_image.size
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You review ad creatives for localization. Return compact JSON only with key `lines`. Each item must contain `text`, `translate`, `surface`, `x`, `y`, `w`, `h`, and optional `align`. `surface` must be one of `overlay`, `packaging`, or `product`. Use `overlay` for creative text added around/over the visual as marketing copy. Use `packaging` for printed text on product bottles, boxes, cans, labels, packaging, or stickers. Use `product` for text physically printed on a product/device/object. Coordinates are percentages from 0 to 100 relative to the whole image and should tightly cover the visible text line. Keep each visible line separate and in strict top-to-bottom order. Set translate=true only for primary marketing claims, feature callouts, CTAs, comparison labels, and instructional copy that are not printed on packaging/product surfaces. Set translate=false for product packaging labels, brand names, URLs, QR-related text, device labels, and tiny legal disclaimers. Use align='center' for centered copy, otherwise align='left'.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the visible text lines from this creative in reading order, with approximate text bounding boxes, translation flags, and whether each line is overlay marketing copy versus printed on packaging/product surfaces."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        raw_items = parsed.get("lines", [])
        if not isinstance(raw_items, list):
            return blocks

        vision_lines: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_items:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                payload = dict(item)
                payload["text"] = text
                payload["translate"] = bool(item.get("translate", False))
            else:
                text = str(item).strip()
                payload = {"text": text, "translate": True}
            normalized = normalize_ocr_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            vision_lines.append(payload)

        if not vision_lines:
            return blocks

        vision_blocks: list[TextBlock] = []
        for line in vision_lines:
            if not isinstance(line, dict):
                continue
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            x = max(0.0, min(100.0, float(line.get("x", 0))))
            y = max(0.0, min(100.0, float(line.get("y", 0))))
            w = max(1.0, min(100.0, float(line.get("w", 1))))
            h = max(1.0, min(100.0, float(line.get("h", 1))))
            left = int(round(image_width * x / 100))
            top = int(round(image_height * y / 100))
            right = int(round(image_width * min(100.0, x + w) / 100))
            bottom = int(round(image_height * min(100.0, y + h) / 100))
            if right <= left or bottom <= top:
                continue
            crop = source_image.crop((left, top, right, bottom))
            align = str(line.get("align", infer_alignment((left, top, right, bottom), image_width))).lower()
            if align not in {"left", "center"}:
                align = infer_alignment((left, top, right, bottom), image_width)
            if align == "center" and left <= image_width * 0.22 and (right - left) >= image_width * 0.22:
                align = "left"
            normalized = text.lower()
            surface = str(line.get("surface", "overlay")).lower()
            if surface not in {"overlay", "packaging", "product"}:
                surface = "overlay"
            translate = bool(line.get("translate", False))
            if normalized.startswith(("with ", "apply ", "cleanse ", "decode ", "share ", "connect ")) and not any(
                token in normalized for token in ("ask.naos", "www.", ".com", "http", "qr")
            ):
                translate = True
            if surface in {"packaging", "product"}:
                translate = False
            vision_blocks.append(
                TextBlock(
                    text=text,
                    role=classify_text_role(text),
                    translate=translate,
                    bbox=(left, top, right, bottom),
                    color=estimate_text_color(crop),
                    font_weight=800 if text.isupper() else 700,
                    align=align,
                    surface=surface,
                    line_boxes=[(left, top, right, bottom)] if surface == "overlay" else [],
                    line_texts=[text] if surface == "overlay" else [],
                )
            )

        if vision_blocks:
            return vision_blocks

        updated = [block.model_copy(update={"translate": False}) for block in blocks]
        for line in vision_lines:
            line_text = str(line.get("text", "")).strip()
            if not line_text:
                continue
            best_index = None
            best_score = 0.0
            for index, block in enumerate(updated):
                score = text_similarity(line_text, block.text)
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None:
                updated[best_index] = updated[best_index].model_copy(update={"text": line_text, "translate": bool(line.get("translate", False))})
        return updated
    except Exception:
        return blocks


def parse_structured_localize_blocks(raw_items: Any, source_image: Image.Image) -> list[TextBlock]:
    if not isinstance(raw_items, list):
        return []
    image_width, image_height = source_image.size
    blocks: list[TextBlock] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        key = normalize_ocr_text(text)
        if not key or key in seen:
            continue
        seen.add(key)

        x = max(0.0, min(100.0, float(item.get("x", 0))))
        y = max(0.0, min(100.0, float(item.get("y", 0))))
        w = max(1.0, min(100.0, float(item.get("w", 1))))
        h = max(1.0, min(100.0, float(item.get("h", 1))))
        left = int(round(image_width * x / 100))
        top = int(round(image_height * y / 100))
        right = int(round(image_width * min(100.0, x + w) / 100))
        bottom = int(round(image_height * min(100.0, y + h) / 100))
        if right <= left or bottom <= top:
            continue

        surface = str(item.get("surface", "overlay")).lower()
        if surface not in {"overlay", "packaging", "product"}:
            surface = "overlay"
        align = str(item.get("align", infer_alignment((left, top, right, bottom), image_width))).lower()
        if align not in {"left", "center"}:
            align = infer_alignment((left, top, right, bottom), image_width)

        translate = bool(item.get("translate", False)) and surface == "overlay"
        word_count = len([part for part in re.split(r"\s+", text) if part.strip()])
        area_ratio = ((right - left) * (bottom - top)) / max(1, image_width * image_height)
        if translate and area_ratio >= 0.20 and word_count >= 12:
            continue
        crop = source_image.crop((left, top, right, bottom))
        font_weight = 800 if str(item.get("font_weight", "")).lower() in {"bold", "700", "800", "900"} or text.isupper() else 700
        blocks.append(
            TextBlock(
                text=text,
                role=str(item.get("role", "headline" if translate else classify_text_role(text))),
                translate=translate,
                bbox=(left, top, right, bottom),
                color=estimate_text_color(crop),
                font_weight=font_weight,
                align=align,
                surface=surface,
                line_boxes=[(left, top, right, bottom)] if surface == "overlay" else [],
                line_texts=[text] if surface == "overlay" else [],
            )
        )
    return sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))


def extract_marketing_blocks_with_gemini_vision(image_path: Path) -> list[TextBlock]:
    if not vertex_available():
        return []
    try:
        source_image = Image.open(image_path).convert("RGB")
        prompt = {
            "task": "Analyze this paid-media creative for localization. Return compact JSON only with key blocks.",
            "schema": {
                "blocks": [
                    {
                        "text": "exact visible text, preserving intended line breaks",
                        "translate": "boolean; true only for overlay marketing copy outside product packaging",
                        "surface": "overlay | packaging | product",
                        "role": "headline | subhead | benefit | cta | disclaimer | product_label",
                        "x": "left percent 0-100",
                        "y": "top percent 0-100",
                        "w": "width percent 0-100",
                        "h": "height percent 0-100",
                        "align": "left | center",
                        "font_weight": "regular | bold",
                    }
                ]
            },
            "rules": [
                "Do not merge unrelated text regions into one block.",
                "Group only visual marketing copy that belongs to the same semantic message and is placed together.",
                "Never mark product packaging, bottle labels, brand logos, SKU text, ml/oz text, SPF/50+/UVA labels, URLs, QR references, legal microcopy, or product names as translate=true.",
                "For catalog/routine creatives, keep each step headline/claim as its own overlay block; do not sweep the product area into a huge block.",
                "Coordinates must tightly cover only the visible text block.",
            ],
        }
        parsed = generate_vertex_gemini_json(
            prompt,
            source_image,
            timeout=int(os.getenv("VERTEX_GEMINI_TIMEOUT", "45")),
        )
        return parse_structured_localize_blocks(parsed.get("blocks", []), source_image)
    except Exception as exc:
        print(f"[vertex] Gemini visual analysis failed: {exc}", flush=True)
        return []


def extract_marketing_blocks_with_openai_vision(image_path: Path) -> list[TextBlock]:
    if not os.getenv("OPENAI_API_KEY"):
        return []

    try:
        source_image = Image.open(image_path).convert("RGB")
        image_width, image_height = source_image.size
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze ad creatives for localization. Return compact JSON only with key `blocks`. "
                        "Each block must contain `text`, `translate`, `surface`, `x`, `y`, `w`, `h`, `align`, and optional `font_weight`. "
                        "`surface` must be one of `overlay`, `packaging`, or `product`. "
                        "Group visual marketing copy that belongs together into a single block and preserve intended line breaks in `text`. "
                        "Do not merge printed packaging/product labels into overlay blocks. "
                        "Set `translate=true` only for overlay marketing copy: headlines, CTA buttons, discount callouts, comparison labels like NEW/OLD, and instructional copy outside packaging. "
                        "Set `translate=false` for anything printed on bottles, boxes, product labels, devices, stickers, legal disclaimers, URLs, QR references, and brand/product names. "
                        "Coordinates are percentages from 0 to 100 and should tightly cover the whole visible text block, not tiny sub-lines."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify only the meaningful text blocks in this creative, separating overlay marketing copy from product or packaging text. Preserve line breaks for each overlay block."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")
        return parse_structured_localize_blocks(parsed.get("blocks", []), source_image)
    except Exception:
        return []


def snap_overlay_blocks_to_ocr(
    blocks: list[TextBlock],
    ocr_blocks: list[TextBlock],
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> list[TextBlock]:
    snapped: list[TextBlock] = []
    for block in blocks:
        if block.surface != "overlay":
            snapped.append(block)
            continue

        expanded = expand_bbox(block.bbox, image_size, max(28, int(max(block.bbox[2] - block.bbox[0], block.bbox[3] - block.bbox[1]) * 0.28)))
        normalized_lines = [normalize_ocr_text(part) for part in block.text.splitlines() if normalize_ocr_text(part)]
        normalized_block = normalize_ocr_text(block.text)
        candidates: list[TextBlock] = []
        for candidate in ocr_blocks:
            center_x = (candidate.bbox[0] + candidate.bbox[2]) / 2
            center_y = (candidate.bbox[1] + candidate.bbox[3]) / 2
            intersects = overlap_fraction(candidate.bbox, expanded) > 0.03 or (
                expanded[0] <= center_x <= expanded[2] and expanded[1] - 32 <= center_y <= expanded[3] + 32
            )
            if not intersects:
                continue
            candidate_normalized = normalize_ocr_text(candidate.text)
            if foreground_bbox and overlap_fraction(candidate.bbox, foreground_bbox) > 0.12 and has_packaging_cues(candidate.text):
                continue
            line_match = any(
                text_similarity(candidate_normalized, line) > 0.42
                or (len(candidate_normalized) >= 3 and candidate_normalized in line)
                or (len(line) >= 3 and line in candidate_normalized)
                for line in normalized_lines
            )
            if not line_match and candidate_normalized:
                line_match = (
                    len(candidate_normalized) >= 3
                    and candidate_normalized in normalized_block
                ) or text_similarity(candidate_normalized, normalized_block) > 0.45
            if not line_match:
                overlap = text_tokens(candidate.text) & text_tokens(block.text)
                line_match = len(overlap) >= 2
            weak_overlap_only_rejected = (
                has_packaging_cues(candidate.text)
                or len(normalize_ocr_text(candidate.text)) <= 2
                or bool(re.fullmatch(r"[\d\s%+.,:-]+", candidate.text.strip()))
            )
            if (
                not line_match
                and not weak_overlap_only_rejected
                and block.translate
                and candidate.translate
                and overlap_fraction(candidate.bbox, expanded) > 0.12
            ):
                line_match = True
            if line_match:
                candidates.append(candidate)

        if candidates:
            bbox = union_bbox([candidate.bbox for candidate in candidates])
            if bbox is not None:
                ordered_candidates = sorted(candidates, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))
                snapped.append(
                    block.model_copy(
                        update={
                            "bbox": bbox,
                            "align": infer_alignment(bbox, image_size[0]),
                            "text": block.text,
                            "line_boxes": [candidate.bbox for candidate in ordered_candidates],
                            "line_texts": [candidate.text.strip() for candidate in ordered_candidates if candidate.text.strip()],
                        }
                    )
                )
                continue
        snapped.append(block)
    return snapped


def stabilize_vision_blocks_with_ocr(
    vision_blocks: list[TextBlock],
    ocr_blocks: list[TextBlock],
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> list[TextBlock]:
    snapped = snap_overlay_blocks_to_ocr(vision_blocks, ocr_blocks, image_size, foreground_bbox)
    image_width, image_height = image_size
    image_area = max(1, image_width * image_height)
    stable: list[TextBlock] = []
    for block in snapped:
        bbox_area = max(1, block.bbox[2] - block.bbox[0]) * max(1, block.bbox[3] - block.bbox[1])
        overlap_with_foreground = overlap_fraction(block.bbox, foreground_bbox) if foreground_bbox else 0.0
        normalized = block.text.lower()
        likely_packaging = (
            block.surface in {"packaging", "product"}
            or has_packaging_cues(block.text)
            or (
                overlap_with_foreground >= 0.35
                and bbox_area <= image_area * 0.10
                and any(token in normalized for token in ("ml", "spf", "h2o", "dr.", "scholl", "bioderma", "sensibio"))
            )
        )
        updates: dict[str, Any] = {}
        if likely_packaging:
            updates.update({"translate": False, "surface": "packaging", "role": "product_label"})
        elif block.surface == "overlay" and not block.line_boxes:
            updates.update({"line_boxes": [block.bbox], "line_texts": [block.text]})
        if block.surface == "overlay" and bbox_area > image_area * 0.62 and len(block.text.split()) <= 4:
            updates.update({"translate": False, "role": "product_label"})
        stable.append(block.model_copy(update=updates) if updates else block)
    return sorted(stable, key=lambda item: (item.bbox[1], item.bbox[0]))


def build_localize_blocks(image_path: Path, source_image: Image.Image) -> list[TextBlock]:
    foreground_bbox = detect_foreground_bbox(source_image)
    raw_ocr_blocks = run_trocr_ocr_on_image(image_path)
    raw_line_blocks = explode_ocr_blocks_to_lines(raw_ocr_blocks, source_image.size)
    vision_block_blocks = extract_marketing_blocks_with_gemini_vision(image_path) or extract_marketing_blocks_with_openai_vision(image_path)
    if vision_block_blocks:
        stable_blocks = stabilize_vision_blocks_with_ocr(vision_block_blocks, raw_line_blocks, source_image.size, foreground_bbox)
        stable_blocks = merge_centered_stacks(stable_blocks, source_image.size)
        stable_blocks = merge_translate_runs(stable_blocks, source_image.size)
        stable_blocks = split_oversized_overlay_blocks(stable_blocks, source_image.size, foreground_bbox)
        stable_blocks = append_missing_translate_ocr_blocks(stable_blocks, raw_line_blocks, source_image.size)
        stable_blocks = suppress_packaging_translation(stable_blocks, source_image.size, foreground_bbox)
        stable_blocks = merge_overlapping_translate_blocks(stable_blocks, source_image.size)
        return dedupe_and_filter_translate_blocks(stable_blocks, source_image.size)

    vision_line_blocks = refine_blocks_with_openai_vision(image_path, raw_line_blocks)
    if vision_line_blocks:
        semantic_tokens = annotate_ocr_tokens_with_vision(raw_line_blocks, vision_line_blocks)
        grouped_blocks = semantic_group_blocks(semantic_tokens, source_image.size)
        grouped_blocks = merge_centered_stacks(grouped_blocks, source_image.size)
        grouped_blocks = merge_translate_runs(grouped_blocks, source_image.size)
        grouped_blocks = split_oversized_overlay_blocks(grouped_blocks, source_image.size, foreground_bbox)
        grouped_blocks = append_missing_translate_ocr_blocks(grouped_blocks, semantic_tokens, source_image.size)
        grouped_blocks = suppress_packaging_translation(grouped_blocks, source_image.size, foreground_bbox)
        grouped_blocks = merge_overlapping_translate_blocks(grouped_blocks, source_image.size)
        return dedupe_and_filter_translate_blocks(grouped_blocks, source_image.size)

    raw_blocks = marketing_filter(raw_line_blocks)
    semantic_tokens = annotate_ocr_tokens_with_vision(raw_blocks, refine_blocks_with_openai_vision(image_path, raw_blocks))
    grouped_blocks = semantic_group_blocks(semantic_tokens, source_image.size)
    grouped_blocks = merge_centered_stacks(grouped_blocks, source_image.size)
    grouped_blocks = merge_translate_runs(grouped_blocks, source_image.size)
    grouped_blocks = split_oversized_overlay_blocks(grouped_blocks, source_image.size, foreground_bbox)
    grouped_blocks = append_missing_translate_ocr_blocks(grouped_blocks, semantic_tokens, source_image.size)
    grouped_blocks = suppress_packaging_translation(grouped_blocks, source_image.size, foreground_bbox)
    grouped_blocks = merge_overlapping_translate_blocks(grouped_blocks, source_image.size)
    return dedupe_and_filter_translate_blocks(grouped_blocks, source_image.size)


def preprocess_image_for_localize(source_image: Image.Image) -> Image.Image:
    normalized = source_image.convert("RGB")
    return normalized.filter(ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=2))


def detect_source_language(blocks: list[TextBlock]) -> str:
    translate_text = "\n".join(block.text for block in blocks if block.translate and block.surface == "overlay").strip()
    if not translate_text:
        return "EN"
    if not os.getenv("OPENAI_API_KEY"):
        normalized = translate_text.lower()
        if any(token in normalized for token in ("und", "mit", "neu", "alt", "gro", "schuh")):
            return "DE"
        if any(token in normalized for token in ("avec", "pour", "votre", "chauss")):
            return "FR"
        return "EN"
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
            temperature=0,
            messages=[
                {"role": "system", "content": "Detect the source language of this marketing copy. Return compact JSON only with key `language` as one of EN, DE, FR, IT, ES, PT, TR, AR, ZH, JA."},
                {"role": "user", "content": json.dumps({"text": translate_text}, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        language = str(payload.get("language", "EN")).upper()
        return language if language in LANGUAGE_NAMES else "EN"
    except Exception:
        return "EN"

def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("Ãƒ", "Ã„", "Ã…", "Ã‚", "Ã¢â‚¬", "Ã¢â‚¬â€œ", "Ã¢â‚¬â„¢")):
        return text
    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def score(candidate: str) -> int:
        penalty = sum(candidate.count(marker) for marker in ("Ãƒ", "Ã„", "Ã…", "Ã‚", "ï¿½")) * 8
        reward = sum(candidate.count(char) for char in "Ã§ÄŸÄ±Ã¶ÅŸÃ¼Ã‡ÄžÄ°Ã–ÅžÃœ") * 2
        return reward - penalty

    return max(candidates, key=score)


def text_changed(source: str, translated: str | None) -> bool:
    if not translated:
        return False
    return normalize_ocr_text(source) != normalize_ocr_text(translated)


def normalize_translation_list(values: Any, source_texts: list[str]) -> list[str]:
    if isinstance(values, list):
        translated = [repair_mojibake(str(item).strip()) for item in values]
    elif isinstance(values, str):
        translated = [repair_mojibake(part.strip()) for part in values.split(" | ") if part.strip()]
    else:
        translated = []
    if len(translated) < len(source_texts):
        translated.extend(source_texts[len(translated):])
    return translated[: len(source_texts)]


def google_gemini_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )


def google_service_account_info() -> dict[str, Any] | None:
    raw_json = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON", "").strip()
        or os.getenv("ADAPTIFAI_VERTEX_SERVICE_ACCOUNT_JSON", "").strip()
    )
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError as exc:
            print(f"[vertex] invalid service account json env: {exc}", flush=True)
            return None
    path_value = (
        os.getenv("ADAPTIFAI_VERTEX_CREDENTIALS_FILE", "").strip()
        or os.getenv("VERTEX_AI_CREDENTIALS_FILE", "").strip()
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    )
    if not path_value:
        return None
    try:
        path = Path(path_value).expanduser()
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        print(f"[vertex] could not read service account file: {exc}", flush=True)
        return None


def vertex_project_id() -> str:
    info = google_service_account_info() or {}
    return (
        os.getenv("VERTEX_AI_PROJECT_ID", "").strip()
        or os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.getenv("GCP_PROJECT", "").strip()
        or str(info.get("project_id", "")).strip()
    )


def vertex_location() -> str:
    return os.getenv("VERTEX_AI_LOCATION", os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")).strip() or "us-central1"


def vertex_imagen_model() -> str:
    return os.getenv("VERTEX_IMAGEN_MODEL", os.getenv("ADAPTIFAI_VERTEX_IMAGEN_MODEL", "imagen-3.0-generate-002")).strip() or "imagen-3.0-generate-002"


def vertex_imagen_edit_model() -> str:
    return os.getenv("VERTEX_IMAGEN_EDIT_MODEL", os.getenv("ADAPTIFAI_VERTEX_IMAGEN_EDIT_MODEL", "imagen-3.0-capability-001")).strip() or "imagen-3.0-capability-001"


def vertex_gemini_model() -> str:
    return os.getenv("VERTEX_GEMINI_MODEL", os.getenv("ADAPTIFAI_VERTEX_GEMINI_MODEL", "gemini-2.5-pro")).strip() or "gemini-2.5-pro"


def vertex_available() -> bool:
    return bool(vertex_project_id() and google_service_account_info())


def vertex_authorized_session() -> Any:
    info = google_service_account_info()
    if not info:
        raise RuntimeError("Vertex service account is not configured.")
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return AuthorizedSession(credentials)


def decode_vertex_imagen_prediction(prediction: dict[str, Any]) -> Image.Image:
    encoded = (
        prediction.get("bytesBase64Encoded")
        or prediction.get("bytes_base64_encoded")
        or prediction.get("image", {}).get("bytesBase64Encoded")
        or prediction.get("image", {}).get("bytes_base64_encoded")
    )
    if not encoded:
        raise ValueError("Vertex Imagen response did not include image bytes.")
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def generate_vertex_imagen_image(
    prompt: str,
    *,
    width: int | None = None,
    height: int | None = None,
    sample_count: int = 1,
) -> tuple[Image.Image, dict[str, Any]]:
    project_id = vertex_project_id()
    if not project_id:
        raise RuntimeError("VERTEX_AI_PROJECT_ID or project_id in service account JSON is required.")
    location = vertex_location()
    model = vertex_imagen_model()
    aspect_ratio = "1:1"
    if width and height:
        ratio = width / max(1, height)
        if ratio >= 1.65:
            aspect_ratio = "16:9"
        elif ratio <= 0.62:
            aspect_ratio = "9:16"
        elif ratio >= 1.18:
            aspect_ratio = "4:3"
        elif ratio <= 0.84:
            aspect_ratio = "3:4"
    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model}:predict"
    )
    response = vertex_authorized_session().post(
        endpoint,
        json={
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": max(1, min(4, int(sample_count))),
                "aspectRatio": aspect_ratio,
                "safetyFilterLevel": os.getenv("VERTEX_IMAGEN_SAFETY_FILTER_LEVEL", "block_some"),
                "personGeneration": os.getenv("VERTEX_IMAGEN_PERSON_GENERATION", "allow_adult"),
            },
        },
        timeout=int(os.getenv("VERTEX_IMAGEN_TIMEOUT", "90")),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Vertex Imagen generate failed with HTTP {response.status_code}: {response.text[:1200]}") from exc
    payload = response.json()
    predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
    if not predictions:
        raise ValueError("Vertex Imagen returned no predictions.")
    image = decode_vertex_imagen_prediction(predictions[0])
    if width and height and image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, {"provider": "vertex", "model": model, "location": location, "aspectRatio": aspect_ratio}


def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_vertex_gemini_json(prompt: dict[str, Any], image: Image.Image | None = None, *, timeout: int = 45) -> dict[str, Any]:
    if not vertex_available():
        return {}
    project_id = vertex_project_id()
    location = vertex_location()
    model = vertex_gemini_model()
    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model}:generateContent"
    )
    parts: list[dict[str, Any]] = [{"text": json.dumps(prompt, ensure_ascii=False)}]
    if image is not None:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": image_to_base64_png(image.convert("RGB")),
                }
            }
        )
    response = vertex_authorized_session().post(
        endpoint,
        json={
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Vertex Gemini failed with HTTP {response.status_code}: {response.text[:1200]}") from exc
    payload = response.json()
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    return extract_json_object(content)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


LANGUAGE_NAMES = {
    "EN": "English",
    "DE": "German",
    "FR": "French",
    "IT": "Italian",
    "ES": "Spanish",
    "PT": "Portuguese",
    "TR": "Turkish",
    "AR": "Arabic",
    "ZH": "Chinese",
    "JA": "Japanese",
}

SHORT_LABEL_TRANSLATIONS: dict[str, dict[str, str]] = {
    "alt": {"TR": "ESKÄ°", "FR": "ANCIEN", "DE": "ALT", "ES": "ANTERIOR", "IT": "PRIMA", "PT": "ANTIGO"},
    "old": {"TR": "ESKÄ°", "FR": "ANCIEN", "DE": "ALT", "ES": "ANTERIOR", "IT": "PRIMA", "PT": "ANTIGO"},
    "neu": {"TR": "YENÄ°", "FR": "NOUVEAU", "DE": "NEU", "ES": "NUEVO", "IT": "NUOVO", "PT": "NOVO"},
    "new": {"TR": "YENÄ°", "FR": "NOUVEAU", "DE": "NEU", "ES": "NUEVO", "IT": "NUOVO", "PT": "NOVO"},
    "shop now": {"TR": "HEMEN AL", "FR": "ACHETER MAINTENANT", "DE": "JETZT KAUFEN", "ES": "COMPRA AHORA", "IT": "ACQUISTA ORA", "PT": "COMPRE AGORA"},
    "defensive": {"TR": "SAVUNMA\nTEKNOLOJÄ°SÄ°", "EN": "DEFENSIVE\nTECHNOLOGY", "DE": "SCHUTZ-\nTECHNOLOGIE", "FR": "TECHNOLOGIE\nDÃ‰FENSIVE", "ES": "TECNOLOGÃA\nDEFENSIVA", "IT": "TECNOLOGIA\nDIFENSIVA", "PT": "TECNOLOGIA\nDEFENSIVA"},
}


def translate_with_gemini(blocks: list[TextBlock], languages: list[str]) -> dict[str, list[str]]:
    source = [block.text for block in blocks if block.translate]
    if not source:
        return {language: [] for language in languages}

    api_key = google_gemini_api_key()
    if not api_key:
        return {language: source for language in languages}

    model = os.getenv("GEMINI_TRANSLATION_MODEL", os.getenv("ADAPTIFAI_GEMINI_TRANSLATION_MODEL", "gemini-2.5-pro")).strip() or "gemini-2.5-pro"
    prompt = {
        "task": "Translate every source string into the requested target language as faithful marketing localization. Treat stacked words as one semantic copy block before translating. Preserve meaning, protected brand/product tokens, metric tokens, [BOLD] tags, style intent by semantic word/phrase meaning, and source line rhythm where natural. Return compact JSON only. Each key must be a requested language code and each value must be an array aligned to the input order.",
        "target_languages": {language: LANGUAGE_NAMES.get(language.upper(), language) for language in languages},
        "strings": source,
    }
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": json.dumps(prompt, ensure_ascii=False)}]}],
                "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
            },
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
        parsed = extract_json_object(content)
        translations = {language: normalize_translation_list(parsed.get(language), source) for language in languages}
        for language in languages:
            language_code = language.upper()
            for index, source_text in enumerate(source):
                mapped = SHORT_LABEL_TRANSLATIONS.get(normalize_ocr_text(source_text), {}).get(language_code)
                if mapped:
                    translations[language][index] = mapped
        return translations
    except Exception as exc:
        print(f"[translation] Gemini fallback failed: {exc}", flush=True)
        return {language: source for language in languages}


def translate_with_gpt4o(blocks: list[TextBlock], languages: list[str]) -> dict[str, list[str]]:
    source = [block.text for block in blocks if block.translate]
    if not source:
        return {language: [] for language in languages}

    if not os.getenv("OPENAI_API_KEY"):
        return translate_with_gemini(blocks, languages)

    client = OpenAI()
    prompt = {
        "task": "Translate every source string into the requested target language as faithful marketing localization, not free transcreation. Treat each source string as one semantic copy block even when words are stacked on separate visual lines; first understand the full sentence/claim, then produce one coherent localized block. Preserve the exact meaning, claim scope, and emphasis. Do not add product names, SKU names, model names, brand names, or new claims that are not present in the source string. Only keep protected brand/product tokens unchanged when they are already present in that source string. Preserve [BOLD]...[/BOLD] tags exactly around the semantically emphasized phrase, even if that phrase moves to a different position in the target language. Preserve style intent by word/phrase meaning, not by absolute source position. Preserve the source line rhythm and line count when it can be done naturally; if the target language needs more characters, keep the same number of lines as much as possible by balancing words across those lines. Return one translated string per source string in the same order. Keep existing brand names, product names, packaging labels, URLs, QR references, Android TV, iPad Pro, H2O, Ask.NAOS.com, Dr.Scholl's and Scholl unchanged unless grammar absolutely requires surrounding words to change. Keep metric tokens exactly unchanged when they appear, including patterns like 24h, 48H, 84%, 88%, 2.4G+5G and Dual-Band. Translate surrounding claim language naturally but faithfully.",
        "target_languages": {language: LANGUAGE_NAMES.get(language.upper(), language) for language in languages},
        "strings": source,
    }
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a localization engine for ad creatives. Return compact JSON only. Each key must be a language code and each value must be an array of translated strings aligned to the input order. Translate into the actual requested language, not transliteration and not source-language copies."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    translations = {language: normalize_translation_list(parsed.get(language), source) for language in languages}

    for language in languages:
        language_code = language.upper()
        for index, source_text in enumerate(source):
            mapped = SHORT_LABEL_TRANSLATIONS.get(normalize_ocr_text(source_text), {}).get(language_code)
            if mapped:
                translations[language][index] = mapped

    for language in languages:
        if language.upper() == "EN":
            continue
        unchanged_indexes = [
            index
            for index, (source_text, translated_text) in enumerate(zip(source, translations.get(language, []), strict=False))
            if normalize_ocr_text(source_text) == normalize_ocr_text(translated_text)
            and not any(token in normalize_ocr_text(source_text) for token in ("24h", "48h", "84%", "88%", "2.4g+5g", "dual-band"))
        ]
        if not unchanged_indexes:
            continue
        retry_prompt = {
            "task": "Translate each string into the target language as ad copy. Do not leave the text in the source language unless it is purely a protected brand or product token. Return compact JSON only with key `translations`.",
            "target_language": LANGUAGE_NAMES.get(language.upper(), language),
            "strings": [source[index] for index in unchanged_indexes],
        }
        try:
            retry_response = client.chat.completions.create(
                model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a localization engine for ad creatives. Return compact JSON only."},
                    {"role": "user", "content": json.dumps(retry_prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            retry_content = retry_response.choices[0].message.content or "{}"
            retry_parsed = json.loads(retry_content)
            retry_values = normalize_translation_list(retry_parsed.get("translations"), [source[index] for index in unchanged_indexes])
            for index, retry_value in zip(unchanged_indexes, retry_values, strict=False):
                translations[language][index] = retry_value
        except Exception:
            continue

    return translations


def apply_translations(blocks: list[TextBlock], translated_strings: list[str], target_language: str) -> tuple[list[TextBlock], str]:
    translated_iter = iter(translated_strings)
    output: list[TextBlock] = []
    editor_parts: list[str] = []
    for block in blocks:
        if block.translate:
            translated = sanitize_bold_markup(repair_mojibake(next(translated_iter, block.text)))
            if block.line_boxes and len([line for line in translated.splitlines() if line.strip()]) < len(block.line_boxes):
                translated = split_text_across_lines(translated, len(block.line_boxes))
            if block.line_boxes:
                translated = translated.replace("[BOLD]", "").replace("[/BOLD]", "")
            translated = preserve_source_metric_tokens(block.text, translated)
            translation_candidates = polish_translated_copy(block, translated, target_language)
            translated = preserve_source_metric_tokens(block.text, translation_candidates[0]["text"])
            source_style_spans = infer_source_style_spans(block.text, block)
            translated_style_spans = infer_translated_style_spans(block.text, translated, source_style_spans, block)
            output.append(
                block.model_copy(
                    update={
                        "translated_text": translated,
                        "source_style_spans": source_style_spans,
                        "translated_style_spans": translated_style_spans,
                        "translation_candidates": translation_candidates,
                        "target_language": target_language,
                    }
                )
            )
            editor_parts.append(translated)
        else:
            output.append(block)
    return output, "\n\n".join(editor_parts)


def preserve_source_metric_tokens(source_text: str, translated_text: str) -> str:
    source_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\s*%|\b\d+\s*[hH]\b|\b\d+(?:[.,]\d+)?G(?:\+\d+(?:[.,]\d+)?G)?\b", source_text)
    output = translated_text
    for token in source_tokens:
        compact = re.sub(r"\s+", "", token)
        if "%" in compact:
            number = compact.replace("%", "")
            output = re.sub(rf"%\s*{re.escape(number)}\b", compact, output)
            output = re.sub(rf"\b{re.escape(number)}\s*%", compact, output)
        elif re.search(r"[hH]$", compact):
            number = re.sub(r"[hH]$", "", compact)
            output = re.sub(rf"\b{re.escape(number)}\s*[hH]\b", compact, output)
        elif "G" in compact.upper():
            output = re.sub(re.escape(compact).replace("G", "[gG]"), compact, output)
    return output


def default_typography_style(block: TextBlock) -> dict[str, Any]:
    return {
        "fontFamily": block.font_family,
        "fontCategory": "sans-serif",
        "fontWeight": block.font_weight,
        "fontSize": block.font_size_estimate,
        "color": block.color if block.color.startswith("#") else "#111111",
        "opacity": 1.0,
        "letterSpacing": 0.0,
        "lineHeight": block.line_height_estimate,
        "alignment": block.align,
        "casing": "uppercase" if block.text.isupper() else "mixed",
        "strokeWidth": 0,
        "strokeFill": None,
    }


def classify_semantic_role(segment: str) -> str:
    normalized = normalize_ocr_text(segment)
    if not normalized:
        return "benefit"
    if "%" in segment:
        return "percentage"
    if any(char.isdigit() for char in segment):
        return "numeric_claim"
    if normalized in {"buy", "shop", "discover", "join", "start", "try", "save", "get", "learn"}:
        return "cta"
    if segment.isupper() and len(segment.split()) <= 4:
        return "condition_or_topic"
    if len(segment.split()) <= 2:
        return "claim_modifier"
    return "benefit"


def split_semantic_segments(text: str) -> list[str]:
    segments: list[str] = []
    for line in text.splitlines():
        parts = re.split(r"(\d+%|\d+[a-zA-Z]*|[%])", line)
        buffer = ""
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"\d+%|\d+[a-zA-Z]*|[%]", part):
                if buffer.strip():
                    segments.append(buffer.strip())
                    buffer = ""
                segments.append(part.strip())
            else:
                buffer += part
        if buffer.strip():
            segments.append(buffer.strip())
    return [segment for segment in segments if segment]


def split_semantic_segments_with_breaks(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line_index, line in enumerate(lines):
        parts = re.split(r"(\d+%|\d+[a-zA-Z]*|[%])", line)
        buffer = ""
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"\d+%|\d+[a-zA-Z]*|[%]", part):
                if buffer.strip():
                    segments.append({"text": buffer.strip(), "forceBreakAfter": False})
                    buffer = ""
                segments.append({"text": part.strip(), "forceBreakAfter": False})
            else:
                buffer += part
        if buffer.strip():
            segments.append({"text": buffer.strip(), "forceBreakAfter": False})
        if segments:
            segments[-1]["forceBreakAfter"] = line_index < len(lines) - 1
    return [segment for segment in segments if segment.get("text")]


def infer_source_style_spans(text: str, block: TextBlock) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    base_style = default_typography_style(block)
    total_segments = max(1, len(split_semantic_segments_with_breaks(text)))
    for item in split_semantic_segments_with_breaks(text):
        segment = item["text"]
        role = classify_semantic_role(segment)
        style = dict(base_style)
        if role in {"percentage", "numeric_claim"}:
            style["fontWeight"] = max(style["fontWeight"], 800)
            style["fontSize"] = int(style["fontSize"] * 1.16)
            style["casing"] = "uppercase"
        elif role in {"claim_modifier", "cta"}:
            style["fontWeight"] = max(style["fontWeight"], 760)
        spans.append(
            {
                "sourceText": segment,
                "semanticRole": role,
                "semanticStyleKey": f"{role}:{normalize_ocr_text(segment)}",
                "style": style,
                "forceBreakAfter": item.get("forceBreakAfter", False),
                "color": style.get("color"),
                "fontSizeRatio": round(style["fontSize"] / max(1, base_style["fontSize"]), 3),
                "fontWeight": style.get("fontWeight"),
                "casing": style.get("casing"),
                "approximatePosition": round(len(spans) / total_segments, 3),
                "styleTransferMode": "semantic_phrase",
            }
        )
    return spans or [{"sourceText": text, "semanticRole": "benefit", "semanticStyleKey": f"benefit:{normalize_ocr_text(text)}", "style": base_style, "styleTransferMode": "semantic_phrase"}]


def infer_translated_style_spans(source_text: str, translated_text: str, source_style_spans: list[dict[str, Any]], block: TextBlock) -> list[dict[str, Any]]:
    segment_items = split_semantic_segments_with_breaks(translated_text)
    if not segment_items:
        return [{"translatedText": translated_text, "matchedSourceRole": "benefit", "style": default_typography_style(block)}]
    source_by_role: dict[str, dict[str, Any]] = {}
    for span in source_style_spans:
        role = span["semanticRole"]
        if role not in source_by_role:
            source_by_role[role] = span
            continue
        current = source_by_role[role]
        current_ratio = float(current.get("fontSizeRatio", 1.0))
        span_ratio = float(span.get("fontSizeRatio", 1.0))
        if span_ratio > current_ratio or int(span.get("fontWeight", 0)) > int(current.get("fontWeight", 0)):
            source_by_role[role] = span
    translated_spans: list[dict[str, Any]] = []
    for item in segment_items:
        segment = item["text"]
        role = classify_semantic_role(segment)
        matched = source_by_role.get(role) or source_style_spans[min(len(translated_spans), len(source_style_spans) - 1)]
        style = dict(matched["style"])
        if role in {"percentage", "numeric_claim"}:
            style["fontSize"] = int(style["fontSize"] * 1.08)
        if segment.isupper():
            style["casing"] = "uppercase"
        translated_spans.append(
            {
                "translatedText": segment,
                "matchedSourceRole": matched["semanticRole"],
                "style": style,
                "sourceSegmentHint": matched["sourceText"],
                "semanticStyleKey": matched.get("semanticStyleKey", f"{matched['semanticRole']}:{normalize_ocr_text(matched['sourceText'])}"),
                "forceBreakAfter": item.get("forceBreakAfter", False),
                "color": style.get("color"),
                "fontSizeRatio": round(style["fontSize"] / max(1, default_typography_style(block)["fontSize"]), 3),
                "fontWeight": style.get("fontWeight"),
                "casing": style.get("casing"),
                "approximatePosition": round(len(translated_spans) / max(1, len(segment_items)), 3),
                "styleTransferMode": "semantic_phrase_not_position",
            }
        )
    return translated_spans


LANGUAGE_FILLER_WORDS: dict[str, set[str]] = {
    "TR": {"ile", "ve", "bir", "iÃ§in", "olarak", "daha", "Ã§ok", "olan", "oranÄ±nda"},
    "DE": {"und", "mit", "fÃ¼r", "die", "der", "das"},
    "FR": {"et", "avec", "pour", "les", "des", "une", "un"},
    "ES": {"y", "con", "para", "los", "las", "una", "un"},
    "IT": {"e", "con", "per", "gli", "le", "una", "un"},
    "PT": {"e", "com", "para", "os", "as", "uma", "um"},
}


def maybe_preserve_uppercase(source_text: str, translated_text: str) -> str:
    return translated_text.upper() if source_text.isupper() else translated_text


def shorten_marketing_copy(text: str, target_language: str, role: str, aggressiveness: int = 1) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filler = LANGUAGE_FILLER_WORDS.get(target_language.upper(), set())
    output_lines: list[str] = []
    for line in lines:
        words = line.split()
        if len(words) <= 2:
            output_lines.append(line)
            continue
        kept: list[str] = []
        for index, word in enumerate(words):
            normalized = normalize_ocr_text(word)
            if aggressiveness >= 1 and normalized in filler and 0 < index < len(words) - 1:
                continue
            if aggressiveness >= 2 and role == "headline" and len(words) > 4 and normalized in {"Ã§ok", "daha", "Ã§oklu", "long", "lang", "mit", "ile"}:
                continue
            kept.append(word)
        candidate = " ".join(kept).strip()
        output_lines.append(candidate or line)
    compact = "\n".join(output_lines).strip()
    if role == "cta" and len(compact.split()) > 3:
        compact = " ".join(compact.split()[:3])
    return compact or text


def rebalance_copy_lines(text: str, target_language: str) -> str:
    connector_words = {
        "TR": {"VE", "Ä°LE", "DA", "DE"},
        "DE": {"UND", "MIT"},
        "FR": {"ET", "AVEC"},
        "ES": {"Y", "CON"},
        "IT": {"E", "CON"},
        "PT": {"E", "COM"},
    }.get(target_language.upper(), set())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return text.strip()
    rebalanced: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        upper_line = line.upper()
        if upper_line in connector_words and index + 1 < len(lines):
            next_line = lines[index + 1]
            rebalanced.append(f"{line} {next_line}".strip())
            index += 2
            continue
        if len(line.split()) == 1 and len(line) <= 3 and rebalanced:
            rebalanced[-1] = f"{rebalanced[-1]} {line}".strip()
            index += 1
            continue
        rebalanced.append(line)
        index += 1
    return "\n".join(rebalanced)


def polish_translated_copy(block: TextBlock, translated_text: str, target_language: str) -> list[dict[str, str]]:
    cleaned = sanitize_bold_markup(repair_mojibake(translated_text)).strip()
    role = block.role
    source_lines = max(1, len(block.line_texts) or cleaned.count("\n") + 1)
    faithful = maybe_preserve_uppercase(block.text, cleaned)
    if role == "headline":
        shorter = maybe_preserve_uppercase(block.text, split_text_across_lines(shorten_marketing_copy(faithful, target_language, role, aggressiveness=1).replace("\n", " "), source_lines))
        compact = maybe_preserve_uppercase(block.text, split_text_across_lines(shorten_marketing_copy(faithful, target_language, role, aggressiveness=2).replace("\n", " "), max(2, min(source_lines, 4))))
    else:
        shorter = maybe_preserve_uppercase(block.text, shorten_marketing_copy(faithful, target_language, role, aggressiveness=1))
        compact = maybe_preserve_uppercase(
            block.text,
            split_text_across_lines(shorten_marketing_copy(faithful, target_language, role, aggressiveness=2), source_lines),
        )
    candidates: list[dict[str, str]] = [
        {"label": "faithful", "text": rebalance_copy_lines(faithful, target_language)},
        {"label": "shorter_marketing", "text": rebalance_copy_lines(shorter, target_language)},
        {"label": "compact_layout_safe", "text": rebalance_copy_lines(compact, target_language)},
    ]
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate["text"].strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append({"label": candidate["label"], "text": normalized})
    return unique or [{"label": "faithful", "text": faithful or block.text}]


def sanitize_bold_markup(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("[/BOLD][BOLD]", "")
    if cleaned.count("[BOLD]") != cleaned.count("[/BOLD]"):
        cleaned = cleaned.replace("[BOLD]", "").replace("[/BOLD]", "")
    return cleaned


def split_text_across_lines(text: str, line_count: int) -> str:
    if line_count <= 1:
        return text.strip()
    words = [word for word in text.replace("\n", " ").split() if word]
    if len(words) <= line_count:
        return "\n".join(words)
    total_chars = sum(len(word) for word in words)
    target = max(1, total_chars // line_count)
    lines: list[str] = []
    current: list[str] = []
    current_chars = 0
    remaining_lines = line_count
    for index, word in enumerate(words):
        remaining_words = len(words) - index
        force_break = remaining_words == remaining_lines
        if current and current_chars + len(word) > target and remaining_lines > 1 and not force_break:
            lines.append(" ".join(current))
            current = [word]
            current_chars = len(word)
            remaining_lines -= 1
            continue
        current.append(word)
        current_chars += len(word)
        if force_break and remaining_lines > 1:
            lines.append(" ".join(current))
            current = []
            current_chars = 0
            remaining_lines -= 1
    if current:
        lines.append(" ".join(current))
    while len(lines) < line_count:
        lines.append("")
    return "\n".join(line for line in lines[:line_count] if line.strip())


def expand_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = bbox
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.strip().lower()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if len(normalized) == 3:
        normalized = "".join(part * 2 for part in normalized)
    if len(normalized) != 6:
        return (17, 17, 17)
    try:
        return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return (17, 17, 17)


def rgb_to_hex(rgb: Sequence[float | int]) -> str:
    values = [max(0, min(255, int(round(float(value))))) for value in list(rgb)[:3]]
    while len(values) < 3:
        values.append(17)
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def build_text_shaped_region_mask(
    image: Image.Image,
    region_box: tuple[int, int, int, int],
    text_color: str,
    font_size_estimate: int,
    line_height_estimate: int,
) -> Image.Image:
    import cv2

    left, top, right, bottom = region_box
    crop = np.array(image.crop(region_box).convert("RGB"))
    region_h, region_w = crop.shape[:2]
    if region_h == 0 or region_w == 0:
        return Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)

    border = np.concatenate(
        [
            crop[:2, :, :].reshape(-1, 3),
            crop[-2:, :, :].reshape(-1, 3),
            crop[:, :2, :].reshape(-1, 3),
            crop[:, -2:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_rgb = np.median(border, axis=0).astype(np.float32)
    text_rgb = np.array(hex_to_rgb(text_color), dtype=np.float32)
    crop_float = crop.astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background_luma = float(np.dot(background_rgb, [0.299, 0.587, 0.114]))
    text_luma = float(np.dot(text_rgb, [0.299, 0.587, 0.114]))
    bg_distance = np.linalg.norm(crop_float - background_rgb[None, None, :], axis=2)
    text_distance = np.linalg.norm(crop_float - text_rgb[None, None, :], axis=2)
    luminance_delta = gray - background_luma
    threshold = max(10.0, float(np.percentile(bg_distance, 55)))
    if text_luma >= background_luma:
        tonal_candidate = luminance_delta >= max(10.0, abs(text_luma - background_luma) * 0.35)
    else:
        tonal_candidate = luminance_delta <= -max(10.0, abs(text_luma - background_luma) * 0.35)

    distance_candidate = bg_distance >= threshold
    text_candidate = text_distance <= max(48.0, float(np.percentile(text_distance, 35)))
    combined = ((distance_candidate & tonal_candidate) | (distance_candidate & text_candidate)).astype(np.uint8) * 255

    kernel_size = max(3, min(9, int(round(max(font_size_estimate, line_height_estimate) / 8)) | 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size + 2, kernel_size + 2))
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
    refined = cv2.dilate(refined, dilate_kernel, iterations=1)
    refined = cv2.medianBlur(refined, 3)

    if np.count_nonzero(refined) < max(12, int(region_w * region_h * 0.01)):
        fallback = np.zeros((region_h, region_w), dtype=np.uint8)
        cv2.rectangle(fallback, (0, 0), (region_w - 1, region_h - 1), 255, thickness=-1)
        refined = fallback

    return Image.fromarray(refined, "L")


def fill_mask_holes(mask_array: np.ndarray) -> np.ndarray:
    import cv2

    binary = (mask_array > 0).astype(np.uint8) * 255
    if binary.size == 0:
        return binary
    flood = binary.copy()
    h, w = binary.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, holes)


def remove_small_mask_components(mask_array: np.ndarray, min_area: int) -> np.ndarray:
    import cv2

    binary = (mask_array > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def build_synthetic_text_mask(image_size: tuple[int, int], block: TextBlock) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    line_boxes = block.line_boxes or [block.bbox]
    line_texts = block.line_texts or [block.text]
    for index, line_box in enumerate(line_boxes):
        text = line_texts[index] if index < len(line_texts) else block.text
        text = (text or "").strip()
        if not text:
            continue
        lines, font_size, line_height = fit_text(
            draw,
            text,
            line_box,
            True,
            preferred_lines=1,
            preferred_font_size=max(10, block.font_size_estimate),
            line_height_ratio=max(1.0, min(1.35, block.line_height_estimate / max(1, block.font_size_estimate))),
        )
        if not lines:
            continue
        total_height = len(lines) * line_height
        y = line_box[1] + max(0, (line_box[3] - line_box[1] - total_height) // 2)
        for line in lines:
            line_width = sum(
                text_width(draw, segment if token_index == 0 else f" {segment}", get_font(font_size, bold=bold))
                for token_index, (segment, bold) in enumerate(line)
            )
            x = line_box[0] if block.align == "left" else line_box[0] + max(0, (line_box[2] - line_box[0] - int(line_width)) // 2)
            draw_rich_line(draw, (x, y), line, font_size, "#ffffff", None, 0)
            y += line_height
    return mask


def build_precise_text_stroke_mask(
    image: Image.Image,
    block: TextBlock,
    *,
    protected_region_mask: Image.Image | None = None,
    dilation_px: int = 7,
    feather_px: int = 2,
    allow_protected_overlap: bool = False,
) -> dict[str, Any]:
    import cv2

    text_lines = block.line_boxes or [block.bbox]
    image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    synthetic_mask = build_synthetic_text_mask(image.size, block)
    synthetic_array_full = np.array(synthetic_mask.convert("L"), dtype=np.uint8)

    adaptive_cluster_full = np.zeros((image.height, image.width), dtype=np.uint8)
    local_contrast_full = np.zeros((image.height, image.width), dtype=np.uint8)
    edge_stroke_full = np.zeros((image.height, image.width), dtype=np.uint8)
    ocr_vision_full = np.zeros((image.height, image.width), dtype=np.uint8)
    support_full = np.zeros((image.height, image.width), dtype=np.uint8)

    for line_box in text_lines:
        left, top, right, bottom = line_box
        pad_x = min(max(6, int((right - left) * 0.08)), 18)
        pad_y = min(max(6, int((bottom - top) * 0.22)), 18)
        region = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(image.width, right + pad_x),
            min(image.height, bottom + pad_y),
        )
        crop = image_rgb[region[1]:region[3], region[0]:region[2]]
        if crop.size == 0:
            continue

        crop_float = crop.astype(np.float32)
        gray_u8 = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = gray_u8.astype(np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        synthetic_crop = synthetic_array_full[region[1]:region[3], region[0]:region[2]]
        border = np.concatenate(
            [
                crop[:2, :, :].reshape(-1, 3),
                crop[-2:, :, :].reshape(-1, 3),
                crop[:, :2, :].reshape(-1, 3),
                crop[:, -2:, :].reshape(-1, 3),
            ],
            axis=0,
        ).astype(np.float32)
        bg_rgb = np.median(border, axis=0)
        bg_distance = np.linalg.norm(crop_float - bg_rgb[None, None, :], axis=2)
        synthetic_support = synthetic_crop > 0
        support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilation_px + 1), max(3, dilation_px + 1)))
        synthetic_support = cv2.dilate(synthetic_support.astype(np.uint8) * 255, support_kernel, iterations=1) > 16
        support_full[region[1]:region[3], region[0]:region[2]] = np.maximum(
            support_full[region[1]:region[3], region[0]:region[2]],
            (synthetic_support.astype(np.uint8) * 255),
        )

        # Adaptive color clustering in LAB/HSV space, constrained by text support.
        pixels = lab.reshape(-1, 3).astype(np.float32)
        if len(pixels) >= 6:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 16, 0.8)
            _compactness, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(lab.shape[:2])
            centers_rgb = cv2.cvtColor(centers.astype(np.uint8).reshape(1, -1, 3), cv2.COLOR_LAB2RGB)[0]
            cluster_scores: list[tuple[float, int]] = []
            for cluster_index, center_rgb in enumerate(centers_rgb):
                cluster_mask = labels == cluster_index
                if not np.any(cluster_mask):
                    continue
                overlap = float(np.logical_and(cluster_mask, synthetic_support).sum()) / max(1, int(cluster_mask.sum()))
                edge_density = float(cv2.Canny(gray_u8, 60, 140)[cluster_mask].mean()) / 255.0
                bg_sep = float(np.linalg.norm(center_rgb.astype(np.float32) - bg_rgb))
                cluster_scores.append((overlap * 0.45 + edge_density * 0.2 + min(1.0, bg_sep / 90.0) * 0.35, cluster_index))
            selected_clusters = {idx for score, idx in cluster_scores if score >= 0.28}
            adaptive_cluster = np.isin(labels, list(selected_clusters)) & synthetic_support
        else:
            adaptive_cluster = synthetic_support

        # Local contrast mask to catch anti-aliased edges and low-opacity glyph pixels.
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.4)
        local_delta = np.abs(gray - blurred)
        local_contrast = (local_delta >= max(6.0, float(np.percentile(local_delta[synthetic_support], 42)) if np.any(synthetic_support) else 8.0)) & synthetic_support

        # Edge stroke mask keeps outline/shadow/stroke structure near glyphs.
        edges = cv2.Canny(gray_u8, 40, 120) > 0
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges.astype(np.uint8) * 255, edge_kernel, iterations=1) > 16
        edge_stroke = edges & synthetic_support

        # OCR/vision pixel support is the synthetic mask itself.
        ocr_vision = synthetic_crop > 180

        adaptive_cluster_full[region[1]:region[3], region[0]:region[2]] = np.maximum(
            adaptive_cluster_full[region[1]:region[3], region[0]:region[2]],
            adaptive_cluster.astype(np.uint8) * 255,
        )
        local_contrast_full[region[1]:region[3], region[0]:region[2]] = np.maximum(
            local_contrast_full[region[1]:region[3], region[0]:region[2]],
            local_contrast.astype(np.uint8) * 255,
        )
        edge_stroke_full[region[1]:region[3], region[0]:region[2]] = np.maximum(
            edge_stroke_full[region[1]:region[3], region[0]:region[2]],
            edge_stroke.astype(np.uint8) * 255,
        )
        ocr_vision_full[region[1]:region[3], region[0]:region[2]] = np.maximum(
            ocr_vision_full[region[1]:region[3], region[0]:region[2]],
            ocr_vision.astype(np.uint8) * 255,
        )

    synthetic_array = synthetic_array_full > 180
    adaptive_cluster_array = adaptive_cluster_full > 0
    local_contrast_array = local_contrast_full > 0
    edge_stroke_array = edge_stroke_full > 0
    ocr_vision_array = ocr_vision_full > 0

    source_text_pixel_mask = adaptive_cluster_array | local_contrast_array | edge_stroke_array | ocr_vision_array
    combined_binary = synthetic_array | source_text_pixel_mask
    combined_array = (combined_binary.astype(np.uint8) * 255)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_array = cv2.morphologyEx(combined_array, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    filled_array = fill_mask_holes(combined_array)
    filled_array = remove_small_mask_components(filled_array, max(18, int(image.width * image.height * 0.00002)))

    dilate_size = max(3, dilation_px * 2 + 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    dilated_array = cv2.dilate(filled_array, dilate_kernel, iterations=1).astype(np.uint8)

    inpaint_binary_mask = Image.fromarray(np.where(dilated_array > 0, 255, 0).astype(np.uint8), "L")
    composite_soft_mask = inpaint_binary_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather_px)))
    binary_array = np.array(inpaint_binary_mask.convert("L"), dtype=np.uint8)

    white_pixels = int((binary_array > 16).sum())
    total_pixels = max(1, binary_array.size)
    white_ratio = white_pixels / total_pixels
    black_ratio = 1.0 - white_ratio
    visible_text_pixels = int(source_text_pixel_mask.sum())
    covered_visible = int(np.logical_and(binary_array > 16, source_text_pixel_mask).sum())
    text_coverage = covered_visible / max(1, visible_text_pixels)
    support_array = support_full > 0
    leakage_pixels = int(np.logical_and(binary_array > 16, ~support_array).sum())
    background_leakage = leakage_pixels / max(1, white_pixels)
    anti_alias_pixels = int(np.logical_and(local_contrast_array, ~ocr_vision_array).sum())
    anti_alias_covered = int(np.logical_and(binary_array > 16, np.logical_and(local_contrast_array, ~ocr_vision_array)).sum())
    anti_alias_coverage = anti_alias_covered / max(1, anti_alias_pixels)
    protected_overlap = 0.0
    protected_overlap_before_subtraction = 0.0
    if protected_region_mask is not None:
        protected_array = np.array(protected_region_mask.resize(image.size).convert("L"), dtype=np.uint8) > 16
        protected_core_u8 = protected_array.astype(np.uint8) * 255
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        for _ in range(5):
            protected_core_u8 = cv2.erode(protected_core_u8, core_kernel, iterations=1)
            if np.logical_and(binary_array > 16, protected_core_u8 > 16).sum() < white_pixels * 0.72:
                break
        protected_core = protected_core_u8 > 16
        protected_overlap_before_subtraction = float(np.logical_and(binary_array > 16, protected_array).sum()) / max(1, white_pixels)
        protected_subtraction_mask = protected_core
        if np.logical_and(binary_array > 16, protected_subtraction_mask).sum() >= white_pixels * 0.72:
            protected_subtraction_mask = np.zeros_like(protected_subtraction_mask, dtype=bool)
        dilated_array = np.where(protected_subtraction_mask, 0, dilated_array).astype(np.uint8)
        inpaint_binary_mask = Image.fromarray(np.where(dilated_array > 0, 255, 0).astype(np.uint8), "L")
        composite_soft_mask = inpaint_binary_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather_px)))
        binary_array = np.array(inpaint_binary_mask.convert("L"), dtype=np.uint8)
        white_pixels = int((binary_array > 16).sum())
        white_ratio = white_pixels / total_pixels
        black_ratio = 1.0 - white_ratio
        covered_visible = int(np.logical_and(binary_array > 16, source_text_pixel_mask).sum())
        text_coverage = covered_visible / max(1, visible_text_pixels)
        leakage_pixels = int(np.logical_and(binary_array > 16, ~support_array).sum())
        background_leakage = leakage_pixels / max(1, white_pixels)
        anti_alias_covered = int(np.logical_and(binary_array > 16, np.logical_and(local_contrast_array, ~ocr_vision_array)).sum())
        anti_alias_coverage = anti_alias_covered / max(1, anti_alias_pixels)
        protected_overlap = float(np.logical_and(binary_array > 16, protected_subtraction_mask).sum()) / max(1, white_pixels)
    raw_bbox = Image.fromarray(combined_array, "L").getbbox()
    rect_area_ratio = 0.0
    if raw_bbox:
        rect_area = max(1, (raw_bbox[2] - raw_bbox[0]) * (raw_bbox[3] - raw_bbox[1]))
        rect_area_ratio = white_pixels / rect_area

    mask_failure_reason = ""
    mask_quality_status = "passed"
    if white_ratio <= 0.001:
        mask_quality_status = "failed"
        mask_failure_reason = "mask_too_small"
    elif text_coverage < 0.72:
        mask_quality_status = "failed"
        mask_failure_reason = "text_not_covered"
    elif anti_alias_coverage < 0.58:
        mask_quality_status = "failed"
        mask_failure_reason = "anti_alias_edges_not_covered"
    elif white_ratio > 0.22:
        mask_quality_status = "failed"
        mask_failure_reason = "mask_too_large"
    elif background_leakage > 0.52:
        mask_quality_status = "failed"
        mask_failure_reason = "excessive_background_leakage"
    elif protected_overlap > 0.42 and not allow_protected_overlap:
        mask_quality_status = "failed"
        mask_failure_reason = "protected_object_overlap"
    elif rect_area_ratio > 0.82:
        mask_quality_status = "failed"
        mask_failure_reason = "excessive_rectangle_mask"

    return {
        "syntheticGlyphMask": Image.fromarray(synthetic_array_full, "L"),
        "adaptiveColorClusterMask": Image.fromarray(adaptive_cluster_full, "L"),
        "localContrastMask": Image.fromarray(local_contrast_full, "L"),
        "edgeStrokeMask": Image.fromarray(edge_stroke_full, "L"),
        "sourceTextPixelMask": Image.fromarray((source_text_pixel_mask.astype(np.uint8) * 255), "L"),
        "combinedBinaryMaskBeforeDilation": Image.fromarray(combined_array, "L"),
        "raw": Image.fromarray(combined_array, "L"),
        "filled": Image.fromarray(filled_array, "L"),
        "dilated": Image.fromarray(dilated_array, "L"),
        "final": inpaint_binary_mask,
        "inpaintMask_binary": inpaint_binary_mask,
        "compositeMask_soft": composite_soft_mask,
        "maskPolarity": "white_inpaint_black_preserve",
        "whitePixelRatio": float(white_ratio),
        "blackPixelRatio": float(black_ratio),
        "textCoverageEstimate": float(text_coverage),
        "antiAliasCoverageEstimate": float(anti_alias_coverage),
        "backgroundLeakageEstimate": float(background_leakage),
        "protectedObjectOverlapEstimate": float(protected_overlap),
        "protectedObjectOverlapBeforeSubtraction": float(protected_overlap_before_subtraction),
        "maskQualityStatus": mask_quality_status,
        "maskFailureReason": mask_failure_reason,
        "maskWarnings": ["protected_subtraction_uncertain_not_applied"] if protected_overlap_before_subtraction > 0.92 and protected_overlap == 0.0 else (["protected_object_overlap_tolerated_for_full_image_inpainting"] if protected_overlap > 0.42 and allow_protected_overlap else []),
        "textPixelDetectionMethodsUsed": [
            "synthetic_glyph_mask",
            "adaptive_color_cluster_mask",
            "local_contrast_mask",
            "edge_stroke_mask",
            "ocr_vision_pixel_mask",
        ],
    }


def collect_token_masks(image: Image.Image, blocks: list[TextBlock], padding: int = 26) -> list[dict[str, Any]]:
    token_masks: list[dict[str, Any]] = []
    for block in blocks:
        block_masks: list[dict[str, Any]] = []
        for region in iter_block_cleanup_regions(image, block, padding):
            block_masks.append(
                {
                    "tokenBox": region["token_box"],
                    "expandedMaskBox": region["expanded_box"],
                    "maskSize": list(region["mask"].size),
                }
            )
        if block_masks:
            token_masks.append({"id": block.id, "masks": block_masks})
    return token_masks


def iter_block_cleanup_regions(image: Image.Image, block: TextBlock, padding: int = 26) -> list[dict[str, Any]]:
    if not block.translate or not text_changed(block.text, block.translated_text):
        return []
    image_size = image.size
    line_regions = block.line_boxes if block.surface == "overlay" and block.line_boxes else [block.clean_box or block.bbox]
    regions: list[dict[str, Any]] = []
    for line_index, region_box in enumerate(line_regions):
        left, top, right, bottom = region_box
        block_width = max(1, right - left)
        block_height = max(1, bottom - top)
        dynamic_padding_x = min(max(10, int(block_width * 0.12)), 28)
        dynamic_padding_y = min(max(10, int(block_height * 0.4)), 24)
        expanded = (
            max(0, left - dynamic_padding_x),
            max(0, top - dynamic_padding_y),
            min(image_size[0], right + dynamic_padding_x),
            min(image_size[1], bottom + dynamic_padding_y),
        )
        local_mask = build_text_shaped_region_mask(
            image,
            expanded,
            block.color,
            block.font_size_estimate,
            block.line_height_estimate,
        )
        mask_bbox = local_mask.getbbox()
        if mask_bbox:
            shrink_pad = max(3, min(10, int(round(max(block.font_size_estimate, block.line_height_estimate) * 0.12))))
            mask_left = max(0, mask_bbox[0] - shrink_pad)
            mask_top = max(0, mask_bbox[1] - shrink_pad)
            mask_right = min(local_mask.size[0], mask_bbox[2] + shrink_pad)
            mask_bottom = min(local_mask.size[1], mask_bbox[3] + shrink_pad)
            cropped_mask = local_mask.crop(
                (
                    mask_left,
                    mask_top,
                    mask_right,
                    mask_bottom,
                )
            )
            cropped_left = max(0, expanded[0] + mask_left)
            cropped_top = max(0, expanded[1] + mask_top)
            expanded = (
                cropped_left,
                cropped_top,
                min(image_size[0], cropped_left + cropped_mask.size[0]),
                min(image_size[1], cropped_top + cropped_mask.size[1]),
            )
            local_mask = cropped_mask
        regions.append(
            {
                "block_id": block.id,
                "line_index": line_index,
                "line_text": block.line_texts[line_index] if line_index < len(block.line_texts) else block.text,
                "token_box": region_box,
                "expanded_box": expanded,
                "mask": local_mask,
                "block_width": block_width,
                "block_height": block_height,
            }
        )
    return regions


def is_overlay_marketing_cleanup_block(block: TextBlock) -> bool:
    return bool(
        block.translate
        and text_changed(block.text, block.translated_text)
        and block.surface == "overlay"
    )


def is_wide_overlay_text_block(block: TextBlock, image_size: tuple[int, int]) -> bool:
    width = max(1, block.bbox[2] - block.bbox[0])
    height = max(1, block.bbox[3] - block.bbox[1])
    area_ratio = (width * height) / max(1, image_size[0] * image_size[1])
    return (
        is_overlay_marketing_cleanup_block(block)
        and width >= image_size[0] * 0.36
        and height <= image_size[1] * 0.32
        and area_ratio >= 0.025
    )


def build_constrained_overlay_text_mask(
    image: Image.Image,
    region_box: tuple[int, int, int, int],
    source_boxes: list[tuple[int, int, int, int]],
    text_color: str,
    font_size_estimate: int,
    line_height_estimate: int,
) -> Image.Image:
    import cv2

    left, top, right, bottom = region_box
    crop = np.array(image.crop(region_box).convert("RGB"), dtype=np.uint8)
    region_h, region_w = crop.shape[:2]
    if region_h == 0 or region_w == 0:
        return Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)

    constraint = np.zeros((region_h, region_w), dtype=np.uint8)
    line_pad_x = max(2, int(round(max(8, font_size_estimate) * 0.22)))
    line_pad_y = max(2, int(round(max(8, line_height_estimate) * 0.24)))
    for box_left, box_top, box_right, box_bottom in source_boxes:
        rel_left = max(0, box_left - left - line_pad_x)
        rel_top = max(0, box_top - top - line_pad_y)
        rel_right = min(region_w, box_right - left + line_pad_x)
        rel_bottom = min(region_h, box_bottom - top + line_pad_y)
        if rel_right > rel_left and rel_bottom > rel_top:
            cv2.rectangle(constraint, (rel_left, rel_top), (rel_right - 1, rel_bottom - 1), 255, thickness=-1)

    if int(np.count_nonzero(constraint)) == 0:
        return Image.fromarray(constraint, "L")

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray_float, (0, 0), 2.4)
    local_delta = np.abs(gray_float - blur)
    constrained_delta = local_delta[constraint > 0]
    constrained_gray = gray[constraint > 0]
    if constrained_delta.size == 0 or constrained_gray.size == 0:
        return Image.fromarray(constraint, "L")

    delta_threshold = max(7.0, float(np.percentile(constrained_delta, 78)))
    bright_threshold = max(184.0, float(np.percentile(constrained_gray, 82)))
    dark_threshold = min(78.0, float(np.percentile(constrained_gray, 18)))
    chroma = crop.astype(np.float32).max(axis=2) - crop.astype(np.float32).min(axis=2)
    constrained_chroma = chroma[constraint > 0]
    chroma_threshold = max(42.0, float(np.percentile(constrained_chroma, 88))) if constrained_chroma.size else 64.0

    edges = cv2.Canny(gray, 36, 118)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    contrast_candidate = local_delta >= delta_threshold
    tonal_candidate = (gray_float >= bright_threshold) | (gray_float <= dark_threshold) | (chroma >= chroma_threshold)
    candidate = ((contrast_candidate | (edges > 0)) & tonal_candidate) & (constraint > 0)

    candidate_u8 = candidate.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate_u8 > 0).astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(candidate_u8)
    constraint_area = max(1, int(np.count_nonzero(constraint)))
    min_area = max(2, int(round((max(8, font_size_estimate) ** 2) * 0.012)))
    max_area = max(12, int(round(constraint_area * 0.18)))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            cleaned[labels == label] = 255

    if int(np.count_nonzero(cleaned)) < max(8, int(round(constraint_area * 0.006))):
        cleaned = candidate_u8

    kernel_size = max(3, min(7, int(round(max(8, line_height_estimate) / 11)) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    cleaned = cv2.bitwise_and(cleaned, constraint)
    return Image.fromarray(cleaned, "L")


def refine_overlay_text_colors(image: Image.Image, blocks: list[TextBlock]) -> None:
    for block in blocks:
        if block.surface != "overlay" or not block.translate:
            continue
        boxes = [tuple(box) for box in (block.line_boxes or [])] or [block.clean_box or block.bbox]
        union = union_bbox(boxes) or (block.clean_box or block.bbox)
        expanded = expand_bbox(union, image.size, max(4, int(round(max(8, block.line_height_estimate) * 0.18))))
        try:
            crop = np.array(image.crop(expanded).convert("RGB"), dtype=np.uint8)
            gray = crop @ np.array([0.299, 0.587, 0.114])
            local_mask = build_constrained_overlay_text_mask(
                image,
                expanded,
                boxes,
                block.color,
                block.font_size_estimate,
                block.line_height_estimate,
            )
            mask = np.array(local_mask.convert("L")) > 16
            if not mask.any():
                continue
            text_gray = gray[mask]
            text_rgb = crop[mask]
            bg_gray = gray[~mask] if (~mask).any() else gray.reshape(-1)
            bg_median = float(np.median(bg_gray)) if bg_gray.size else float(np.median(gray))
            bright_gap = float(np.percentile(text_gray, 92) - bg_median)
            dark_gap = float(bg_median - np.percentile(text_gray, 8))
            if bright_gap > max(18.0, dark_gap * 0.85):
                selected = text_rgb[text_gray >= np.percentile(text_gray, 82)]
            elif dark_gap > 18.0:
                selected = text_rgb[text_gray <= np.percentile(text_gray, 18)]
            else:
                continue
            if selected.size:
                block.color = rgb_to_hex(np.median(selected.reshape(-1, 3), axis=0))
        except Exception:
            continue


def build_overlay_block_cleanup_region(image: Image.Image, block: TextBlock) -> dict[str, Any]:
    width, height = image.size
    source_boxes = [tuple(box) for box in (block.line_boxes or [])] or [block.clean_box or block.bbox]
    union = union_bbox(source_boxes) or (block.clean_box or block.bbox)
    left, top, right, bottom = union
    block_w = max(1, right - left)
    block_h = max(1, bottom - top)
    wide = is_wide_overlay_text_block(block, image.size)
    pad_x = max(6, min(int(width * (0.055 if wide else 0.028)), int(block_w * (0.22 if wide else 0.16))))
    pad_y = max(5, min(int(height * (0.055 if wide else 0.032)), int(block_h * (0.55 if wide else 0.35))))
    expanded = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    mask = Image.new("L", (max(1, expanded[2] - expanded[0]), max(1, expanded[3] - expanded[1])), 0)
    shaped = build_text_shaped_region_mask(
        image,
        expanded,
        block.color,
        block.font_size_estimate,
        block.line_height_estimate,
    )
    try:
        crop_arr = np.array(image.crop(expanded).convert("RGB"), dtype=np.float32)
        luma = crop_arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        chroma = crop_arr.max(axis=2) - crop_arr.min(axis=2)
        solid_context = float(np.std(luma)) < 22.0 and float(np.std(chroma)) < 18.0
    except Exception:
        solid_context = False
    constrained = build_constrained_overlay_text_mask(
        image,
        expanded,
        source_boxes,
        block.color,
        block.font_size_estimate,
        block.line_height_estimate,
    )
    shaped_ratio = float((np.array(shaped) > 16).mean()) if shaped.width and shaped.height else 0.0
    constrained_ratio = float((np.array(constrained) > 16).mean()) if constrained.width and constrained.height else 0.0
    if not solid_context and shaped_ratio > max(0.32, constrained_ratio * 3.5):
        shaped = Image.new("L", shaped.size, 0)
    if solid_context:
        draw = ImageDraw.Draw(mask)
        for box in source_boxes:
            box_left, box_top, box_right, box_bottom = box
            rel = (
                max(0, box_left - expanded[0] - pad_x // 2),
                max(0, box_top - expanded[1] - pad_y // 2),
                min(mask.width, box_right - expanded[0] + pad_x // 2),
                min(mask.height, box_bottom - expanded[1] + pad_y // 2),
            )
            draw.rectangle(rel, fill=255)
        mask = ImageChops.lighter(mask, shaped)
    else:
        mask = constrained
    mask = mask.filter(ImageFilter.MaxFilter(size=5 if not solid_context else (5 if wide else 3)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(3, block_h // 18))))
    return {
        "block_id": block.id,
        "line_index": -1,
        "line_text": " ".join(block.line_texts or [block.text]),
        "token_box": union,
        "expanded_box": expanded,
        "mask": mask,
        "block_width": block_w,
        "block_height": block_h,
        "overlay_marketing_cleanup": True,
        "wideOverlayCleanup": wide,
        "solidContextCleanup": solid_context,
        "shapedMaskRatio": shaped_ratio,
        "constrainedMaskRatio": constrained_ratio,
    }


def build_text_mask(image: Image.Image, blocks: list[TextBlock], padding: int = 26) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    has_large_overlay = False
    for block in blocks:
        for region in iter_block_cleanup_regions(image, block, padding):
            expanded = region["expanded_box"]
            local_mask = region["mask"]
            if (block.bbox[2] - block.bbox[0]) > image.size[0] * 0.55 and (block.bbox[3] - block.bbox[1]) < image.size[1] * 0.28:
                has_large_overlay = True
            mask.paste(ImageChops.lighter(mask.crop(expanded), local_mask), expanded)
    max_filter_size = 13 if has_large_overlay else 17
    blur_radius = 0.8 if has_large_overlay else 1.2
    mask = mask.filter(ImageFilter.MaxFilter(size=max_filter_size))
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def build_soft_background_fill(image: Image.Image, mask: Image.Image) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=26)).convert("RGBA")
    original = image.convert("RGBA")
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=10)).convert("L")
    return Image.composite(blurred, original, soft_mask).convert("RGB")


def reconstruct_overlay_background(image: Image.Image, blocks: list[TextBlock]) -> Image.Image:
    working = image.convert("RGBA")
    rgb = np.array(image.convert("RGB"))
    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text) or block.surface != "overlay":
            continue
        regions = block.line_boxes or [block.clean_box or block.bbox]
        for left, top, right, bottom in regions:
            width = max(1, right - left)
            height = max(1, bottom - top)
            pad_x = min(max(10, int(width * 0.14)), 28)
            pad_y = min(max(8, int(height * 0.35)), 22)
            region = (
                max(0, left - pad_x),
                max(0, top - pad_y),
                min(image.width, right + pad_x),
                min(image.height, bottom + pad_y),
            )
            region_w = region[2] - region[0]
            region_h = region[3] - region[1]
            if region_w <= 0 or region_h <= 0:
                continue
            sample_h = min(max(6, height // 2), 20)
            sample_w = min(max(6, width // 6), 20)
            gap_y = min(max(6, height // 3), 18)
            gap_x = min(max(6, width // 8), 18)
            top_start = max(0, region[1] - gap_y - sample_h)
            top_end = max(0, region[1] - gap_y)
            bottom_start = min(image.height, region[3] + gap_y)
            bottom_end = min(image.height, region[3] + gap_y + sample_h)
            left_start = max(0, region[0] - gap_x - sample_w)
            left_end = max(0, region[0] - gap_x)
            right_start = min(image.width, region[2] + gap_x)
            right_end = min(image.width, region[2] + gap_x + sample_w)
            top_strip = rgb[top_start:top_end, region[0]:region[2], :]
            bottom_strip = rgb[bottom_start:bottom_end, region[0]:region[2], :]
            left_strip = rgb[region[1]:region[3], left_start:left_end, :]
            right_strip = rgb[region[1]:region[3], right_start:right_end, :]

            prefer_horizontal = width > image.width * 0.45
            if prefer_horizontal and left_strip.size and right_strip.size:
                left_color = np.median(left_strip, axis=1)
                right_color = np.median(right_strip, axis=1)
                t = np.linspace(0.0, 1.0, region_w, dtype=np.float32)[None, :, None]
                gradient = left_color[:, None, :] * (1.0 - t) + right_color[:, None, :] * t
                fill_array = np.clip(gradient, 0, 255).astype(np.uint8)
            elif top_strip.size and bottom_strip.size:
                top_color = np.median(top_strip, axis=0)
                bottom_color = np.median(bottom_strip, axis=0)
                t = np.linspace(0.0, 1.0, region_h, dtype=np.float32)[:, None, None]
                gradient = top_color[None, :, :] * (1.0 - t) + bottom_color[None, :, :] * t
                fill_array = np.clip(gradient, 0, 255).astype(np.uint8)
            elif left_strip.size and right_strip.size:
                left_color = np.median(left_strip, axis=1)
                right_color = np.median(right_strip, axis=1)
                t = np.linspace(0.0, 1.0, region_w, dtype=np.float32)[None, :, None]
                gradient = left_color[:, None, :] * (1.0 - t) + right_color[:, None, :] * t
                fill_array = np.clip(gradient, 0, 255).astype(np.uint8)
            else:
                base_color = rgb[region[1]:region[3], region[0]:region[2], :].mean(axis=(0, 1))
                fill_array = np.full((region_h, region_w, 3), np.clip(base_color, 0, 255).astype(np.uint8), dtype=np.uint8)

            fill = Image.fromarray(fill_array, "RGB").filter(ImageFilter.GaussianBlur(radius=4 if prefer_horizontal else 6)).convert("RGBA")
            region_mask = Image.new("L", (region_w, region_h), 0)
            region_draw = ImageDraw.Draw(region_mask)
            radius = max(5, int(min(region_w, region_h) * 0.12))
            region_draw.rounded_rectangle((0, 0, region_w, region_h), radius=radius, fill=220)
            working.alpha_composite(Image.composite(fill, working.crop(region), region_mask), dest=(region[0], region[1]))
    return working.convert("RGB")


def reconstruct_large_overlay_bands(image: Image.Image, blocks: list[TextBlock]) -> Image.Image:
    working = image.convert("RGBA")
    rgb = np.array(image.convert("RGB"))
    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text) or block.surface != "overlay":
            continue
        block_width = block.bbox[2] - block.bbox[0]
        if block_width <= image.width * 0.45:
            continue
        region = block.clean_box or compute_clean_box(block.bbox, image.size, large_block=True)
        region_w = region[2] - region[0]
        region_h = region[3] - region[1]
        if region_w <= 0 or region_h <= 0:
            continue

        sample_h = min(max(10, region_h // 5), 28)
        sample_w = min(max(10, region_w // 8), 28)
        gap_y = min(max(8, region_h // 8), 20)
        gap_x = min(max(8, region_w // 10), 20)
        top_strip = rgb[max(0, region[1] - sample_h):region[1], region[0]:region[2], :]
        bottom_strip = rgb[region[3]:min(image.height, region[3] + sample_h), region[0]:region[2], :]
        left_strip = rgb[region[1]:region[3], max(0, region[0] - sample_w - gap_x):max(0, region[0] - gap_x), :]
        right_strip = rgb[region[1]:region[3], min(image.width, region[2] + gap_x):min(image.width, region[2] + gap_x + sample_w), :]
        prefer_horizontal = region_w > region_h * 2.2
        if prefer_horizontal and left_strip.size and right_strip.size:
            left_color = np.median(left_strip.reshape(-1, 3), axis=0)
            right_color = np.median(right_strip.reshape(-1, 3), axis=0)
            t = np.linspace(0.0, 1.0, region_w, dtype=np.float32)[None, :, None]
            gradient = left_color[None, None, :] * (1.0 - t) + right_color[None, None, :] * t
            fill_array = np.repeat(np.clip(gradient, 0, 255).astype(np.uint8), region_h, axis=0)
        elif top_strip.size and bottom_strip.size:
            top_color = np.median(top_strip.reshape(-1, 3), axis=0)
            bottom_color = np.median(bottom_strip.reshape(-1, 3), axis=0)
            t = np.linspace(0.0, 1.0, region_h, dtype=np.float32)[:, None, None]
            gradient = top_color[None, None, :] * (1.0 - t) + bottom_color[None, None, :] * t
            fill_array = np.repeat(np.clip(gradient, 0, 255).astype(np.uint8), region_w, axis=1)
        elif left_strip.size and right_strip.size:
            left_color = np.median(left_strip.reshape(-1, 3), axis=0)
            right_color = np.median(right_strip.reshape(-1, 3), axis=0)
            t = np.linspace(0.0, 1.0, region_w, dtype=np.float32)[None, :, None]
            gradient = left_color[None, None, :] * (1.0 - t) + right_color[None, None, :] * t
            fill_array = np.repeat(np.clip(gradient, 0, 255).astype(np.uint8), region_h, axis=0)
        else:
            base_color = rgb[region[1]:region[3], region[0]:region[2], :].mean(axis=(0, 1))
            fill_array = np.full((region_h, region_w, 3), np.clip(base_color, 0, 255).astype(np.uint8), dtype=np.uint8)

        fill = Image.fromarray(fill_array, "RGB").filter(ImageFilter.GaussianBlur(radius=3)).convert("RGBA")
        region_mask = Image.new("L", (region_w, region_h), 0)
        region_draw = ImageDraw.Draw(region_mask)
        region_draw.rounded_rectangle((0, 0, region_w, region_h), radius=max(8, int(min(region_w, region_h) * 0.08)), fill=255)
        working.alpha_composite(Image.composite(fill, working.crop(region), region_mask), dest=(region[0], region[1]))
    return working.convert("RGB")


def compute_mask_foreground_overlap(
    mask: Image.Image,
    expanded_box: tuple[int, int, int, int],
    protected_region_mask: Image.Image | None,
    *,
    erosion_radius: int = 0,
) -> float:
    if protected_region_mask is None:
        return 0.0
    mask_array = np.array(mask.convert("L")) > 16
    total = int(mask_array.sum())
    if total == 0:
        return 0.0
    protected_crop = np.array(protected_region_mask.crop(expanded_box).convert("L")) > 16
    if erosion_radius > 0 and protected_crop.any():
        import cv2

        kernel_size = max(3, erosion_radius * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        protected_crop = cv2.erode(protected_crop.astype(np.uint8) * 255, kernel, iterations=1) > 16
    overlap_pixels = int(np.logical_and(mask_array, protected_crop).sum())
    return overlap_pixels / total


def analyze_cleanup_region(
    image: Image.Image,
    block: TextBlock,
    region: dict[str, Any],
    foreground_bbox: tuple[int, int, int, int] | None = None,
    protected_region_mask: Image.Image | None = None,
    protected_meta: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    import cv2

    expanded = region["expanded_box"]
    mask = region["mask"]
    crop = np.array(image.crop(expanded).convert("RGB"))
    mask_array = np.array(mask) > 16
    if crop.size == 0 or not mask_array.any():
        return {"strategy": "opencv", "area_ratio": 0.0, "mask_density": 0.0, "contrast": 0.0, "texture_variance": 0.0}
    region_area = crop.shape[0] * crop.shape[1]
    area_ratio = region_area / max(1, image.size[0] * image.size[1])
    mask_density = float(mask_array.mean())
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    contrast = float(gray[mask_array].std()) if mask_array.any() else 0.0
    texture_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    is_headline = block.role == "headline"
    bbox_foreground_overlap = overlap_fraction(expanded, foreground_bbox) if foreground_bbox else 0.0
    mask_foreground_overlap = compute_mask_foreground_overlap(mask, expanded, protected_region_mask)
    core_mask_foreground_overlap = compute_mask_foreground_overlap(mask, expanded, protected_region_mask, erosion_radius=3)
    large_text = area_ratio >= 0.022 or block.font_size_estimate >= 28 or block_height_ratio(block) >= 0.06
    flat_background = texture_variance <= 10 and contrast <= 16 and mask_density <= 0.38
    protected_region_ratio = max(0.0, 1.0 - mask_density)
    generative_allowed_reason = ""
    generative_blocked_reason = ""
    refinement_uncertain = bool((protected_meta or {}).get("refinementUncertain", False))
    if (
        large_text
        and mask_foreground_overlap <= 0.40
        and core_mask_foreground_overlap <= 0.26
        and protected_region_ratio >= 0.45
        and (contrast >= 16 or texture_variance >= 16 or area_ratio >= 0.02)
    ):
        strategy = "generative"
        generative_allowed_reason = "large text overlaps protected boundary moderately but editable text mask is still mostly background-safe"
    elif mask_foreground_overlap > 0.2 and core_mask_foreground_overlap > 0.12:
        strategy = "conservative"
        generative_blocked_reason = "text mask overlaps protected foreground"
    elif mask_foreground_overlap <= 0.42 and core_mask_foreground_overlap <= 0.12 and large_text and (contrast >= 16 or texture_variance >= 16):
        strategy = "generative"
        generative_allowed_reason = "text mask only grazes protected boundary; core protected overlap is low"
    elif is_headline and large_text and (contrast >= 18 or texture_variance >= 18 or area_ratio >= 0.03):
        strategy = "generative"
        generative_allowed_reason = "large high-contrast headline with object-safe text mask"
    elif large_text and mask_foreground_overlap <= 0.06 and (contrast >= 22 or texture_variance >= 24):
        strategy = "generative"
        generative_allowed_reason = "large complex text region and mask is object-safe"
    elif flat_background and not is_headline:
        strategy = "sampled"
    else:
        strategy = "conservative"
        generative_blocked_reason = "background simple enough for conservative cleanup"
    if refinement_uncertain and mask_foreground_overlap <= 0.06 and strategy != "sampled":
        strategy = "generative"
        generative_allowed_reason = "protected mask refinement uncertain but text mask overlap is low"
        generative_blocked_reason = ""
    return {
        "strategy": strategy,
        "area_ratio": float(area_ratio),
        "mask_density": float(mask_density),
        "contrast": float(contrast),
        "texture_variance": float(texture_variance),
        "foreground_overlap": float(mask_foreground_overlap),
        "bbox_foreground_overlap": float(bbox_foreground_overlap),
        "mask_foreground_overlap": float(mask_foreground_overlap),
        "core_mask_foreground_overlap": float(core_mask_foreground_overlap),
        "protected_region_ratio": float(protected_region_ratio),
        "generative_allowed_reason": generative_allowed_reason,
        "generative_blocked_reason": generative_blocked_reason,
        "protected_mask_refinement_method": str((protected_meta or {}).get("protectedMaskRefinementMethod", "")),
        "protected_mask_refinement_uncertain": refinement_uncertain,
    }


def block_height_ratio(block: TextBlock) -> float:
    height = max(1, block.bbox[3] - block.bbox[1])
    width = max(1, block.bbox[2] - block.bbox[0])
    return height / max(1, width)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def detect_residual_ocr_similarity(cleaned_crop: Image.Image, source_text_hint: str | None) -> tuple[float, list[str]]:
    if not source_text_hint or not normalize_ocr_text(source_text_hint):
        return 0.0, []
    try:
        with tempfile.TemporaryDirectory(prefix="adaptifai-residual-ocr-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            upscaled = cleaned_crop.convert("RGB").resize(
                (max(64, cleaned_crop.width * 2), max(64, cleaned_crop.height * 2)),
                Image.Resampling.LANCZOS,
            )
            temp_path = temp_dir / "residual.png"
            upscaled.save(temp_path, "PNG")
            detected = run_trocr_ocr_on_image(temp_path)
            texts = [block.text for block in detected if normalize_ocr_text(block.text)]
            if not texts:
                return 0.0, []
            source_norm = normalize_ocr_text(source_text_hint)
            similarity = max(
                text_similarity(source_text_hint, text)
                * min(1.0, (len(normalize_ocr_text(text)) / max(1, len(source_norm))) * 1.4)
                for text in texts
            )
            return float(similarity), texts
    except Exception:
        return 0.0, []


def extract_source_word_targets(block: TextBlock) -> list[dict[str, str]]:
    source_text = " ".join(block.line_texts or [block.text])
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-zÃ€-Ã¿0-9%]+", source_text):
        normalized = normalize_match_text(token)
        if not normalized or normalized.isdigit() or len(normalized) < 4:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append({"raw": token, "normalized": normalized})
    return candidates


def run_ocr_on_crop(cleaned_crop: Image.Image) -> list[str]:
    try:
        with tempfile.TemporaryDirectory(prefix="adaptifai-cleaned-ocr-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            base = cleaned_crop.convert("RGB").resize(
                (max(128, cleaned_crop.width * 2), max(128, cleaned_crop.height * 2)),
                Image.Resampling.LANCZOS,
            )
            gray = ImageOps.grayscale(base)
            boosted = ImageEnhance.Contrast(gray).enhance(3.4)
            inverted = ImageOps.invert(boosted)
            binary = boosted.point(lambda value: 255 if value > 188 else 0).convert("L")
            variants = {
                "rgb": base,
                "boosted": boosted.convert("RGB"),
                "inverted": inverted.convert("RGB"),
                "binary": binary.convert("RGB"),
            }
            texts: list[str] = []
            seen: set[str] = set()
            for variant_name, variant in variants.items():
                temp_path = temp_dir / f"cleaned-crop-{variant_name}.png"
                variant.save(temp_path, "PNG")
                detected = run_trocr_ocr_on_image(temp_path)
                for block in detected:
                    normalized = normalize_match_text(block.text)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    texts.append(block.text)
            return texts
    except Exception:
        return []


def detect_ocr_boxes_on_crop(crop: Image.Image) -> list[dict[str, Any]]:
    try:
        detector = load_ocr_detector()
        ocr_image, scale = fit_for_ocr(crop.convert("RGB"))
        detections = detector.readtext(
            np.array(ocr_image),
            detail=1,
            paragraph=False,
            batch_size=int(os.getenv("ADAPTIFAI_OCR_BATCH_SIZE", "1")),
            width_ths=float(os.getenv("ADAPTIFAI_OCR_WIDTH_THS", "0.7")),
            decoder=os.getenv("ADAPTIFAI_EASYOCR_DECODER", "greedy"),
        )
    except Exception:
        return []

    boxes: list[dict[str, Any]] = []
    for points, detected_text, confidence in detections:
        text = str(detected_text or "").strip()
        if not text:
            continue
        xs = [int(point[0] / max(scale, 1e-6)) for point in points]
        ys = [int(point[1] / max(scale, 1e-6)) for point in points]
        left = max(0, min(xs))
        top = max(0, min(ys))
        right = min(crop.width, max(xs))
        bottom = min(crop.height, max(ys))
        if right <= left or bottom <= top:
            continue
        boxes.append(
            {
                "text": text,
                "normalized": normalize_match_text(text),
                "bbox": [left, top, right, bottom],
                "confidence": float(confidence),
            }
        )
    return boxes


def _fill_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    import cv2

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.max() <= 1:
        binary_mask = binary_mask * 255
    flood = binary_mask.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary_mask, flood_inv)


def build_residual_word_focus_boxes(
    block: TextBlock,
    crop_box: tuple[int, int, int, int],
    target_terms: list[str],
    cleaned_ocr_boxes: list[dict[str, Any]],
    source_ocr_boxes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    left, top, _, _ = crop_box
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()

    def add_box(box: list[int], word: str, source: str, confidence: float = 0.0) -> None:
        x1, y1, x2, y2 = [int(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            return
        key = (x1, y1, x2, y2, normalize_match_text(word))
        if key in seen:
            return
        seen.add(key)
        boxes.append(
            {
                "word": word,
                "normalized": normalize_match_text(word),
                "bbox": [x1, y1, x2, y2],
                "fullImageBox": [x1 + left, y1 + top, x2 + left, y2 + top],
                "source": source,
                "confidence": float(confidence),
            }
        )

    for detection in cleaned_ocr_boxes + source_ocr_boxes:
        normalized = str(detection.get("normalized") or normalize_match_text(str(detection.get("text", ""))))
        if not normalized:
            continue
        matching_terms = [
            term
            for term in target_terms
            if term
            and (
                term == normalized
                or normalized == term
                or (term in normalized and len(normalized) <= max(len(term) * 2, len(term) + 5))
                or (normalized in term and len(term) <= max(len(normalized) * 2, len(normalized) + 5))
                or SequenceMatcher(None, normalized, term).ratio() >= 0.72
            )
        ]
        if not matching_terms:
            continue
        best_term = max(matching_terms, key=lambda term: SequenceMatcher(None, normalized, term).ratio())
        if len(normalized) > max(len(best_term) * 2, len(best_term) + 5) and normalized != best_term:
            continue
        add_box(list(detection.get("bbox", [0, 0, 0, 0])), str(detection.get("text", best_term)), "ocr", float(detection.get("confidence", 0.0)))

    for line_text, line_box in zip(block.line_texts or [], block.line_boxes or []):
        normalized_line = normalize_match_text(line_text)
        if not normalized_line:
            continue
        words = [match for match in re.finditer(r"[A-Za-zÃ€-Ã¿0-9%]+", line_text)]
        if not words:
            continue
        line_left, line_top, line_right, line_bottom = line_box
        line_width = max(1, line_right - line_left)
        for match in words:
            word = match.group(0)
            normalized_word = normalize_match_text(word)
            if not normalized_word or not any(term in normalized_word or normalized_word in term for term in target_terms):
                continue
            span_start = match.start() / max(1, len(line_text))
            span_end = match.end() / max(1, len(line_text))
            pad_x = max(8, int(line_width * 0.018))
            pad_y = max(6, int((line_bottom - line_top) * 0.10))
            add_box(
                [
                    max(0, int(line_left - left + span_start * line_width) - pad_x),
                    max(0, int(line_top - top) - pad_y),
                    max(0, int(line_left - left + span_end * line_width) + pad_x),
                    max(0, int(line_bottom - top) + pad_y),
                ],
                word,
                "source_line_projection",
                0.62,
            )
    return boxes


def build_residual_text_mask(
    source_image: Image.Image,
    cleaned_image: Image.Image,
    block: TextBlock,
    base_mask: Image.Image,
    failed_words: list[str] | None,
    residual_ocr_texts: list[str] | None,
) -> tuple[Image.Image, dict[str, Any]]:
    import cv2

    crop_box = block.clean_box or block.bbox
    left, top, right, bottom = crop_box
    if right <= left or bottom <= top:
        empty = Image.new("L", source_image.size, 0)
        return empty, {"generated": False, "reason": "empty_crop"}

    original_crop = np.array(source_image.crop(crop_box).convert("RGB"))
    cleaned_crop = np.array(cleaned_image.crop(crop_box).convert("RGB"))
    base_mask_crop = np.array(base_mask.crop(crop_box).convert("L"))
    if original_crop.size == 0 or cleaned_crop.size == 0:
        empty = Image.new("L", source_image.size, 0)
        return empty, {"generated": False, "reason": "empty_pixels"}

    base_mask_bool = base_mask_crop > 16
    if not np.any(base_mask_bool):
        empty = Image.new("L", source_image.size, 0)
        return empty, {"generated": False, "reason": "no_base_mask"}

    original_gray = cv2.cvtColor(original_crop, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(cleaned_crop, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(original_gray, cleaned_gray)
    original_edges = cv2.Canny(original_gray, 50, 150)
    cleaned_edges = cv2.Canny(cleaned_gray, 50, 150)
    similarity_map = diff < max(18, int(np.percentile(diff[base_mask_bool], 62))) if np.any(base_mask_bool) else diff < 18
    edge_overlap = np.logical_and(original_edges > 0, cleaned_edges > 0)
    low_contrast_delta = np.abs(
        cv2.GaussianBlur(original_gray.astype(np.float32), (0, 0), 1.2)
        - cv2.GaussianBlur(cleaned_gray.astype(np.float32), (0, 0), 1.2)
    ) < 14.0
    residual_candidate = np.logical_and(base_mask_bool, np.logical_or(similarity_map, np.logical_or(edge_overlap, low_contrast_delta)))

    source_terms = [normalize_match_text(word) for word in (failed_words or []) if normalize_match_text(word)]
    residual_terms: list[str] = []
    for text in residual_ocr_texts or []:
        normalized = normalize_match_text(text)
        if not normalized:
            continue
        if source_terms and not any(SequenceMatcher(None, normalized, term).ratio() >= 0.62 or term in normalized or normalized in term for term in source_terms):
            continue
        residual_terms.append(normalized)
    target_terms = [term for term in dict.fromkeys(source_terms + residual_terms) if term]

    source_ocr_boxes = detect_ocr_boxes_on_crop(source_image.crop(crop_box))
    cleaned_ocr_boxes = detect_ocr_boxes_on_crop(cleaned_image.crop(crop_box))
    word_boxes = build_residual_word_focus_boxes(block, crop_box, target_terms, cleaned_ocr_boxes, source_ocr_boxes)
    focus_mask = np.zeros_like(base_mask_crop, dtype=np.uint8)
    matched_boxes: list[list[int]] = []
    max_word_height = 0
    for word_box in word_boxes:
        bx1, by1, bx2, by2 = word_box["bbox"]
        max_word_height = max(max_word_height, by2 - by1)
        pad_x = max(5, int((bx2 - bx1) * 0.06))
        pad_y = max(4, int((by2 - by1) * 0.16))
        matched_boxes.append([bx1, by1, bx2, by2])
        focus_mask[max(0, by1 - pad_y):min(focus_mask.shape[0], by2 + pad_y), max(0, bx1 - pad_x):min(focus_mask.shape[1], bx2 + pad_x)] = 255

    residual_glyph_mask = np.logical_and(base_mask_bool, focus_mask > 0) if matched_boxes else base_mask_bool
    if matched_boxes:
        residual_candidate = np.logical_and(residual_candidate, focus_mask > 0)
        residual_candidate = np.logical_or(residual_candidate, residual_glyph_mask)

    local_context = cv2.dilate((focus_mask if matched_boxes else base_mask_crop).astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1) > 0
    local_ring = np.logical_and(local_context, ~base_mask_bool)
    if np.any(local_ring):
        cleaned_lab = cv2.cvtColor(cleaned_crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        bg_lab = np.median(cleaned_lab[local_ring], axis=0)
        lab_delta = np.linalg.norm(cleaned_lab - bg_lab[None, None, :], axis=2)
        local_delta_values = lab_delta[local_ring]
        color_threshold = max(18.0, float(np.percentile(local_delta_values, 88)) if local_delta_values.size else 18.0)
        color_deviation = np.logical_and(base_mask_bool, lab_delta > color_threshold)
        if matched_boxes:
            color_deviation = np.logical_and(color_deviation, focus_mask > 0)
        residual_candidate = np.logical_or(residual_candidate, color_deviation)
    else:
        color_deviation = np.zeros_like(base_mask_bool)

    contrast_expansion = np.zeros_like(base_mask_crop, dtype=np.uint8)
    if matched_boxes:
        candidate_seed = (residual_candidate.astype(np.uint8) * 255)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_band = cv2.dilate(candidate_seed, edge_kernel, iterations=1) > 0
        weak_strokes = np.logical_and.reduce(
            (
                focus_mask > 0,
                base_mask_bool,
                np.logical_or(cleaned_edges > 0, original_edges > 0),
                edge_band,
            )
        )
        contrast_expansion[weak_strokes] = 255
        residual_candidate = np.logical_or(residual_candidate, weak_strokes)

    candidate_u8 = (residual_candidate.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel)
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate_u8 = _fill_mask_holes(candidate_u8)
    residual_height = max(max_word_height, int(max(block.font_size_estimate, block.line_height_estimate) * 0.7))
    dilation_px = max(3, min(10, int(round(residual_height * 0.10))))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
    candidate_u8 = cv2.dilate(candidate_u8, dilation_kernel, iterations=1)
    candidate_u8 = np.where(base_mask_crop > 0, candidate_u8, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, 8)
    filtered = np.zeros_like(candidate_u8)
    min_area = max(18, int(base_mask_bool.sum() * 0.002))
    max_area = max(min_area + 1, int(base_mask_bool.sum() * 0.45))
    kept_components = 0
    for label in range(1, num_labels):
        area_component = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area_component <= max_area:
            filtered[labels == label] = 255
            kept_components += 1
    if kept_components:
        candidate_u8 = filtered

    area = int((candidate_u8 > 0).sum())
    base_area = int(base_mask_bool.sum())
    coverage_by_word: list[dict[str, Any]] = []
    for word_box in word_boxes:
        bx1, by1, bx2, by2 = word_box["bbox"]
        word_region = candidate_u8[max(0, by1):min(candidate_u8.shape[0], by2), max(0, bx1):min(candidate_u8.shape[1], bx2)]
        base_region = base_mask_crop[max(0, by1):min(base_mask_crop.shape[0], by2), max(0, bx1):min(base_mask_crop.shape[1], bx2)] > 0
        coverage_by_word.append(
            {
                "word": word_box["word"],
                "normalized": word_box["normalized"],
                "bbox": word_box["fullImageBox"],
                "source": word_box["source"],
                "coverage": float((word_region > 0).sum() / max(1, int(base_region.sum()))),
            }
        )
    false_positive_area = int(np.logical_and(candidate_u8 > 0, focus_mask == 0).sum()) if matched_boxes else 0
    false_positive_estimate = false_positive_area / max(1, area)
    full_mask = Image.new("L", source_image.size, 0)
    full_mask.paste(Image.fromarray(candidate_u8, "L"), crop_box[:2])
    return full_mask, {
        "generated": area > 24,
        "reason": "" if area > 24 else "residual_mask_too_small",
        "matchedBoxes": matched_boxes,
        "residualWordBoxes": word_boxes,
        "area": area,
        "baseArea": base_area,
        "coverageRatio": (area / base_area) if base_area else 0.0,
        "ocrDetections": cleaned_ocr_boxes,
        "sourceOcrDetections": source_ocr_boxes,
        "residualGlyphMask": Image.fromarray((residual_glyph_mask.astype(np.uint8) * 255), "L"),
        "residualColorDeviationMask": Image.fromarray((color_deviation.astype(np.uint8) * 255), "L"),
        "residualArtifactMaskExpanded": Image.fromarray(candidate_u8, "L"),
        "coverageByWord": coverage_by_word,
        "falsePositiveEstimate": float(false_positive_estimate),
        "dilationPx": dilation_px,
    }


def evaluate_block_cleanup_visibility(source_image: Image.Image, cleaned_image: Image.Image, block: TextBlock) -> dict[str, Any]:
    import cv2

    crop_box = block.clean_box or block.bbox
    cleaned_crop = cleaned_image.crop(crop_box).convert("RGB")
    source_word_targets = extract_source_word_targets(block)
    residual_ocr = run_ocr_on_crop(cleaned_crop)
    residual_similarity, residual_similarity_texts = detect_residual_ocr_similarity(
        cleaned_crop,
        " ".join(block.line_texts or [block.text]),
    )
    combined_region = build_combined_block_cleanup_region(source_image, block)
    ghosting_score = 0.0
    residual_text_score = 1.0
    if combined_region is not None:
        breakdown = score_cleanup_candidate_detailed(
            source_image.crop(combined_region["expanded_box"]).convert("RGB"),
            cleaned_image.crop(combined_region["expanded_box"]).convert("RGB"),
            combined_region["mask"],
            source_text_hint=" ".join(block.line_texts or [block.text]),
            candidate_name="final-clean-visibility-gate",
        )
        ghosting_score = float(breakdown.get("ghostingScore", 0.0))
        residual_text_score = float(breakdown.get("residualTextScore", 1.0))
    line_ghosting_deltas: list[dict[str, float | int]] = []
    color_variance_profile = {
        "maskedVariance": 0.0,
        "backgroundVariance": 0.0,
        "varianceRatio": 1.0,
        "matchesBackgroundNoise": True,
        "maskedRingColorDelta": 0.0,
        "sourceRingColorDelta": 0.0,
        "meanLumaDelta": 0.0,
        "matchesColorContinuity": True,
    }
    if combined_region is not None:
        region_mask = np.array(combined_region["mask"].convert("L"), dtype=np.uint8) > 0
        if np.any(region_mask):
            ring = np.logical_and(
                cv2.dilate(region_mask.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1).astype(bool),
                ~region_mask,
            )
            cleaned_region = np.array(cleaned_image.crop(combined_region["expanded_box"]).convert("RGB"), dtype=np.float32)
            source_region = np.array(source_image.crop(combined_region["expanded_box"]).convert("RGB"), dtype=np.float32)
            if np.any(ring):
                masked_variance = float(np.mean(np.var(cleaned_region[region_mask], axis=0)))
                background_variance = float(np.mean(np.var(cleaned_region[ring], axis=0)))
                variance_ratio = masked_variance / max(background_variance, 1e-6)
                masked_color = cleaned_region[region_mask].mean(axis=0)
                ring_color = cleaned_region[ring].mean(axis=0)
                source_mask_color = source_region[region_mask].mean(axis=0)
                source_ring_color = source_region[ring].mean(axis=0)
                masked_ring_color_delta = float(np.linalg.norm(masked_color - ring_color))
                source_ring_color_delta = float(np.linalg.norm(source_mask_color - source_ring_color))
                cleaned_gray_region = cv2.cvtColor(cleaned_region.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
                mean_luma_delta = float(abs(cleaned_gray_region[region_mask].mean() - cleaned_gray_region[ring].mean()))
                color_variance_profile = {
                    "maskedVariance": masked_variance,
                    "backgroundVariance": background_variance,
                    "varianceRatio": variance_ratio,
                    "matchesBackgroundNoise": 0.28 <= variance_ratio <= 3.2,
                    "maskedRingColorDelta": masked_ring_color_delta,
                    "sourceRingColorDelta": source_ring_color_delta,
                    "meanLumaDelta": mean_luma_delta,
                    "matchesColorContinuity": masked_ring_color_delta <= max(18.0, source_ring_color_delta * 0.62 + 8.0) and mean_luma_delta <= max(12.0, source_ring_color_delta * 0.28 + 6.0),
                }
    for region in iter_block_cleanup_regions(source_image, block):
        local_mask = (np.array(region["mask"].convert("L")) > 0).astype(np.uint8)
        if not np.any(local_mask):
            continue
        ring = np.logical_and(
            cv2.dilate(local_mask, np.ones((7, 7), np.uint8), iterations=1).astype(bool),
            ~local_mask.astype(bool),
        )
        if not ring.any():
            continue
        source_gray = np.array(source_image.crop(region["expanded_box"]).convert("L"), dtype=np.float32)
        cleaned_gray = np.array(cleaned_image.crop(region["expanded_box"]).convert("L"), dtype=np.float32)
        mask_bool = local_mask.astype(bool)
        source_delta = float((source_gray[ring].mean() - source_gray[mask_bool].mean()) / 255.0)
        cleaned_delta = float((cleaned_gray[ring].mean() - cleaned_gray[mask_bool].mean()) / 255.0)
        ratio = cleaned_delta / max(source_delta, 1e-6)
        line_ghosting_deltas.append(
            {
                "lineIndex": int(region["line_index"]),
                "sourceDelta": source_delta,
                "cleanedDelta": cleaned_delta,
                "ratio": ratio,
            }
        )
    normalized_residual = [normalize_match_text(text) for text in residual_ocr]
    failed_source_words: list[str] = []
    for target in source_word_targets:
        normalized_target = target["normalized"]
        if any(
            normalized_target in text
            or text in normalized_target
            or SequenceMatcher(a=normalized_target, b=text).ratio() >= 0.78
            for text in normalized_residual
            if text
        ):
            failed_source_words.append(target["raw"])
    visual_ghosting_detected = any(
        entry["sourceDelta"] >= 0.12 and (entry["cleanedDelta"] >= 0.006 or entry["ratio"] >= 0.015)
        for entry in line_ghosting_deltas
    )
    readable = (
        bool(failed_source_words)
        or residual_similarity >= 0.4
        or ghosting_score >= 0.12
        or residual_text_score <= 0.88
        or visual_ghosting_detected
        or not color_variance_profile["matchesBackgroundNoise"]
        or not color_variance_profile.get("matchesColorContinuity", True)
    )
    return {
        "cleanupStatus": "failed" if readable else "passed",
        "residualSourceOCR": residual_ocr or residual_similarity_texts,
        "failedSourceWords": failed_source_words,
        "residualSimilarity": float(residual_similarity),
        "ghostingScore": ghosting_score,
        "residualTextScore": residual_text_score,
        "lineGhostingDeltas": line_ghosting_deltas,
        "visualGhostingDetected": visual_ghosting_detected,
        "colorVarianceProfile": color_variance_profile,
        "cropBox": list(crop_box),
    }


def build_combined_block_cleanup_region(image: Image.Image, block: TextBlock, padding: int = 26) -> dict[str, Any] | None:
    if is_overlay_marketing_cleanup_block(block):
        return build_overlay_block_cleanup_region(image, block)
    regions = iter_block_cleanup_regions(image, block, padding)
    if not regions:
        return None
    union_box = union_bbox([tuple(region["expanded_box"]) for region in regions])
    if union_box is None:
        return None
    union_left, union_top, union_right, union_bottom = union_box
    union_mask = Image.new("L", (max(1, union_right - union_left), max(1, union_bottom - union_top)), 0)
    for region in regions:
        expanded = region["expanded_box"]
        local_mask = region["mask"]
        offset = (
            expanded[0] - union_left,
            expanded[1] - union_top,
            expanded[2] - union_left,
            expanded[3] - union_top,
        )
        current = union_mask.crop(offset)
        union_mask.paste(ImageChops.lighter(current, local_mask), (offset[0], offset[1]))
    union_mask = union_mask.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=0.8))
    return {
        "block_id": block.id,
        "line_index": -1,
        "line_text": " ".join(block.line_texts or [block.text]),
        "token_box": block.bbox,
        "expanded_box": union_box,
        "mask": union_mask,
        "block_width": max(1, block.bbox[2] - block.bbox[0]),
        "block_height": max(1, block.bbox[3] - block.bbox[1]),
    }


def is_large_marketing_headline_block(block: TextBlock, image_size: tuple[int, int]) -> bool:
    width = max(1, block.bbox[2] - block.bbox[0])
    height = max(1, block.bbox[3] - block.bbox[1])
    area_ratio = (width * height) / max(1, image_size[0] * image_size[1])
    multi_line = len(block.line_boxes) >= 2 or len((block.text or "").splitlines()) >= 2
    return (
        block.translate
        and text_changed(block.text, block.translated_text)
        and block.surface == "overlay"
        and (multi_line or is_wide_overlay_text_block(block, image_size))
        and width >= image_size[0] * 0.38
        and area_ratio >= 0.025
    )


def resolve_cleanup_provider(block: TextBlock, image_size: tuple[int, int]) -> str:
    """
    Risk-based provider routing (Protocol V2.2 Phase 2).
    LaMa is NOT treated as universal primary.
    Maskless text-removal providers are experimental only.
    """
    configured = os.getenv("ADAPTIFAI_CLEANUP_PROVIDER", "").strip().lower()
    if configured in {"lama", "sdxl_controlnet"}:
        return "huggingface"
    if configured in {"openai", "huggingface"}:
        return configured

    # Risk-based routing: estimate risk from block properties
    block_width = max(1, block.bbox[2] - block.bbox[0])
    block_height = max(1, block.bbox[3] - block.bbox[1])
    area_ratio = (block_width * block_height) / max(1, image_size[0] * image_size[1])

    if is_large_marketing_headline_block(block, image_size):
        # HIGH risk â€” prefer mask-capable specialist or SDXL
        routes = route_cleanup_providers(RiskLevel.HIGH, block, image_size)
        for route in routes:
            if route.provider == "huggingface" and get_huggingface_cleanup_token():
                return "huggingface"
            if route.provider == "openai" and os.getenv("OPENAI_API_KEY"):
                return "openai"
        return "huggingface"

    if area_ratio > 0.10:
        # MEDIUM risk
        routes = route_cleanup_providers(RiskLevel.MEDIUM, block, image_size)
        for route in routes:
            if route.provider == "huggingface" and get_huggingface_cleanup_token():
                return "huggingface"
            if route.provider == "openai" and os.getenv("OPENAI_API_KEY"):
                return "openai"
        return "openai"

    # LOW risk â€” OpenAI primary, HF fallback
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "huggingface"


def get_cleanup_provider_availability(provider: str) -> tuple[bool, str]:
    provider = provider.lower()
    if provider == "huggingface":
        return (bool(get_huggingface_cleanup_token()), "huggingface_not_configured" if not get_huggingface_cleanup_token() else "")
    if provider == "openai":
        return (bool(os.getenv("OPENAI_API_KEY")), "openai_not_configured" if not os.getenv("OPENAI_API_KEY") else "")
    return False, "unknown_cleanup_provider"

def prepare_huggingface_cleanup_mask(
    editable_mask: Image.Image,
    *,
    dilation_px: int = 15,
    feather_px: int = 5,
    block: TextBlock | None = None,
    image: Image.Image | None = None,
) -> Image.Image:
    """
    Prepare cleanup mask with stroke-width-based dilation (Protocol V2.2 Phase 3).
    dilationPx = clamp(strokeWidth * 0.8 to strokeWidth * 1.5, min 4px, max 18px)
    """
    import cv2

    # Use stroke-width-based dilation if block info available
    if block is not None and image is not None:
        dilation_px = compute_dilation_px(block, image)
        feather_px = max(2, dilation_px // 3)

    mask_array = np.array(editable_mask.convert("L"), dtype=np.uint8)
    binary = np.where(mask_array > 16, 255, 0).astype(np.uint8)
    if dilation_px > 0:
        kernel_size = max(3, dilation_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.dilate(binary, kernel, iterations=1)
    if feather_px > 0:
        binary = cv2.GaussianBlur(binary, (0, 0), feather_px)
    return Image.fromarray(binary, "L")


def compute_dilation_px(block: TextBlock, image: Image.Image) -> int:
    """
    Stroke-width-based dilation (Protocol V2.2):
    dilationPx = clamp(strokeWidth * 0.8 to strokeWidth * 1.5, min 4px, max 18px)
    """
    font_size = block.font_size_estimate or 16
    stroke_width = max(2.0, font_size * 0.08)
    dilation_low = stroke_width * 0.8
    dilation_high = stroke_width * 1.5
    dilation = (dilation_low + dilation_high) / 2.0
    return int(max(4, min(18, round(dilation))))


def encode_png_bytes(image: Image.Image, *, mode: str | None = None) -> bytes:
    output = io.BytesIO()
    encoded = image.convert(mode) if mode else image
    encoded.save(output, format="PNG")
    return output.getvalue()


def decode_huggingface_image(content: bytes) -> Image.Image | None:
    import cv2

    image_array = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, "RGB")


def encode_base64_png(image: Image.Image, *, mode: str | None = None) -> str:
    return base64.b64encode(encode_png_bytes(image, mode=mode)).decode("utf-8")


def prepare_huggingface_request_image(image: Image.Image, mask: Image.Image, *, max_side: int = 768) -> dict[str, Any]:
    import cv2

    original_image = image.convert("RGB")
    original_mask = mask.convert("L")
    width, height = original_image.size
    longest_side = max(width, height)
    if longest_side <= 1024:
        return {
            "image": original_image,
            "mask": original_mask,
            "scaled": False,
            "originalSize": (width, height),
            "requestSize": (width, height),
        }

    scale = min(1.0, max_side / float(longest_side))
    request_width = max(64, int(round(width * scale)))
    request_height = max(64, int(round(height * scale)))
    request_width = max(64, request_width - (request_width % 8))
    request_height = max(64, request_height - (request_height % 8))
    resized_image = original_image.resize((request_width, request_height), Image.Resampling.LANCZOS)
    mask_array = np.array(original_mask, dtype=np.uint8)
    resized_mask_array = cv2.resize(mask_array, (request_width, request_height), interpolation=cv2.INTER_NEAREST)
    resized_mask = Image.fromarray(np.where(resized_mask_array > 127, 255, 0).astype(np.uint8), "L")
    return {
        "image": resized_image,
        "mask": resized_mask,
        "scaled": True,
        "originalSize": (width, height),
        "requestSize": (request_width, request_height),
    }


def upscale_huggingface_result(result_image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    import cv2

    if result_image.size == target_size:
        return result_image.convert("RGB")
    bgr = cv2.cvtColor(np.array(result_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    upscaled = cv2.resize(bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, "RGB")


def call_huggingface_inpainting_model(
    session: requests.Session,
    token: str,
    model_id: str,
    image: Image.Image,
    mask: Image.Image,
    *,
    prompt: str,
    negative_prompt: str,
    timeout_seconds: int = 35,
) -> requests.Response:
    api_root = os.getenv("ADAPTIFAI_HF_INPAINT_API_ROOT", "https://api-inference.huggingface.co/models").rstrip("/")
    url = f"{api_root}/{model_id}"
    payload = {
        "inputs": {
            "image": encode_base64_png(image, mode="RGB"),
            "mask": encode_base64_png(mask, mode="L"),
        },
        "parameters": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": int(os.getenv("ADAPTIFAI_HF_INPAINT_STEPS", "50")),
            "guidance_scale": float(os.getenv("ADAPTIFAI_HF_INPAINT_GUIDANCE", "7.5")),
        },
    }
    return session.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=timeout_seconds,
    )


def cleanup_via_huggingface_v3(
    image: Image.Image,
    editable_mask: Image.Image,
    *,
    validation_callback: Any | None = None,
    stage_label: str = "cleanup",
) -> dict[str, Any]:
    token = get_huggingface_cleanup_token()
    if not token:
        return {
            "success": False,
            "provider": "huggingface",
            "failureReason": "huggingface_not_configured",
            "modelAttempts": [],
        }

    session = get_huggingface_cleanup_session()
    request_mask = prepare_huggingface_cleanup_mask(
        editable_mask,
        dilation_px=int(os.getenv("ADAPTIFAI_HF_MASK_DILATION", "15")),
        feather_px=int(os.getenv("ADAPTIFAI_HF_MASK_FEATHER", "5")),
    )
    request_bundle = prepare_huggingface_request_image(
        image.convert("RGB"),
        request_mask,
        max_side=int(os.getenv("ADAPTIFAI_HF_REQUEST_MAX_SIDE", "768")),
    )
    request_image = request_bundle["image"]
    request_mask = request_bundle["mask"]
    prompt = os.getenv(
        "ADAPTIFAI_HF_INPAINT_PROMPT",
        "seamless high-quality fabric texture, studio lighting, matching colors, clean background",
    )
    negative_prompt = os.getenv(
        "ADAPTIFAI_HF_INPAINT_NEGATIVE_PROMPT",
        "text, words, letters, messy, blurry, blue smudges, artifacts",
    )
    timeout_seconds = int(os.getenv("ADAPTIFAI_HF_INPAINT_TIMEOUT", "35"))
    loading_retry_limit = int(os.getenv("ADAPTIFAI_HF_LOADING_RETRIES", "2"))

    attempts: list[dict[str, Any]] = []
    transitions: list[str] = []
    best_effort_image: Image.Image | None = None
    best_effort_validation: dict[str, Any] | None = None
    best_effort_score = -1.0

    for model_index, model_id in enumerate(HF_INPAINT_MODELS):
        next_model = HF_INPAINT_MODELS[model_index + 1] if model_index + 1 < len(HF_INPAINT_MODELS) else None
        retry_index = 0
        while True:
            attempt_log: dict[str, Any] = {
                "stage": stage_label,
                "model": model_id,
                "retryIndex": retry_index,
                "requestMaskDilationPx": int(os.getenv("ADAPTIFAI_HF_MASK_DILATION", "15")),
                "requestMaskFeatherPx": int(os.getenv("ADAPTIFAI_HF_MASK_FEATHER", "5")),
                "requestImagePreview": request_image.copy(),
                "requestMaskPreview": request_mask.copy(),
                "requestImageSize": list(request_image.size),
                "scaledRequest": bool(request_bundle["scaled"]),
            }
            try:
                response = call_huggingface_inpainting_model(
                    session,
                    token,
                    model_id,
                    request_image,
                    request_mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                attempt_log["statusCode"] = None
                attempt_log["failureReason"] = f"request_error:{exc.__class__.__name__}"
                attempts.append(attempt_log)
                if next_model:
                    transition = f"Model {model_id} failed, switching to {next_model}..."
                    print(transition)
                    transitions.append(transition)
                break

            attempt_log["statusCode"] = response.status_code
            if response.status_code == 503 and retry_index < loading_retry_limit:
                attempt_log["failureReason"] = "model_loading"
                attempts.append(attempt_log)
                time.sleep(12)
                retry_index += 1
                continue

            if response.status_code == 200:
                result_image = decode_huggingface_image(response.content)
                if result_image is None:
                    attempt_log["failureReason"] = "invalid_image_response"
                    attempts.append(attempt_log)
                else:
                    result_image = upscale_huggingface_result(result_image, tuple(request_bundle["originalSize"]))
                    validation_result = validation_callback(result_image) if callable(validation_callback) else {}
                    attempt_log["validation"] = validation_result
                    attempt_log["apiSuccessResultPreview"] = result_image.copy()
                    cleanup_passed = not validation_result or validation_result.get("cleanupStatus") == "passed"
                    if cleanup_passed:
                        attempt_log["success"] = True
                        attempts.append(attempt_log)
                        return {
                            "success": True,
                            "provider": "huggingface",
                            "actualModel": model_id,
                            "image": result_image,
                            "validation": validation_result,
                            "modelAttempts": attempts,
                            "requestImagePreview": request_image.copy(),
                            "requestMaskPreview": request_mask.copy(),
                            "apiSuccessResultPreview": result_image.copy(),
                            "cleanupCascadeLog": transitions,
                        }
                    validation_score = (
                        max(0.0, 1.0 - float(validation_result.get("ghostingScore", 1.0)))
                        + max(0.0, float(validation_result.get("residualTextScore", 0.0)))
                        + max(0.0, 1.0 - len(validation_result.get("failedSourceWords", [])) * 0.08)
                    )
                    if validation_score > best_effort_score:
                        best_effort_score = validation_score
                        best_effort_image = result_image.copy()
                        best_effort_validation = validation_result
                    attempt_log["failureReason"] = "cleanup_success_gate_failed"
                    attempts.append(attempt_log)
                if next_model:
                    transition = f"Model {model_id} failed, switching to {next_model}..."
                    print(transition)
                    transitions.append(transition)
                break

            attempt_log["failureReason"] = f"http_{response.status_code}"
            try:
                attempt_log["errorBody"] = response.json()
            except Exception:
                attempt_log["errorBody"] = response.text
            attempts.append(attempt_log)
            if next_model:
                transition = f"Model {model_id} failed, switching to {next_model}..."
                print(transition)
                transitions.append(transition)
            break

    return {
        "success": False,
        "provider": "huggingface",
        "failureReason": "cleanup_failed",
        "bestEffortImage": best_effort_image,
        "bestEffortValidation": best_effort_validation or {},
        "modelAttempts": attempts,
        "requestImagePreview": request_image.copy(),
        "requestMaskPreview": request_mask.copy(),
        "cleanupCascadeLog": transitions,
    }


def prepare_protected_region_mask(source: Image.Image, exclusion_mask: Image.Image | None = None) -> tuple[Image.Image, dict[str, Any]]:
    requested_backend = os.getenv("ADAPTIFAI_PROTECTED_REGION_BACKEND", "groundingdino").lower()
    available = False
    failure_reason = ""
    if requested_backend in {"groundingdino", "yolo"}:
        try:
            if requested_backend == "groundingdino":
                __import__("groundingdino")
            else:
                __import__("ultralytics")
            available = True
        except Exception:
            failure_reason = f"{requested_backend}_not_configured"
    protected_mask, meta = detect_protected_region_mask(source, exclusion_mask=exclusion_mask)
    meta.update(
        {
            "requestedProtectedRegionBackend": requested_backend,
            "actualProtectedRegionBackend": "heuristic_fallback" if not available else requested_backend,
            "protectedRegionBackendAvailable": available,
            "protectedRegionBackendFailureReason": failure_reason,
        }
    )
    return protected_mask, meta


def build_full_image_block_mask(
    image: Image.Image,
    block: TextBlock,
    *,
    protected_region_mask: Image.Image | None = None,
    dilation_px: int = 7,
    feather_px: int = 2,
    allow_protected_overlap: bool = False,
) -> dict[str, Any]:
    empty = Image.new("L", image.size, 0)
    combined_region = build_combined_block_cleanup_region(image, block)
    if combined_region is None:
        fallback = build_ocr_guided_openai_mask_fallback(
            image,
            block,
            protected_region_mask=protected_region_mask,
            dilation_px=dilation_px,
            feather_px=feather_px,
        )
        if fallback is not None:
            return fallback
        return {
            "raw": empty,
            "filled": empty,
            "dilated": empty,
            "final": empty,
            "maskPolarity": "white_inpaint_black_preserve",
            "whitePixelRatio": 0.0,
            "blackPixelRatio": 1.0,
            "textCoverageEstimate": 0.0,
            "backgroundLeakageEstimate": 0.0,
            "maskQualityStatus": "failed",
            "maskFailureReason": "mask_too_small",
        }
    bundle = build_precise_text_stroke_mask(
        image,
        block,
        protected_region_mask=protected_region_mask,
        dilation_px=dilation_px,
        feather_px=feather_px,
        allow_protected_overlap=allow_protected_overlap,
    )
    binary_mask = bundle.get("inpaintMask_binary")
    binary_white_ratio = float(bundle.get("whitePixelRatio", 0.0))
    binary_has_pixels = isinstance(binary_mask, Image.Image) and bool(np.any(np.array(binary_mask.convert("L")) > 0))
    if bundle.get("maskQualityStatus") == "passed" and binary_has_pixels and binary_white_ratio > 0.0005:
        return bundle
    fallback = build_ocr_guided_openai_mask_fallback(
        image,
        block,
        protected_region_mask=protected_region_mask,
        dilation_px=dilation_px,
        feather_px=feather_px,
    )
    if fallback is not None:
        return fallback
    return bundle


def build_ocr_guided_openai_mask_fallback(
    image: Image.Image,
    block: TextBlock,
    *,
    protected_region_mask: Image.Image | None = None,
    dilation_px: int = 7,
    feather_px: int = 2,
) -> dict[str, Any] | None:
    import cv2

    boxes = list(block.line_boxes or [])
    if not boxes:
        boxes = [block.clean_box or block.bbox]
    try:
        full_image_ocr_boxes = detect_ocr_boxes_on_crop(image)
    except Exception:
        full_image_ocr_boxes = []
    expanded_block_box = expand_bbox(block.clean_box or block.bbox, image.size, 18)
    footer_boxes: list[tuple[int, int, int, int]] = []
    for detection in full_image_ocr_boxes:
        det_box = tuple(int(v) for v in detection.get("bbox", []))
        if len(det_box) != 4:
            continue
        left, top, right, bottom = det_box
        width = max(1, right - left)
        height = max(1, bottom - top)
        normalized = normalize_match_text(str(detection.get("text", "")))
        if not normalized:
            continue
        overlaps_block = overlap_fraction(det_box, expanded_block_box) > 0.05
        footer_like = (
            bottom >= int(image.height * 0.78)
            and height <= max(24, int(image.height * 0.06))
            and width <= int(image.width * 0.72)
            and left <= int(image.width * 0.45)
        )
        if overlaps_block or footer_like:
            footer_boxes.append(det_box)
    boxes.extend(footer_boxes)
    if not boxes:
        return None

    mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
    for left, top, right, bottom in boxes:
        if right <= left or bottom <= top:
            continue
        pad_x = max(4, int(round(max(1, right - left) * 0.02)))
        pad_y = max(4, int(round(max(1, bottom - top) * 0.10)))
        x1 = max(0, left - pad_x)
        y1 = max(0, top - pad_y)
        x2 = min(image.width, right + pad_x)
        y2 = min(image.height, bottom + pad_y)
        mask_array[y1:y2, x1:x2] = 255

    if not np.any(mask_array):
        return None

    kernel_size = max(3, min(41, dilation_px * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    if protected_region_mask is not None:
        before_protected_pixels = int((mask_array > 0).sum())
        protected = np.array(protected_region_mask.resize(image.size).convert("L"), dtype=np.uint8) > 16
        protected_subtracted = mask_array.copy()
        protected_subtracted[protected] = 0
        after_protected_pixels = int((protected_subtracted > 0).sum())
        if before_protected_pixels > 0 and after_protected_pixels >= max(24, int(before_protected_pixels * 0.12)):
            mask_array = protected_subtracted
    binary_mask = Image.fromarray(np.where(mask_array > 0, 255, 0).astype(np.uint8), "L")
    soft_mask = binary_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather_px))).convert("L")
    white_ratio = float((np.array(binary_mask, dtype=np.uint8) > 0).mean())
    if white_ratio <= 0.0005:
        return None
    empty_mask = Image.new("L", image.size, 0)
    return {
        "raw": binary_mask.copy(),
        "filled": binary_mask.copy(),
        "dilated": binary_mask.copy(),
        "final": soft_mask.copy(),
        "inpaintMask_binary": binary_mask.copy(),
        "compositeMask_soft": soft_mask.copy(),
        "syntheticGlyphMask": binary_mask.copy(),
        "adaptiveColorClusterMask": empty_mask.copy(),
        "localContrastMask": empty_mask.copy(),
        "edgeStrokeMask": empty_mask.copy(),
        "sourceTextPixelMask": binary_mask.copy(),
        "combinedBinaryMaskBeforeDilation": binary_mask.copy(),
        "maskPolarity": "white_inpaint_black_preserve",
        "whitePixelRatio": white_ratio,
        "blackPixelRatio": 1.0 - white_ratio,
        "textCoverageEstimate": 0.82,
        "antiAliasCoverageEstimate": 0.72,
        "backgroundLeakageEstimate": min(0.32, white_ratio * 1.35),
        "protectedObjectOverlapEstimate": 0.0,
        "textPixelDetectionMethodsUsed": ["ocr_line_box_fallback"],
        "maskQualityStatus": "passed",
        "maskFailureReason": "",
        "maskWarnings": ["ocr_line_box_fallback_used"],
    }


def build_deep_fill_mask(
    image: Image.Image,
    block: TextBlock,
    protected_region_mask: Image.Image | None,
    *,
    padding: int = 20,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = block.clean_box or block.bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    mask_array = np.zeros((image.height, image.width), dtype=np.uint8)
    if right > left and bottom > top:
        mask_array[top:bottom, left:right] = 255
    protected_removed = 0
    if protected_region_mask is not None:
        import cv2

        protected = np.array(protected_region_mask.resize(image.size).convert("L"), dtype=np.uint8) > 16
        protected_core_u8 = protected.astype(np.uint8) * 255
        core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        mask_area = max(1, int((mask_array > 0).sum()))
        for _ in range(5):
            protected_core_u8 = cv2.erode(protected_core_u8, core_kernel, iterations=1)
            if np.logical_and(mask_array > 0, protected_core_u8 > 16).sum() < mask_area * 0.72:
                break
        subtraction_mask = protected_core_u8 > 16
        if np.logical_and(mask_array > 0, subtraction_mask).sum() >= mask_area * 0.72:
            subtraction_mask = np.zeros_like(subtraction_mask, dtype=bool)
        protected_removed = int(np.logical_and(mask_array > 0, subtraction_mask).sum())
        mask_array[subtraction_mask] = 0
    area = int((mask_array > 0).sum())
    return Image.fromarray(mask_array, "L"), {
        "deepFillBox": [left, top, right, bottom],
        "deepFillArea": area,
        "protectedPixelsRemoved": protected_removed,
        "generated": area > 24,
    }


def block_uses_deep_fill(block: TextBlock, image_size: tuple[int, int]) -> bool:
    return (block.bbox[3] - block.bbox[1]) / max(1, image_size[1]) > 0.15


def build_cleanup_failed_preview(image: Image.Image, message: str) -> Image.Image:
    preview = image.convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (20, 16, 16, 0))
    draw = ImageDraw.Draw(overlay)
    band_height = max(120, int(preview.height * 0.16))
    band_top = max(0, (preview.height - band_height) // 2)
    draw.rectangle((0, band_top, preview.width, band_top + band_height), fill=(121, 28, 48, 220))
    font = get_font(max(28, min(56, preview.width // 18)), bold=True)
    text = message.upper()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.text(
        ((preview.width - text_width) / 2, band_top + (band_height - text_height) / 2 - text_bbox[1]),
        text,
        fill=(255, 245, 245, 255),
        font=font,
    )
    return Image.alpha_composite(preview, overlay).convert("RGB")


def score_cleanup_candidate_detailed(
    original_crop: Image.Image,
    cleaned_crop: Image.Image,
    local_mask: Image.Image,
    protected_crop_mask: Image.Image | None = None,
    *,
    source_text_hint: str | None = None,
    candidate_name: str = "",
) -> dict[str, Any]:
    import cv2

    original = np.array(original_crop.convert("RGB"))
    cleaned = np.array(cleaned_crop.convert("RGB"))
    mask = np.array(local_mask) > 16
    if original.shape[:2] != cleaned.shape[:2]:
        shared_h = min(original.shape[0], cleaned.shape[0], mask.shape[0])
        shared_w = min(original.shape[1], cleaned.shape[1], mask.shape[1])
        original = original[:shared_h, :shared_w]
        cleaned = cleaned[:shared_h, :shared_w]
        mask = mask[:shared_h, :shared_w]
        if protected_crop_mask is not None:
            protected_crop_mask = protected_crop_mask.crop((0, 0, shared_w, shared_h))
    if original.size == 0 or cleaned.size == 0 or not mask.any():
        return {
            "editableMaskInsideChange": 0.0,
            "protectedRegionChange": 0.0,
            "unmaskedBackgroundChange": 0.0,
            "residualTextInMask": 1.0,
            "edgeContinuityAroundMask": 0.0,
            "ocrResidualScore": 1.0,
            "ghostingScore": 1.0,
            "ghostingPenalty": 1.0,
            "blurPenalty": 1.0,
            "textureLossPenalty": 1.0,
            "artifactPenalty": 1.0,
            "naturalnessScore": 0.0,
            "gradientContinuityScore": 0.0,
            "textureRealismScore": 0.0,
            "noiseDistributionScore": 0.0,
            "residualTextScore": 0.0,
            "backgroundContinuityScore": 0.0,
            "protectedRegionChangeScore": 0.0,
            "objectPreservationScore": 0.0,
            "edgeArtifactScore": 0.0,
            "blurScore": 0.0,
            "colorShiftScore": 0.0,
            "structureChangeScore": 0.0,
            "finalScore": 0.0,
            "hardReject": True,
            "hardRejectReasons": ["empty crop or mask"],
            "rejectionReasons": ["empty crop or mask"],
            "whyOpenCVLost": "",
            "whyGenerativeWon": "",
        }
    cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Canny(cleaned_gray, 80, 160) > 0
    original_edge_map = cv2.Canny(original_gray, 80, 160) > 0
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dilated, ~mask)
    if not ring.any():
        ring = np.logical_not(mask)
    protected = np.zeros_like(mask, dtype=bool)
    if protected_crop_mask is not None:
        protected = np.array(protected_crop_mask.convert("L")) > 16
    protected_only = np.logical_and(protected, ~mask)
    background_only = np.logical_and(~mask, ~protected)
    if not background_only.any():
        background_only = ring

    abs_diff = np.abs(cleaned.astype(np.float32) - original.astype(np.float32))
    editable_change = float(abs_diff[mask].mean() / 255.0) if mask.any() else 0.0
    protected_change = float(abs_diff[protected_only].mean() / 255.0) if protected_only.any() else 0.0
    background_change = float(abs_diff[background_only].mean() / 255.0) if background_only.any() else 0.0

    cleaned_mask_pixels = cleaned[mask]
    ring_pixels = original[ring]
    background_pixels_cleaned = cleaned[background_only]
    background_pixels_original = original[background_only]

    original_mask_variance = float(cv2.Laplacian(original_gray, cv2.CV_32F)[mask].var()) if mask.any() else 0.0
    cleaned_mask_variance = float(cv2.Laplacian(cleaned_gray, cv2.CV_32F)[mask].var()) if mask.any() else 0.0
    ring_variance = float(cv2.Laplacian(original_gray, cv2.CV_32F)[ring].var()) if ring.any() else original_mask_variance

    color_gap = 0.0
    if len(background_pixels_cleaned) and len(background_pixels_original):
        color_gap = float(np.linalg.norm(background_pixels_cleaned.mean(axis=0) - background_pixels_original.mean(axis=0)))
    elif len(cleaned_mask_pixels) and len(ring_pixels):
        color_gap = float(np.linalg.norm(cleaned_mask_pixels.mean(axis=0) - ring_pixels.mean(axis=0)))

    residual_edges_cleaned = float(edge_map[mask].mean())
    residual_edges_original = max(0.02, float(original_edge_map[mask].mean()))
    residual_ratio = residual_edges_cleaned / residual_edges_original
    ocr_residual_score, residual_ocr_texts = detect_residual_ocr_similarity(cleaned_crop, source_text_hint)
    cleaned_mask_mean = float(cleaned_gray[mask].mean()) if mask.any() else 0.0
    ring_mean = float(original_gray[ring].mean()) if ring.any() else float(original_gray[mask].mean())
    original_delta = abs(float(original_gray[mask].mean()) - ring_mean) / 255.0 if mask.any() else 0.0
    cleaned_delta = abs(cleaned_mask_mean - ring_mean) / 255.0
    low_contrast_ghost = _clamp01(cleaned_delta / max(0.06, original_delta if original_delta > 0 else 0.12))
    ghosting_score = _clamp01(max(ocr_residual_score * 0.82, residual_ratio * 0.4, low_contrast_ghost * 0.55))
    residual_text_score = _clamp01((1.0 - ocr_residual_score) * 0.62 + (1.0 - ghosting_score) * 0.38)

    cleaned_laplacian = float(cv2.Laplacian(cleaned_gray, cv2.CV_32F)[background_only].var()) if background_only.any() else 0.0
    original_laplacian = float(cv2.Laplacian(original_gray, cv2.CV_32F)[background_only].var()) if background_only.any() else 0.0
    blur_gap = abs(cleaned_laplacian - original_laplacian)
    blur_score = _clamp01(1.0 - min(1.0, blur_gap / 2200.0))
    blur_penalty = _clamp01(
        max(
            0.0,
            1.0 - (cleaned_mask_variance / max(1.0, ring_variance)),
            1.0 - (cleaned_mask_variance / max(1.0, original_mask_variance)),
        )
    )

    protected_structure_gap = 0.0
    if protected_only.any():
        protected_structure_gap = float(
            np.abs(
                cv2.Laplacian(cleaned_gray, cv2.CV_32F)[protected_only]
                - cv2.Laplacian(original_gray, cv2.CV_32F)[protected_only]
            ).mean()
        )
    structure_change_score = _clamp01(1.0 - min(1.0, protected_structure_gap / 90.0))

    border = np.logical_xor(dilated, cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool))
    seam_penalty = 0.0
    if border.any():
        seam_penalty = float(np.abs(cleaned.astype(np.float32) - original.astype(np.float32))[border].mean())
    edge_artifact_score = _clamp01(1.0 - min(1.0, seam_penalty / 70.0))
    artifact_penalty = _clamp01(1.0 - edge_artifact_score)
    edge_continuity = edge_artifact_score
    color_shift_score = _clamp01(1.0 - min(1.0, color_gap / 120.0))
    protected_region_change_score = _clamp01(1.0 - min(1.0, protected_change / 0.18))
    object_preservation_score = _clamp01(0.6 * protected_region_change_score + 0.4 * structure_change_score)
    background_continuity_score = _clamp01(
        0.45 * (1.0 - min(1.0, background_change / 0.22))
        + 0.3 * color_shift_score
        + 0.25 * blur_score
    )

    texture_realism_score = _clamp01(1.0 - abs(cleaned_mask_variance - ring_variance) / max(800.0, ring_variance * 3.0 + 1.0))
    noise_distribution_score = _clamp01(1.0 - abs(float(cleaned[mask].std()) - float(original[ring].std() if ring.any() else original[mask].std())) / 90.0)
    gradient_ring = cv2.Sobel(original_gray, cv2.CV_32F, 1, 1, ksize=3)
    gradient_clean = cv2.Sobel(cleaned_gray, cv2.CV_32F, 1, 1, ksize=3)
    gradient_gap = float(np.abs(gradient_clean[border] - gradient_ring[border]).mean()) if border.any() else 0.0
    gradient_continuity_score = _clamp01(1.0 - min(1.0, gradient_gap / 55.0))
    texture_loss_penalty = _clamp01(1.0 - texture_realism_score)
    naturalness_score = _clamp01(
        0.25 * background_continuity_score
        + 0.2 * texture_realism_score
        + 0.2 * noise_distribution_score
        + 0.15 * gradient_continuity_score
        + 0.2 * (1.0 - blur_penalty)
    )

    final_score = (
        residual_text_score * 0.35
        + naturalness_score * 0.30
        + object_preservation_score * 0.20
        + edge_artifact_score * 0.10
        - blur_penalty * 0.15
        - ghosting_score * 0.20
    )
    if candidate_name.startswith("generative:") and residual_text_score >= 0.58 and naturalness_score >= 0.7 and protected_region_change_score >= 0.9:
        final_score += 0.08
    if candidate_name.startswith("generative:") and residual_text_score >= 0.34 and naturalness_score >= 0.62 and protected_region_change_score >= 0.92:
        final_score += 0.12
    if candidate_name == "reconstruction-touchup" and blur_penalty > 0.9:
        final_score -= 0.18 * blur_penalty
    final_score = _clamp01(final_score)

    hard_reject_reasons: list[str] = []
    if protected_region_change_score < 0.34:
        hard_reject_reasons.append("protected region changed too much")
    if object_preservation_score < 0.30:
        hard_reject_reasons.append("object preservation score too low")
    if structure_change_score < 0.24:
        hard_reject_reasons.append("structure changed too much")
    if candidate_name == "reconstruction-touchup" and blur_penalty > 0.36 and ghosting_score > 0.16:
        hard_reject_reasons.append("reconstruction hard reject: blur and ghosting both high")
    if candidate_name == "reconstruction-touchup" and (ocr_residual_score > 0.3 or ghosting_score > 0.3):
        hard_reject_reasons.append("reconstruction hard reject: source text still readable after cleanup")

    rejection_reasons: list[str] = []
    if residual_text_score < 0.35:
        rejection_reasons.append("residual text remains in editable mask")
    if ocr_residual_score > 0.45:
        rejection_reasons.append("ocr still detects source text")
    if background_continuity_score < 0.42:
        rejection_reasons.append("background continuity is weak")
    if edge_artifact_score < 0.42:
        rejection_reasons.append("edge artifacts visible around cleaned mask")
    if protected_region_change_score < 0.52:
        rejection_reasons.append("protected region changed noticeably")
    if blur_penalty > 0.34:
        rejection_reasons.append("blur penalty too high")
    if ghosting_score > 0.22:
        rejection_reasons.append("ghosting remains visible")

    return {
        "editableMaskInsideChange": editable_change,
        "protectedRegionChange": protected_change,
        "unmaskedBackgroundChange": background_change,
        "residualTextInMask": float(residual_edges_cleaned),
        "edgeContinuityAroundMask": edge_continuity,
        "ocrResidualScore": float(ocr_residual_score),
        "ocrResidualTexts": residual_ocr_texts,
        "ghostingScore": float(ghosting_score),
        "ghostingPenalty": float(ghosting_score),
        "blurPenalty": float(blur_penalty),
        "textureLossPenalty": float(texture_loss_penalty),
        "artifactPenalty": float(artifact_penalty),
        "naturalnessScore": float(naturalness_score),
        "gradientContinuityScore": float(gradient_continuity_score),
        "textureRealismScore": float(texture_realism_score),
        "noiseDistributionScore": float(noise_distribution_score),
        "residualTextScore": residual_text_score,
        "backgroundContinuityScore": background_continuity_score,
        "protectedRegionChangeScore": protected_region_change_score,
        "objectPreservationScore": object_preservation_score,
        "edgeArtifactScore": edge_artifact_score,
        "blurScore": blur_score,
        "colorShiftScore": color_shift_score,
        "structureChangeScore": structure_change_score,
        "finalScore": final_score,
        "hardReject": bool(hard_reject_reasons),
        "hardRejectReasons": hard_reject_reasons,
        "rejectionReasons": rejection_reasons,
        "whyOpenCVLost": "reconstruction candidate penalized for blur/ghosting bias" if candidate_name == "reconstruction-touchup" and (blur_penalty > 0.24 or ghosting_score > 0.2) else "",
        "whyGenerativeWon": "generative candidate preserved object regions while improving naturalness" if candidate_name.startswith("generative:") and final_score > 0.7 else "",
    }


def score_cleanup_candidate(original_crop: Image.Image, cleaned_crop: Image.Image, local_mask: Image.Image) -> float:
    return float(score_cleanup_candidate_detailed(original_crop, cleaned_crop, local_mask).get("finalScore", 0.0))


def cleanup_block_with_sampled_fill(image: Image.Image, region: dict[str, Any]) -> Image.Image:
    expanded = region["expanded_box"]
    local_mask = region["mask"]
    crop = image.crop(expanded).convert("RGB")
    rebuilt = reconstruct_overlay_background(
        image,
        [TextBlock(text="", role="body", translate=True, bbox=region["token_box"], clean_box=expanded, color="#111111", surface="overlay")],
    ).crop(expanded)
    feather = local_mask.filter(ImageFilter.GaussianBlur(radius=2)).convert("L")
    return Image.composite(rebuilt, crop, feather)


def cleanup_block_with_masked_blur(image: Image.Image, region: dict[str, Any]) -> Image.Image:
    expanded = region["expanded_box"]
    local_mask = region["mask"]
    crop = image.crop(expanded).convert("RGB")
    blur_radius = max(6, min(18, int(round(max(crop.size) * 0.03))))
    softened = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    feather = local_mask.filter(ImageFilter.GaussianBlur(radius=3)).convert("L")
    return Image.composite(softened, crop, feather)


def cleanup_line_with_mask_guided_reconstruction(image: Image.Image, region: dict[str, Any]) -> Image.Image:
    import cv2

    expanded = region["expanded_box"]
    local_mask = region["mask"].convert("L")
    crop = image.crop(expanded).convert("RGB")
    crop_np = np.array(crop)
    mask_np = (np.array(local_mask) > 16).astype(np.uint8) * 255
    if crop_np.size == 0 or mask_np.max() == 0:
        return crop

    rebuilt = reconstruct_overlay_background(
        image,
        [TextBlock(text="", role="body", translate=True, bbox=region["token_box"], clean_box=expanded, color="#111111", surface="overlay")],
    ).crop(expanded).convert("RGB")
    rebuilt_np = np.array(rebuilt, dtype=np.uint8)
    blurred = cv2.bilateralFilter(rebuilt_np, d=9, sigmaColor=36, sigmaSpace=18)
    blended = cv2.addWeighted(blurred, 0.82, rebuilt_np, 0.18, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = cv2.dilate(mask_np, kernel, iterations=1) - mask_np
    if np.count_nonzero(ring) > 12:
        ring_pixels = crop_np[ring > 0]
        ring_mean = ring_pixels.mean(axis=0)
        mask_pixels = blended[mask_np > 0]
        if len(mask_pixels):
            mask_mean = mask_pixels.mean(axis=0)
            shift = ring_mean - mask_mean
            adjusted = blended.astype(np.float32)
            adjusted[mask_np > 0] = np.clip(adjusted[mask_np > 0] + shift * 0.45, 0, 255)
            blended = adjusted.astype(np.uint8)

    feather = local_mask.filter(ImageFilter.GaussianBlur(radius=2)).convert("L")
    return Image.composite(Image.fromarray(blended, "RGB"), crop, feather)


def cleanup_region_with_opencv_inpaint(image: Image.Image, region: dict[str, Any]) -> Image.Image:
    import cv2

    expanded = region["expanded_box"]
    local_mask = region["mask"].convert("L")
    crop = image.crop(expanded).convert("RGB")
    crop_np = np.array(crop, dtype=np.uint8)
    mask_np = (np.array(local_mask) > 12).astype(np.uint8) * 255
    if crop_np.size == 0 or int(np.count_nonzero(mask_np)) == 0:
        return crop
    block_h = int(region.get("block_height") or crop.height)
    radius = max(3, min(11, int(round(block_h * 0.18))))
    kernel_size = max(3, min(9, radius | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    telea = cv2.inpaint(crop_np, mask_np, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(crop_np, mask_np, max(3, radius - 1), cv2.INPAINT_NS)
    blended = cv2.addWeighted(telea, 0.72, ns, 0.28, 0)
    feather = Image.fromarray(mask_np, "L").filter(ImageFilter.GaussianBlur(radius=2))
    return Image.composite(Image.fromarray(blended, "RGB"), crop, feather)


def prepare_generative_edit_context(
    image: Image.Image,
    local_mask: Image.Image,
    context: dict[str, Any],
    protected_region_mask: Image.Image | None,
    block: TextBlock,
) -> dict[str, Any] | None:
    expanded = context["expanded_box"]
    local_mask_l = local_mask.convert("L")
    local_mask_array = np.array(local_mask_l) > 16
    if not local_mask_array.any():
        return None

    protected_crop = Image.new("L", local_mask_l.size, 0)
    if protected_region_mask is not None:
        protected_crop = protected_region_mask.crop(expanded).convert("L")
    protected_array = np.array(protected_crop) > 16

    editable_array = np.logical_and(local_mask_array, ~protected_array)
    editable_pixels = int(editable_array.sum())
    total_pixels = int(local_mask_array.sum())
    if editable_pixels < max(24, int(total_pixels * 0.12)):
        return {
            "blocked": True,
            "reason": "editable mask too small after protected subtraction",
            "editable_mask_preview": Image.fromarray((editable_array.astype(np.uint8) * 255), "L"),
            "protected_subtracted_mask_preview": Image.fromarray((editable_array.astype(np.uint8) * 255), "L"),
        }

    import cv2

    editable_uint8 = editable_array.astype(np.uint8) * 255
    dilation = max(1, min(4, int(round(block.font_size_estimate * 0.04))))
    kernel_size = max(3, dilation * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(editable_uint8, kernel, iterations=1)
    if protected_array.any():
        dilated[np.logical_and(dilated > 16, protected_array)] = 0
    if np.count_nonzero(dilated) < max(24, editable_pixels):
        dilated = editable_uint8

    ys, xs = np.where(dilated > 16)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "blocked": True,
            "reason": "editable mask vanished after refinement",
            "editable_mask_preview": Image.fromarray(editable_uint8, "L"),
            "protected_subtracted_mask_preview": Image.fromarray(dilated, "L"),
        }

    is_block_level = int(context.get("line_index", 0)) == -1
    if is_block_level:
        pad_x = min(56, max(18, int(round(block.font_size_estimate * 0.42))))
        pad_y = min(42, max(14, int(round(block.line_height_estimate * 0.38))))
    else:
        pad_x = min(18, max(6, int(round(block.font_size_estimate * 0.22))))
        pad_y = min(16, max(6, int(round(block.line_height_estimate * 0.22))))
    x1 = max(0, int(xs.min()) - pad_x)
    y1 = max(0, int(ys.min()) - pad_y)
    x2 = min(local_mask_l.width, int(xs.max()) + 1 + pad_x)
    y2 = min(local_mask_l.height, int(ys.max()) + 1 + pad_y)

    safe_crop_box = (
        expanded[0] + x1,
        expanded[1] + y1,
        expanded[0] + x2,
        expanded[1] + y2,
    )
    crop = image.crop(safe_crop_box).convert("RGB")
    editable_mask_crop = Image.fromarray(dilated[y1:y2, x1:x2], "L")
    protected_crop_local = protected_crop.crop((x1, y1, x2, y2)).convert("L")
    feather_mask = editable_mask_crop.filter(ImageFilter.GaussianBlur(radius=2)).convert("L")

    return {
        "blocked": False,
        "safe_crop_box": safe_crop_box,
        "safe_crop_local_box": (x1, y1, x2, y2),
        "original_crop": crop,
        "editable_mask": editable_mask_crop,
        "protected_crop_mask": protected_crop_local,
        "editable_mask_preview": editable_mask_crop,
        "protected_subtracted_mask_preview": editable_mask_crop,
        "feather_mask": feather_mask,
    }


def pad_image_and_mask_for_edit(
    crop: Image.Image,
    editable_mask: Image.Image,
    protected_crop_mask: Image.Image,
    *,
    min_side: int = 256,
) -> dict[str, Any]:
    width, height = crop.size
    target_width = max(min_side, width)
    target_height = max(min_side, height)
    pad_left = max(0, (target_width - width) // 2)
    pad_top = max(0, (target_height - height) // 2)
    padded_crop = Image.new("RGB", (target_width, target_height), tuple(np.array(crop).reshape(-1, 3).mean(axis=0).astype(np.uint8)))
    padded_crop.paste(crop, (pad_left, pad_top))

    padded_editable = Image.new("L", (target_width, target_height), 0)
    padded_editable.paste(editable_mask.convert("L"), (pad_left, pad_top))

    padded_protected = Image.new("L", (target_width, target_height), 0)
    padded_protected.paste(protected_crop_mask.convert("L"), (pad_left, pad_top))

    return {
        "crop": padded_crop,
        "editable_mask": padded_editable,
        "protected_crop_mask": padded_protected,
        "paste_offset": (pad_left, pad_top),
        "original_size": (width, height),
    }


def build_openai_edit_mask(editable_mask: Image.Image) -> Image.Image:
    alpha = editable_mask.convert("L")
    rgba_mask = Image.new("RGBA", editable_mask.size, (255, 255, 255, 255))
    rgba_mask.putalpha(ImageChops.invert(alpha))
    return rgba_mask


def _sanitize_inpaint_result(
    source: Image.Image,
    composited: Image.Image,
    mask: Image.Image,
    divergence_threshold: float = 60.0,
) -> Image.Image:
    """
    Detect and suppress AI hallucination in inpainted regions.

    Uses two independent signals:
    1. Color divergence: inpainted mean color vs surrounding border median.
    2. Edge-density ratio: if the inpainted region has significantly more
       edges/structure than the surrounding source area, the model hallucinated
       new imagery (icons, objects, text) â€” suppressed even when colours match.

    When either signal triggers, the inpainted region is blended back toward
    the local median background color.
    """
    try:
        mask_arr = np.array(mask.convert("L"))
        mask_bool = mask_arr > 16
        if not np.any(mask_bool):
            return composited

        src_arr = np.array(source, dtype=np.float32)
        comp_arr = np.array(composited, dtype=np.float32)

        # Dilate the mask to get a thin border strip in the source (non-masked)
        border_dilated = np.array(
            mask.filter(ImageFilter.MaxFilter(31)).convert("L")
        ) > 16
        border_only = border_dilated & ~mask_bool

        if not np.any(border_only):
            return composited

        border_colors = src_arr[border_only]
        border_median = np.median(border_colors, axis=0)

        inpainted_colors = comp_arr[mask_bool]
        inpainted_mean = np.mean(inpainted_colors, axis=0)
        color_divergence = float(np.linalg.norm(inpainted_mean - border_median))

        # â”€â”€ Edge-density hallucination detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Compute per-channel gradient magnitude (Sobel-like) for both the
        # composited inpainted region and the source border ring.
        def _edge_density(arr_f32: np.ndarray, region_bool: np.ndarray) -> float:
            gray = arr_f32[..., 0] * 0.299 + arr_f32[..., 1] * 0.587 + arr_f32[..., 2] * 0.114
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            grad = np.sqrt(gy ** 2 + gx ** 2)
            vals = grad[region_bool]
            return float(vals.mean()) if vals.size > 0 else 0.0

        inpaint_edge = _edge_density(comp_arr, mask_bool)
        border_edge = _edge_density(src_arr, border_only)
        # Ratio > 2.5 means the inpainted area has 2.5Ã— more structure than
        # the surrounding area â€” a strong hallucination signal.
        edge_ratio = inpaint_edge / max(border_edge, 1.0)
        structural_hallucination = edge_ratio > 2.5 and inpaint_edge > 8.0

        if color_divergence <= divergence_threshold and not structural_hallucination:
            return composited

        # Determine blend strength from the dominant signal
        color_blend = (color_divergence - divergence_threshold) / 80.0 if color_divergence > divergence_threshold else 0.0
        edge_blend = min(0.90, (edge_ratio - 2.5) / 4.0) if structural_hallucination else 0.0
        blend = min(0.92, max(color_blend, edge_blend))

        fill = np.full_like(comp_arr, border_median)
        comp_arr[mask_bool] = (1.0 - blend) * comp_arr[mask_bool] + blend * fill[mask_bool]
        return Image.fromarray(comp_arr.clip(0, 255).astype(np.uint8))
    except Exception:
        return composited


def run_openai_full_image_cleanup(image: Image.Image, editable_mask: Image.Image, prompt: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"success": False, "provider": "openai", "failureReason": "openai_not_configured"}
    model = os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
    with tempfile.TemporaryDirectory(prefix="adaptifai-openai-full-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        rgba_mask = build_openai_edit_mask(editable_mask)
        image_path, mask_path, alpha_preview_path = export_openai_edit_debug_artifacts(temp_dir, image.convert("RGB"), rgba_mask)
        request_meta = validate_openai_edit_request(image_path, mask_path, image.convert("RGB"), rgba_mask, model)
        if not request_meta["valid"]:
            return {
                "success": False,
                "provider": "openai",
                "failureReason": "openai_request_invalid",
                "requestMeta": request_meta,
                "requestImagePreview": image.convert("RGB"),
                "requestMaskPreview": rgba_mask,
                "requestMaskAlphaPreview": rgba_mask.getchannel("A"),
            }
        try:
            client = OpenAI()
            with image_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
                response = client.images.edit(
                    model=model,
                    image=image_file,
                    mask=mask_file,
                    prompt=prompt,
                    n=1,
                    output_format="png",
                )
            if response.data and getattr(response.data[0], "b64_json", None):
                result = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))).convert("RGB")
                if result.size != image.size:
                    result = result.resize(image.size, Image.Resampling.LANCZOS)
                feather = editable_mask.convert("L")
                composited = Image.composite(result, image.convert("RGB"), feather)
                # Hallucination safeguard: if AI result diverges dramatically from surrounding
                # background in the masked region, blend back toward a local median fill.
                composited = _sanitize_inpaint_result(image.convert("RGB"), composited, feather)
                return {
                    "success": True,
                    "provider": "openai",
                    "image": composited,
                    "requestMeta": request_meta,
                    "requestImagePreview": image.convert("RGB"),
                    "requestMaskPreview": rgba_mask,
                    "requestMaskAlphaPreview": rgba_mask.getchannel("A"),
                    "apiSuccessResultPreview": result.copy(),
                    "postCompositePreview": composited.copy(),
                }
            return {"success": False, "provider": "openai", "failureReason": "openai_no_image_returned", "requestMeta": request_meta}
        except Exception as exc:
            return {
                "success": False,
                "provider": "openai",
                "failureReason": "openai_api_error",
                "requestMeta": request_meta,
                "apiError": extract_openai_error_payload(exc, request_meta, model),
                "requestImagePreview": image.convert("RGB"),
                "requestMaskPreview": rgba_mask,
                "requestMaskAlphaPreview": rgba_mask.getchannel("A"),
            }


def run_vertex_full_image_cleanup(image: Image.Image, editable_mask: Image.Image, prompt: str) -> dict[str, Any]:
    if not vertex_available():
        return {"success": False, "provider": "vertex", "failureReason": "vertex_not_configured"}
    try:
        project_id = vertex_project_id()
        location = vertex_location()
        model = vertex_imagen_edit_model()
        endpoint = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{location}/publishers/google/models/{model}:predict"
        )
        binary_mask = Image.fromarray((np.array(editable_mask.convert("L")) > 16).astype(np.uint8) * 255, "L")
        mask_rgb = Image.merge("RGB", (binary_mask, binary_mask, binary_mask))
        response = vertex_authorized_session().post(
            endpoint,
            json={
                "instances": [
                    {
                        "prompt": prompt,
                        "referenceImages": [
                            {
                                "referenceType": "REFERENCE_TYPE_RAW",
                                "referenceId": 1,
                                "referenceImage": {"bytesBase64Encoded": image_to_base64_png(image.convert("RGB"))},
                            },
                            {
                                "referenceType": "REFERENCE_TYPE_MASK",
                                "referenceId": 2,
                                "referenceImage": {"bytesBase64Encoded": image_to_base64_png(mask_rgb)},
                                "maskImageConfig": {
                                    "maskMode": "MASK_MODE_USER_PROVIDED",
                                    "dilation": float(os.getenv("VERTEX_IMAGEN_MASK_DILATION", "0.02")),
                                },
                            },
                        ],
                    }
                ],
                "parameters": {
                    "sampleCount": 1,
                    "editMode": os.getenv("VERTEX_IMAGEN_INPAINT_EDIT_MODE", "EDIT_MODE_INPAINT_REMOVAL"),
                    "editConfig": {"baseSteps": int(os.getenv("VERTEX_IMAGEN_EDIT_STEPS", "35"))},
                    "safetyFilterLevel": os.getenv("VERTEX_IMAGEN_SAFETY_FILTER_LEVEL", "block_some"),
                    "personGeneration": os.getenv("VERTEX_IMAGEN_PERSON_GENERATION", "allow_adult"),
                },
            },
            timeout=int(os.getenv("VERTEX_IMAGEN_TIMEOUT", "45")),
        )
        response.raise_for_status()
        payload = response.json()
        predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
        if not predictions:
            return {"success": False, "provider": "vertex", "failureReason": "vertex_no_image_returned"}
        result = decode_vertex_imagen_prediction(predictions[0]).convert("RGB")
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        feather = editable_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=2))
        composited = Image.composite(result, image.convert("RGB"), feather)
        return {
            "success": True,
            "provider": "vertex",
            "image": composited,
            "model": model,
            "location": location,
            "requestImagePreview": image.convert("RGB"),
            "requestMaskAlphaPreview": binary_mask,
            "apiSuccessResultPreview": result.copy(),
            "postCompositePreview": composited.copy(),
        }
    except requests.HTTPError as exc:
        return {
            "success": False,
            "provider": "vertex",
            "failureReason": "vertex_api_error",
            "apiError": {"statusCode": getattr(exc.response, "status_code", None), "body": getattr(exc.response, "text", "")[:1200] if getattr(exc, "response", None) is not None else str(exc)},
        }
    except Exception as exc:
        return {"success": False, "provider": "vertex", "failureReason": f"vertex_error:{exc.__class__.__name__}", "apiError": {"message": str(exc)[:1200]}}


def build_overlay_provider_composite_mask(image_size: tuple[int, int], block: TextBlock) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    boxes = [tuple(box) for box in (block.line_boxes or [])] or [block.clean_box or block.bbox]
    for left, top, right, bottom in boxes:
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        pad_x = max(3, int(round(box_w * 0.025)))
        pad_y = max(3, int(round(box_h * 0.08)))
        draw.rounded_rectangle(
            (
                max(0, left - pad_x),
                max(0, top - pad_y),
                min(image_size[0], right + pad_x),
                min(image_size[1], bottom + pad_y),
            ),
            radius=max(3, min(10, box_h // 12)),
            fill=255,
        )
    return mask.filter(ImageFilter.GaussianBlur(radius=4)).convert("L")


def composite_overlay_provider_result(source: Image.Image, provider_result: Image.Image, block: TextBlock, request_mask: Image.Image | None = None) -> Image.Image:
    soft_mask = (
        request_mask.convert("L").filter(ImageFilter.MaxFilter(size=17)).filter(ImageFilter.GaussianBlur(radius=4))
        if request_mask is not None
        else build_overlay_provider_composite_mask(source.size, block)
    )
    try:
        import cv2

        hard_mask = (np.array(soft_mask.convert("L")) > 16).astype(np.uint8) * 255
        if int(np.count_nonzero(hard_mask)) == 0:
            return Image.composite(provider_result.convert("RGB"), source.convert("RGB"), soft_mask)
        boxes = [tuple(box) for box in (block.line_boxes or [])] or [block.clean_box or block.bbox]
        union = union_bbox(boxes) or (block.clean_box or block.bbox)
        center = (int((union[0] + union[2]) / 2), int((union[1] + union[3]) / 2))
        src_np = cv2.cvtColor(np.array(provider_result.convert("RGB")), cv2.COLOR_RGB2BGR)
        dst_np = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
        cloned = cv2.seamlessClone(src_np, dst_np, hard_mask, center, cv2.NORMAL_CLONE)
        return Image.fromarray(cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.composite(provider_result.convert("RGB"), source.convert("RGB"), soft_mask)


def provider_raw_result_safe_for_overlay(source: Image.Image, raw: Image.Image, block: TextBlock) -> bool:
    try:
        mask = build_overlay_provider_composite_mask(source.size, block)
        mask_np = np.array(mask.convert("L")) > 16
        if not mask_np.any():
            return False
        source_np = np.array(source.convert("RGB"), dtype=np.uint8)
        raw_np = np.array(raw.resize(source.size, Image.Resampling.LANCZOS).convert("RGB"), dtype=np.uint8)
        source_luma = source_np @ np.array([0.299, 0.587, 0.114])
        raw_luma = raw_np @ np.array([0.299, 0.587, 0.114])
        new_bright = np.logical_and(raw_luma > 218, source_luma < 198)
        new_bright_ratio = float(new_bright[mask_np].mean())
        diff = np.abs(raw_np.astype(np.float32) - source_np.astype(np.float32)).mean(axis=2)
        large_change_ratio = float((diff[mask_np] > 72).mean())
        return new_bright_ratio < 0.008 and large_change_ratio < 0.45
    except Exception:
        return False


def cleanup_overlay_marketing_with_full_image_provider(
    image: Image.Image,
    block: TextBlock,
    region: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    request_mask = Image.new("L", image.size, 0)
    request_mask.paste(region["mask"].convert("L"), region["expanded_box"])
    request_mask = request_mask.filter(ImageFilter.MaxFilter(size=3))
    composite_mask = build_overlay_provider_composite_mask(image.size, block)
    attempts: list[dict[str, Any]] = []

    if os.getenv("OPENAI_API_KEY"):
        openai_result = run_openai_full_image_cleanup(image, request_mask, prompt)
        attempts.append(openai_result)
        raw_candidate = openai_result.get("apiSuccessResultPreview")
        raw_safe = isinstance(raw_candidate, Image.Image) and provider_raw_result_safe_for_overlay(image, raw_candidate, block)
        raw = raw_candidate if raw_safe else None
        if openai_result.get("success") and isinstance(raw, Image.Image):
            composited = composite_overlay_provider_result(image, raw.convert("RGB"), block, request_mask)
            return {
                "success": True,
                "provider": "openai",
                "image": composited,
                "requestMaskPreview": request_mask,
                "compositeMaskPreview": composite_mask,
                "attempts": attempts,
                **{key: value for key, value in openai_result.items() if key not in {"image"}},
            }

    if vertex_available():
        vertex_result = run_vertex_full_image_cleanup(image, request_mask, prompt)
        attempts.append(vertex_result)
        raw_candidate = vertex_result.get("apiSuccessResultPreview")
        raw = raw_candidate if isinstance(raw_candidate, Image.Image) and provider_raw_result_safe_for_overlay(image, raw_candidate, block) else None
        if vertex_result.get("success") and isinstance(raw, Image.Image):
            composited = composite_overlay_provider_result(image, raw.convert("RGB"), block, request_mask)
            return {
                "success": True,
                "provider": "vertex",
                "image": composited,
                "requestMaskPreview": request_mask,
                "compositeMaskPreview": composite_mask,
                "attempts": attempts,
                **{key: value for key, value in vertex_result.items() if key not in {"image"}},
            }

    return {
        "success": False,
        "provider": "none",
        "failureReason": "no_full_image_provider_succeeded",
        "requestMaskPreview": request_mask,
        "compositeMaskPreview": composite_mask,
        "attempts": attempts,
    }


def run_huggingface_full_image_cleanup(
    image: Image.Image,
    editable_mask: Image.Image,
    *,
    validation_callback: Any | None = None,
    stage_label: str = "cleanup",
) -> dict[str, Any]:
    result = cleanup_via_huggingface_v3(
        image,
        editable_mask,
        validation_callback=validation_callback,
        stage_label=stage_label,
    )
    return {
        **result,
        "provider": "huggingface",
        "huggingFaceInputImagePreview": image.convert("RGB").copy(),
        "huggingFaceMaskPreview": prepare_huggingface_cleanup_mask(
            editable_mask,
            dilation_px=int(os.getenv("ADAPTIFAI_HF_MASK_DILATION", "15")),
            feather_px=int(os.getenv("ADAPTIFAI_HF_MASK_FEATHER", "5")),
        ),
        "huggingFaceOutputPreview": (
            result.get("image").copy()
            if isinstance(result.get("image"), Image.Image)
            else result.get("bestEffortImage").copy()
            if isinstance(result.get("bestEffortImage"), Image.Image)
            else None
        ),
    }


def export_openai_edit_debug_artifacts(
    temp_dir: Path,
    crop: Image.Image,
    rgba_mask: Image.Image,
) -> tuple[Path, Path, Path]:
    image_path = temp_dir / "request_image.png"
    mask_path = temp_dir / "request_mask.png"
    alpha_preview_path = temp_dir / "request_mask_alpha.png"
    crop.save(image_path, format="PNG")
    rgba_mask.save(mask_path, format="PNG")
    rgba_mask.getchannel("A").save(alpha_preview_path, format="PNG")
    return image_path, mask_path, alpha_preview_path


def validate_openai_edit_request(image_path: Path, mask_path: Path, crop: Image.Image, rgba_mask: Image.Image, model: str | None = None) -> dict[str, Any]:
    alpha = np.array(rgba_mask.getchannel("A"))
    transparent_pixels = int((alpha == 0).sum())
    opaque_pixels = int((alpha == 255).sum())
    meta = {
        "model": model or os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1",
        "imageCropWidth": crop.size[0],
        "imageCropHeight": crop.size[1],
        "maskWidth": rgba_mask.size[0],
        "maskHeight": rgba_mask.size[1],
        "imageFileFormat": "PNG",
        "maskFileFormat": "PNG",
        "imageFileSize": image_path.stat().st_size if image_path.exists() else 0,
        "maskFileSize": mask_path.stat().st_size if mask_path.exists() else 0,
        "maskMode": rgba_mask.mode,
        "maskChannels": len(rgba_mask.getbands()),
        "maskAlphaMin": int(alpha.min()) if alpha.size else 255,
        "maskAlphaMax": int(alpha.max()) if alpha.size else 255,
        "transparentPixelCount": transparent_pixels,
        "opaquePixelCount": opaque_pixels,
        "maskBandNames": list(rgba_mask.getbands()),
        "valid": True,
        "validationErrors": [],
    }
    if crop.size != rgba_mask.size:
        meta["validationErrors"].append("image and mask dimensions differ")
    if rgba_mask.mode != "RGBA":
        meta["validationErrors"].append("mask is not RGBA")
    if transparent_pixels == 0:
        meta["validationErrors"].append("mask has no transparent editable pixels")
    if opaque_pixels == 0:
        meta["validationErrors"].append("mask has no opaque protected pixels")
    if meta["maskFileSize"] >= 4 * 1024 * 1024:
        meta["validationErrors"].append("mask file exceeds 4MB")
    if meta["imageFileSize"] >= 4 * 1024 * 1024:
        meta["validationErrors"].append("image file exceeds 4MB")
    meta["valid"] = not meta["validationErrors"]
    return meta


def extract_openai_error_payload(exc: Exception, request_meta: dict[str, Any], model: str) -> dict[str, Any]:
    error_payload = {
        "type": getattr(exc, "type", None),
        "code": getattr(exc, "code", None),
        "message": getattr(exc, "message", str(exc)),
        "param": getattr(exc, "param", None),
        "model": model,
        "imageCropWidth": request_meta.get("imageCropWidth"),
        "imageCropHeight": request_meta.get("imageCropHeight"),
        "maskWidth": request_meta.get("maskWidth"),
        "maskHeight": request_meta.get("maskHeight"),
        "imageFileFormat": request_meta.get("imageFileFormat"),
        "maskFileFormat": request_meta.get("maskFileFormat"),
        "imageFileSize": request_meta.get("imageFileSize"),
        "maskFileSize": request_meta.get("maskFileSize"),
        "maskMode": request_meta.get("maskMode"),
        "maskChannels": request_meta.get("maskChannels"),
        "maskBandNames": request_meta.get("maskBandNames"),
        "maskAlphaMin": request_meta.get("maskAlphaMin"),
        "maskAlphaMax": request_meta.get("maskAlphaMax"),
        "transparentPixelCount": request_meta.get("transparentPixelCount"),
        "opaquePixelCount": request_meta.get("opaquePixelCount"),
    }
    body = getattr(exc, "body", None)
    if body is not None:
        error_payload["body"] = body
    response = getattr(exc, "response", None)
    if response is not None:
        error_payload["responseStatusCode"] = getattr(response, "status_code", None)
        try:
            error_payload["responseBody"] = response.json()
        except Exception:
            try:
                error_payload["responseText"] = response.text
            except Exception:
                pass
    return error_payload


def cleanup_block_with_generative_fill_candidates(
    image: Image.Image,
    block: TextBlock,
    mask: Image.Image,
    context: dict[str, Any],
    protected_region_mask: Image.Image | None = None,
    prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    if not localize_generative_cleanup_enabled():
        return []
    provider = os.getenv("ADAPTIFAI_GENERATIVE_CLEANUP_PROVIDER", "auto").lower()
    prepared = prepare_generative_edit_context(image, mask, context, protected_region_mask, block)
    attempts: list[dict[str, Any]] = []
    if prepared is None:
        return attempts
    if prepared.get("blocked"):
        attempts.append(
            {
                "model": "generative-blocked",
                "attempted": False,
                "success": False,
                "rejected": True,
                "rejectionReasons": [str(prepared.get("reason", "editable mask unavailable"))],
                "editableMaskPreview": prepared.get("editable_mask_preview"),
                "protectedSubtractedMaskPreview": prepared.get("protected_subtracted_mask_preview"),
            }
        )
        return attempts

    crop = prepared["original_crop"]
    editable_mask_crop = prepared["editable_mask"]
    protected_crop_mask = prepared["protected_crop_mask"]
    prompt = prompt_override or (
        "Remove only the text pixels inside the transparent mask. Reconstruct only the masked text area using the surrounding background. "
        "Do not modify any unmasked pixels. Preserve all people, products, shoes, logos, packaging, lighting, shadows, textures and background outside the mask exactly. "
        "Do not add text. Do not add blur, panels, gradients, rectangles or new objects. "
        "Return the clean image only."
    )
    if provider in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
        deduped_models = [os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"]
        try:
            with tempfile.TemporaryDirectory(prefix="adaptifai-openai-clean-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                padded = pad_image_and_mask_for_edit(crop, editable_mask_crop, protected_crop_mask, min_side=256)
                request_crop = padded["crop"]
                request_editable_mask = padded["editable_mask"]
                request_protected_mask = padded["protected_crop_mask"]
                pad_left, pad_top = padded["paste_offset"]
                original_width, original_height = padded["original_size"]
                rgba_mask = build_openai_edit_mask(request_editable_mask)
                image_path, mask_path, alpha_preview_path = export_openai_edit_debug_artifacts(temp_dir, request_crop, rgba_mask)
                request_meta = validate_openai_edit_request(image_path, mask_path, request_crop, rgba_mask, deduped_models[0])
                client = OpenAI()
                for model in deduped_models:
                    attempt: dict[str, Any] = {
                        "model": model,
                        "attempted": True,
                        "success": False,
                        "rejected": False,
                        "rejectionReasons": [],
                        "requestImagePreview": request_crop.copy(),
                        "requestMaskPreview": rgba_mask.copy(),
                        "requestMaskAlphaPreview": rgba_mask.getchannel("A").copy(),
                        "requestMeta": {**request_meta, "model": model},
                        "originalCropPreview": request_crop.copy(),
                        "editableMaskPreview": request_editable_mask.copy(),
                        "protectedSubtractedMaskPreview": request_editable_mask.copy(),
                    }
                    if not request_meta["valid"]:
                        attempt["rejected"] = True
                        attempt["rejectionReasons"] = list(request_meta["validationErrors"])
                        attempts.append(attempt)
                        continue
                    try:
                        with image_path.open("rb") as image_file, mask_path.open("rb") as mask_file:
                            response = client.images.edit(
                                model=model,
                                image=image_file,
                                mask=mask_file,
                                prompt=prompt,
                                n=1,
                                output_format="png",
                            )
                        if response.data and getattr(response.data[0], "b64_json", None):
                            result_bytes = base64.b64decode(response.data[0].b64_json)
                            result = Image.open(io.BytesIO(result_bytes)).convert("RGB")
                            if result.size != request_crop.size:
                                result = result.resize(request_crop.size, Image.Resampling.LANCZOS)
                            unpadded_result = result.crop((pad_left, pad_top, pad_left + original_width, pad_top + original_height)).convert("RGB")
                            unpadded_editable_mask = request_editable_mask.crop((pad_left, pad_top, pad_left + original_width, pad_top + original_height)).convert("L")
                            unpadded_protected_mask = request_protected_mask.crop((pad_left, pad_top, pad_left + original_width, pad_top + original_height)).convert("L")
                            protected_before = float(
                                np.abs(np.array(crop, dtype=np.float32) - np.array(unpadded_result, dtype=np.float32))[np.array(unpadded_protected_mask) > 16].mean() / 255.0
                            ) if np.any(np.array(unpadded_protected_mask) > 16) else 0.0
                            local_feather = unpadded_editable_mask.filter(ImageFilter.GaussianBlur(radius=2)).convert("L")
                            composited = Image.composite(unpadded_result, crop, local_feather)
                            diff_image = ImageChops.difference(crop.convert("RGB"), composited.convert("RGB"))
                            protected_after = float(
                                np.abs(np.array(crop, dtype=np.float32) - np.array(composited, dtype=np.float32))[np.array(protected_crop_mask) > 16].mean() / 255.0
                            ) if np.any(np.array(unpadded_protected_mask) > 16) else 0.0
                            attempt["success"] = True
                            attempt["image"] = composited
                            attempt["apiSuccessResultPreview"] = result.copy()
                            attempt["openAIResultCropPreview"] = unpadded_result.copy()
                            attempt["editableMaskPreview"] = editable_mask_crop.copy()
                            attempt["protectedSubtractedMaskPreview"] = unpadded_editable_mask.copy()
                            attempt["postCompositePreview"] = composited.copy()
                            attempt["diffPreview"] = diff_image
                            attempt["protectedChangeBeforeComposite"] = protected_before
                            attempt["protectedChangeAfterComposite"] = protected_after
                        else:
                            attempt["rejected"] = True
                            attempt["rejectionReasons"] = ["no image returned"]
                    except Exception as exc:
                        error_payload = extract_openai_error_payload(exc, request_meta, model)
                        attempt["rejected"] = True
                        attempt["rejectionReasons"] = [f"api error: {exc.__class__.__name__}"]
                        attempt["apiError"] = error_payload
                    attempts.append(attempt)
        except Exception as exc:
            attempts.append(
                {
                    "model": deduped_models[0] if deduped_models else os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1",
                    "attempted": True,
                    "success": False,
                    "rejected": True,
                    "rejectionReasons": [f"temporary directory failure: {exc.__class__.__name__}"],
                }
            )
            if provider == "openai":
                return attempts
    if provider in {"vertex", "auto", "openai"} and vertex_available() and not any(attempt.get("success") for attempt in attempts):
        try:
            vertex_result = run_vertex_full_image_cleanup(crop, editable_mask_crop, prompt)
            attempt: dict[str, Any] = {
                "model": vertex_imagen_edit_model(),
                "attempted": True,
                "success": bool(vertex_result.get("success")),
                "rejected": not bool(vertex_result.get("success")),
                "rejectionReasons": [] if vertex_result.get("success") else [str(vertex_result.get("failureReason", "vertex_failed"))],
                "provider": "vertex",
                "image": vertex_result.get("image"),
                "requestImagePreview": vertex_result.get("requestImagePreview", crop.copy()),
                "requestMaskAlphaPreview": vertex_result.get("requestMaskAlphaPreview", editable_mask_crop.copy()),
                "apiSuccessResultPreview": vertex_result.get("apiSuccessResultPreview"),
                "postCompositePreview": vertex_result.get("postCompositePreview"),
                "apiError": vertex_result.get("apiError"),
            }
            attempts.append(attempt)
        except Exception as exc:
            attempts.append(
                {
                    "model": vertex_imagen_edit_model(),
                    "attempted": True,
                    "success": False,
                    "rejected": True,
                    "rejectionReasons": [f"vertex fallback failed: {exc.__class__.__name__}"],
                    "provider": "vertex",
                }
            )
    return attempts


def cleanup_block_with_generative_fill(image: Image.Image, block: TextBlock, mask: Image.Image, context: dict[str, Any]) -> Image.Image | None:
    attempts = cleanup_block_with_generative_fill_candidates(image, block, mask, context)
    for attempt in attempts:
        result = attempt.get("image")
        if attempt.get("success") and isinstance(result, Image.Image):
            return result
    return None


def polish_overlay_regions(image: Image.Image, original: Image.Image, blocks: list[TextBlock]) -> Image.Image:
    polished = image.convert("RGBA")
    original_rgba = original.convert("RGBA")
    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text) or block.surface != "overlay":
            continue
        left, top, right, bottom = block.bbox
        width = max(1, right - left)
        height = max(1, bottom - top)
        if width > original.width * 0.3 or height > original.height * 0.18:
            continue
        pad_x = min(max(8, int(width * 0.05)), 18)
        pad_y = min(max(6, int(height * 0.08)), 16)
        region = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(original.width, right + pad_x),
            min(original.height, bottom + pad_y),
        )
        crop = original_rgba.crop(region).filter(ImageFilter.GaussianBlur(radius=10))
        region_mask = Image.new("L", (region[2] - region[0], region[3] - region[1]), 0)
        region_draw = ImageDraw.Draw(region_mask)
        radius = max(6, int(min(region_mask.size) * 0.12))
        region_draw.rounded_rectangle((0, 0, region_mask.size[0], region_mask.size[1]), radius=radius, fill=210)
        polished.alpha_composite(Image.composite(crop, polished.crop(region), region_mask), dest=(region[0], region[1]))
    return polished.convert("RGB")


def build_clean_background(
    image: Image.Image,
    blocks: list[TextBlock],
    cleanup_strength: int = 100,
    *,
    return_debug: bool = False,
) -> Image.Image | tuple[Image.Image, dict[str, Any]]:
    mask = build_text_mask(image, blocks, int(os.getenv("ADAPTIFAI_INPAINT_PADDING", "26")))
    mask_bbox = mask.getbbox()
    if not mask_bbox:
        empty_debug = {
            "blockLineGroups": [],
            "lineCleanupRegions": [],
            "lineMasks": [],
            "lineCleanupStrategies": [],
            "lineCleanupQualityScores": [],
            "foregroundOverlapScores": [],
            "cleanupWarnings": [],
            "maskPolarity": "white_inpaint_black_preserve",
            "textCoverageEstimate": 0.0,
            "backgroundLeakageEstimate": 0.0,
            "maskQualityStatus": "failed",
            "maskFailureReason": "mask_too_small",
            "maskWhitePixelRatio": 0.0,
            "textStrokeMaskRawImage": None,
            "textStrokeMaskFilledImage": None,
            "textStrokeMaskDilatedImage": None,
            "textStrokeMaskFinalImage": None,
        }
        return (image.copy(), empty_debug) if return_debug else image.copy()
    working = image.convert("RGB")
    foreground_bbox = detect_foreground_bbox(image)
    protected_region_mask, protected_meta = prepare_protected_region_mask(image, exclusion_mask=mask)
    debug_info: dict[str, Any] = {
        "blockLineGroups": [],
        "lineCleanupRegions": [],
        "lineMasks": [],
        "lineCleanupStrategies": [],
        "lineCleanupQualityScores": [],
        "foregroundOverlapScores": [],
        "cleanupWarnings": [],
        "generativeAttemptPreviews": [],
        "rawForegroundBoxes": protected_meta.get("rawForegroundBoxes", []),
        "protectedMaskRefinementMethod": protected_meta.get("protectedMaskRefinementMethod", ""),
        "protectedRegionMaskImage": protected_region_mask,
        "maskPolarity": "white_inpaint_black_preserve",
        "textCoverageEstimate": 0.0,
        "backgroundLeakageEstimate": 0.0,
        "maskQualityStatus": "pending",
        "maskFailureReason": "",
        "maskWhitePixelRatio": 0.0,
        "textStrokeMaskRawImage": None,
        "textStrokeMaskFilledImage": None,
        "textStrokeMaskDilatedImage": None,
        "textStrokeMaskFinalImage": None,
    }
    for block in blocks:
        if block.translate and text_changed(block.text, block.translated_text):
            debug_info["blockLineGroups"].append(
                {
                    "id": block.id,
                    "bbox": list(block.bbox),
                    "lines": [
                        {"index": index, "text": line_text, "bbox": list(block.line_boxes[index]) if index < len(block.line_boxes) else None}
                        for index, line_text in enumerate(block.line_texts or [block.text])
                    ],
                }
            )
    if localize_fast_cleanup_enabled():
        for block in blocks:
            if not block.translate or not text_changed(block.text, block.translated_text):
                continue
            cleanup_regions = (
                [combined_region]
                if is_overlay_marketing_cleanup_block(block)
                and (combined_region := build_combined_block_cleanup_region(working, block, int(os.getenv("ADAPTIFAI_INPAINT_PADDING", "26")))) is not None
                else iter_block_cleanup_regions(working, block, int(os.getenv("ADAPTIFAI_INPAINT_PADDING", "26")))
            )
            for region in cleanup_regions:
                expanded = region["expanded_box"]
                local_mask = region["mask"].convert("L")
                if local_mask.getbbox() is None:
                    continue
                original_crop = working.crop(expanded).convert("RGB")
                used_generative = False
                if is_overlay_marketing_cleanup_block(block) and not bool(region.get("solidContextCleanup")):
                    prompt = (
                        "Remove only the original source marketing text inside the mask and reconstruct the masked area from the surrounding image. "
                        "Preserve the person's skin, product, packaging, logo, lighting, texture, grain, shadows and every unmasked pixel. "
                        "Do not add new text, do not translate here, do not blur, do not create panels, rectangles, cream, lotion, strokes, marks, labels or objects. "
                        "The cleaned area must look like the text was never printed on the image."
                    )
                    provider_result = cleanup_overlay_marketing_with_full_image_provider(working, block, region, prompt)
                    debug_info["generativeAttemptPreviews"].append(
                        {
                            "id": block.id,
                            "lineIndex": region["line_index"],
                            "name": f"fast-full-image:{provider_result.get('provider', 'unknown')}",
                            "requestImagePreview": working.copy(),
                            "requestMaskPreview": provider_result.get("requestMaskPreview"),
                            "requestMaskAlphaPreview": provider_result.get("requestMaskAlphaPreview"),
                            "apiSuccessResultPreview": provider_result.get("apiSuccessResultPreview"),
                            "postCompositePreview": provider_result.get("postCompositePreview"),
                            "apiError": provider_result.get("apiError"),
                            "compositeMaskPreview": provider_result.get("compositeMaskPreview"),
                        }
                    )
                    if provider_result.get("success") and isinstance(provider_result.get("image"), Image.Image):
                        provider_image = provider_result["image"].convert("RGB")
                        cleaned_crop = provider_image.crop(expanded).convert("RGB")
                        used_generative = True
                    else:
                        cleaned_crop = cleanup_region_with_opencv_inpaint(working, region).convert("RGB")
                else:
                    cleaned_crop = (
                        cleanup_region_with_opencv_inpaint(working, region)
                        if is_overlay_marketing_cleanup_block(block)
                        else cleanup_line_with_mask_guided_reconstruction(working, region)
                    ).convert("RGB")
                if is_overlay_marketing_cleanup_block(block):
                    working.paste(cleaned_crop, expanded)
                else:
                    feather = local_mask.filter(ImageFilter.GaussianBlur(radius=3)).convert("L")
                    working.paste(Image.composite(cleaned_crop, original_crop, feather), expanded)
                mask_coverage = float((np.array(local_mask) > 16).mean())
                debug_info["lineCleanupRegions"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "lineText": region["line_text"],
                        "tokenBox": list(region["token_box"]),
                        "expandedBox": list(expanded),
                    }
                )
                debug_info["lineMasks"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "maskSize": list(local_mask.size),
                        "maskCoverage": mask_coverage,
                    }
                )
                debug_info["foregroundOverlapScores"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "score": 0.0,
                        "bboxForegroundOverlap": 0.0,
                        "maskForegroundOverlap": 0.0,
                        "coreMaskForegroundOverlap": 0.0,
                        "protectedRegionRatio": 0.0,
                    }
                )
                debug_info["lineCleanupStrategies"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "lineText": region.get("line_text"),
                        "requestedStrategy": "fast-cpu",
                        "selectedStrategy": "fast-generative-inpaint" if used_generative else ("fast-opencv-inpaint" if is_overlay_marketing_cleanup_block(block) else "fast-reconstruction-touchup"),
                        "candidateScores": [],
                        "selectedCandidate": "fast-generative-inpaint" if used_generative else ("fast-opencv-inpaint" if is_overlay_marketing_cleanup_block(block) else "fast-reconstruction-touchup"),
                        "rejectedCandidates": [],
                        "hardRejectReasons": [],
                        "whySelectedOverOpenCV": "CPU fast cleanup avoids residual OCR scoring and generative cleanup in production.",
                    }
                )
                debug_info["lineCleanupQualityScores"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "score": 1.0,
                        "strategy": "fast-reconstruction-touchup",
                        "scoreBreakdown": {"fastCleanup": True, "maskCoverage": mask_coverage},
                    }
                )
        debug_info["cleanupWarnings"].append(
            {
                "id": "localize-fast-cleanup",
                "lineIndex": -1,
                "warning": "CPU fast cleanup skipped heavy candidate scoring, residual OCR, and generative cleanup.",
            }
        )
        cleaned = working.convert("RGB")
        return (cleaned, debug_info) if return_debug else cleaned

    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue
        if is_large_marketing_headline_block(block, image.size):
            cleaned_large_block, large_block_meta = apply_large_block_primary_cleanup(
                working,
                block,
                protected_region_mask,
                debug_info,
            )
            working = cleaned_large_block.convert("RGB")
            for key in (
                "requestedCleanupProvider",
                "actualCleanupProvider",
                "providerAvailable",
                "providerFailureReason",
                "maskDilationPx",
                "maskFeatherPx",
                "inpaintingInputMode",
            ):
                if key in large_block_meta:
                    debug_info[key] = large_block_meta[key]
            debug_info["cleanupWarnings"].append(
                {
                    "id": block.id,
                    "lineIndex": -1,
                    "warning": "Large headline block used block-level cleanup pipeline",
                    "selectedStrategy": large_block_meta.get("selectedStrategy"),
                    "cleanupPassed": bool(large_block_meta.get("cleanupPassed", False)),
                    "reason": large_block_meta.get("reason", ""),
                }
            )
            continue
        for region in iter_block_cleanup_regions(working, block, int(os.getenv("ADAPTIFAI_INPAINT_PADDING", "26"))):
            expanded = region["expanded_box"]
            local_mask = region["mask"]
            original_crop = working.crop(expanded).convert("RGB")
            analysis = analyze_cleanup_region(working, block, region, foreground_bbox, protected_region_mask, protected_meta)
            strategy = str(analysis["strategy"])
            radius = max(3, min(12, int(round(max(block.font_size_estimate, block.line_height_estimate) * 0.28))))
            region_key = f"{block.id or 'block'}:{region['line_index']}"
            debug_info["lineCleanupRegions"].append(
                {
                    "id": block.id,
                    "lineIndex": region["line_index"],
                    "lineText": region["line_text"],
                    "tokenBox": list(region["token_box"]),
                    "expandedBox": list(expanded),
                }
            )
            debug_info["lineMasks"].append(
                {
                    "id": block.id,
                    "lineIndex": region["line_index"],
                    "maskSize": list(local_mask.size),
                    "maskCoverage": float((np.array(local_mask) > 16).mean()),
                }
            )
            debug_info["foregroundOverlapScores"].append(
                {
                    "id": block.id,
                    "lineIndex": region["line_index"],
                    "score": float(analysis["mask_foreground_overlap"]),
                    "bboxForegroundOverlap": float(analysis["bbox_foreground_overlap"]),
                    "maskForegroundOverlap": float(analysis["mask_foreground_overlap"]),
                    "coreMaskForegroundOverlap": float(analysis.get("core_mask_foreground_overlap", 0.0)),
                    "protectedRegionRatio": float(analysis["protected_region_ratio"]),
                }
            )
            protected_crop_mask = protected_region_mask.crop(expanded).convert("L") if protected_region_mask is not None else None
            candidates: list[dict[str, Any]] = []

            def add_candidate(
                name: str,
                candidate_crop: Image.Image | None,
                *,
                model: str | None = None,
                attempted: bool = True,
                success: bool = True,
                rejection_reasons: list[str] | None = None,
            ) -> None:
                if candidate_crop is None:
                    candidates.append(
                        {
                            "name": name,
                            "model": model or name,
                            "image": None,
                            "score": 0.0,
                            "scoreBreakdown": {},
                            "attempted": attempted,
                            "success": success,
                            "rejected": True,
                            "rejectionReasons": rejection_reasons or ["no image candidate"],
                            "hardRejectReasons": rejection_reasons or ["no image candidate"],
                        }
                    )
                    return
                breakdown = score_cleanup_candidate_detailed(
                    original_crop,
                    candidate_crop,
                    local_mask,
                    protected_crop_mask,
                    source_text_hint=region.get("line_text"),
                    candidate_name=name,
                )
                candidates.append(
                    {
                        "name": name,
                        "model": model or name,
                        "image": candidate_crop.convert("RGB"),
                        "score": float(breakdown.get("finalScore", 0.0)),
                        "scoreBreakdown": breakdown,
                        "attempted": attempted,
                        "success": success,
                        "rejected": bool(breakdown.get("hardReject", False)),
                        "rejectionReasons": list(breakdown.get("rejectionReasons", [])),
                        "hardRejectReasons": list(breakdown.get("hardRejectReasons", [])),
                    }
                )

            if strategy == "generative":
                for attempt in cleanup_block_with_generative_fill_candidates(working, block, local_mask, region, protected_region_mask):
                    if any(
                        key in attempt
                        for key in (
                            "requestImagePreview",
                            "requestMaskPreview",
                            "requestMaskAlphaPreview",
                            "originalCropPreview",
                            "openAIResultCropPreview",
                            "apiSuccessResultPreview",
                            "editableMaskPreview",
                            "protectedSubtractedMaskPreview",
                            "postCompositePreview",
                            "requestMeta",
                            "apiError",
                        )
                    ):
                        debug_info["generativeAttemptPreviews"].append(
                            {
                                "id": block.id,
                                "lineIndex": region["line_index"],
                                "name": f"generative:{attempt.get('model', 'unknown')}",
                                "requestImagePreview": attempt.get("requestImagePreview"),
                                "requestMaskPreview": attempt.get("requestMaskPreview"),
                                "requestMaskAlphaPreview": attempt.get("requestMaskAlphaPreview"),
                                "originalCropPreview": attempt.get("originalCropPreview"),
                                "openAIResultCropPreview": attempt.get("openAIResultCropPreview"),
                                "apiSuccessResultPreview": attempt.get("apiSuccessResultPreview"),
                                "editableMaskPreview": attempt.get("editableMaskPreview"),
                                "protectedSubtractedMaskPreview": attempt.get("protectedSubtractedMaskPreview"),
                                "postCompositePreview": attempt.get("postCompositePreview"),
                                "diffPreview": attempt.get("diffPreview"),
                                "protectedChangeBeforeComposite": attempt.get("protectedChangeBeforeComposite"),
                                "protectedChangeAfterComposite": attempt.get("protectedChangeAfterComposite"),
                                "requestMeta": attempt.get("requestMeta"),
                                "apiError": attempt.get("apiError"),
                            }
                        )
                    add_candidate(
                        f"generative:{attempt.get('model', 'unknown')}",
                        attempt.get("image"),
                        model=str(attempt.get("model", "unknown")),
                        attempted=bool(attempt.get("attempted", True)),
                        success=bool(attempt.get("success", False)),
                        rejection_reasons=list(attempt.get("rejectionReasons", [])),
                    )
                add_candidate("conservative-inpaint", cleanup_line_with_mask_guided_reconstruction(working, region))
                if float(analysis["texture_variance"]) <= 14 and float(analysis["contrast"]) <= 18:
                    add_candidate("sampled", cleanup_block_with_sampled_fill(working, region))
                add_candidate("reconstruction-touchup", cleanup_line_with_mask_guided_reconstruction(working, region))
            elif float(analysis["foreground_overlap"]) > 0.08:
                add_candidate("conservative-inpaint", cleanup_line_with_mask_guided_reconstruction(working, region))
                add_candidate("reconstruction-touchup", cleanup_line_with_mask_guided_reconstruction(working, region))
            elif strategy == "sampled":
                add_candidate("sampled", cleanup_block_with_sampled_fill(working, region))
                add_candidate("conservative-inpaint", cleanup_line_with_mask_guided_reconstruction(working, region))
                add_candidate("reconstruction-touchup", cleanup_line_with_mask_guided_reconstruction(working, region))
                for attempt in cleanup_block_with_generative_fill_candidates(working, block, local_mask, region, protected_region_mask):
                    if any(
                        key in attempt
                        for key in (
                            "requestImagePreview",
                            "requestMaskPreview",
                            "requestMaskAlphaPreview",
                            "originalCropPreview",
                            "openAIResultCropPreview",
                            "apiSuccessResultPreview",
                            "editableMaskPreview",
                            "protectedSubtractedMaskPreview",
                            "postCompositePreview",
                            "requestMeta",
                            "apiError",
                        )
                    ):
                        debug_info["generativeAttemptPreviews"].append(
                            {
                                "id": block.id,
                                "lineIndex": region["line_index"],
                                "name": f"generative:{attempt.get('model', 'unknown')}",
                                "requestImagePreview": attempt.get("requestImagePreview"),
                                "requestMaskPreview": attempt.get("requestMaskPreview"),
                                "requestMaskAlphaPreview": attempt.get("requestMaskAlphaPreview"),
                                "originalCropPreview": attempt.get("originalCropPreview"),
                                "openAIResultCropPreview": attempt.get("openAIResultCropPreview"),
                                "apiSuccessResultPreview": attempt.get("apiSuccessResultPreview"),
                                "editableMaskPreview": attempt.get("editableMaskPreview"),
                                "protectedSubtractedMaskPreview": attempt.get("protectedSubtractedMaskPreview"),
                                "postCompositePreview": attempt.get("postCompositePreview"),
                                "diffPreview": attempt.get("diffPreview"),
                                "protectedChangeBeforeComposite": attempt.get("protectedChangeBeforeComposite"),
                                "protectedChangeAfterComposite": attempt.get("protectedChangeAfterComposite"),
                                "requestMeta": attempt.get("requestMeta"),
                                "apiError": attempt.get("apiError"),
                            }
                        )
                    add_candidate(
                        f"generative:{attempt.get('model', 'unknown')}",
                        attempt.get("image"),
                        model=str(attempt.get("model", "unknown")),
                        attempted=bool(attempt.get("attempted", True)),
                        success=bool(attempt.get("success", False)),
                        rejection_reasons=list(attempt.get("rejectionReasons", [])),
                    )
            else:
                add_candidate("conservative-inpaint", cleanup_line_with_mask_guided_reconstruction(working, region))
                add_candidate("reconstruction-touchup", cleanup_line_with_mask_guided_reconstruction(working, region))
                if float(analysis["texture_variance"]) <= 12 and float(analysis["contrast"]) <= 16 and block.role != "headline":
                    add_candidate("sampled", cleanup_block_with_sampled_fill(working, region))
                if float(analysis["foreground_overlap"]) <= 0.08:
                    for attempt in cleanup_block_with_generative_fill_candidates(working, block, local_mask, region, protected_region_mask):
                        if any(
                            key in attempt
                            for key in (
                                "requestImagePreview",
                                "requestMaskPreview",
                                "requestMaskAlphaPreview",
                                "originalCropPreview",
                                "openAIResultCropPreview",
                                "apiSuccessResultPreview",
                                "editableMaskPreview",
                                "protectedSubtractedMaskPreview",
                                "postCompositePreview",
                                "requestMeta",
                                "apiError",
                            )
                        ):
                            debug_info["generativeAttemptPreviews"].append(
                                {
                                    "id": block.id,
                                    "lineIndex": region["line_index"],
                                    "name": f"generative:{attempt.get('model', 'unknown')}",
                                    "requestImagePreview": attempt.get("requestImagePreview"),
                                    "requestMaskPreview": attempt.get("requestMaskPreview"),
                                    "requestMaskAlphaPreview": attempt.get("requestMaskAlphaPreview"),
                                    "originalCropPreview": attempt.get("originalCropPreview"),
                                    "openAIResultCropPreview": attempt.get("openAIResultCropPreview"),
                                    "apiSuccessResultPreview": attempt.get("apiSuccessResultPreview"),
                                    "editableMaskPreview": attempt.get("editableMaskPreview"),
                                    "protectedSubtractedMaskPreview": attempt.get("protectedSubtractedMaskPreview"),
                                    "postCompositePreview": attempt.get("postCompositePreview"),
                                    "diffPreview": attempt.get("diffPreview"),
                                    "protectedChangeBeforeComposite": attempt.get("protectedChangeBeforeComposite"),
                                    "protectedChangeAfterComposite": attempt.get("protectedChangeAfterComposite"),
                                    "requestMeta": attempt.get("requestMeta"),
                                    "apiError": attempt.get("apiError"),
                                }
                            )
                        add_candidate(
                            f"generative:{attempt.get('model', 'unknown')}",
                            attempt.get("image"),
                            model=str(attempt.get("model", "unknown")),
                            attempted=bool(attempt.get("attempted", True)),
                            success=bool(attempt.get("success", False)),
                            rejection_reasons=list(attempt.get("rejectionReasons", [])),
                        )

            if not candidates:
                continue
            viable_candidates = [item for item in candidates if item.get("image") is not None and not item.get("rejected")]
            if not viable_candidates:
                viable_candidates = [item for item in candidates if item.get("image") is not None]
            viable_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            best_candidate = viable_candidates[0]
            best_name = str(best_candidate["name"])
            best_crop = best_candidate["image"]
            best_score = float(best_candidate.get("score", 0.0))

            opencv_candidate = next((item for item in viable_candidates if item["name"] == "reconstruction-touchup"), None)
            why_selected = "highest weighted score across cleanup candidates"
            if best_name.startswith("generative:") and opencv_candidate is not None:
                residual_gain = (
                    float(best_candidate["scoreBreakdown"].get("residualTextScore", 0.0))
                    - float(opencv_candidate["scoreBreakdown"].get("residualTextScore", 0.0))
                )
                protected_change = float(best_candidate["scoreBreakdown"].get("protectedRegionChange", 0.0))
                why_selected = (
                    f"{best_name} selected because residualTextScore improved by {residual_gain * 100:.0f}% "
                    f"while protectedRegionChange stayed at {protected_change:.3f}"
                )
            elif best_name == "reconstruction-touchup":
                strongest_generative = next((item for item in viable_candidates if str(item["name"]).startswith("generative:")), None)
                if strongest_generative is not None:
                    why_selected = (
                        f"reconstruction-touchup selected because {strongest_generative['name']} scored lower on weighted region-aware cleanup "
                        f"({float(strongest_generative.get('score', 0.0)):.3f} vs {best_score:.3f})"
                    )
                    generative_breakdown = strongest_generative.get("scoreBreakdown", {})
                    if generative_breakdown.get("whyOpenCVLost"):
                        why_selected += f" / {generative_breakdown['whyOpenCVLost']}"
            debug_info["lineCleanupStrategies"].append(
                {
                    "id": block.id,
                    "lineIndex": region["line_index"],
                    "lineText": region.get("line_text"),
                    "requestedStrategy": strategy,
                    "selectedStrategy": best_name,
                    "bboxForegroundOverlap": float(analysis["bbox_foreground_overlap"]),
                    "maskForegroundOverlap": float(analysis["mask_foreground_overlap"]),
                    "coreMaskForegroundOverlap": float(analysis.get("core_mask_foreground_overlap", 0.0)),
                    "protectedRegionRatio": float(analysis["protected_region_ratio"]),
                    "generativeSafeCropBox": list(expanded),
                    "generativeAllowedReason": str(analysis["generative_allowed_reason"]),
                    "generativeBlockedReason": str(analysis["generative_blocked_reason"]),
                    "candidateScores": [
                        {
                            "name": candidate["name"],
                            "model": candidate.get("model"),
                            "score": float(candidate.get("score", 0.0)),
                            "scoreBreakdown": candidate.get("scoreBreakdown", {}),
                            "attempted": bool(candidate.get("attempted", True)),
                            "success": bool(candidate.get("success", candidate.get("image") is not None)),
                            "rejected": bool(candidate.get("rejected", False)),
                            "rejectionReasons": list(candidate.get("rejectionReasons", [])),
                        }
                        for candidate in candidates
                    ],
                    "selectedCandidate": best_name,
                    "rejectedCandidates": [candidate["name"] for candidate in candidates if candidate.get("rejected")],
                    "hardRejectReasons": [
                        {"name": candidate["name"], "reasons": list(candidate.get("hardRejectReasons", []))}
                        for candidate in candidates
                        if candidate.get("hardRejectReasons")
                    ],
                    "whyOpenCVLost": next(
                        (candidate.get("scoreBreakdown", {}).get("whyOpenCVLost", "") for candidate in candidates if candidate["name"] == "reconstruction-touchup"),
                        "",
                    ),
                    "whyGenerativeWon": next(
                        (
                            candidate.get("scoreBreakdown", {}).get("whyGenerativeWon", "")
                            for candidate in candidates
                            if str(candidate["name"]).startswith("generative:")
                        ),
                        "",
                    ),
                    "whySelectedOverOpenCV": why_selected,
                }
            )
            debug_info["lineCleanupQualityScores"].append(
                {
                    "id": block.id,
                    "lineIndex": region["line_index"],
                    "score": float(best_score),
                    "strategy": best_name,
                    "scoreBreakdown": best_candidate.get("scoreBreakdown", {}),
                }
            )
            if best_score < 0.5:
                debug_info["cleanupWarnings"].append(
                    {
                        "id": block.id,
                        "lineIndex": region["line_index"],
                        "warning": "Low cleanup quality",
                        "strategy": best_name,
                        "score": float(best_score),
                    }
                )
            feather_radius = 4 if best_name == "sampled" else 3
            feather = local_mask.filter(ImageFilter.GaussianBlur(radius=feather_radius)).convert("L")
            working.paste(Image.composite(best_crop, original_crop, feather), expanded)

    cleaned = working.convert("RGB")

    if cleanup_strength >= 100:
        return (cleaned, debug_info) if return_debug else cleaned

    alpha = max(0.0, min(1.0, cleanup_strength / 100))
    original = image.convert("RGBA")
    restored = cleaned.convert("RGBA")
    rgba_mask = mask.point(lambda value: int(value * alpha)).convert("L")
    blended = Image.composite(restored, original, rgba_mask)
    final = blended.convert("RGB")
    return (final, debug_info) if return_debug else final


def attempt_block_level_generative_cleanup(
    image: Image.Image,
    block: TextBlock,
    protected_region_mask: Image.Image | None,
    cleanup_debug: dict[str, Any],
) -> tuple[Image.Image | None, dict[str, Any]]:
    region = build_combined_block_cleanup_region(image, block)
    if region is None:
        return None, {
            "blockCleanupStatus": "failed",
            "blockLevelFallbackAttempted": False,
            "blockLevelFallbackSelected": False,
            "reason": "no combined cleanup region",
        }

    prompt = (
        "Remove only the visible text inside the transparent mask area. "
        "Fill it with the exact same background colour and texture that borders the mask â€” do NOT generate new images, products, labels, or patterns. "
        "Preserve everything outside the mask exactly as-is. "
        "Do not add text. Do not blur. Return only the clean continuation of the existing background."
    )
    original_crop = image.crop(region["expanded_box"]).convert("RGB")
    local_mask = region["mask"].convert("L")
    protected_crop_mask = protected_region_mask.crop(region["expanded_box"]).convert("L") if protected_region_mask is not None else None
    attempts = cleanup_block_with_generative_fill_candidates(
        image,
        block,
        local_mask,
        region,
        protected_region_mask,
        prompt_override=prompt,
    )
    candidates: list[dict[str, Any]] = []
    for attempt in attempts:
        if any(
            key in attempt
            for key in (
                "requestImagePreview",
                "requestMaskPreview",
                "requestMaskAlphaPreview",
                "originalCropPreview",
                "openAIResultCropPreview",
                "apiSuccessResultPreview",
                "editableMaskPreview",
                "protectedSubtractedMaskPreview",
                "postCompositePreview",
                "requestMeta",
                "apiError",
            )
        ):
            cleanup_debug["generativeAttemptPreviews"].append(
                {
                    "id": block.id,
                    "lineIndex": -1,
                    "name": f"block-fallback:{attempt.get('model', 'unknown')}",
                    "requestImagePreview": attempt.get("requestImagePreview"),
                    "requestMaskPreview": attempt.get("requestMaskPreview"),
                    "requestMaskAlphaPreview": attempt.get("requestMaskAlphaPreview"),
                    "originalCropPreview": attempt.get("originalCropPreview"),
                    "openAIResultCropPreview": attempt.get("openAIResultCropPreview"),
                    "apiSuccessResultPreview": attempt.get("apiSuccessResultPreview"),
                    "editableMaskPreview": attempt.get("editableMaskPreview"),
                    "protectedSubtractedMaskPreview": attempt.get("protectedSubtractedMaskPreview"),
                    "postCompositePreview": attempt.get("postCompositePreview"),
                    "diffPreview": attempt.get("diffPreview"),
                    "protectedChangeBeforeComposite": attempt.get("protectedChangeBeforeComposite"),
                    "protectedChangeAfterComposite": attempt.get("protectedChangeAfterComposite"),
                    "requestMeta": attempt.get("requestMeta"),
                    "apiError": attempt.get("apiError"),
                }
            )
        candidate_crop = attempt.get("image")
        if not isinstance(candidate_crop, Image.Image):
            candidates.append(
                {
                    "name": f"block-generative:{attempt.get('model', 'unknown')}",
                    "score": 0.0,
                    "rejected": True,
                    "rejectionReasons": list(attempt.get("rejectionReasons", [])),
                    "scoreBreakdown": {},
                    "image": None,
                }
            )
            continue
        breakdown = score_cleanup_candidate_detailed(
            original_crop,
            candidate_crop,
            local_mask,
            protected_crop_mask,
            source_text_hint=" ".join(block.line_texts or [block.text]),
            candidate_name=f"block-generative:{attempt.get('model', 'unknown')}",
        )
        candidates.append(
            {
                "name": f"block-generative:{attempt.get('model', 'unknown')}",
                "score": float(breakdown.get("finalScore", 0.0)),
                "rejected": bool(breakdown.get("hardReject", False)),
                "rejectionReasons": list(breakdown.get("rejectionReasons", [])),
                "scoreBreakdown": breakdown,
                "image": candidate_crop.convert("RGB"),
            }
        )

    viable = [candidate for candidate in candidates if candidate.get("image") is not None and not candidate.get("rejected")]
    if not viable:
        cleanup_debug["cleanupWarnings"].append(
            {
                "id": block.id,
                "lineIndex": -1,
                "warning": "Block-level fallback failed",
                "candidates": [
                    {"name": candidate["name"], "rejectionReasons": candidate.get("rejectionReasons", [])}
                    for candidate in candidates
                ],
            }
        )
        return None, {
            "blockCleanupStatus": "failed",
            "blockLevelFallbackAttempted": True,
            "blockLevelFallbackSelected": False,
            "reason": "no viable generative block-level candidate",
            "candidates": candidates,
        }

    viable.sort(key=lambda candidate: float(candidate.get("score", 0.0)), reverse=True)
    best = viable[0]
    full_image = image.convert("RGB").copy()
    full_image.paste(best["image"], region["expanded_box"])
    return full_image, {
        "blockCleanupStatus": "passed",
        "blockLevelFallbackAttempted": True,
        "blockLevelFallbackSelected": True,
        "reason": f"{best['name']} selected after block-level fallback",
        "selectedCandidate": best["name"],
        "candidates": candidates,
    }


def apply_large_block_primary_cleanup(
    image: Image.Image,
    block: TextBlock,
    protected_region_mask: Image.Image | None,
    cleanup_debug: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    requested_provider = resolve_cleanup_provider(block, image.size)
    provider_available, provider_failure_reason = get_cleanup_provider_availability(requested_provider)
    dynamic_dilation_px = max(12, min(18, int(round(max(block.font_size_estimate, block.line_height_estimate) * 0.12))))
    base_dilation_px = int(os.getenv("ADAPTIFAI_LARGE_BLOCK_MASK_DILATION", str(dynamic_dilation_px)))
    feather_px = int(os.getenv("ADAPTIFAI_LARGE_BLOCK_MASK_FEATHER", "6"))
    cleanup_debug["requestedCleanupProvider"] = requested_provider
    cleanup_debug["providerAvailable"] = provider_available
    cleanup_debug["providerFailureReason"] = provider_failure_reason
    cleanup_debug["maskFeatherPx"] = feather_px
    cleanup_debug["inpaintingInputMode"] = "full_image"
    cleanup_debug["cleanupCascadeEnabled"] = True
    cleanup_debug["firstPassProvider"] = requested_provider
    cleanup_debug["firstPassStatus"] = "not_started"
    cleanup_debug["residualTextDetectedAfterFirstPass"] = False
    cleanup_debug["residualWordsAfterFirstPass"] = []
    cleanup_debug["residualMaskGenerated"] = False
    cleanup_debug["secondPassProvider"] = requested_provider
    cleanup_debug["secondPassStatus"] = "not_attempted"
    cleanup_debug["residualTextDetectedAfterSecondPass"] = False
    cleanup_debug["residualWordsAfterSecondPass"] = []
    cleanup_debug["sdxlFallbackAttempted"] = False
    cleanup_debug["sdxlFallbackStatus"] = "not_attempted"
    cleanup_debug["finalCleanupQualityStatus"] = "pending"

    if not provider_available:
        cleanup_debug["cleanupWarnings"].append(
            {
                "id": block.id,
                "lineIndex": -1,
                "warning": "Primary cleanup provider unavailable",
                "requestedCleanupProvider": requested_provider,
                "providerFailureReason": provider_failure_reason,
            }
        )
        return image.convert("RGB"), {
            "requestedCleanupProvider": requested_provider,
            "actualCleanupProvider": None,
            "providerAvailable": False,
            "providerFailureReason": provider_failure_reason,
            "maskDilationPx": base_dilation_px,
            "maskFeatherPx": feather_px,
            "inpaintingInputMode": "full_image",
            "selectedStrategy": "provider_not_configured",
            "cleanupPassed": False,
            "reason": provider_failure_reason,
        }

    prompt = (
        "Remove only the text from the masked region and fill it with the exact same "
        "background colour, texture and pattern that immediately surrounds that area. "
        "Do NOT generate new imagery, products, labels, patterns, or objects. "
        "Do NOT change anything outside the masked area. "
        "Do not blur, smear, or leave ghosting. "
        "The result must look like the text was never there â€” only the background visible."
    )
    dilation_attempts = sorted({max(8, base_dilation_px - 3), base_dilation_px, min(24, base_dilation_px + 3)})
    attempt_summaries: list[dict[str, Any]] = []
    best_attempt: dict[str, Any] | None = None
    last_mask_bundle: dict[str, Any] | None = None

    for dilation_px in dilation_attempts:
        full_mask_bundle = build_full_image_block_mask(
            image,
            block,
            protected_region_mask=protected_region_mask,
            dilation_px=dilation_px,
            feather_px=feather_px,
            allow_protected_overlap=True,
        )
        last_mask_bundle = full_mask_bundle
        attempt_summary: dict[str, Any] = {
            "dilationPx": dilation_px,
            "maskQualityStatus": full_mask_bundle.get("maskQualityStatus"),
            "maskFailureReason": full_mask_bundle.get("maskFailureReason"),
            "textCoverageEstimate": full_mask_bundle.get("textCoverageEstimate"),
            "antiAliasCoverageEstimate": full_mask_bundle.get("antiAliasCoverageEstimate"),
            "backgroundLeakageEstimate": full_mask_bundle.get("backgroundLeakageEstimate"),
            "protectedObjectOverlapEstimate": full_mask_bundle.get("protectedObjectOverlapEstimate"),
            "firstPassStatus": "not_started",
            "residualTextDetectedAfterFirstPass": False,
            "residualWordsAfterFirstPass": [],
            "residualOCRAfterFirstPass": [],
            "residualMaskGenerated": False,
            "secondPassProvider": requested_provider,
            "secondPassStatus": "not_attempted",
            "residualTextDetectedAfterSecondPass": False,
            "residualWordsAfterSecondPass": [],
            "residualOCRAfterSecondPass": [],
            "sdxlFallbackAttempted": False,
            "sdxlFallbackStatus": "not_attempted",
        }
        if full_mask_bundle.get("maskQualityStatus") != "passed":
            attempt_summary["providerQualityStatus"] = "mask_failed"
            attempt_summaries.append(attempt_summary)
            continue

        binary_mask = full_mask_bundle["inpaintMask_binary"]
        provider_result = (
            run_huggingface_full_image_cleanup(
                image,
                binary_mask,
                validation_callback=lambda candidate: evaluate_block_cleanup_visibility(image, candidate, block),
                stage_label=f"{block.id or 'block'}:d{dilation_px}:pass1",
            )
            if requested_provider == "huggingface"
            else run_openai_full_image_cleanup(image, binary_mask, prompt)
        )

        attempt_summary["provider"] = provider_result.get("provider")
        attempt_summary["providerFailureReason"] = provider_result.get("failureReason", "")
        if any(
            key in provider_result
            for key in (
                "requestImagePreview",
                "requestMaskPreview",
                "requestMaskAlphaPreview",
                "apiSuccessResultPreview",
                "postCompositePreview",
                "requestMeta",
                "apiError",
                "huggingFaceInputImagePreview",
                "huggingFaceMaskPreview",
                "huggingFaceOutputPreview",
            )
        ):
            cleanup_debug["generativeAttemptPreviews"].append(
                {
                    "id": block.id,
                    "lineIndex": -1,
                    "name": f"provider:{provider_result.get('provider', 'unknown')}:d{dilation_px}",
                    "requestImagePreview": provider_result.get("requestImagePreview"),
                    "requestMaskPreview": provider_result.get("requestMaskPreview"),
                    "requestMaskAlphaPreview": provider_result.get("requestMaskAlphaPreview"),
                    "apiSuccessResultPreview": provider_result.get("apiSuccessResultPreview"),
                    "postCompositePreview": provider_result.get("postCompositePreview"),
                    "requestMeta": provider_result.get("requestMeta"),
                    "apiError": provider_result.get("apiError"),
                    "huggingFaceInputImagePreview": provider_result.get("huggingFaceInputImagePreview"),
                    "huggingFaceMaskPreview": provider_result.get("huggingFaceMaskPreview"),
                    "huggingFaceOutputPreview": provider_result.get("huggingFaceOutputPreview"),
                    "modelAttempts": provider_result.get("modelAttempts", []),
                }
            )

        selected_first_pass = provider_result.get("image") if provider_result.get("success") else provider_result.get("bestEffortImage")
        if isinstance(selected_first_pass, Image.Image):
            first_pass_image = selected_first_pass.convert("RGB")
            attempt_summary["firstPassImage"] = first_pass_image
            attempt_summary["firstPassProvider"] = provider_result.get("provider", requested_provider)
            first_visibility = dict(provider_result.get("validation") or provider_result.get("bestEffortValidation") or {})
            if not first_visibility:
                first_visibility = evaluate_block_cleanup_visibility(image, first_pass_image, block)
            attempt_summary["firstPassStatus"] = first_visibility["cleanupStatus"]
            attempt_summary["residualTextDetectedAfterFirstPass"] = first_visibility["cleanupStatus"] != "passed"
            attempt_summary["residualWordsAfterFirstPass"] = list(first_visibility["failedSourceWords"])
            attempt_summary["residualOCRAfterFirstPass"] = list(first_visibility["residualSourceOCR"])
            attempt_summary["modelAttempts"] = provider_result.get("modelAttempts", [])

            candidate_image = first_pass_image
            visibility = first_visibility
            max_cleanup_passes = max(1, int(os.getenv("ADAPTIFAI_MAX_CLEANUP_PASSES", "3")))
            pass_records: list[dict[str, Any]] = [
                {
                    "passIndex": 1,
                    "provider": provider_result.get("provider", requested_provider),
                    "status": first_visibility["cleanupStatus"],
                    "residualWords": list(first_visibility["failedSourceWords"]),
                    "residualOCR": list(first_visibility["residualSourceOCR"]),
                    "ghostingScore": first_visibility["ghostingScore"],
                    "residualTextScore": first_visibility["residualTextScore"],
                    "residualMaskArea": int((np.array(binary_mask.convert("L")) > 0).sum()),
                    "improvement": 0.0,
                    "modelAttempts": list(provider_result.get("modelAttempts", [])),
                }
            ]
            residual_mask = binary_mask
            residual_meta: dict[str, Any] = {"generated": False}
            previous_visibility = first_visibility
            stopped_reason = "cleanup_passed" if visibility["cleanupStatus"] == "passed" else "max_passes_reached"
            for pass_index in range(2, max_cleanup_passes + 1):
                if visibility["cleanupStatus"] == "passed":
                    stopped_reason = "cleanup_passed"
                    break
                residual_mask, residual_meta = build_residual_text_mask(
                    image,
                    candidate_image,
                    block,
                    binary_mask,
                    visibility["failedSourceWords"],
                    visibility["residualSourceOCR"],
                )
                attempt_summary["residualMaskGenerated"] = bool(residual_meta.get("generated"))
                attempt_summary["residualMaskMeta"] = residual_meta
                if pass_index == 2:
                    attempt_summary["residualMaskImage"] = residual_mask
                    attempt_summary["residualGlyphMaskAfterFirstPassImage"] = residual_meta.get("residualGlyphMask")
                    attempt_summary["residualArtifactMaskExpandedImage"] = residual_meta.get("residualArtifactMaskExpanded")
                    attempt_summary["residualWordBoxesAfterFirstPass"] = residual_meta.get("residualWordBoxes", [])
                    attempt_summary["residualMaskCoverageByWord"] = residual_meta.get("coverageByWord", [])
                    attempt_summary["residualMaskFalsePositiveEstimate"] = residual_meta.get("falsePositiveEstimate", 0.0)
                residual_area = int(residual_meta.get("area", 0))
                residual_area_ratio = float(residual_meta.get("coverageRatio", 0.0))
                if not residual_meta.get("generated"):
                    stopped_reason = str(residual_meta.get("reason", "residual_mask_not_generated"))
                    if pass_index == 2:
                        attempt_summary["secondPassStatus"] = stopped_reason
                    break
                if residual_area_ratio > float(os.getenv("ADAPTIFAI_MAX_RESIDUAL_MASK_RATIO", "0.92")):
                    stopped_reason = "residual_mask_too_large"
                    if pass_index == 2:
                        attempt_summary["secondPassStatus"] = stopped_reason
                    break

                pass_result = (
                    run_huggingface_full_image_cleanup(
                        candidate_image,
                        residual_mask,
                        validation_callback=lambda candidate: evaluate_block_cleanup_visibility(image, candidate, block),
                        stage_label=f"{block.id or 'block'}:d{dilation_px}:pass{pass_index}",
                    )
                    if requested_provider == "huggingface"
                    else run_openai_full_image_cleanup(candidate_image, residual_mask, prompt)
                )
                if pass_index == 2:
                    attempt_summary["secondPassProvider"] = pass_result.get("provider", requested_provider)
                selected_pass_image = pass_result.get("image") if pass_result.get("success") else pass_result.get("bestEffortImage")
                if not isinstance(selected_pass_image, Image.Image):
                    stopped_reason = pass_result.get("failureReason", "provider_failed")
                    if pass_index == 2:
                        attempt_summary["secondPassStatus"] = stopped_reason
                    break
                pass_image = selected_pass_image.convert("RGB")
                pass_visibility = dict(pass_result.get("validation") or pass_result.get("bestEffortValidation") or {})
                if not pass_visibility:
                    pass_visibility = evaluate_block_cleanup_visibility(image, pass_image, block)
                improvement = (
                    (len(previous_visibility["failedSourceWords"]) - len(pass_visibility["failedSourceWords"])) * 1.0
                    + max(0.0, previous_visibility["ghostingScore"] - pass_visibility["ghostingScore"]) * 2.0
                    + max(0.0, pass_visibility["residualTextScore"] - previous_visibility["residualTextScore"])
                )
                pass_records.append(
                    {
                        "passIndex": pass_index,
                        "provider": pass_result.get("provider", "lama"),
                        "status": pass_visibility["cleanupStatus"],
                        "residualWords": list(pass_visibility["failedSourceWords"]),
                        "residualOCR": list(pass_visibility["residualSourceOCR"]),
                        "ghostingScore": pass_visibility["ghostingScore"],
                        "residualTextScore": pass_visibility["residualTextScore"],
                        "residualMaskArea": residual_area,
                        "residualMaskCoverageRatio": residual_area_ratio,
                        "improvement": round(float(improvement), 4),
                        "modelAttempts": pass_result.get("modelAttempts", []),
                    }
                )
                if pass_index == 2:
                    attempt_summary["secondPassImage"] = pass_image
                    attempt_summary["secondPassStatus"] = pass_visibility["cleanupStatus"]
                    attempt_summary["residualTextDetectedAfterSecondPass"] = pass_visibility["cleanupStatus"] != "passed"
                    attempt_summary["residualWordsAfterSecondPass"] = list(pass_visibility["failedSourceWords"])
                    attempt_summary["residualOCRAfterSecondPass"] = list(pass_visibility["residualSourceOCR"])
                improved_pass = (
                    pass_visibility["cleanupStatus"] == "passed"
                    or improvement > 0.025
                    or len(pass_visibility["failedSourceWords"]) <= len(visibility["failedSourceWords"])
                )
                if improved_pass:
                    candidate_image = pass_image
                    visibility = pass_visibility
                    previous_visibility = pass_visibility
                else:
                    stopped_reason = "no_residual_improvement"
                    break
            else:
                stopped_reason = "max_passes_reached"
            attempt_summary["cleanupPassesRun"] = len(pass_records)
            attempt_summary["residualWordsByPass"] = [{"passIndex": item["passIndex"], "words": item["residualWords"]} for item in pass_records]
            attempt_summary["residualMaskAreaByPass"] = [{"passIndex": item["passIndex"], "area": item["residualMaskArea"]} for item in pass_records]
            attempt_summary["cleanupImprovementByPass"] = [{"passIndex": item["passIndex"], "improvement": item["improvement"]} for item in pass_records]
            attempt_summary["stoppedReason"] = stopped_reason
            attempt_summary["cleanupPassRecords"] = pass_records

            all_model_attempts: list[dict[str, Any]] = []
            for record in pass_records:
                all_model_attempts.extend(list(record.get("modelAttempts", [])))
            all_model_attempts.extend(list(provider_result.get("modelAttempts", [])))
            sdxl_attempts = [
                item for item in all_model_attempts
                if str(item.get("model")) == "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            ]
            attempt_summary["sdxlFallbackAttempted"] = bool(sdxl_attempts)
            attempt_summary["sdxlFallbackStatus"] = (
                "passed"
                if any(item.get("success") for item in sdxl_attempts)
                else str(sdxl_attempts[-1].get("failureReason", "not_attempted")) if sdxl_attempts else "not_attempted"
            )

            attempt_summary["providerQualityStatus"] = visibility["cleanupStatus"]
            attempt_summary["residualSourceOCR"] = visibility["residualSourceOCR"]
            attempt_summary["failedSourceWords"] = visibility["failedSourceWords"]
            attempt_summary["ghostingScore"] = visibility["ghostingScore"]
            attempt_summary["residualTextScore"] = visibility["residualTextScore"]
            attempt_summary["visualGhostingDetected"] = visibility["visualGhostingDetected"]
            attempt_summary["image"] = candidate_image
            if visibility["cleanupStatus"] == "passed":
                attempt_summary["selectionScore"] = 2.0 + max(0.0, 1.0 - visibility["ghostingScore"])
            else:
                source_word_count = max(1, len(extract_source_word_targets(block)))
                removed_word_ratio = max(0.0, (source_word_count - len(visibility["failedSourceWords"])) / source_word_count)
                attempt_summary["selectionScore"] = max(
                    0.0,
                    0.45
                    + removed_word_ratio * 0.55
                    + max(0.0, 1.0 - visibility["ghostingScore"]) * 0.2
                    + visibility["residualTextScore"] * 0.15,
                )
        else:
            attempt_summary["providerQualityStatus"] = "provider_failed"
            attempt_summary["selectionScore"] = 0.0

        attempt_summaries.append(attempt_summary)

    cleanup_debug["dilationAttempts"] = [
        {
            "dilationPx": attempt["dilationPx"],
            "maskQualityStatus": attempt.get("maskQualityStatus"),
            "maskFailureReason": attempt.get("maskFailureReason"),
            "providerQualityStatus": attempt.get("providerQualityStatus"),
            "residualSourceOCR": attempt.get("residualSourceOCR", []),
            "ghostingScore": attempt.get("ghostingScore"),
        }
        for attempt in attempt_summaries
    ]
    cleanup_debug["residualTextAfterEachAttempt"] = [
        {"dilationPx": attempt["dilationPx"], "residualSourceOCR": attempt.get("residualSourceOCR", [])}
        for attempt in attempt_summaries
    ]
    cleanup_debug["ghostingScoreAfterEachAttempt"] = [
        {"dilationPx": attempt["dilationPx"], "ghostingScore": attempt.get("ghostingScore")}
        for attempt in attempt_summaries
    ]

    valid_attempts = [attempt for attempt in attempt_summaries if isinstance(attempt.get("image"), Image.Image)]
    if valid_attempts:
        best_attempt = max(valid_attempts, key=lambda attempt: float(attempt.get("selectionScore", 0.0)))

    selected_bundle = last_mask_bundle if last_mask_bundle is not None else {
        "maskPolarity": "white_inpaint_black_preserve",
        "textCoverageEstimate": 0.0,
        "antiAliasCoverageEstimate": 0.0,
        "backgroundLeakageEstimate": 0.0,
        "protectedObjectOverlapEstimate": 0.0,
        "maskQualityStatus": "failed",
        "maskFailureReason": "mask_failed",
        "whitePixelRatio": 0.0,
    }
    if best_attempt is not None:
        selected_bundle = build_full_image_block_mask(
            image,
            block,
            protected_region_mask=protected_region_mask,
            dilation_px=int(best_attempt["dilationPx"]),
            feather_px=feather_px,
            allow_protected_overlap=True,
        )

    cleanup_debug["maskDilationPx"] = int(best_attempt["dilationPx"]) if best_attempt is not None else base_dilation_px
    cleanup_debug["selectedDilationPx"] = int(best_attempt["dilationPx"]) if best_attempt is not None else None
    cleanup_debug["bestCleanupAttempt"] = {
        "dilationPx": best_attempt.get("dilationPx"),
        "providerQualityStatus": best_attempt.get("providerQualityStatus"),
        "ghostingScore": best_attempt.get("ghostingScore"),
        "failedSourceWords": best_attempt.get("failedSourceWords", []),
    } if best_attempt is not None else None
    cleanup_debug["providerQualityStatus"] = best_attempt.get("providerQualityStatus", "provider_quality_failed") if best_attempt is not None else "provider_quality_failed"
    cleanup_debug["cleanupSelectedReason"] = ""
    cleanup_debug["maskPolarity"] = selected_bundle.get("maskPolarity")
    cleanup_debug["textCoverageEstimate"] = selected_bundle.get("textCoverageEstimate")
    cleanup_debug["antiAliasCoverageEstimate"] = selected_bundle.get("antiAliasCoverageEstimate")
    cleanup_debug["backgroundLeakageEstimate"] = selected_bundle.get("backgroundLeakageEstimate")
    cleanup_debug["protectedObjectOverlapEstimate"] = selected_bundle.get("protectedObjectOverlapEstimate")
    cleanup_debug["maskQualityStatus"] = selected_bundle.get("maskQualityStatus")
    cleanup_debug["maskFailureReason"] = selected_bundle.get("maskFailureReason")
    cleanup_debug["maskWhitePixelRatio"] = selected_bundle.get("whitePixelRatio")
    cleanup_debug["textPixelDetectionMethodsUsed"] = selected_bundle.get("textPixelDetectionMethodsUsed", [])
    cleanup_debug["textStrokeMaskRawImage"] = selected_bundle.get("raw")
    cleanup_debug["textStrokeMaskFilledImage"] = selected_bundle.get("filled")
    cleanup_debug["textStrokeMaskDilatedImage"] = selected_bundle.get("dilated")
    cleanup_debug["textStrokeMaskFinalImage"] = selected_bundle.get("final")
    cleanup_debug["syntheticGlyphMaskImage"] = selected_bundle.get("syntheticGlyphMask")
    cleanup_debug["adaptiveColorClusterMaskImage"] = selected_bundle.get("adaptiveColorClusterMask")
    cleanup_debug["localContrastMaskImage"] = selected_bundle.get("localContrastMask")
    cleanup_debug["edgeStrokeMaskImage"] = selected_bundle.get("edgeStrokeMask")
    cleanup_debug["sourceTextPixelMaskImage"] = selected_bundle.get("sourceTextPixelMask")
    cleanup_debug["combinedBinaryMaskBeforeDilationImage"] = selected_bundle.get("combinedBinaryMaskBeforeDilation")
    cleanup_debug["inpaintMaskBinaryImage"] = selected_bundle.get("inpaintMask_binary")
    cleanup_debug["compositeMaskSoftImage"] = selected_bundle.get("compositeMask_soft")
    cleanup_debug["compositeMaskAvailable"] = selected_bundle.get("compositeMask_soft") is not None
    cleanup_debug["adaptiveTextPixelDetection"] = True
    cleanup_debug["maskTypeUsedForLaMa"] = "binary"
    cleanup_debug["firstPassProvider"] = best_attempt.get("firstPassProvider", requested_provider) if best_attempt is not None else requested_provider
    cleanup_debug["firstPassStatus"] = best_attempt.get("firstPassStatus", "not_started") if best_attempt is not None else "not_started"
    cleanup_debug["residualTextDetectedAfterFirstPass"] = bool(best_attempt.get("residualTextDetectedAfterFirstPass", False)) if best_attempt is not None else False
    cleanup_debug["residualWordsAfterFirstPass"] = best_attempt.get("residualWordsAfterFirstPass", []) if best_attempt is not None else []
    cleanup_debug["residualMaskGenerated"] = bool(best_attempt.get("residualMaskGenerated", False)) if best_attempt is not None else False
    cleanup_debug["secondPassProvider"] = best_attempt.get("secondPassProvider", requested_provider) if best_attempt is not None else requested_provider
    cleanup_debug["secondPassStatus"] = best_attempt.get("secondPassStatus", "not_attempted") if best_attempt is not None else "not_attempted"
    cleanup_debug["residualTextDetectedAfterSecondPass"] = bool(best_attempt.get("residualTextDetectedAfterSecondPass", False)) if best_attempt is not None else False
    cleanup_debug["residualWordsAfterSecondPass"] = best_attempt.get("residualWordsAfterSecondPass", []) if best_attempt is not None else []
    cleanup_debug["sdxlFallbackAttempted"] = bool(best_attempt.get("sdxlFallbackAttempted", False)) if best_attempt is not None else False
    cleanup_debug["sdxlFallbackStatus"] = best_attempt.get("sdxlFallbackStatus", "not_attempted") if best_attempt is not None else "not_attempted"
    cleanup_debug["finalCleanupQualityStatus"] = best_attempt.get("providerQualityStatus", "provider_quality_failed") if best_attempt is not None else "provider_quality_failed"
    cleanup_debug["cleanupPassesRun"] = best_attempt.get("cleanupPassesRun", 0) if best_attempt is not None else 0
    cleanup_debug["residualWordsByPass"] = best_attempt.get("residualWordsByPass", []) if best_attempt is not None else []
    cleanup_debug["residualMaskAreaByPass"] = best_attempt.get("residualMaskAreaByPass", []) if best_attempt is not None else []
    cleanup_debug["cleanupImprovementByPass"] = best_attempt.get("cleanupImprovementByPass", []) if best_attempt is not None else []
    cleanup_debug["stoppedReason"] = best_attempt.get("stoppedReason", "") if best_attempt is not None else "no_cleanup_attempt"
    cleanup_debug["residualWordBoxesAfterFirstPass"] = best_attempt.get("residualWordBoxesAfterFirstPass", []) if best_attempt is not None else []
    cleanup_debug["residualMaskCoverageByWord"] = best_attempt.get("residualMaskCoverageByWord", []) if best_attempt is not None else []
    cleanup_debug["residualMaskFalsePositiveEstimate"] = best_attempt.get("residualMaskFalsePositiveEstimate", 0.0) if best_attempt is not None else 0.0
    cleanup_debug["sdxlFallbackConfigured"] = bool(best_attempt.get("sdxlFallbackConfigured", False)) if best_attempt is not None else False
    cleanup_debug["sdxlControlType"] = best_attempt.get("sdxlControlType") if best_attempt is not None else os.getenv("ADAPTIFAI_SDXL_CONTROL_TYPE", "canny")
    cleanup_debug["sdxlModelId"] = best_attempt.get("sdxlModelId") if best_attempt is not None else os.getenv("ADAPTIFAI_SDXL_INPAINT_MODEL")
    cleanup_debug["controlNetModelId"] = best_attempt.get("controlNetModelId") if best_attempt is not None else os.getenv("ADAPTIFAI_SDXL_CONTROLNET_MODEL")
    cleanup_debug["sdxlFailureReason"] = best_attempt.get("sdxlFailureReason", "") if best_attempt is not None else ""
    cleanup_debug["deepFillUsedForSdxl"] = bool(best_attempt.get("deepFillUsedForSdxl", False)) if best_attempt is not None else False
    cleanup_debug["deepFillMeta"] = best_attempt.get("deepFillMeta", {}) if best_attempt is not None else {}
    cleanup_debug["firstPassLamaOutputImage"] = best_attempt.get("firstPassImage") if best_attempt is not None else None
    cleanup_debug["residualTextMaskImage"] = best_attempt.get("residualMaskImage") if best_attempt is not None else None
    cleanup_debug["residualGlyphMaskAfterFirstPassImage"] = best_attempt.get("residualGlyphMaskAfterFirstPassImage") if best_attempt is not None else None
    cleanup_debug["residualArtifactMaskExpandedImage"] = best_attempt.get("residualArtifactMaskExpandedImage") if best_attempt is not None else None
    cleanup_debug["secondPassLamaOutputImage"] = best_attempt.get("secondPassImage") if best_attempt is not None else None
    cleanup_debug["finalCleanedCandidateImage"] = best_attempt.get("image") if best_attempt is not None else None
    cleanup_debug["residualOCRAfterFirstPass"] = best_attempt.get("residualOCRAfterFirstPass", []) if best_attempt is not None else []
    cleanup_debug["residualOCRAfterSecondPass"] = best_attempt.get("residualOCRAfterSecondPass", []) if best_attempt is not None else []
    cleanup_debug["cleanupCascadeLog"] = [
        {
            "dilationPx": attempt.get("dilationPx"),
            "firstPassProvider": attempt.get("firstPassProvider", requested_provider),
            "firstPassStatus": attempt.get("firstPassStatus", "not_started"),
            "residualTextDetectedAfterFirstPass": bool(attempt.get("residualTextDetectedAfterFirstPass", False)),
            "residualWordsAfterFirstPass": attempt.get("residualWordsAfterFirstPass", []),
            "residualMaskGenerated": bool(attempt.get("residualMaskGenerated", False)),
            "secondPassProvider": attempt.get("secondPassProvider", requested_provider),
            "secondPassStatus": attempt.get("secondPassStatus", "not_attempted"),
            "residualTextDetectedAfterSecondPass": bool(attempt.get("residualTextDetectedAfterSecondPass", False)),
            "residualWordsAfterSecondPass": attempt.get("residualWordsAfterSecondPass", []),
            "sdxlFallbackAttempted": bool(attempt.get("sdxlFallbackAttempted", False)),
            "sdxlFallbackStatus": attempt.get("sdxlFallbackStatus", "not_attempted"),
            "finalCleanupQualityStatus": attempt.get("providerQualityStatus", "provider_quality_failed"),
            "cleanupPassesRun": attempt.get("cleanupPassesRun", 0),
            "residualWordsByPass": attempt.get("residualWordsByPass", []),
            "residualMaskAreaByPass": attempt.get("residualMaskAreaByPass", []),
            "cleanupImprovementByPass": attempt.get("cleanupImprovementByPass", []),
            "stoppedReason": attempt.get("stoppedReason", ""),
            "sdxlFallbackConfigured": bool(attempt.get("sdxlFallbackConfigured", False)),
            "sdxlFailureReason": attempt.get("sdxlFailureReason", ""),
            "deepFillUsedForSdxl": bool(attempt.get("deepFillUsedForSdxl", False)),
            "deepFillMeta": attempt.get("deepFillMeta", {}),
        }
        for attempt in attempt_summaries
    ]

    if best_attempt is None:
        reason = "provider_quality_failed" if any(attempt.get("maskQualityStatus") == "passed" for attempt in attempt_summaries) else (selected_bundle.get("maskFailureReason") or "mask_failed")
        cleanup_debug["providerFailureReason"] = reason
        cleanup_debug["cleanupSelectedReason"] = "no valid cleanup attempt passed provider quality gate"
        cleanup_debug["cleanupWarnings"].append(
            {
                "id": block.id,
                "lineIndex": -1,
                "warning": "Large-block cleanup failed across all dilation attempts",
                "reason": reason,
            }
        )
        return image.convert("RGB"), {
            "requestedCleanupProvider": requested_provider,
            "actualCleanupProvider": requested_provider if reason == "provider_quality_failed" else None,
            "providerAvailable": True,
            "providerFailureReason": reason,
            "maskDilationPx": cleanup_debug["maskDilationPx"],
            "maskFeatherPx": feather_px,
            "inpaintingInputMode": "full_image",
            "selectedStrategy": reason,
            "cleanupPassed": False,
            "reason": reason,
        }

    cleanup_debug["actualCleanupProvider"] = requested_provider
    cleanup_debug["providerFailureReason"] = "" if best_attempt.get("providerQualityStatus") == "passed" else "cleanup_failed"
    cleanup_debug["cleanupSelectedReason"] = (
        f"dilation {best_attempt['dilationPx']} selected with provider quality status {best_attempt.get('providerQualityStatus')} "
        f"and ghosting score {best_attempt.get('ghostingScore')}"
    )
    candidate_image = best_attempt["image"].convert("RGB")
    if best_attempt.get("providerQualityStatus") == "passed":
        cleanup_debug["cleanupWarnings"].append(
            {
                "id": block.id,
                "lineIndex": -1,
                "warning": "Large-block cleanup accepted",
                "strategy": requested_provider,
                "selectedDilationPx": best_attempt["dilationPx"],
            }
        )
        return candidate_image, {
            "requestedCleanupProvider": requested_provider,
            "actualCleanupProvider": requested_provider,
            "providerAvailable": True,
            "providerFailureReason": "",
            "maskDilationPx": int(best_attempt["dilationPx"]),
            "maskFeatherPx": feather_px,
            "inpaintingInputMode": "full_image",
            "selectedStrategy": requested_provider,
            "cleanupPassed": True,
            "reason": cleanup_debug["cleanupSelectedReason"],
        }

    cleanup_debug["cleanupWarnings"].append(
        {
            "id": block.id,
            "lineIndex": -1,
            "warning": "Large-block cleanup rejected by validation",
            "strategy": requested_provider,
            "selectedDilationPx": best_attempt["dilationPx"],
            "failedSourceWords": best_attempt.get("failedSourceWords", []),
            "residualSourceOCR": best_attempt.get("residualSourceOCR", []),
            "visualGhostingDetected": best_attempt.get("visualGhostingDetected", False),
        }
    )
    return candidate_image, {
        "requestedCleanupProvider": requested_provider,
        "actualCleanupProvider": requested_provider,
        "providerAvailable": True,
        "providerFailureReason": "cleanup_failed",
        "maskDilationPx": int(best_attempt["dilationPx"]),
        "maskFeatherPx": feather_px,
        "inpaintingInputMode": "full_image",
        "selectedStrategy": "cleanup_failed",
        "cleanupPassed": False,
        "reason": "source text still visible after provider cleanup",
    }


def enforce_localize_cleanup_gate(
    source_image: Image.Image,
    cleaned_image: Image.Image,
    blocks: list[TextBlock],
    cleanup_debug: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    protected_region_mask = cleanup_debug.get("protectedRegionMaskImage")
    gated_image = cleaned_image.convert("RGB")
    block_statuses: list[dict[str, Any]] = []
    residual_source_ocr: list[dict[str, Any]] = []
    failed_source_words: list[dict[str, Any]] = []
    block_level_fallback_attempted = False
    block_level_fallback_selected = False

    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue
        visibility = evaluate_block_cleanup_visibility(source_image, gated_image, block)
        residual_source_ocr.append(
            {
                "id": block.id,
                "texts": visibility["residualSourceOCR"],
                "similarity": visibility["residualSimilarity"],
                "ghostingScore": visibility["ghostingScore"],
                "residualTextScore": visibility["residualTextScore"],
                "lineGhostingDeltas": visibility["lineGhostingDeltas"],
                "visualGhostingDetected": visibility["visualGhostingDetected"],
            }
        )
        failed_source_words.append(
            {
                "id": block.id,
                "words": visibility["failedSourceWords"],
            }
        )
        if visibility["cleanupStatus"] == "passed":
            block_statuses.append(
                {
                    "id": block.id,
                    "cleanupStatus": "passed",
                    "blockLevelFallbackAttempted": False,
                    "blockLevelFallbackSelected": False,
                    "failedSourceWords": [],
                }
            )
            continue

        if is_large_marketing_headline_block(block, source_image.size):
            block_statuses.append(
                {
                    "id": block.id,
                    "cleanupStatus": "failed",
                    "blockLevelFallbackAttempted": False,
                    "blockLevelFallbackSelected": False,
                    "failedSourceWords": visibility["failedSourceWords"],
                    "reason": "source text still visible after cleanup",
                }
            )
            cleanup_debug["cleanupWarnings"].append(
                {
                    "id": block.id,
                    "lineIndex": -1,
                    "warning": "source text still visible after cleanup",
                    "failedSourceWords": visibility["failedSourceWords"],
                    "residualSourceOCR": visibility["residualSourceOCR"],
                    "ghostingScore": visibility["ghostingScore"],
                    "residualTextScore": visibility["residualTextScore"],
                    "lineGhostingDeltas": visibility["lineGhostingDeltas"],
                }
            )
            continue

        block_level_fallback_attempted = True
        fallback_image, fallback_meta = attempt_block_level_generative_cleanup(
            gated_image,
            block,
            protected_region_mask if isinstance(protected_region_mask, Image.Image) else None,
            cleanup_debug,
        )
        if isinstance(fallback_image, Image.Image):
            gated_image = fallback_image
            visibility = evaluate_block_cleanup_visibility(source_image, gated_image, block)
            residual_source_ocr[-1] = {
                "id": block.id,
                "texts": visibility["residualSourceOCR"],
                "similarity": visibility["residualSimilarity"],
                "ghostingScore": visibility["ghostingScore"],
                "residualTextScore": visibility["residualTextScore"],
                "lineGhostingDeltas": visibility["lineGhostingDeltas"],
                "visualGhostingDetected": visibility["visualGhostingDetected"],
            }
            failed_source_words[-1] = {
                "id": block.id,
                "words": visibility["failedSourceWords"],
            }
        fallback_selected = bool(fallback_meta.get("blockLevelFallbackSelected", False))
        if fallback_selected:
            block_level_fallback_selected = True

        block_statuses.append(
            {
                "id": block.id,
                "cleanupStatus": visibility["cleanupStatus"],
                "blockLevelFallbackAttempted": True,
                "blockLevelFallbackSelected": fallback_selected and visibility["cleanupStatus"] == "passed",
                "failedSourceWords": visibility["failedSourceWords"],
                "reason": "source text still visible after cleanup" if visibility["cleanupStatus"] != "passed" else fallback_meta.get("reason", ""),
            }
        )
        if visibility["cleanupStatus"] != "passed":
            cleanup_debug["cleanupWarnings"].append(
                {
                    "id": block.id,
                    "lineIndex": -1,
                    "warning": "source text still visible after cleanup",
                    "failedSourceWords": visibility["failedSourceWords"],
                    "residualSourceOCR": visibility["residualSourceOCR"],
                    "ghostingScore": visibility["ghostingScore"],
                    "residualTextScore": visibility["residualTextScore"],
                    "lineGhostingDeltas": visibility["lineGhostingDeltas"],
                }
            )

    any_block_failed = any(status_entry.get("cleanupStatus") != "passed" for status_entry in block_statuses)
    cleanup_status = "cleanup_failed" if any_block_failed else "passed"
    return gated_image, {
        "cleanupStatus": cleanup_status,
        "blockCleanupStatus": block_statuses,
        "residualSourceOCR": residual_source_ocr,
        "failedSourceWords": failed_source_words,
        "blockLevelFallbackAttempted": block_level_fallback_attempted,
        "blockLevelFallbackSelected": block_level_fallback_selected,
        "finalRenderSkippedReason": "source text still visible after cleanup" if cleanup_status != "passed" else "",
    }


def parse_bold_markup(text: str) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("[BOLD]", cursor)
        if start == -1:
            remaining = text[cursor:]
            if remaining:
                segments.append((remaining, False))
            break
        if start > cursor:
            segments.append((text[cursor:start], False))
        end = text.find("[/BOLD]", start)
        if end == -1:
            segments.append((text[start + 6 :], True))
            break
        segments.append((text[start + 6 : end], True))
        cursor = end + 7
    return [(segment, bold) for segment, bold in segments if segment]


def get_font(size: int, bold: bool = False, family: str | None = None, category: str | None = None, weight: int | None = None):
    requested_weight = int(weight or (700 if bold else 400))
    google_font = get_google_font_file(family, bold, category, requested_weight)
    if google_font is not None:
        try:
            return ImageFont.truetype(str(google_font), size)
        except OSError as exc:
            print(f"[fonts] Google font load failed for {google_font}: {exc}", flush=True)

    local_cache_key = (f"{family or ''}:{normalize_font_category(category)}:{size}:{requested_weight}", bold)
    if local_cache_key in LOCAL_FONT_FILE_CACHE:
        return LOCAL_FONT_FILE_CACHE[local_cache_key]

    if bold:
        font_names = [
            family or "",
            *(
                ["C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/bahnschrift.ttf"]
                if requested_weight >= 800
                else []
            ),
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        font_names = [
            family or "",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for font_name in font_names:
        if not font_name:
            continue
        try:
            font = ImageFont.truetype(font_name, size)
            LOCAL_FONT_FILE_CACHE[local_cache_key] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default()
    LOCAL_FONT_FILE_CACHE[local_cache_key] = font
    return font


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


V62_CALIBRATED_POINT_SIZE_CACHE: dict[tuple[str, int, bool, str, int], int] = {}


def v62_line_sample_text(tokens: list[dict[str, Any]]) -> str:
    sample = " ".join(str(token.get("text") or "").strip() for token in tokens if str(token.get("text") or "").strip())
    return sample or "Hg0123456789"


def v62_calibrated_point_size(
    draw: ImageDraw.ImageDraw,
    tokens: list[dict[str, Any]],
    target_pixel_height: int,
    *,
    font_weight: int,
    font_category: str,
    bold: bool,
) -> int:
    target = max(1, int(target_pixel_height))
    sample = v62_line_sample_text(tokens)
    cache_key = (sample[:80], target, bool(bold), normalize_font_category(font_category), int(font_weight))
    if cache_key in V62_CALIBRATED_POINT_SIZE_CACHE:
        return V62_CALIBRATED_POINT_SIZE_CACHE[cache_key]

    best_size = target
    best_delta = 10**9
    upper = max(12, min(240, int(target * 3.0) + 12))
    for size in range(1, upper + 1):
        font = get_font(size=size, bold=bold, category=font_category, weight=font_weight)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), sample, font=font)
            pixel_height = max(1, int(bottom - top))
        except Exception:
            pixel_height = size
        delta = abs(pixel_height - target)
        if delta < best_delta or (delta == best_delta and pixel_height >= target and size < best_size):
            best_delta = delta
            best_size = size
        if pixel_height >= target and delta <= 1:
            best_size = size
            break

    V62_CALIBRATED_POINT_SIZE_CACHE[cache_key] = max(1, int(best_size))
    return max(1, int(best_size))


def build_wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    size: int,
    max_width: int,
    *,
    line_height_ratio: float = 1.08,
    max_lines: int | None = None,
) -> tuple[list[list[tuple[str, bool]]], int]:
    regular_font = get_font(size, bold=False)
    bold_font = get_font(size, bold=True)
    segments = parse_bold_markup(text)
    words: list[tuple[str, bool]] = []
    for segment, bold in segments:
        split = segment.replace("\n", " \n ").split(" ")
        for piece in split:
            if piece:
                words.append((piece, bold))

    lines: list[list[tuple[str, bool]]] = [[]]
    current_width = 0.0
    for token, bold in words:
        if token == "\n":
            lines.append([])
            current_width = 0
            continue
        token_text = token if not lines[-1] else f" {token}"
        font = bold_font if bold else regular_font
        width = text_width(draw, token_text, font)
        if lines[-1] and current_width + width > max_width:
            lines.append([(token, bold)])
            current_width = text_width(draw, token, font)
        else:
            lines[-1].append((token, bold))
            current_width += text_width(draw, token_text, font)
    filtered = [line for line in lines if line]
    if max_lines and len(filtered) > max_lines:
        overflow_tail = filtered[max_lines - 1 :]
        merged_tail: list[tuple[str, bool]] = []
        for line in overflow_tail:
            merged_tail.extend(line)
        filtered = filtered[: max_lines - 1] + [merged_tail]
    line_height = int(size * line_height_ratio)
    return filtered, line_height

def draw_rich_line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], segments: list[tuple[str, bool]], size: int, color: str, stroke_fill: str | None, stroke_width: int) -> None:
    x, y = xy
    for index, (text, bold) in enumerate(segments):
        token = text if index == 0 else f" {text}"
        font = get_font(size, bold=bold)
        draw.text((x, y), token, fill=color, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill or color)
        x += int(text_width(draw, token, font))


def tokenize_style_span(span: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(span.get("translatedText") or span.get("sourceText") or "").strip()
    text = text.replace("[BOLD]", "").replace("[/BOLD]", "").strip()
    if not text:
        return []
    role = str(span.get("matchedSourceRole") or span.get("semanticRole") or "benefit")
    style = dict(span.get("style", {}))
    source_word_id = str(span.get("sourceWordId") or span.get("source_word_id") or "").strip()
    raw_source_word_ids = span.get("sourceWordIds") or span.get("source_word_ids") or []
    source_word_ids = [str(value).strip() for value in raw_source_word_ids if str(value).strip()] if isinstance(raw_source_word_ids, list) else []
    if source_word_id and source_word_id not in source_word_ids:
        source_word_ids.insert(0, source_word_id)
    source_word_styles = [item for item in (span.get("sourceWordStyles") or []) if isinstance(item, dict)]

    def source_id_for_style(source_style: dict[str, Any]) -> str:
        return str(source_style.get("id") or "").strip()

    def source_style_for_tokens(tokens: list[str]) -> list[dict[str, Any] | None]:
        if not tokens or not source_word_styles:
            return [None for _ in tokens]
        assignments: list[dict[str, Any] | None] = [None for _ in tokens]
        used_source_indexes: set[int] = set()
        normalized_sources = [normalize_ocr_text(str(item.get("text") or "")) for item in source_word_styles]

        for token_index, token_text in enumerate(tokens):
            normalized_token = normalize_ocr_text(token_text)
            if not normalized_token:
                continue
            for source_index, normalized_source in enumerate(normalized_sources):
                if source_index in used_source_indexes:
                    continue
                if normalized_source and normalized_source == normalized_token:
                    assignments[token_index] = source_word_styles[source_index]
                    used_source_indexes.add(source_index)
                    break

        remaining_sources = [
            source_style
            for source_index, source_style in enumerate(source_word_styles)
            if source_index not in used_source_indexes
        ]
        remaining_cursor = 0
        for token_index, current in enumerate(assignments):
            if current is not None:
                continue
            if remaining_cursor < len(remaining_sources):
                assignments[token_index] = remaining_sources[remaining_cursor]
                remaining_cursor += 1
        return assignments

    def token_style_from_source(source_style: dict[str, Any] | None, fallback_style: dict[str, Any]) -> dict[str, Any]:
        if source_style:
            return style_from_source_word_style(source_style, fallback_style)
        if len(source_word_styles) > 1:
            return majority_source_style(source_word_styles, fallback_style)
        return dict(fallback_style)

    def token_source_ids_from_source(source_style: dict[str, Any] | None) -> list[str]:
        source_id = source_id_for_style(source_style or {})
        if source_id:
            return [source_id]
        return source_word_ids

    force_break_after = bool(span.get("forceBreakAfter", False))
    split_tokens_preview = [token for token in text.split() if token]
    punctuation_only = bool(re.fullmatch(r"[!?.,;:%]+", text))
    source_line_indexes = {
        int(source_style.get("lineIndex") or 0)
        for source_style in source_word_styles
        if isinstance(source_style, dict)
    }
    keep_whole = role in {"percentage", "numeric_claim", "brand/product_name"} and len(source_word_styles) <= 1 and len(split_tokens_preview) <= 2
    if force_break_after and len(split_tokens_preview) > 1 and not punctuation_only and (len(source_word_styles) <= 1 or len(source_line_indexes) == 1):
        keep_whole = True
    if "[BOLD]" in text or "[/BOLD]" in text:
        bold_tokens: list[dict[str, Any]] = []
        for segment_text, segment_bold in parse_bold_markup(text):
            segment_style = dict(style)
            if segment_bold:
                segment_style["fontWeight"] = max(700, int(segment_style.get("fontWeight") or 700))
            segment_tokens = [token for token in segment_text.split() if token]
            assignments = source_style_for_tokens(segment_tokens)
            for token, source_style in zip(segment_tokens, assignments, strict=False):
                if token:
                    token_style = token_style_from_source(source_style, segment_style)
                    bold_tokens.append({"text": token, "style": token_style, "role": role, "sourceWordIds": token_source_ids_from_source(source_style), "forceBreakAfter": False})
        if bold_tokens:
            bold_tokens[-1]["forceBreakAfter"] = force_break_after
        return bold_tokens
    if keep_whole:
        token_style = majority_source_style(source_word_styles, style) if source_word_styles else style
        return [{"text": text, "style": token_style, "role": role, "sourceWordIds": source_word_ids, "forceBreakAfter": force_break_after, "noSpaceBefore": punctuation_only}]
    split_tokens = split_tokens_preview
    assignments = source_style_for_tokens(split_tokens)
    tokens = [
        {
            "text": token,
            "style": token_style_from_source(source_style, style),
            "role": role,
            "sourceWordIds": token_source_ids_from_source(source_style),
            "forceBreakAfter": False,
            "noSpaceBefore": bool(re.fullmatch(r"[!?.,;:%]+", token)),
        }
        for token, source_style in zip(split_tokens, assignments, strict=False)
    ]
    if tokens:
        tokens[-1]["forceBreakAfter"] = force_break_after
    return tokens


def get_font_for_style(style: dict[str, Any], size: int):
    weight = int(style.get("fontWeight", 700))
    family = str(style.get("fontFamily") or "").strip() or None
    category = str(style.get("fontCategory") or "sans-serif").strip() or "sans-serif"
    return get_font(size, bold=weight >= 700, family=family, category=category, weight=weight)


def span_token_width(draw: ImageDraw.ImageDraw, token: dict[str, Any], *, first_in_line: bool, scale_factor: float) -> float:
    base_size = max(8, int(token["style"].get("fontSize", 16)))
    size = max(8, int(round(base_size * scale_factor)))
    font = get_font_for_style(token["style"], size)
    text = token["text"] if first_in_line else f" {token['text']}"
    return text_width(draw, text, font)


def span_token_metrics(token: dict[str, Any], scale_factor: float) -> dict[str, Any]:
    base_size = max(8, int(token["style"].get("fontSize", 16)))
    size = max(8, int(round(base_size * scale_factor)))
    font = get_font_for_style(token["style"], size)
    try:
        ascent, descent = font.getmetrics()
    except Exception:
        ascent, descent = size, max(2, size // 4)
    return {"font": font, "size": size, "ascent": ascent, "descent": descent, "style": token["style"], "role": token["role"], "text": token["text"]}


def build_styled_lines(
    draw: ImageDraw.ImageDraw,
    spans: list[dict[str, Any]],
    max_width: int,
    scale_factor: float,
    *,
    honor_force_breaks: bool = True,
) -> list[list[dict[str, Any]]]:
    tokens: list[dict[str, Any]] = []
    for span in spans:
        tokens.extend(tokenize_style_span(span))
    if not tokens:
        return []
    lines: list[list[dict[str, Any]]] = [[]]
    cursor_x = 0.0
    for token in tokens:
        width = span_token_width(draw, token, first_in_line=not lines[-1], scale_factor=scale_factor)
        if lines[-1] and cursor_x + width > max_width:
            lines.append([token])
            cursor_x = span_token_width(draw, token, first_in_line=True, scale_factor=scale_factor)
        else:
            lines[-1].append(token)
            cursor_x += width
        if honor_force_breaks and token.get("forceBreakAfter") and lines[-1]:
            lines.append([])
            cursor_x = 0.0
    return [line for line in lines if line]


def measure_styled_layout(
    draw: ImageDraw.ImageDraw,
    lines: list[list[dict[str, Any]]],
    scale_factor: float,
    line_height_ratio: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    total_height = 0
    widest = 0
    line_layouts: list[dict[str, Any]] = []
    for line in lines:
        line_width = 0.0
        metrics = [span_token_metrics(token, scale_factor) for token in line]
        max_ascent = max((item["ascent"] for item in metrics), default=0)
        max_descent = max((item["descent"] for item in metrics), default=0)
        max_size = max((item["size"] for item in metrics), default=12)
        line_height = max(int(round(max_size * line_height_ratio)), max_ascent + max_descent)
        for index, metric in enumerate(metrics):
            token_text = metric["text"] if index == 0 else f" {metric['text']}"
            line_width += text_width(draw, token_text, metric["font"])
        widest = max(widest, int(round(line_width)))
        total_height += line_height
        line_layouts.append(
            {
                "tokens": metrics,
                "lineWidth": int(round(line_width)),
                "lineHeight": line_height,
                "maxAscent": max_ascent,
                "maxDescent": max_descent,
            }
        )
    return widest, total_height, line_layouts


def fit_styled_spans(
    draw: ImageDraw.ImageDraw,
    spans: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    base_typography: dict[str, Any],
    preferred_line_count: int | None = None,
) -> dict[str, Any]:
    max_width = max(24, box[2] - box[0])
    max_height = max(16, box[3] - box[1])
    base_sizes = [max(10, int(span.get("style", {}).get("fontSize", base_typography.get("fontSize", 16)))) for span in spans] or [max(10, int(base_typography.get("fontSize", 16)))]
    min_scale = max(0.34, 10 / max(base_sizes))
    best: dict[str, Any] | None = None
    for scale_factor in np.linspace(1.0, min_scale, 28):
        lines = build_styled_lines(draw, spans, max_width, float(scale_factor))
        if not lines:
            continue
        widest, total_height, line_layouts = measure_styled_layout(
            draw,
            lines,
            float(scale_factor),
            max(1.0, min(1.5, float(base_typography.get("lineHeight", 18)) / max(1.0, float(base_typography.get("fontSize", 16))))),
        )
        overflow = max(0, widest - max_width) + max(0, total_height - max_height)
        line_count_penalty = 0
        if preferred_line_count:
            line_count_penalty = abs(len(lines) - max(1, preferred_line_count)) * 42
        score = float(scale_factor * 1000 - overflow * 5 - max(0, len(lines) - max(1, len(spans))) * 12 - line_count_penalty)
        candidate = {
            "scaleFactor": float(scale_factor),
            "lines": line_layouts,
            "widest": widest,
            "totalHeight": total_height,
            "overflow": overflow,
            "preferredLineCount": preferred_line_count,
            "lineCountDelta": abs(len(lines) - preferred_line_count) if preferred_line_count else 0,
        }
        if overflow == 0:
            if best is None or score > best["score"]:
                candidate["score"] = score
                best = candidate
        elif best is None:
            candidate["score"] = score
            best = candidate
    if best is None:
        return {"scaleFactor": 1.0, "lines": [], "widest": 0, "totalHeight": 0, "overflow": max_width}
    return best


def assess_layout_quality(
    layout: dict[str, Any],
    box: tuple[int, int, int, int],
    spans: list[dict[str, Any]],
    block: TextBlock,
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    box_width = max(1, box[2] - box[0])
    box_height = max(1, box[3] - box[1])
    scale_factor = float(layout.get("scaleFactor", 1.0))
    widest = max(1, int(layout.get("widest", 0)))
    total_height = max(1, int(layout.get("totalHeight", 0)))
    overflow = int(layout.get("overflow", 0))
    occupancy_x = widest / box_width
    occupancy_y = total_height / box_height
    occupancy = occupancy_x * occupancy_y
    line_count = len(layout.get("lines", []))
    preferred_lines = max(1, len(block.line_texts) or len([line for line in (block.translated_text or "").splitlines() if line.strip()]) or 1)
    score = 100.0
    if overflow > 0:
        warnings.append("overflow")
        score -= min(40.0, overflow / 8)
    if scale_factor < 0.72:
        warnings.append("font scaled down heavily")
        score -= (0.72 - scale_factor) * 70
    if occupancy_x < 0.42:
        warnings.append("line width underused")
        score -= (0.42 - occupancy_x) * 24
    if occupancy_x > 0.98:
        warnings.append("line width crowded")
        score -= (occupancy_x - 0.98) * 60
    if occupancy_y < 0.28 and block.role in {"headline", "benefit"}:
        warnings.append("box too empty")
        score -= (0.28 - occupancy_y) * 22
    if occupancy_y > 0.92:
        warnings.append("box too dense")
        score -= (occupancy_y - 0.92) * 60
    if abs(line_count - preferred_lines) > 2:
        warnings.append("line count far from source rhythm")
        score -= abs(line_count - preferred_lines) * 6
    if layout.get("lines"):
        last_line = layout["lines"][-1]
        last_tokens = [token.get("text", "") for token in last_line.get("tokens", [])]
        if len(last_tokens) == 1 and len(last_tokens[0].split()) == 1 and line_count > 1:
            warnings.append("orphan last line")
            score -= 12
    return max(0.0, score), warnings


def select_translation_candidate_for_layout(
    draw: ImageDraw.ImageDraw,
    block: TextBlock,
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    base_typography = default_typography_style(block)
    candidates = block.translation_candidates or [{"label": "faithful", "text": block.translated_text or block.text}]
    evaluations: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_text = str(candidate.get("text", "")).strip()
        if not candidate_text:
            continue
        spans = infer_translated_style_spans(block.text, candidate_text, block.source_style_spans or infer_source_style_spans(block.text, block), block)
        preferred_line_count = max(1, len(block.line_texts) or len([line for line in block.text.splitlines() if line.strip()]) or 1)
        layout = fit_styled_spans(draw, spans, box, base_typography, preferred_line_count=preferred_line_count)
        quality_score, warnings = assess_layout_quality(layout, box, spans, block)
        faithful_token_count = max(1, len((block.translated_text or block.text).replace("\n", " ").split()))
        candidate_token_count = len(candidate_text.replace("\n", " ").split())
        semantic_coverage = candidate_token_count / faithful_token_count
        if semantic_coverage < 0.78:
            warnings = warnings + ["semantic coverage reduced"]
            quality_score -= (0.78 - semantic_coverage) * 80
        evaluations.append(
            {
                "label": candidate.get("label", f"candidate_{index + 1}"),
                "text": candidate_text,
                "spans": spans,
                "layout": layout,
                "qualityScore": round(quality_score, 3),
                "warnings": warnings,
                "semanticCoverage": round(semantic_coverage, 3),
            }
        )
    if not evaluations:
        fallback_text = block.translated_text or block.text
        fallback_spans = infer_translated_style_spans(block.text, fallback_text, block.source_style_spans or infer_source_style_spans(block.text, block), block)
        preferred_line_count = max(1, len(block.line_texts) or len([line for line in block.text.splitlines() if line.strip()]) or 1)
        fallback_layout = fit_styled_spans(draw, fallback_spans, box, base_typography, preferred_line_count=preferred_line_count)
        return {
            "label": "faithful",
            "text": fallback_text,
            "spans": fallback_spans,
            "layout": fallback_layout,
            "qualityScore": 0.0,
            "warnings": ["no candidates"],
            "reason": "fallback faithful candidate used",
            "candidates": [],
        }
    best = max(
        evaluations,
        key=lambda item: (
            item["qualityScore"],
            1 if item["label"] == "shorter_marketing" else 0,
            1 if item["label"] == "faithful" else 0,
        ),
    )
    reason = "selected highest layout quality candidate"
    if best["label"] == "shorter_marketing":
        reason = "selected shorter marketing candidate for better visual fit"
    elif best["label"] == "compact_layout_safe":
        reason = "selected compact candidate because box fit was tighter"
    elif best["label"] == "faithful":
        reason = "selected faithful candidate because layout quality was sufficient"
    best["reason"] = reason
    best["candidates"] = [
        {
            "label": item["label"],
            "text": item["text"],
            "qualityScore": item["qualityScore"],
            "warnings": item["warnings"],
            "semanticCoverage": item.get("semanticCoverage", 1.0),
        }
        for item in evaluations
    ]
    return best


def analyze_semantic_grouping(blocks: list[TextBlock], image_size: tuple[int, int]) -> dict[str, Any]:
    headline_blocks = [block for block in blocks if block.translate and block.role == "headline" and block.surface == "overlay"]
    if not headline_blocks:
        return {
            "semanticBlockGroupingStatus": "no_headline_detected",
            "headlineMergedIntoSingleBlock": False,
            "groupedLineIds": [],
            "groupingReason": "No translatable headline block detected.",
            "groupingConfidence": 0.0,
            "rejectedMergeReasons": [],
            "blockType": None,
        }
    primary = max(
        headline_blocks,
        key=lambda block: ((block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1]), len(block.line_texts)),
    )
    grouped_line_ids = [f"{primary.id}:line-{index}" for index, _ in enumerate(primary.line_texts)]
    competing = [
        block for block in headline_blocks
        if block.id != primary.id and overlap_fraction(block.bbox, primary.bbox) >= 0.18
    ]
    merged = len(primary.line_texts) > 1 and not competing
    confidence = min(
        1.0,
        0.45
        + min(0.35, len(primary.line_texts) * 0.14)
        + min(0.2, ((primary.bbox[2] - primary.bbox[0]) / max(1, image_size[0])) * 0.3),
    )
    rejected_reasons: list[str] = []
    if competing:
        rejected_reasons.append("multiple overlapping headline blocks still present")
    if len(primary.line_texts) <= 1:
        rejected_reasons.append("headline block contains only one line")
    return {
        "semanticBlockGroupingStatus": "passed" if merged else "partial",
        "headlineMergedIntoSingleBlock": merged,
        "groupedLineIds": grouped_line_ids,
        "groupingReason": "Merged visually consistent headline lines into one semantic block." if merged else "Headline grouping remains fragmented or ambiguous.",
        "groupingConfidence": round(confidence, 3),
        "rejectedMergeReasons": rejected_reasons,
        "blockType": primary.role,
    }


def draw_rich_text_token(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Any,
    stroke_width: int,
    stroke_fill: Any,
    italic: bool,
) -> None:
    if not italic or not hasattr(draw, "_image"):
        draw.text(position, text, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
        return

    x, y = position
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    pad = max(4, stroke_width + 3)
    shear = 0.22
    shear_px = int(round(height * shear))
    layer_width = width + pad * 2 + shear_px + 2
    layer_height = height + pad * 2 + 2
    layer = Image.new("RGBA", (layer_width, layer_height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )
    try:
        sheared = layer.transform(
            (layer_width, layer_height),
            Image.Transform.AFFINE,
            (1, shear, -shear * layer_height, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )
        draw._image.alpha_composite(sheared, (int(round(x)) - pad, int(round(y)) - pad))
    except Exception:
        draw.text(position, text, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)


def draw_rich_text_decorations(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    *,
    style: dict[str, Any],
    fill: Any,
) -> None:
    if not (style.get("isUnderlined") or style.get("isStrikethrough")):
        return
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return
    color = style.get("decorationColor") or style.get("color") or fill or "#111111"
    source_height = int(style.get("sourcePixelHeight") or style.get("fontSize") or max(1, bottom - top))
    line_width = max(1, int(round(source_height * 0.07)))
    if style.get("isUnderlined"):
        y = bottom + max(1, int(round(source_height * 0.08)))
        draw.line((left, y, right, y), fill=color, width=line_width)
    if style.get("isStrikethrough"):
        y = top + int(round((bottom - top) * 0.54))
        draw.line((left, y, right, y), fill=color, width=line_width)


def choose_render_fill_for_style(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    style: dict[str, Any],
    fallback_fill: Any,
) -> Any:
    if not (style.get("hasTextBackground") and style.get("backgroundColor")):
        return fallback_fill
    if style.get("color"):
        return style.get("color")
    if not hasattr(draw, "_image"):
        return fallback_fill

    left, top, right, bottom = bbox
    image = draw._image.convert("RGB")
    left = max(0, min(image.width, int(left)))
    right = max(left + 1, min(image.width, int(right)))
    top = max(0, min(image.height, int(top)))
    bottom = max(top + 1, min(image.height, int(bottom)))
    sample = np.array(image.crop((left, top, right, bottom)), dtype=np.float32)
    if sample.size == 0:
        return fallback_fill

    observed_bg = tuple(int(value) for value in np.median(sample.reshape(-1, 3), axis=0))
    source_bg = parse_hex_color(str(style.get("backgroundColor") or "#ffffff"), fallback=(255, 255, 255))
    source_fg = parse_hex_color(str(style.get("color") or "#111111"), fallback=(17, 17, 17))
    bg_distance = float(np.linalg.norm(np.array(observed_bg, dtype=np.float32) - np.array(source_bg, dtype=np.float32)))
    source_fg_contrast = contrast_ratio(observed_bg, source_fg)
    source_bg_as_text_contrast = contrast_ratio(observed_bg, source_bg)

    if bg_distance <= 42.0 and source_fg_contrast >= 2.2:
        return style.get("color") or fallback_fill
    if source_bg_as_text_contrast > source_fg_contrast:
        return style.get("backgroundColor") or fallback_fill
    return style.get("color") or fallback_fill


def render_styled_spans(
    draw: ImageDraw.ImageDraw,
    layout: dict[str, Any],
    box: tuple[int, int, int, int],
    *,
    alignment: str,
    style: dict[str, Any] | None = None,
    precise_inline: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_height = int(layout.get("totalHeight", 0))
    if layout.get("verticalOverflowAllowed"):
        y = box[1]
    else:
        y = box[1] + max(0, (box[3] - box[1] - total_height) // 2)

    rendered_spans: list[dict[str, Any]] = []
    span_render_boxes: list[dict[str, Any]] = []
    shadow = style.get("shadow") if style else None

    for line_index, line in enumerate(layout.get("lines", [])):
        line_width = int(line.get("lineWidth", 0))
        
        # KURAL: Sol hizalamada padding s?f?r
        if alignment == "left":
            x = box[0]
        elif layout.get("blockCenterX") is not None:
            x = int(round(float(layout["blockCenterX"]) - (line_width / 2)))
        else:
            x = box[0] + max(0, (box[2] - box[0] - line_width) // 2)

        baseline = y + int(line.get("maxAscent", 0))
        line_font_size = line.get("lineFontSize", 16)
        for token_index, token in enumerate(line.get("tokens", [])):
            token_text = token["text"]
            font = token["font"]
            token_style = token.get("style", {})
            fill = token_style.get("color", "#111111")
            
            stroke_width = max(0, min(14, int(round(float(token_style.get("strokeWidth") or 0)))))
            stroke_fill = token_style.get("strokeFill") or fill
            
            # Use Pillow native stroke/transparent fill instead of the old 8-direction loop.
            if token_style.get("fillTransparent"):
                fill = (0, 0, 0, 0)

            try:
                advance = float(token.get("xAdvance") or (draw.textlength(token_text, font=font) + draw.textlength(" ", font=font)))
            except AttributeError:
                advance = float(token.get("xAdvance") or (text_width(draw, token_text, font) + text_width(draw, " ", font)))

            left, top, right, bottom = draw.textbbox(
                (x, baseline - token["ascent"]),
                token_text,
                font=font,
                stroke_width=stroke_width,
            )
            fill = choose_render_fill_for_style(draw, (int(left), int(top), int(right), int(bottom)), token_style, fill)
            if token_style.get("hasTextBackground") and token_style.get("backgroundColor"):
                bg_pad_x = max(1, int(round(float(token_style.get("sourcePixelHeight") or line_font_size or 16) * 0.12)))
                bg_pad_y = max(1, int(round(float(token_style.get("sourcePixelHeight") or line_font_size or 16) * 0.10)))
                bg_left = max(0, int(round(x)) - bg_pad_x)
                bg_top = max(0, int(top) - bg_pad_y)
                bg_right = min(getattr(draw._image, "width", int(right)), int(round(x + advance)) + bg_pad_x)
                bg_bottom = min(getattr(draw._image, "height", int(bottom)), int(bottom) + bg_pad_y)
                if bg_right > bg_left and bg_bottom > bg_top:
                    draw.rectangle((bg_left, bg_top, bg_right, bg_bottom), fill=token_style.get("backgroundColor"))
            if shadow and shadow.get("enabled"):
                shadow_fill = shadow.get("color", "#000000")
                shadow_offset = int(shadow.get("offset", 2))
                draw.text((x + shadow_offset, baseline - token["ascent"] + shadow_offset), token_text, fill=shadow_fill, font=font)
            
            text_y = baseline - token["ascent"]
            if token_style.get("fillTransparent") and stroke_width > 0 and hasattr(draw, "_image"):
                overlay = Image.new("RGBA", draw._image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                draw_rich_text_token(
                    overlay_draw,
                    (x, text_y),
                    token_text,
                    fill=(0, 0, 0, 0),
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                    italic=bool(token_style.get("isItalic")),
                )
                draw._image.alpha_composite(overlay)
            else:
                draw_rich_text_token(
                    draw,
                    (x, text_y),
                    token_text,
                    fill=fill,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                    italic=bool(token_style.get("isItalic")),
                )
            draw_rich_text_decorations(
                draw,
                (int(left), int(top), int(right), int(bottom)),
                style=token_style,
                fill=stroke_fill if token_style.get("fillTransparent") else fill,
            )
            
            rendered_spans.append({
                "lineIndex": line_index,
                "text": token["text"],
                "role": token["role"],
                "style": token["style"],
            })
            span_render_boxes.append({
                "lineIndex": line_index,
                "text": token["text"],
                "bbox": [left, top, right, bottom],
            })
            
            # KURAL: Explicit Space xAdvance
            if precise_inline or "xAdvance" in token:
                x += advance
            else:
                try:
                    x += draw.textlength(token_text, font=font)
                except AttributeError:
                    x += text_width(draw, token_text, font)
            
        y += int(line.get("lineHeight", 0)) + int(line_font_size * 0.25)

    return rendered_spans, span_render_boxes


def serialize_styled_layout(layout: dict[str, Any]) -> dict[str, Any]:
    serialized_lines: list[dict[str, Any]] = []
    for line in layout.get("lines", []):
        serialized_lines.append(
            {
                "lineWidth": line.get("lineWidth", 0),
                "lineHeight": line.get("lineHeight", 0),
                "lineFontSize": line.get("lineFontSize", 0),
                "lineTop": line.get("lineTop", 0),
                "lineBottom": line.get("lineBottom", 0),
                "maxAscent": line.get("maxAscent", 0),
                "maxDescent": line.get("maxDescent", 0),
                "tokens": [
                    {
                        "text": token.get("text", ""),
                        "size": token.get("size", 0),
                        "xAdvance": token.get("xAdvance", 0),
                        "role": token.get("role", ""),
                        "style": token.get("style", {}),
                    }
                    for token in line.get("tokens", [])
                ],
            }
        )
    return {
        "scaleFactor": layout.get("scaleFactor", 1.0),
        "widest": layout.get("widest", 0),
        "totalHeight": layout.get("totalHeight", 0),
        "overflow": layout.get("overflow", 0),
        "typesetting": layout.get("typesetting"),
        "blockCenterX": layout.get("blockCenterX"),
        "lines": serialized_lines,
    }


def source_spans_to_renderable(source_style_spans: list[dict[str, Any]], block: TextBlock) -> list[dict[str, Any]]:
    if not source_style_spans:
        base_style = default_typography_style(block)
        return [
            {
                "translatedText": block.text,
                "matchedSourceRole": "benefit",
                "style": base_style,
                "sourceSegmentHint": block.text,
                "forceBreakAfter": False,
            }
        ]
    return [
        {
            "translatedText": span.get("sourceText", block.text),
            "matchedSourceRole": span.get("semanticRole", "benefit"),
            "style": span.get("style", default_typography_style(block)),
            "sourceSegmentHint": span.get("sourceText", block.text),
            "forceBreakAfter": span.get("forceBreakAfter", False),
        }
        for span in source_style_spans
    ]


def build_render_audit_artifacts(
    image_size: tuple[int, int],
    blocks: list[TextBlock],
    render_plan: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source_canvas = Image.new("RGBA", image_size, (255, 255, 255, 0))
    translated_canvas = Image.new("RGBA", image_size, (255, 255, 255, 0))
    source_draw = ImageDraw.Draw(source_canvas)
    translated_draw = ImageDraw.Draw(translated_canvas)
    renderable_blocks = filter_render_blocks(blocks, dict(render_plan or {}))

    block_plans: list[dict[str, Any]] = []
    overflow_warnings: list[dict[str, Any]] = []
    selected_font_scale: list[dict[str, Any]] = []
    styled_span_layout: list[dict[str, Any]] = []
    span_render_boxes: list[dict[str, Any]] = []
    source_style_spans_summary: list[dict[str, Any]] = []
    translated_style_spans_summary: list[dict[str, Any]] = []

    for block in renderable_blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue

        block_plan = (render_plan or {}).get(block.id or "", {})
        bbox = tuple(block_plan.get("finalTextBox") or render_bbox_for_block(block, image_size, 0, 0))
        base_typography = default_typography_style(block)
        source_spans = block.source_style_spans or infer_source_style_spans(block.text, block)
        renderable_source_spans = source_spans_to_renderable(source_spans, block)
        preferred_line_count = max(1, len(block.line_texts) or len([line for line in block.text.splitlines() if line.strip()]) or 1)
        source_layout = fit_styled_spans(source_draw, renderable_source_spans, bbox, base_typography, preferred_line_count=preferred_line_count)
        _, source_boxes = render_styled_spans(source_draw, source_layout, bbox, alignment=block.align)

        candidate_choice = select_translation_candidate_for_layout(translated_draw, block, bbox)
        translated_layout = candidate_choice["layout"]
        rendered_spans, translated_boxes = render_styled_spans(translated_draw, translated_layout, bbox, alignment=block.align)
        quality_score, warnings = assess_layout_quality(translated_layout, bbox, candidate_choice["spans"], block)

        source_style_spans_summary.append({"id": block.id, "spans": source_spans})
        translated_style_spans_summary.append({"id": block.id, "spans": candidate_choice["spans"]})
        styled_span_layout.append({"id": block.id, "layout": serialize_styled_layout(translated_layout)})
        span_render_boxes.append({"id": block.id, "sourceBoxes": source_boxes, "translatedBoxes": translated_boxes})
        selected_font_scale.append({"id": block.id, "scaleFactor": translated_layout.get("scaleFactor", 1.0)})
        if warnings:
            overflow_warnings.append({"id": block.id, "warnings": warnings})

        block_plans.append(
            {
                "id": block.id,
                "sourceText": block.text,
                "translatedText": candidate_choice["text"],
                "sourceStyleSpans": source_spans,
                "translatedStyleSpans": candidate_choice["spans"],
                "styledSpanLayout": serialize_styled_layout(translated_layout),
                "spanLines": [[token.get("text", "") for token in line.get("tokens", [])] for line in translated_layout.get("lines", [])],
                "spanRenderBoxes": translated_boxes,
                "selectedFontScale": translated_layout.get("scaleFactor", 1.0),
                "lineBreakStrategy": "semantic_span_layout",
                "lineCountPreservation": {
                    "preferred": preferred_line_count,
                    "actual": len(translated_layout.get("lines", [])),
                    "delta": translated_layout.get("lineCountDelta", abs(len(translated_layout.get("lines", [])) - preferred_line_count)),
                },
                "overflowWarnings": warnings,
                "selectionReason": candidate_choice.get("reason", ""),
                "selectedTranslationCandidate": {
                    "label": candidate_choice.get("label", "faithful"),
                    "text": candidate_choice.get("text", block.translated_text or block.text),
                    "qualityScore": candidate_choice.get("qualityScore", 0.0),
                },
                "translationCandidates": candidate_choice.get("candidates", []),
                "actualRenderedSpans": rendered_spans,
                "renderQualityScore": round(quality_score, 3),
            }
        )

    source_typography_extracted = {
        "fontSize": True,
        "fontWeight": True,
        "color": True,
        "casing": True,
        "alignment": True,
        "lineHeight": True,
        "textBoxPosition": True,
        "relativeVisualHierarchy": True,
        "fontFamily": "google_fonts_assisted_match" if google_fonts_enabled() else "approximate_system_match",
        "letterSpacing": "approximate_uniform",
        "multiStyleSpans": any(len(item.get("spans", [])) > 1 for item in source_style_spans_summary),
    }
    semantic_style_mapping = {
        "enabled": True,
        "mode": "semantic_phrase_word_mapping",
        "colorAware": True,
        "fontFamilyAware": "google_fonts_assisted_match" if google_fonts_enabled() else "approximate_system_match",
        "letterSpacingAware": "approximate_uniform",
        "notes": "Styles follow semantic phrase/word meaning rather than source position; line-count preservation is scored against the source visual rhythm.",
    }
    severe_render_warnings = {
        "overflow",
        "font scaled down heavily",
        "semantic coverage reduced",
    }
    if not block_plans:
        render_quality_status = "failed"
    elif any(plan["renderQualityScore"] < 55 for plan in block_plans):
        render_quality_status = "failed"
    elif (
        semantic_style_mapping["enabled"]
        and min(plan["renderQualityScore"] for plan in block_plans) >= 85
        and not any(
            warning in severe_render_warnings
            for item in overflow_warnings
            for warning in item.get("warnings", [])
        )
    ):
        render_quality_status = "passed"
    else:
        render_quality_status = "partial"

    return {
        "translatedPreviewImage": translated_canvas,
        "sourceReferenceImage": source_canvas,
        "renderPlan": block_plans,
        "styleAwareRenderingEnabled": bool(block_plans),
        "sourceTypographyExtracted": source_typography_extracted,
        "sourceStyleSpans": source_style_spans_summary,
        "translatedStyleSpans": translated_style_spans_summary,
        "semanticStyleMapping": semantic_style_mapping,
        "styledSpanLayout": styled_span_layout,
        "spanRenderBoxes": span_render_boxes,
        "overflowWarnings": overflow_warnings,
        "selectedFontScale": selected_font_scale,
        "lineBreakStrategy": "semantic_span_layout",
        "renderQualityStatus": render_quality_status,
    }


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: tuple[int, int, int, int],
    fit_bounds: bool,
    *,
    preferred_lines: int | None = None,
    preferred_font_size: int | None = None,
    line_height_ratio: float = 1.08,
) -> tuple[list[list[tuple[str, bool]]], int, int]:
    left, top, right, bottom = bbox
    max_width = max(24, right - left)
    max_height = max(16, bottom - top)
    plain_text = text.replace("[BOLD]", "").replace("[/BOLD]", "")
    longest_token = max((len(token) for token in plain_text.replace("\n", " ").split() if token), default=8)
    estimated_lines = max(1, plain_text.count("\n") + 1)
    start = preferred_font_size or max(
        18,
        min(
            int(max_height * 0.74),
            int(max_width / max(3.8, longest_token * 0.52)),
            int((max_height / estimated_lines) * 0.92),
            84,
        ),
    )
    minimum = max(10, int(start * 0.48)) if fit_bounds else max(10, start - 2)
    line_candidates = [
        preferred_lines,
        preferred_lines + 1 if preferred_lines else None,
        preferred_lines + 2 if preferred_lines else None,
        None,
    ]
    seen_candidates: list[int | None] = []
    for candidate in line_candidates:
        if candidate not in seen_candidates:
            seen_candidates.append(candidate)

    best: tuple[float, list[list[tuple[str, bool]]], int, int] | None = None
    for size in range(start, minimum - 1, -1):
        for max_lines in seen_candidates:
            lines, line_height = build_wrapped_lines(
                draw,
                text,
                size,
                max_width,
                line_height_ratio=line_height_ratio,
                max_lines=max_lines,
            )
            used_height = len(lines) * line_height
            widest = max(
                (
                    sum(
                        text_width(draw, segment if index == 0 else f" {segment}", get_font(size, bold=bold))
                        for index, (segment, bold) in enumerate(line)
                    )
                    for line in lines
                ),
                default=0,
            )
            overflow = max(0, used_height - max_height) + max(0, widest - max_width)
            line_penalty = abs(len(lines) - (preferred_lines or len(lines)))
            if overflow == 0:
                score = float(size * 100 - line_penalty * 8 - max(0, max_height - used_height) * 0.08)
                if best is None or score > best[0]:
                    best = (score, lines, size, line_height)
            elif best is None:
                score = float(-overflow - line_penalty * 12 + size)
                best = (score, lines, size, line_height)
        if best is not None and best[2] == size and best[0] > 0:
            return best[1], best[2], best[3]
    if best is not None:
        return best[1], best[2], best[3]
    lines, line_height = build_wrapped_lines(draw, text, minimum, max_width, line_height_ratio=line_height_ratio)
    return lines, minimum, line_height


def render_bbox_for_block(block: TextBlock, image_size: tuple[int, int], x_offset: int, y_offset: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = block.bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    if block.align == "center":
        pad_x = int(width * 0.28)
        pad_y = int(height * 0.18)
        if width > image_size[0] * 0.55 and height < image_size[1] * 0.22:
            pad_x = int(width * 0.04)
            pad_y = int(height * 0.08)
    else:
        pad_x = int(width * 0.08)
        pad_y = int(height * 0.10)
        if left > image_size[0] * 0.55:
            pad_left = 0
            pad_right = int(width * 0.04)
            return (
                max(0, left + x_offset),
                max(0, top - pad_y + y_offset),
                min(image_size[0], right + pad_right + x_offset),
                min(image_size[1], bottom + pad_y + y_offset),
            )
    return (
        max(0, left - pad_x + x_offset),
        max(0, top - pad_y + y_offset),
        min(image_size[0], right + pad_x + x_offset),
        min(image_size[1], bottom + pad_y + y_offset),
    )


def average_cleanup_score(block_id: str | None, cleanup_debug: dict[str, Any]) -> tuple[float, float, float]:
    scores = [float(item["score"]) for item in cleanup_debug.get("lineCleanupQualityScores", []) if item.get("id") == block_id]
    overlaps = [float(item["score"]) for item in cleanup_debug.get("foregroundOverlapScores", []) if item.get("id") == block_id]
    warnings = [item for item in cleanup_debug.get("cleanupWarnings", []) if item.get("id") == block_id]
    mean_score = sum(scores) / len(scores) if scores else 1.0
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    warning_penalty = min(0.3, 0.08 * len(warnings))
    return mean_score, mean_overlap, warning_penalty


def estimate_block_background_complexity(image: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    import cv2

    crop = np.array(image.crop(bbox).convert("RGB"))
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def sample_panel_color(image: Image.Image, bbox: tuple[int, int, int, int]) -> str:
    left, top, right, bottom = bbox
    pad = max(10, int(min(right - left, bottom - top) * 0.08))
    ring = image.crop(
        (
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        )
    ).convert("RGB")
    ring_np = np.array(ring)
    if ring_np.size == 0:
        return "#f4f1ea"
    color = np.median(ring_np.reshape(-1, 3), axis=0)
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in color))


def relative_luminance(color: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        srgb = value / 255.0
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(value) for value in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color_a: tuple[int, int, int], color_b: tuple[int, int, int]) -> float:
    lum_a = relative_luminance(color_a)
    lum_b = relative_luminance(color_b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def choose_text_color(background_hex: str, preferred_hex: str | None = None) -> tuple[str, float]:
    background = hex_to_rgb(background_hex)
    candidates = [("#111111", contrast_ratio(background, hex_to_rgb("#111111"))), ("#ffffff", contrast_ratio(background, hex_to_rgb("#ffffff")))]
    if preferred_hex and preferred_hex.startswith("#"):
        preferred_ratio = contrast_ratio(background, hex_to_rgb(preferred_hex))
        candidates.append((preferred_hex, preferred_ratio))
    best = max(candidates, key=lambda item: item[1])
    return best


def build_overlay_style_config(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    strategy: str,
    block: TextBlock,
    dominant_color: str,
) -> dict[str, Any]:
    complexity = estimate_block_background_complexity(image, bbox)
    base_rgb = hex_to_rgb(dominant_color)
    bg_luma = relative_luminance(base_rgb)
    is_headline = block.role in {"headline", "cta"}
    if bg_luma > 0.62:
        wash_color = "#1c2430"
    elif bg_luma < 0.34:
        wash_color = "#f6f8fb"
    else:
        wash_color = dominant_color
    opacity = 0.28
    blur_radius = 10
    feather_radius = 12
    padding = max(18, int(min(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.08))
    shadow = {"enabled": False, "offset": 0, "blur": 0, "color": "#000000", "opacity": 0.0}
    if strategy == "gradient_overlay":
        opacity = 0.34 if is_headline else 0.28
        blur_radius = 8 if complexity < 70 else 12
        feather_radius = 18
    elif strategy == "overlay_panel":
        opacity = 0.42 if is_headline else 0.34
        blur_radius = 10
        feather_radius = 16
    elif strategy == "soft_blur_panel":
        opacity = 0.26 if is_headline else 0.22
        blur_radius = 16 if complexity > 45 else 12
        feather_radius = 14
    elif strategy == "reposition_to_safe_area":
        opacity = 0.24 if is_headline else 0.2
        blur_radius = 14
        feather_radius = 14
    text_color, contrast = choose_text_color(wash_color, block.color if block.color.startswith("#") else None)
    if contrast < 4.6:
        text_color = "#111111" if text_color.lower() != "#111111" else "#ffffff"
        contrast = contrast_ratio(hex_to_rgb(wash_color), hex_to_rgb(text_color))
    if contrast < 6.0 and is_headline:
        shadow = {"enabled": True, "offset": 2, "blur": 0, "color": "#000000" if text_color.lower() == "#ffffff" else "#ffffff", "opacity": 0.18}
    return {
        "type": strategy,
        "opacity": round(opacity, 3),
        "blurRadius": blur_radius,
        "featherRadius": feather_radius,
        "dominantColor": wash_color,
        "textColor": text_color,
        "shadow": shadow,
        "padding": padding,
        "contrastScore": round(contrast, 3),
        "direction": "vertical" if (bbox[3] - bbox[1]) >= (bbox[2] - bbox[0]) else "horizontal",
    }


def build_overlay_backdrop(
    base: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    style: dict[str, Any],
) -> Image.Image:
    overlay = base.copy().convert("RGBA")
    left, top, right, bottom = bbox
    panel_w = max(1, right - left)
    panel_h = max(1, bottom - top)
    radius = max(8, int(min(panel_w, panel_h) * 0.08))
    feather_radius = int(style.get("featherRadius", 12))
    panel_mask = Image.new("L", (panel_w, panel_h), 0)
    panel_draw = ImageDraw.Draw(panel_mask)
    panel_draw.rounded_rectangle((feather_radius // 2, feather_radius // 2, max(feather_radius // 2 + 1, panel_w - feather_radius // 2), max(feather_radius // 2 + 1, panel_h - feather_radius // 2)), radius=radius, fill=235)
    panel_mask = panel_mask.filter(ImageFilter.GaussianBlur(radius=max(2, feather_radius / 2)))
    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    panel_draw_rgba = ImageDraw.Draw(panel)
    rgb = hex_to_rgb(str(style.get("dominantColor", "#d9e0ea")))
    opacity = float(style.get("opacity", 0.3))
    blur_radius = int(style.get("blurRadius", 10))
    kind = str(style.get("type", "overlay_panel"))
    direction = str(style.get("direction", "vertical"))
    if kind == "gradient_overlay":
        if direction == "horizontal":
            for col in range(panel_w):
                distance = abs((col / max(1, panel_w - 1)) - 0.5) * 2
                alpha = int((opacity * 255) * (1.0 - 0.45 * distance))
                panel_draw_rgba.line((col, 0, col, panel_h), fill=(rgb[0], rgb[1], rgb[2], alpha))
        else:
            for row in range(panel_h):
                distance = abs((row / max(1, panel_h - 1)) - 0.5) * 2
                alpha = int((opacity * 255) * (1.0 - 0.45 * distance))
                panel_draw_rgba.line((0, row, panel_w, row), fill=(rgb[0], rgb[1], rgb[2], alpha))
    elif kind == "soft_blur_panel":
        blurred = base.crop(bbox).filter(ImageFilter.GaussianBlur(radius=blur_radius)).convert("RGBA")
        panel.alpha_composite(blurred)
        veil = Image.new("RGBA", (panel_w, panel_h), (rgb[0], rgb[1], rgb[2], int(opacity * 255)))
        panel.alpha_composite(veil)
    else:
        blurred = base.crop(bbox).filter(ImageFilter.GaussianBlur(radius=max(6, blur_radius - 2))).convert("RGBA")
        panel.alpha_composite(blurred)
        panel_draw_rgba.rounded_rectangle((0, 0, panel_w, panel_h), radius=radius, fill=(rgb[0], rgb[1], rgb[2], int(opacity * 255)))
    overlay.alpha_composite(Image.composite(panel, overlay.crop(bbox), panel_mask), dest=(left, top))
    return overlay.convert("RGB")


def find_safe_area_candidates(
    image: Image.Image,
    block: TextBlock,
    foreground_bbox: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    box = block.bbox
    width = max(180, box[2] - box[0])
    height = max(120, box[3] - box[1])
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []
    candidates.append(("top-right", (max(0, image.width - width - 48), 48, min(image.width, image.width - 48), min(image.height, 48 + height))))
    candidates.append(("bottom-left", (48, max(0, image.height - height - 48), min(image.width, 48 + width), min(image.height, image.height - 48))))
    candidates.append(("bottom-right", (max(0, image.width - width - 48), max(0, image.height - height - 48), min(image.width, image.width - 48), min(image.height, image.height - 48))))
    candidates.append(("top-left", (48, 48, min(image.width, 48 + width), min(image.height, 48 + height))))
    scored: list[dict[str, Any]] = []
    for label, candidate in candidates:
        overlap = overlap_fraction(candidate, foreground_bbox) if foreground_bbox else 0.0
        complexity = estimate_block_background_complexity(image, candidate)
        score = (1.0 - min(1.0, overlap * 2.5)) + max(0.0, 1.0 - min(1.0, complexity / 180.0))
        scored.append({"label": label, "bbox": list(candidate), "overlap": overlap, "complexity": complexity, "score": score})
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def decide_block_render_strategy(
    image: Image.Image,
    block: TextBlock,
    cleanup_debug: dict[str, Any],
    foreground_bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    mean_score, mean_overlap, warning_penalty = average_cleanup_score(block.id, cleanup_debug)
    complexity = estimate_block_background_complexity(image, block.bbox)
    area_ratio = ((block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])) / max(1, image.width * image.height)
    residual_risk = max(0.0, 1.0 - mean_score)
    importance = 0.14 if block.role in {"headline", "cta"} else 0.05
    cleanup_confidence = max(0.0, min(1.0, mean_score - warning_penalty - min(0.35, mean_overlap * 0.8) - min(0.22, complexity / 420.0) - min(0.18, area_ratio * 1.8) + importance))
    panel_color = sample_panel_color(image, block.bbox)
    safe_areas = find_safe_area_candidates(image, block, foreground_bbox)

    # Always use clean_replace â€” overlay/panel fallbacks produce visible boxes
    # that look wrong to the user. Render the translated text directly at the
    # original position regardless of cleanup confidence.
    strategy = "clean_replace"
    overlay_style: dict[str, Any] | str = "none"
    final_text_box = list(block.bbox)
    reason = "clean_replace (overlay fallbacks disabled)"

    return {
        "id": block.id,
        "sourceText": block.text,
        "translatedText": block.translated_text or block.text,
        "boundingBox": list(block.bbox),
        "cleanBox": list(block.clean_box) if block.clean_box else None,
        "cleanupConfidence": round(cleanup_confidence, 4),
        "cleanupStrategy": "line_level_mask_guided_cleanup",
        "renderStrategy": strategy,
        "overlayStyle": overlay_style if isinstance(overlay_style, dict) else {"type": "none"},
        "finalTextBox": final_text_box,
        "reason": reason,
        "safeAreaCandidates": safe_areas,
        "panelColor": panel_color,
        "contrastScore": overlay_style.get("contrastScore", 0.0) if isinstance(overlay_style, dict) else 0.0,
        "sourceStyleSpans": block.source_style_spans,
        "translatedStyleSpans": block.translated_style_spans,
        "typography": default_typography_style(block),
        "styleMappingReason": "semantic role mapping used for translated style spans",
        "fallbackReason": "" if strategy == "clean_replace" else reason,
    }


def render_text_into_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: tuple[int, int, int, int],
    *,
    align: str,
    color: str,
    image_size: tuple[int, int],
    preserve_bold: bool,
    fit_bounds: bool,
    block: TextBlock | None = None,
    style: dict[str, Any] | None = None,
) -> None:
    render_text = text.strip()
    if not render_text:
        return
    if not preserve_bold:
        render_text = render_text.replace("[BOLD]", "").replace("[/BOLD]", "")
    if style:
        render_bbox = bbox
        inset = int(style.get("padding", 0))
        render_bbox = (
            min(render_bbox[2] - 1, render_bbox[0] + inset),
            min(render_bbox[3] - 1, render_bbox[1] + inset),
            max(render_bbox[0] + 1, render_bbox[2] - inset),
            max(render_bbox[1] + 1, render_bbox[3] - inset),
        )
    else:
        temp_block = TextBlock(text=render_text, role="headline", translate=True, bbox=bbox, color=color, align=align)
        render_bbox = render_bbox_for_block(temp_block, image_size, 0, 0)
    preferred_lines = len(block.line_texts) if block and block.line_texts else max(1, render_text.count("\n") + 1)
    preferred_font_size = block.font_size_estimate if block else None
    line_height_ratio = (
        max(1.0, min(1.42, block.line_height_estimate / max(1, block.font_size_estimate)))
        if block and block.font_size_estimate > 0
        else 1.08
    )
    lines, font_size, line_height = fit_text(
        draw,
        render_text,
        render_bbox,
        fit_bounds,
        preferred_lines=preferred_lines,
        preferred_font_size=preferred_font_size,
        line_height_ratio=line_height_ratio,
    )
    total_height = len(lines) * line_height
    y = render_bbox[1] + max(0, (render_bbox[3] - render_bbox[1] - total_height) // 2)
    fill = style.get("textColor", color) if style else (color if color.startswith("#") else "#111111")
    stroke_fill = None
    stroke_width = 0
    shadow = style.get("shadow") if style else None
    for line in lines:
        line_width = sum(
            text_width(draw, segment if index == 0 else f" {segment}", get_font(font_size, bold=bold))
            for index, (segment, bold) in enumerate(line)
        )
        x = render_bbox[0] if align == "left" else render_bbox[0] + max(0, (render_bbox[2] - render_bbox[0] - int(line_width)) // 2)
        if shadow and shadow.get("enabled"):
            shadow_fill = shadow.get("color", "#000000")
            shadow_offset = int(shadow.get("offset", 2))
            draw_rich_line(draw, (x + shadow_offset, y + shadow_offset), line, font_size, shadow_fill, None, 0)
        draw_rich_line(draw, (x, y), line, font_size, fill, stroke_fill, stroke_width)
        y += line_height


def should_suppress_render_block(candidate: TextBlock, container: TextBlock) -> bool:
    if candidate.id == container.id:
        return False
    if not candidate.translate or not container.translate:
        return False
    if candidate.surface != "overlay" or container.surface != "overlay":
        return False
    if candidate.role != container.role:
        return False

    candidate_w = candidate.bbox[2] - candidate.bbox[0]
    candidate_h = candidate.bbox[3] - candidate.bbox[1]
    container_w = container.bbox[2] - container.bbox[0]
    container_h = container.bbox[3] - container.bbox[1]
    if container_w < candidate_w or container_h < candidate_h:
        return False

    coverage = overlap_fraction(candidate.bbox, container.bbox)
    if coverage < 0.72:
        return False

    candidate_text = normalize_ocr_text(candidate.text)
    container_text = normalize_ocr_text(container.text)
    if not candidate_text or not container_text:
        return False
    shared_tokens = text_tokens(candidate.text) & text_tokens(container.text)
    if candidate_text in container_text and shared_tokens:
        return True
    if text_similarity(candidate_text, container_text) >= 0.6 and len(shared_tokens) >= 1:
        return True
    return False


def filter_render_blocks(blocks: list[TextBlock], render_plan: dict[str, dict[str, Any]] | None = None) -> list[TextBlock]:
    survivors: list[TextBlock] = []
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            -((block.bbox[2] - block.bbox[0]) * (block.bbox[3] - block.bbox[1])),
            block.bbox[1],
            block.bbox[0],
        ),
    )
    for block in sorted_blocks:
        suppressed_by: TextBlock | None = None
        for existing in survivors:
            if should_suppress_render_block(block, existing):
                suppressed_by = existing
                break
        if suppressed_by is not None:
            if render_plan and block.id and block.id in render_plan:
                render_plan[block.id]["renderSuppressed"] = True
                render_plan[block.id]["renderSuppressedReason"] = f"suppressed as nested duplicate of {suppressed_by.id}"
            continue
        survivors.append(block)
        if render_plan and block.id and block.id in render_plan:
            render_plan[block.id]["renderSuppressed"] = False
            render_plan[block.id]["renderSuppressedReason"] = ""
    return sorted(survivors, key=lambda block: (block.bbox[1], block.bbox[0]))

def render_translated_text(
    base: Image.Image,
    blocks: list[TextBlock],
    x_offset: int = 0,
    y_offset: int = 0,
    preserve_bold: bool = True,
    fit_bounds: bool = True,
    render_plan: dict[str, dict[str, Any]] | None = None,
) -> Image.Image:
    output = base.copy()
    draw = ImageDraw.Draw(output)
    renderable_blocks = filter_render_blocks(blocks, render_plan)
    for block in renderable_blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue
        text = (block.translated_text or block.text).strip()
        fill = block.color if block.color.startswith("#") else "#111111"
        plan = (render_plan or {}).get(block.id or "")
        bbox = tuple(plan["finalTextBox"]) if plan and plan.get("finalTextBox") else render_bbox_for_block(block, output.size, x_offset, y_offset)
        strategy = plan.get("renderStrategy") if plan else "clean_replace"
        overlay_style = plan.get("overlayStyle") if plan else {"type": "none"}
        if strategy in {"overlay_panel", "gradient_overlay", "soft_blur_panel", "reposition_to_safe_area"}:
            overlay_bbox = bbox
            output = build_overlay_backdrop(output, overlay_bbox, style=overlay_style or {"type": "overlay_panel"})
            draw = ImageDraw.Draw(output)
        # When background cleanup was incomplete (confidence below threshold),
        # paint a local background-matched patch before rendering text so the
        # translated text is not drawn on top of residual original text.
        if block.cleanup_confidence < 0.70:
            try:
                bx0, by0, bx1, by1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                img_w, img_h = output.size
                # Expand patch slightly to cover nearby design elements (brackets etc.)
                pad = max(12, int(min(bx1 - bx0, by1 - by0) * 0.18))
                px0 = max(0, bx0 - pad)
                py0 = max(0, by0 - pad)
                px1 = min(img_w, bx1 + pad)
                py1 = min(img_h, by1 + pad)
                # Sample a border ring around the patch for median background colour
                arr = np.array(output.convert("RGB"), dtype=np.float32)
                ring_w = max(4, pad)
                ring_mask = np.zeros((img_h, img_w), dtype=bool)
                # top/bottom/left/right strips outside the patch
                ring_mask[max(0, py0 - ring_w):py0, px0:px1] = True
                ring_mask[py1:min(img_h, py1 + ring_w), px0:px1] = True
                ring_mask[py0:py1, max(0, px0 - ring_w):px0] = True
                ring_mask[py0:py1, px1:min(img_w, px1 + ring_w)] = True
                if np.any(ring_mask):
                    bg_color = tuple(int(v) for v in np.median(arr[ring_mask], axis=0))
                else:
                    bg_color = (255, 255, 255)
                draw.rectangle([px0, py0, px1, py1], fill=bg_color)
            except Exception:
                pass
        candidate_choice = select_translation_candidate_for_layout(draw, block, bbox)
        selected_text = candidate_choice["text"]
        selected_spans = candidate_choice["spans"]
        styled_layout = candidate_choice["layout"]
        block.overflow_warning = bool(styled_layout.get("overflow", 0) > 0)
        if selected_spans:
            if plan is not None:
                plan["translationCandidates"] = candidate_choice.get("candidates", [])
                plan["selectedTranslationCandidate"] = {
                    "label": candidate_choice["label"],
                    "text": selected_text,
                    "qualityScore": candidate_choice["qualityScore"],
                }
                plan["selectionReason"] = candidate_choice["reason"]
            block.translated_text = selected_text
            block.translated_style_spans = selected_spans
            block.overflow_warning = bool(styled_layout.get("overflow", 0) > 0)
            rendered_spans, span_render_boxes = render_styled_spans(
                draw,
                styled_layout,
                bbox,
                alignment=block.align,
                style=overlay_style if isinstance(overlay_style, dict) else None,
            )
            if plan is not None:
                plan["styledSpanLayout"] = serialize_styled_layout(styled_layout)
                plan["spanLines"] = [
                    [token["text"] for token in line.get("tokens", [])]
                    for line in styled_layout.get("lines", [])
                ]
                plan["spanRenderBoxes"] = span_render_boxes
                plan["spanScaleFactor"] = styled_layout.get("scaleFactor", 1.0)
                plan["spanOverflowWarnings"] = candidate_choice.get("warnings", []) + (["Styled span overflow"] if block.overflow_warning else [])
                plan["actualRenderedSpans"] = rendered_spans
        else:
            plain_text = selected_text.replace("[BOLD]", "").replace("[/BOLD]", "")
            preview_lines, _, preview_line_height = fit_text(
                draw,
                plain_text,
                bbox,
                fit_bounds,
                preferred_lines=len(block.line_texts) if block.line_texts else None,
                preferred_font_size=block.font_size_estimate,
                line_height_ratio=max(1.0, min(1.42, block.line_height_estimate / max(1, block.font_size_estimate))),
            )
            block.overflow_warning = len(preview_lines) * preview_line_height > max(1, bbox[3] - bbox[1])
            render_text_into_bbox(
                draw,
                selected_text,
                bbox,
                align=block.align,
                color=fill,
                image_size=output.size,
                preserve_bold=preserve_bold,
                fit_bounds=fit_bounds,
                block=block,
                style=overlay_style if isinstance(overlay_style, dict) else None,
            )
            if plan is not None:
                plan["translationCandidates"] = candidate_choice.get("candidates", [])
                plan["selectedTranslationCandidate"] = {
                    "label": candidate_choice["label"],
                    "text": selected_text,
                    "qualityScore": candidate_choice["qualityScore"],
                }
                plan["selectionReason"] = candidate_choice["reason"]
                plan["styledSpanLayout"] = None
                plan["spanLines"] = [[plain_text]]
                plan["spanRenderBoxes"] = []
                plan["spanScaleFactor"] = 1.0
                plan["spanOverflowWarnings"] = candidate_choice.get("warnings", []) + (["Plain text overflow"] if block.overflow_warning else [])
                plan["actualRenderedSpans"] = []
    return output


def build_debug_payload(
    raw_ocr_blocks: list[TextBlock],
    semantic_blocks: list[TextBlock],
    *,
    token_masks: list[dict[str, Any]] | None = None,
    cleaned_background_preview: str | None = None,
    final_render_plan: list[dict[str, Any]] | None = None,
    block_line_groups: list[dict[str, Any]] | None = None,
    line_cleanup_regions: list[dict[str, Any]] | None = None,
    line_masks: list[dict[str, Any]] | None = None,
    line_cleanup_strategies: list[dict[str, Any]] | None = None,
    line_cleanup_quality_scores: list[dict[str, Any]] | None = None,
    foreground_overlap_scores: list[dict[str, Any]] | None = None,
    cleanup_warnings: list[dict[str, Any]] | None = None,
    safe_area_candidates: list[dict[str, Any]] | None = None,
    raw_foreground_boxes: list[list[int]] | None = None,
    protected_region_mask_preview: str | None = None,
    refined_protected_region_mask: str | None = None,
    protected_mask_refinement_method: str | None = None,
    generative_attempt_previews: list[dict[str, Any]] | None = None,
    cleanup_status: str | None = None,
    block_cleanup_status: list[dict[str, Any]] | None = None,
    residual_source_ocr: list[dict[str, Any]] | None = None,
    failed_source_words: list[dict[str, Any]] | None = None,
    block_level_fallback_attempted: bool = False,
    block_level_fallback_selected: bool = False,
    final_render_skipped_reason: str | None = None,
    requested_cleanup_provider: str | None = None,
    actual_cleanup_provider: str | None = None,
    provider_available: bool | None = None,
    provider_failure_reason: str | None = None,
    mask_dilation_px: int | None = None,
    mask_feather_px: int | None = None,
    inpainting_input_mode: str | None = None,
    mask_polarity: str | None = None,
    text_coverage_estimate: float | None = None,
    background_leakage_estimate: float | None = None,
    mask_quality_status: str | None = None,
    mask_failure_reason: str | None = None,
    mask_white_pixel_ratio: float | None = None,
    text_stroke_mask_paths: dict[str, str] | None = None,
    anti_alias_coverage_estimate: float | None = None,
    protected_object_overlap_estimate: float | None = None,
    dilation_attempts: list[dict[str, Any]] | None = None,
    selected_dilation_px: int | None = None,
    best_cleanup_attempt: dict[str, Any] | None = None,
    provider_quality_status: str | None = None,
    residual_text_after_each_attempt: list[dict[str, Any]] | None = None,
    ghosting_score_after_each_attempt: list[dict[str, Any]] | None = None,
    cleanup_selected_reason: str | None = None,
    cleanup_cascade_enabled: bool | None = None,
    first_pass_provider: str | None = None,
    first_pass_status: str | None = None,
    residual_text_detected_after_first_pass: bool | None = None,
    residual_words_after_first_pass: list[str] | None = None,
    residual_mask_generated: bool | None = None,
    second_pass_provider: str | None = None,
    second_pass_status: str | None = None,
    residual_text_detected_after_second_pass: bool | None = None,
    residual_words_after_second_pass: list[str] | None = None,
    sdxl_fallback_attempted: bool | None = None,
    sdxl_fallback_status: str | None = None,
    final_cleanup_quality_status: str | None = None,
    mask_type_used_for_lama: str | None = None,
    composite_mask_available: bool | None = None,
    adaptive_text_pixel_detection: bool | None = None,
    text_pixel_detection_methods_used: list[str] | None = None,
    first_pass_lama_output: str | None = None,
    residual_text_mask_preview: str | None = None,
    second_pass_lama_output: str | None = None,
    final_cleaned_candidate: str | None = None,
    residual_ocr_after_first_pass: list[str] | None = None,
    residual_ocr_after_second_pass: list[str] | None = None,
    cleanup_cascade_log: list[dict[str, Any]] | None = None,
    cleanup_passes_run: int | None = None,
    residual_words_by_pass: list[dict[str, Any]] | None = None,
    residual_mask_area_by_pass: list[dict[str, Any]] | None = None,
    cleanup_improvement_by_pass: list[dict[str, Any]] | None = None,
    stopped_reason: str | None = None,
    residual_word_boxes_after_first_pass: list[dict[str, Any]] | None = None,
    residual_glyph_mask_after_first_pass: str | None = None,
    residual_artifact_mask_expanded: str | None = None,
    residual_mask_coverage_by_word: list[dict[str, Any]] | None = None,
    residual_mask_false_positive_estimate: float | None = None,
    sdxl_fallback_configured: bool | None = None,
    sdxl_control_type: str | None = None,
    sdxl_model_id: str | None = None,
    control_net_model_id: str | None = None,
    sdxl_failure_reason: str | None = None,
    style_aware_rendering_enabled: bool | None = None,
    source_typography_extracted: dict[str, Any] | None = None,
    source_style_spans_summary: list[dict[str, Any]] | None = None,
    translated_style_spans_summary: list[dict[str, Any]] | None = None,
    semantic_style_mapping: dict[str, Any] | None = None,
    styled_span_layout_summary: list[dict[str, Any]] | None = None,
    span_render_boxes_summary: list[dict[str, Any]] | None = None,
    overflow_warnings_summary: list[dict[str, Any]] | None = None,
    selected_font_scale: list[dict[str, Any]] | None = None,
    line_break_strategy: str | None = None,
    render_quality_status: str | None = None,
    translated_text_render_preview: str | None = None,
    source_text_style_reference: str | None = None,
    render_plan_preview: str | None = None,
    semantic_block_grouping_status: str | None = None,
    headline_merged_into_single_block: bool | None = None,
    grouped_line_ids: list[str] | None = None,
    grouping_reason: str | None = None,
    grouping_confidence: float | None = None,
    rejected_merge_reasons: list[str] | None = None,
    block_type: str | None = None,
) -> dict[str, Any]:
    return {
        "ocrTokens": [block.model_dump() for block in raw_ocr_blocks],
        "semanticBlocks": [block.model_dump() for block in semantic_blocks],
        "blockLineGroups": block_line_groups or [],
        "lineCleanupRegions": line_cleanup_regions or [],
        "lineMasks": line_masks or [],
        "lineCleanupStrategies": line_cleanup_strategies or [],
        "lineCleanupQualityScores": line_cleanup_quality_scores or [],
        "foregroundOverlapScores": foreground_overlap_scores or [],
        "tokenMasks": token_masks or [],
        "expandedCleanBoxes": [
            {"id": block.id, "clean_box": block.clean_box, "bbox": block.bbox}
            for block in semantic_blocks
            if block.clean_box
        ],
        "cleanedBackgroundPreview": cleaned_background_preview,
        "finalLayoutBoxes": [
            {
                "id": block.id,
                "bbox": block.bbox,
                "line_boxes": block.line_boxes,
                "alignment": block.align,
                "font_size_estimate": block.font_size_estimate,
                "line_height_estimate": block.line_height_estimate,
            }
            for block in semantic_blocks
        ],
        "textOverflowWarnings": [
            {"id": block.id, "text": block.translated_text or block.text}
            for block in semantic_blocks
            if block.overflow_warning
        ],
        "cleanupWarnings": cleanup_warnings or [],
        "safeAreaCandidates": safe_area_candidates or [],
        "rawForegroundBoxes": raw_foreground_boxes or [],
        "protectedRegionMaskPreview": protected_region_mask_preview,
        "refinedProtectedRegionMask": refined_protected_region_mask,
        "protectedMaskRefinementMethod": protected_mask_refinement_method,
        "textMaskProtectedOverlap": foreground_overlap_scores or [],
        "generativeAttemptPreviews": generative_attempt_previews or [],
        "cleanupStatus": cleanup_status,
        "blockCleanupStatus": block_cleanup_status or [],
        "residualSourceOCR": residual_source_ocr or [],
        "failedSourceWords": failed_source_words or [],
        "blockLevelFallbackAttempted": block_level_fallback_attempted,
        "blockLevelFallbackSelected": block_level_fallback_selected,
        "finalRenderSkippedReason": final_render_skipped_reason,
        "requestedCleanupProvider": requested_cleanup_provider,
        "actualCleanupProvider": actual_cleanup_provider,
        "providerAvailable": provider_available,
        "providerFailureReason": provider_failure_reason,
        "maskDilationPx": mask_dilation_px,
        "maskFeatherPx": mask_feather_px,
        "maskPolarity": mask_polarity,
        "textCoverageEstimate": text_coverage_estimate,
        "backgroundLeakageEstimate": background_leakage_estimate,
        "maskQualityStatus": mask_quality_status,
        "maskFailureReason": mask_failure_reason,
        "maskWhitePixelRatio": mask_white_pixel_ratio,
        "textStrokeMaskPaths": text_stroke_mask_paths or {},
        "antiAliasCoverageEstimate": anti_alias_coverage_estimate,
        "protectedObjectOverlapEstimate": protected_object_overlap_estimate,
        "dilationAttempts": dilation_attempts or [],
        "selectedDilationPx": selected_dilation_px,
        "bestCleanupAttempt": best_cleanup_attempt,
        "providerQualityStatus": provider_quality_status,
        "residualTextAfterEachAttempt": residual_text_after_each_attempt or [],
        "ghostingScoreAfterEachAttempt": ghosting_score_after_each_attempt or [],
        "cleanupSelectedReason": cleanup_selected_reason,
        "cleanupCascadeEnabled": cleanup_cascade_enabled,
        "firstPassProvider": first_pass_provider,
        "firstPassStatus": first_pass_status,
        "residualTextDetectedAfterFirstPass": residual_text_detected_after_first_pass,
        "residualWordsAfterFirstPass": residual_words_after_first_pass or [],
        "residualMaskGenerated": residual_mask_generated,
        "secondPassProvider": second_pass_provider,
        "secondPassStatus": second_pass_status,
        "residualTextDetectedAfterSecondPass": residual_text_detected_after_second_pass,
        "residualWordsAfterSecondPass": residual_words_after_second_pass or [],
        "sdxlFallbackAttempted": sdxl_fallback_attempted,
        "sdxlFallbackStatus": sdxl_fallback_status,
        "finalCleanupQualityStatus": final_cleanup_quality_status,
        "maskTypeUsedForLaMa": mask_type_used_for_lama,
        "compositeMaskAvailable": composite_mask_available,
        "adaptiveTextPixelDetection": adaptive_text_pixel_detection,
        "textPixelDetectionMethodsUsed": text_pixel_detection_methods_used or [],
        "firstPassLamaOutput": first_pass_lama_output,
        "residualTextMask": residual_text_mask_preview,
        "secondPassLamaOutput": second_pass_lama_output,
        "finalCleanedCandidate": final_cleaned_candidate,
        "residualOCRAfterFirstPass": residual_ocr_after_first_pass or [],
        "residualOCRAfterSecondPass": residual_ocr_after_second_pass or [],
        "cleanupCascadeLog": cleanup_cascade_log or [],
        "cleanupPassesRun": cleanup_passes_run,
        "residualWordsByPass": residual_words_by_pass or [],
        "residualMaskAreaByPass": residual_mask_area_by_pass or [],
        "cleanupImprovementByPass": cleanup_improvement_by_pass or [],
        "stoppedReason": stopped_reason,
        "residualWordBoxesAfterFirstPass": residual_word_boxes_after_first_pass or [],
        "residualGlyphMaskAfterFirstPass": residual_glyph_mask_after_first_pass,
        "residualArtifactMaskExpanded": residual_artifact_mask_expanded,
        "residualMaskCoverageByWord": residual_mask_coverage_by_word or [],
        "residualMaskFalsePositiveEstimate": residual_mask_false_positive_estimate,
        "sdxlFallbackConfigured": sdxl_fallback_configured,
        "sdxlControlType": sdxl_control_type,
        "sdxlModelId": sdxl_model_id,
        "controlNetModelId": control_net_model_id,
        "sdxlFailureReason": sdxl_failure_reason,
        "inpaintingInputMode": inpainting_input_mode,
        "styleAwareRenderingEnabled": style_aware_rendering_enabled,
        "sourceTypographyExtracted": source_typography_extracted or {},
        "sourceStyleSpans": source_style_spans_summary or [],
        "translatedStyleSpans": translated_style_spans_summary or [],
        "semanticStyleMapping": semantic_style_mapping or {},
        "styledSpanLayout": styled_span_layout_summary or [],
        "spanRenderBoxes": span_render_boxes_summary or [],
        "overflowWarnings": overflow_warnings_summary or [],
        "selectedFontScale": selected_font_scale or [],
        "lineBreakStrategy": line_break_strategy,
        "renderQualityStatus": render_quality_status,
        "translatedTextRenderPreview": translated_text_render_preview,
        "sourceTextStyleReference": source_text_style_reference,
        "renderPlanPreview": render_plan_preview,
        "semanticBlockGroupingStatus": semantic_block_grouping_status,
        "headlineMergedIntoSingleBlock": headline_merged_into_single_block,
        "groupedLineIds": grouped_line_ids or [],
        "groupingReason": grouping_reason,
        "groupingConfidence": grouping_confidence,
        "rejectedMergeReasons": rejected_merge_reasons or [],
        "blockType": block_type,
        "finalRenderPlan": final_render_plan or [],
    }


def save_image_output(image: Image.Image, target: Path, extension: str) -> None:
    if extension == "pdf":
        image.convert("RGB").save(target, "PDF", resolution=100.0)
    elif extension == "jpeg":
        image.convert("RGB").save(target, "JPEG", quality=92)
    elif extension == "webp":
        image.convert("RGB").save(target, "WEBP", quality=92)
    else:
        image.save(target, "PNG")


def image_to_png_bytes(image: Image.Image) -> io.BytesIO:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def is_localize_marketing_edit_block(
    block: TextBlock,
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> bool:
    if not block.translate or not text_changed(block.text, block.translated_text):
        return False
    if block.surface != "overlay":
        return False
    width, height = image_size
    block_width = max(1, block.bbox[2] - block.bbox[0])
    block_height = max(1, block.bbox[3] - block.bbox[1])
    overlap_with_foreground = overlap_fraction(block.bbox, foreground_bbox) if foreground_bbox else 0.0
    center_x = (block.bbox[0] + block.bbox[2]) / 2
    product_label_like = (
        has_packaging_cues(block.text)
        and block_width <= max(1, width) * 0.5
        and block_height <= max(1, height) * 0.25
        and (overlap_with_foreground >= 0.22 or center_x >= max(1, width) * 0.45)
    )
    return not product_label_like


def localize_edit_boxes_for_block(block: TextBlock, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    regions = block.line_boxes if block.surface == "overlay" and block.line_boxes else [block.clean_box or block.bbox]
    boxes: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom in regions:
        if right <= left or bottom <= top:
            continue
        if normalize_ocr_text(block.text) == "defensive":
            bottom = min(height, bottom + max(block.line_height_estimate, bottom - top) + 6)
        pad_x = max(3, min(16, int(round(max(block.font_size_estimate, right - left) * 0.035))))
        pad_y = max(2, min(10, int(round(max(block.line_height_estimate, bottom - top) * 0.08))))
        boxes.append(
            (
                max(0, left - pad_x),
                max(0, top - pad_y),
                min(width, right + pad_x),
                min(height, bottom + pad_y),
            )
        )
    return boxes


def build_openai_localize_edit_mask(
    image: Image.Image,
    blocks: list[TextBlock],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    edit_luma = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(edit_luma)
    for block in blocks:
        if not is_localize_marketing_edit_block(block, image.size, foreground_bbox):
            continue
        for box in localize_edit_boxes_for_block(block, image.size):
            draw.rectangle(box, fill=255)
    edit_luma = edit_luma.filter(ImageFilter.GaussianBlur(radius=0.35))
    alpha = ImageChops.invert(edit_luma).point(lambda value: 0 if value < 18 else value)
    mask = Image.new("RGBA", image.size, (255, 255, 255, 255))
    mask.putalpha(alpha)
    return mask


def openai_output_format_for(extension: str) -> str:
    normalized = extension.lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpeg"
    if normalized == "webp":
        return "webp"
    return "png"


def nearest_openai_edit_size(width: int, height: int) -> str:
    ratio = width / max(1, height)
    candidates = {
        "1024x1024": 1.0,
        "1024x1536": 1024 / 1536,
        "1536x1024": 1536 / 1024,
    }
    return min(candidates, key=lambda key: abs(candidates[key] - ratio))


def decode_openai_image_response(response: Any) -> Image.Image:
    item = response.data[0] if getattr(response, "data", None) else None
    if item is None:
        raise RuntimeError("OpenAI image edit returned no image data.")
    encoded = getattr(item, "b64_json", None)
    if encoded:
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    url = getattr(item, "url", None)
    if url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    raise RuntimeError("OpenAI image edit returned neither base64 nor URL image data.")


def create_openai_image_edit_with_fallback(
    client: OpenAI,
    base_kwargs: dict[str, Any],
    streams: list[io.BytesIO],
    *,
    output_format: str | None,
    size: str,
    width: int,
    height: int,
) -> tuple[Any, dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {"size": size, "output_format": output_format, "response_format": "b64_json"},
        {"size": size, "output_format": output_format},
        {"size": size},
    ]
    fallback_size = nearest_openai_edit_size(width, height)
    if size != fallback_size:
        variants.append({"size": fallback_size})

    last_error: Exception | None = None
    for variant in variants:
        request_kwargs = dict(base_kwargs)
        request_kwargs.update({key: value for key, value in variant.items() if value is not None})
        for stream in streams:
            stream.seek(0)
        try:
            return client.images.edit(**request_kwargs), variant
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if any(fragment in message for fragment in ["unknown parameter", "unsupported", "invalid size", "must be one of"]):
                continue
            raise
    raise RuntimeError(f"OpenAI image edit failed after compatible retries: {last_error}") from last_error


def build_openai_localize_prompt(
    blocks: list[TextBlock],
    target_language: str,
    source_language: str,
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> str:
    source_items: list[dict[str, Any]] = []
    for block in blocks:
        if not is_localize_marketing_edit_block(block, image_size, foreground_bbox):
            continue
        source_lines = block.line_texts or [block.text]
        replacement_lines = [line.strip() for line in (block.translated_text or block.text).splitlines() if line.strip()]
        line_boxes = localize_edit_boxes_for_block(block, image_size)
        source_items.append(
            {
                "source_text": block.text,
                "replacement_text": block.translated_text or block.text,
                "exact_visible_text_required": block.translated_text or block.text,
                "bbox": list(block.bbox),
                "line_mapping": [
                    {
                        "source_line": source_lines[index] if index < len(source_lines) else block.text,
                        "replacement_line": replacement_lines[index] if index < len(replacement_lines) else "",
                        "editable_box": list(line_boxes[index]) if index < len(line_boxes) else list(block.bbox),
                    }
                    for index in range(max(len(source_lines), len(replacement_lines), len(line_boxes), 1))
                ],
                "source_style_spans": block.source_style_spans,
                "translated_style_spans": block.translated_style_spans,
                "style": {
                    "font_weight": block.font_weight,
                    "font_size_estimate": block.font_size_estimate,
                    "color": block.color,
                    "alignment": block.align,
                    "uppercase": block.text.isupper(),
                },
            }
        )
    protected_items = [
        {
            "text": block.text,
            "bbox": list(block.bbox),
            "surface": block.surface,
            "reason": "product_or_packaging_text" if block.surface in {"packaging", "product"} or has_packaging_cues(block.text) else "non_marketing_text",
        }
        for block in blocks
        if not any(item.get("source_text") == block.text for item in source_items)
        and (block.surface in {"packaging", "product"} or has_packaging_cues(block.text) or not block.translate)
    ][:30]

    return (
        "Edit the provided ad creative for strict marketing text localization.\n"
        f"Source language: {source_language}. Target language: {LANGUAGE_NAMES.get(target_language.upper(), target_language)} ({target_language}).\n"
        "Hard contract:\n"
        "1. Replace ONLY the editable source marketing text items listed in Text replacements JSON.\n"
        "2. Use each replacement_text exactly. Do not paraphrase, summarize, reorder, add product names, add SKU names, or invent new copy.\n"
        "2B. The final visible localized text for each item must match exact_visible_text_required character-for-character except for line breaks inserted only to fit. Do not add synonyms, filler words, explanatory words, extra suffixes, or target-language variants that are not present in exact_visible_text_required.\n"
        "2A. Preserve the exact original canvas, crop, camera framing, zoom level, face/product scale, object positions, background and aspect ratio. Do not reframe, zoom in, zoom out, retouch the whole photo, beautify the subject, or regenerate the full creative.\n"
        "3. Source words may be stacked visually but form one sentence. Preserve the semantic sentence in replacement_text while keeping the source line count and visual rhythm as much as possible.\n"
        "4. If replacement_text has line breaks, keep those line breaks. If line_mapping is present, map each replacement_line to the corresponding source_line/editable_box.\n"
        "5. Transfer style by semantic word/phrase meaning, not by absolute position: if the emphasized source word moves from the end to the beginning in the target language, the target word at the beginning receives that style. Source blue text stays blue in the corresponding translated words; source white text on a yellow highlight stays white on the same yellow highlight; font weight, casing, size hierarchy, alignment, background color, and spacing must match the source region.\n"
        "6. Do not create any new text background rectangle, white box, highlight band, label panel, sticker, or callout unless the source text already had that same visual background in the exact editable region. If the source text is directly on the photo/background, render the replacement directly on the same photo/background.\n"
        "7. Product packaging text is immutable. Do not translate, redraw, blur, clean, repaint, or alter any text printed on bottles, boxes, labels, devices, stickers, logos, brand marks, QR codes, legal text, ingredient text, or UI chrome.\n"
        "8. Preserve the product, packaging, person, objects, background, lighting, shadows, composition, and all unlisted pixels unchanged. Decorative/safe-zone corner brackets, arrows, frame lines, dividers and icons are not text; preserve them as design elements and do not turn them into quote marks or characters.\n"
        "9. The mask marks the only editable marketing text regions. Treat everything outside the mask as read-only.\n"
        "10. Do not add new logos, watermarks, badges, product names, extra copy, or new design containers.\n"
        "Return a finished localized image only.\n"
        f"Text replacements JSON:\n{json.dumps(source_items, ensure_ascii=False, indent=2)}\n"
        f"Protected non-editable text hints JSON:\n{json.dumps(protected_items, ensure_ascii=False, indent=2)}"
    )


def build_openai_localize_cleanup_prompt(
    blocks: list[TextBlock],
    target_language: str,
    source_language: str,
    image_size: tuple[int, int],
    foreground_bbox: tuple[int, int, int, int] | None = None,
) -> str:
    source_items = [
        {
            "source_text": block.text,
            "bbox": list(block.bbox),
            "line_boxes": [list(box) for box in (block.line_boxes or [])],
        }
        for block in blocks
        if is_localize_marketing_edit_block(block, image_size, foreground_bbox)
    ]
    return (
        "Remove the listed source marketing text from this ad creative.\n"
        f"Source language: {source_language}. Target language later: {LANGUAGE_NAMES.get(target_language.upper(), target_language)} ({target_language}).\n"
        "Hard contract:\n"
        "1. Remove only the visible source marketing text pixels inside the transparent/editable mask.\n"
        "2. Do not add translated text. Do not add any new text at all.\n"
        "3. Reconstruct the original background/photo/design behind the removed text naturally, matching texture, lighting, grain, shadows and color.\n"
        "4. Preserve the exact original canvas, crop, camera framing, zoom level, face/product scale, object positions, background and aspect ratio.\n"
        "5. Preserve safe-zone corner brackets, arrows, frame lines, icons, product packaging, logos, legal text and all unmasked pixels.\n"
        "6. Do not add boxes, panels, cream/lotion strokes, labels, stickers, artifacts, watermarks or objects.\n"
        "Return the same creative with only the source marketing text removed.\n"
        f"Source marketing text to remove JSON:\n{json.dumps(source_items, ensure_ascii=False, indent=2)}"
    )


def render_localize_with_openai_image_edit(
    source_image: Image.Image,
    translated_blocks: list[TextBlock],
    target_language: str,
    source_language: str,
    output_format: str,
) -> tuple[Image.Image, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI image localization.")

    model = os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
    quality = os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium").strip().lower() or "medium"
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "medium"

    extension = normalize_output_format(output_format)
    foreground_bbox = detect_foreground_bbox(source_image)
    prompt_blocks = prepare_model_localize_blocks(translated_blocks)
    prompt = build_openai_localize_cleanup_prompt(prompt_blocks, target_language, source_language, source_image.size, foreground_bbox)
    mask_image = build_openai_localize_edit_mask(source_image, prompt_blocks, foreground_bbox)
    image_file = image_to_png_bytes(source_image)
    mask_file = io.BytesIO()
    mask_image.save(mask_file, format="PNG")
    mask_file.seek(0)

    client = OpenAI(api_key=api_key)
    response, api_variant = create_openai_image_edit_with_fallback(
        client,
        {
            "model": model,
            "image": ("creative.png", image_file, "image/png"),
            "mask": ("marketing-text-mask.png", mask_file, "image/png"),
            "prompt": prompt,
            "quality": quality,
        },
        [image_file, mask_file],
        output_format=openai_output_format_for(extension),
        size="auto",
        width=source_image.width,
        height=source_image.height,
    )
    cleaned = decode_openai_image_response(response)
    if cleaned.size != source_image.size:
        cleaned = cleaned.resize(source_image.size, Image.Resampling.LANCZOS)
    edit_luma = ImageChops.invert(mask_image.getchannel("A")).convert("L")
    composite_luma = edit_luma.filter(ImageFilter.MaxFilter(size=61)).filter(ImageFilter.GaussianBlur(radius=6))
    try:
        source_np = np.array(source_image.convert("RGB"), dtype=np.float32)
        rendered_np = np.array(cleaned.convert("RGB"), dtype=np.float32)
        protected = np.array(composite_luma) <= 8
        protected_change = float(np.abs(source_np - rendered_np).mean(axis=2)[protected].mean() / 255.0) if protected.any() else 0.0
    except Exception:
        protected_change = 1.0
    used_composite_guard = protected_change > float(os.getenv("ADAPTIFAI_LOCALIZE_MODEL_MAX_PROTECTED_CHANGE", "0.025"))
    if used_composite_guard:
        cleaned = Image.composite(cleaned.convert("RGB"), source_image.convert("RGB"), composite_luma)
    rendered = render_translated_text(cleaned.convert("RGB"), prompt_blocks, render_plan=None)

    return rendered, {
        "provider": "openai",
        "model": model,
        "quality": quality,
        "apiVariant": api_variant,
        "maskedBlocks": len([block for block in prompt_blocks if is_localize_marketing_edit_block(block, source_image.size, foreground_bbox)]),
        "maskMode": "strict_marketing_line_boxes",
        "styleContract": "model_cleanup_deterministic_text_render",
        "outputFormat": openai_output_format_for(extension),
        "protectedChange": protected_change,
        "usedCompositeGuard": used_composite_guard,
    }


def prepare_model_localize_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    prepared: list[TextBlock] = []
    for block in blocks:
        if not block.translate or not block.translation_candidates:
            prepared.append(block)
            continue
        candidates = [
            preserve_source_metric_tokens(block.text, str(candidate.get("text", "")).strip())
            for candidate in block.translation_candidates
            if str(candidate.get("text", "")).strip()
        ]
        if not candidates:
            prepared.append(block)
            continue
        source_len = max(1, len((block.text or "").replace("\n", " ")))
        current = block.translated_text or block.text
        selected = current
        if len(current.replace("\n", " ")) > source_len * 0.92:
            selected = min(candidates, key=lambda value: (len(value.replace("\n", " ")), value.count("\n")))
        prepared.append(block.model_copy(update={"translated_text": selected}))
    return prepared


def union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def bbox_from_polygon(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if not points:
        return (0, 0, 1, 1)
    xs = [int(point[0]) for point in points]
    ys = [int(point[1]) for point in points]
    return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)


def polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_mask(image_size: tuple[int, int], polygons: list[list[tuple[int, int]]], *, dilation_px: int = 0) -> Image.Image:
    import cv2

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([(int(x), int(y)) for x, y in polygon], fill=255)
    if dilation_px > 0:
        arr = np.array(mask, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1))
        arr = cv2.dilate(arr, kernel, iterations=1)
        mask = Image.fromarray(arr, "L")
    return mask


def polygon_mask_overlap_fraction(polygons: list[list[tuple[int, int]]], product_mask: Image.Image) -> float:
    if not polygons:
        return 0.0
    text_mask = polygon_mask(product_mask.size, polygons)
    text_arr = np.array(text_mask.convert("L")) > 16
    if not np.any(text_arr):
        return 0.0
    product_arr = np.array(product_mask.convert("L")) > 16
    return float(np.logical_and(text_arr, product_arr).sum() / max(1, text_arr.sum()))


def google_vision_available() -> bool:
    return bool(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
        or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    )


def google_vision_word_blocks(image_path: Path, image_size: tuple[int, int]) -> list[TextBlock]:
    if not google_vision_available():
        return []
    try:
        from google.cloud import vision  # type: ignore
        from google.oauth2 import service_account  # type: ignore
    except Exception as exc:
        print(f"[vision] google-cloud-vision unavailable: {exc}", flush=True)
        return []
    try:
        credentials = None
        inline_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if inline_json:
            credentials = service_account.Credentials.from_service_account_info(json.loads(inline_json))
        client = vision.ImageAnnotatorClient(credentials=credentials) if credentials else vision.ImageAnnotatorClient()
        content = image_path.read_bytes()
        response = client.document_text_detection(image=vision.Image(content=content))
        if response.error.message:
            print(f"[vision] document_text_detection error: {response.error.message}", flush=True)
            return []
    except Exception as exc:
        print(f"[vision] document_text_detection failed: {exc}", flush=True)
        return []

    words: list[TextBlock] = []
    word_index = 0
    try:
        pages = response.full_text_annotation.pages
    except Exception:
        pages = []
    for page in pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    text = "".join(symbol.text for symbol in word.symbols).strip()
                    if not text:
                        continue
                    vertices = word.bounding_box.vertices
                    polygon = [(int(vertex.x or 0), int(vertex.y or 0)) for vertex in vertices]
                    symbol_polygons: list[list[tuple[int, int]]] = []
                    for symbol in word.symbols:
                        symbol_vertices = symbol.bounding_box.vertices
                        symbol_polygon = [(int(vertex.x or 0), int(vertex.y or 0)) for vertex in symbol_vertices]
                        if len(symbol_polygon) >= 3 and polygon_area(symbol_polygon) > 1:
                            symbol_polygons.append(symbol_polygon)
                    if len(polygon) < 3 or polygon_area(polygon) <= 2:
                        continue
                    bbox = bbox_from_polygon(polygon)
                    word_index += 1
                    words.append(
                        TextBlock(
                            id=f"v5-word-{word_index}",
                            text=text,
                            role=classify_text_role(text),
                            translate=True,
                            bbox=bbox,
                            clean_box=bbox,
                            polygon=polygon,
                            line_polygons=[polygon],
                            symbol_polygons=symbol_polygons,
                            line_boxes=[bbox],
                            line_texts=[text],
                            color="#111111",
                            font_weight=800 if text.isupper() else 700,
                            font_size_estimate=estimate_font_size_from_bbox(bbox),
                            line_height_estimate=estimate_line_height_from_bbox(bbox),
                            align=infer_alignment(bbox, image_size[0]),
                            surface="overlay",
                        )
                    )
    return words


def bbox_area(box: tuple[int, int, int, int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def split_ocr_line_to_word_boxes(text: str, bbox: tuple[int, int, int, int]) -> list[tuple[str, tuple[int, int, int, int]]]:
    words = [part for part in re.split(r"\s+", str(text or "").strip()) if part]
    if not words:
        return []
    left, top, right, bottom = bbox
    width = max(1, right - left)
    weights = [max(1, len(normalize_ocr_text(word)) or len(word)) for word in words]
    total_weight = max(1, sum(weights))
    cursor = float(left)
    results: list[tuple[str, tuple[int, int, int, int]]] = []
    for index, (word, weight) in enumerate(zip(words, weights, strict=False)):
        if index == len(words) - 1:
            word_right = right
        else:
            word_right = int(round(cursor + width * (weight / total_weight)))
        word_left = int(round(cursor))
        if word_right <= word_left:
            word_right = min(right, word_left + 1)
        results.append((word, (max(left, word_left), top, min(right, word_right), bottom)))
        cursor = float(word_right)
    return results


def easyocr_supplemental_v5_word_blocks(
    image_path: Path,
    image_size: tuple[int, int],
    existing_words: list[TextBlock],
) -> list[TextBlock]:
    try:
        detector = load_ocr_detector()
        image = Image.open(image_path).convert("RGB")
        ocr_image, scale = fit_for_ocr(image)
        detections = detector.readtext(
            np.array(ocr_image),
            detail=1,
            paragraph=False,
            batch_size=int(os.getenv("ADAPTIFAI_OCR_BATCH_SIZE", "1")),
            width_ths=float(os.getenv("ADAPTIFAI_OCR_WIDTH_THS", "0.7")),
            decoder=os.getenv("ADAPTIFAI_EASYOCR_DECODER", "greedy"),
        )
    except Exception as exc:
        print(f"[vision] EasyOCR supplemental pass unavailable: {exc}", flush=True)
        return []

    existing_boxes = [word.bbox for word in existing_words if word.bbox]
    supplemental: list[TextBlock] = []
    min_confidence = float(os.getenv("ADAPTIFAI_V5_SUPPLEMENTAL_OCR_MIN_CONFIDENCE", "0.18"))
    word_index = 0
    for points, detected_text, confidence in detections:
        text = repair_mojibake(str(detected_text or "").strip())
        if not text or float(confidence or 0.0) < min_confidence:
            continue
        xs = [int(point[0] / scale) for point in points]
        ys = [int(point[1] / scale) for point in points]
        line_box = (
            max(0, min(xs)),
            max(0, min(ys)),
            min(image_size[0], max(xs)),
            min(image_size[1], max(ys)),
        )
        if line_box[2] <= line_box[0] or line_box[3] <= line_box[1]:
            continue
        duplicates_existing = any(
            overlap_fraction(line_box, existing_box) >= 0.45 or overlap_fraction(existing_box, line_box) >= 0.55
            for existing_box in existing_boxes
        )
        if duplicates_existing:
            if "%" in text and not any("%" in str(word.text or "") and overlap_fraction(line_box, word.bbox) >= 0.12 for word in existing_words):
                symbol_width = max(4, int((line_box[2] - line_box[0]) * 0.18))
                symbol_box = (max(line_box[0], line_box[2] - symbol_width), line_box[1], line_box[2], line_box[3])
                polygon = [
                    (symbol_box[0], symbol_box[1]),
                    (symbol_box[2], symbol_box[1]),
                    (symbol_box[2], symbol_box[3]),
                    (symbol_box[0], symbol_box[3]),
                ]
                word_index += 1
                supplemental.append(
                    TextBlock(
                        id=f"v5-easyocr-word-{word_index}",
                        text="%",
                        role="numeric_claim",
                        translate=True,
                        bbox=symbol_box,
                        clean_box=symbol_box,
                        polygon=polygon,
                        line_polygons=[polygon],
                        symbol_polygons=[polygon],
                        line_boxes=[symbol_box],
                        line_texts=["%"],
                        color="#111111",
                        font_weight=800,
                        font_size_estimate=estimate_font_size_from_bbox(symbol_box),
                        line_height_estimate=estimate_line_height_from_bbox(symbol_box),
                        align=infer_alignment(symbol_box, image_size[0]),
                        surface="overlay",
                    )
                )
            continue
        for word_text, word_box in split_ocr_line_to_word_boxes(text, line_box):
            if is_decorative_or_numeric_only(word_text) and len(normalize_ocr_text(word_text)) <= 1:
                continue
            polygon = [(word_box[0], word_box[1]), (word_box[2], word_box[1]), (word_box[2], word_box[3]), (word_box[0], word_box[3])]
            word_index += 1
            supplemental.append(
                TextBlock(
                    id=f"v5-easyocr-word-{word_index}",
                    text=word_text,
                    role=classify_text_role(word_text),
                    translate=True,
                    bbox=word_box,
                    clean_box=word_box,
                    polygon=polygon,
                    line_polygons=[polygon],
                    symbol_polygons=[polygon],
                    line_boxes=[word_box],
                    line_texts=[word_text],
                    color="#111111",
                    font_weight=800 if word_text.isupper() else 700,
                    font_size_estimate=estimate_font_size_from_bbox(word_box),
                    line_height_estimate=estimate_line_height_from_bbox(word_box),
                    align=infer_alignment(word_box, image_size[0]),
                    surface="overlay",
                )
            )
    return supplemental


def supplement_v5_ocr_words(image_path: Path, image_size: tuple[int, int], words: list[TextBlock]) -> tuple[list[TextBlock], dict[str, Any]]:
    if os.getenv("ADAPTIFAI_DISABLE_SUPPLEMENTAL_OCR", "0") == "1":
        return words, {"supplementalProvider": "disabled", "supplementalWordCount": 0}
    image_area = max(1, image_size[0] * image_size[1])
    detected_area = sum(bbox_area(word.bbox) for word in words)
    area_ratio = detected_area / image_area
    should_run = area_ratio < float(os.getenv("ADAPTIFAI_V5_SUPPLEMENTAL_OCR_AREA_RATIO", "0.16")) or len(words) < int(os.getenv("ADAPTIFAI_V5_SUPPLEMENTAL_OCR_MIN_WORDS", "8"))
    if not should_run:
        return words, {"supplementalProvider": "skipped", "supplementalWordCount": 0, "visionAreaRatio": area_ratio}
    supplemental = easyocr_supplemental_v5_word_blocks(image_path, image_size, words)
    if not supplemental:
        return words, {"supplementalProvider": "easyocr", "supplementalWordCount": 0, "visionAreaRatio": area_ratio}
    merged = list(words)
    existing_boxes = [word.bbox for word in merged]
    added = 0
    for candidate in supplemental:
        duplicate = any(
            overlap_fraction(candidate.bbox, existing_box) >= 0.45 or overlap_fraction(existing_box, candidate.bbox) >= 0.55
            for existing_box in existing_boxes
        )
        if duplicate:
            continue
        merged.append(candidate)
        existing_boxes.append(candidate.bbox)
        added += 1
    return merged, {
        "supplementalProvider": "easyocr",
        "supplementalWordCount": added,
        "visionAreaRatio": area_ratio,
    }


def is_v5_numeric_bypass_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    return bool(re.fullmatch(r"[\d]+(?:[.)])?", cleaned))


def is_v5_step_marker_word(word: TextBlock, words: list[TextBlock], image_size: tuple[int, int]) -> bool:
    if not is_v5_numeric_bypass_text(word.text):
        return False
    box = word.bbox
    height = max(1, box[3] - box[1])
    width = max(1, box[2] - box[0])
    if height < max(26, int(image_size[1] * 0.045)):
        return False
    if width > height * 1.15:
        return False
    cy = (box[1] + box[3]) / 2.0
    right_context: list[TextBlock] = []
    for other in words:
        if other is word or is_v5_numeric_bypass_text(other.text):
            continue
        other_box = other.bbox
        other_height = max(1, other_box[3] - other_box[1])
        other_cy = (other_box[1] + other_box[3]) / 2.0
        same_band = abs(other_cy - cy) <= max(height, other_height) * 1.35
        starts_to_right = other_box[0] >= box[2] + max(14, int(height * 0.8))
        if same_band and starts_to_right:
            right_context.append(other)
    if not right_context:
        return False
    nearest_context_left = min(item.bbox[0] for item in right_context)
    return box[0] <= image_size[0] * 0.32 and nearest_context_left > box[2]


def detect_v5_step_marker_words(words: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    explicit = [word for word in words if is_v5_step_marker_word(word, words, image_size)]
    candidates: list[TextBlock] = []
    for word in words:
        if not is_v5_numeric_bypass_text(word.text):
            continue
        cleaned = re.sub(r"\D", "", str(word.text or ""))
        if not cleaned or len(cleaned) > 2:
            continue
        value = int(cleaned)
        if value < 1 or value > 12:
            continue
        box = word.bbox
        height = max(1, box[3] - box[1])
        width = max(1, box[2] - box[0])
        if height < max(18, int(image_size[1] * 0.018)):
            continue
        if width > height * 1.45:
            continue
        candidates.append(word)

    if len(candidates) < 2:
        detected = explicit
    else:
        heights = [max(1, item.bbox[3] - item.bbox[1]) for item in candidates]
        median_height = sorted(heights)[len(heights) // 2]
        similar = [
            item
            for item in candidates
            if 0.45 <= (max(1, item.bbox[3] - item.bbox[1]) / max(1, median_height)) <= 2.2
        ]
        distinct_values = {int(re.sub(r"\D", "", str(item.text or ""))) for item in similar}
        if len(similar) >= 2 and len(distinct_values) >= 2:
            detected = explicit + similar
        else:
            detected = explicit

    unique: list[TextBlock] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for word in detected:
        key = (str(word.text), word.bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(word)
    return unique


def looks_like_numeric_marketing_claim_word(word: TextBlock, image_size: tuple[int, int]) -> bool:
    text = str(word.text or "").strip().lower()
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False
    if "%" in text or re.search(r"\d+\s*(h|hr|hrs|hour|hours|saat|sa)", normalized):
        return True
    box = word.bbox
    height = max(1, box[3] - box[1])
    center_y = (box[1] + box[3]) / 2.0
    return bool(re.search(r"\d", normalized)) and height >= image_size[1] * 0.035 and image_size[1] * 0.12 <= center_y <= image_size[1] * 0.82


def is_possible_v5_step_marker_candidate(word: TextBlock, image_size: tuple[int, int]) -> bool:
    if not is_v5_numeric_bypass_text(word.text):
        return False
    cleaned = re.sub(r"\D", "", str(word.text or ""))
    if not cleaned or len(cleaned) > 2:
        return False
    value = int(cleaned)
    if value < 1 or value > 12:
        return False
    box = word.bbox
    height = max(1, box[3] - box[1])
    width = max(1, box[2] - box[0])
    return height >= max(18, int(image_size[1] * 0.018)) and width <= height * 1.45


def v5_words_sharing_line(word: TextBlock, words: list[TextBlock]) -> list[TextBlock]:
    box = word.bbox
    height = max(1, box[3] - box[1])
    cy = (box[1] + box[3]) / 2.0
    shared: list[TextBlock] = []
    for other in words:
        other_box = other.bbox
        other_height = max(1, other_box[3] - other_box[1])
        other_cy = (other_box[1] + other_box[3]) / 2.0
        if abs(other_cy - cy) <= max(height, other_height) * 0.72:
            shared.append(other)
    return shared


def has_instructional_line_context(word: TextBlock, words: list[TextBlock]) -> bool:
    line_text = " ".join(item.text for item in sorted(v5_words_sharing_line(word, words), key=lambda item: item.bbox[0]))
    return is_instructional_context_text(line_text)


def is_probable_v5_ocr_noise_near_step_marker(word: TextBlock, step_markers: list[TextBlock], image_size: tuple[int, int]) -> bool:
    text = str(word.text or "").strip()
    if not text or normalize_ocr_text(text):
        return False
    if len(text) > 3 or not step_markers:
        return False
    box = word.bbox
    height = max(1, box[3] - box[1])
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    for marker in step_markers:
        marker_box = marker.bbox
        marker_height = max(1, marker_box[3] - marker_box[1])
        marker_center_x = (marker_box[0] + marker_box[2]) / 2.0
        marker_center_y = (marker_box[1] + marker_box[3]) / 2.0
        if abs(center_y - marker_center_y) <= max(height, marker_height) * 1.1 and abs(center_x - marker_center_x) <= max(
            marker_height * 2.4, image_size[0] * 0.12
        ):
            return True
    return False


def group_v5_polygon_words(words: list[TextBlock], image: Image.Image) -> tuple[list[TextBlock], list[TextBlock], dict[str, Any]]:
    if not words:
        return [], [], {"provider": "google-cloud-vision", "wordCount": 0, "mode": "polygon"}
    product_mask = build_sam_or_vision_product_mask(image)
    overlay_words: list[TextBlock] = []
    protected_words: list[TextBlock] = []
    product_overlap_threshold = float(os.getenv("ADAPTIFAI_V5_PRODUCT_POLYGON_OVERLAP", "0.55"))
    for word in words:
        overlap = polygon_mask_overlap_fraction([word.polygon] if word.polygon else [], product_mask)
        if overlap >= product_overlap_threshold:
            protected_words.append(word.model_copy(update={"translate": False, "surface": "packaging"}))
            continue
        keep_as_overlay = (
            should_translate_ocr_overlay_block(word, image.size)
            or has_instructional_line_context(word, words)
            or looks_like_numeric_marketing_claim_word(word, image.size)
            or is_possible_v5_step_marker_candidate(word, image.size)
        )
        overlay_words.append(word.model_copy(update={"translate": True, "surface": "overlay"}))

    numeric_bypass_words = detect_v5_step_marker_words(overlay_words, image.size)
    numeric_bypass_keys = {(word.text, word.bbox) for word in numeric_bypass_words}
    textual_overlay_words = [
        word
        for word in overlay_words
        if (word.text, word.bbox) not in numeric_bypass_keys
        and not is_v5_numeric_bypass_text(word.text)
        and not is_probable_v5_ocr_noise_near_step_marker(word, numeric_bypass_words, image.size)
    ]
    ordered = sorted(textual_overlay_words, key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2, item.bbox[0]))
    lines: list[list[TextBlock]] = []
    for word in ordered:
        cy = (word.bbox[1] + word.bbox[3]) / 2
        h = max(1, word.bbox[3] - word.bbox[1])
        matched: list[TextBlock] | None = None
        for line in lines:
            line_box = union_bbox([item.bbox for item in line]) or line[0].bbox
            line_cy = (line_box[1] + line_box[3]) / 2
            line_h = max(1, line_box[3] - line_box[1])
            if abs(cy - line_cy) <= max(h, line_h) * 0.62:
                matched = line
                break
        if matched is None:
            lines.append([word])
        else:
            matched.append(word)

    split_lines: list[list[TextBlock]] = []
    for line in lines:
        sorted_line = sorted(line, key=lambda item: item.bbox[0])
        if len(sorted_line) <= 1:
            split_lines.append(sorted_line)
            continue
        heights = [max(1, item.bbox[3] - item.bbox[1]) for item in sorted_line]
        median_height = sorted(heights)[len(heights) // 2]
        current_run = [sorted_line[0]]
        for item in sorted_line[1:]:
            gap = item.bbox[0] - current_run[-1].bbox[2]
            if gap >= max(24, int(median_height * 1.35)):
                split_lines.append(current_run)
                current_run = [item]
            else:
                current_run.append(item)
        if current_run:
            split_lines.append(current_run)
    lines = split_lines
    lines.sort(key=lambda line: min(item.bbox[1] for item in line))
    groups: list[list[TextBlock]] = []
    for line in lines:
        line_box = union_bbox([item.bbox for item in line]) or line[0].bbox
        line_text = " ".join(item.text for item in line)
        best_group_index: int | None = None
        best_gap: float | None = None
        for group_index, prev in enumerate(groups):
            prev_box = union_bbox([item.bbox for item in prev]) or prev[-1].bbox
            prev_text = " ".join(item.text for item in prev)
            gap_y = line_box[1] - prev_box[3]
            if gap_y < -max(10, (line_box[3] - line_box[1]) * 0.5):
                continue
            overlap_x = max(0, min(line_box[2], prev_box[2]) - max(line_box[0], prev_box[0]))
            min_width = max(1, min(line_box[2] - line_box[0], prev_box[2] - prev_box[0]))
            same_column = overlap_x >= min_width * 0.20 or abs(line_box[0] - prev_box[0]) <= image.width * 0.075
            normalized_pair = normalize_ocr_text(prev_text + " " + line_text)
            semantic_continuation = (
                not prev_text.strip().isupper()
                or not line_text.strip().isupper()
                or any(
                    term in normalized_pair
                    for term in (
                        "use",
                        "with",
                        "apply",
                        "remove",
                        "cleanse",
                        "rinse",
                        "benefits",
                        "look",
                        "recommended",
                        "acne",
                        "prone",
                        "adult",
                        "skin",
                        "tragen",
                        "spray",
                        "schuhe",
                    )
                )
            )
            if same_column and semantic_continuation and gap_y <= max(24, (line_box[3] - line_box[1]) * 1.25):
                if best_gap is None or gap_y < best_gap:
                    best_group_index = group_index
                    best_gap = gap_y
        if best_group_index is not None:
            groups[best_group_index].extend(line)
        else:
            groups.append(line)

    split_groups: list[list[TextBlock]] = []
    for group in groups:
        ordered_group = sorted(group, key=lambda item: item.bbox[0])
        heights = [max(1, item.bbox[3] - item.bbox[1]) for item in ordered_group]
        median_height = sorted(heights)[len(heights) // 2] if heights else 1
        vertical_span = max(item.bbox[3] for item in ordered_group) - min(item.bbox[1] for item in ordered_group)
        gaps = [
            ordered_group[index + 1].bbox[0] - ordered_group[index].bbox[2]
            for index in range(len(ordered_group) - 1)
        ]
        should_split_columns = (
            len(ordered_group) >= 2
            and vertical_span <= median_height * 1.8
            and bool(gaps)
            and min(gaps) >= max(28, median_height * 2.4)
        )
        if should_split_columns:
            split_groups.extend([[word] for word in ordered_group])
        else:
            split_groups.append(group)
    groups = split_groups

    blocks: list[TextBlock] = []
    for numeric_index, word in enumerate(sorted(numeric_bypass_words, key=lambda item: (item.bbox[1], item.bbox[0])), start=1):
        block = TextBlock(
            id=f"v5-numeric-{numeric_index}",
            text=word.text,
            role="numeric_claim",
            translate=False,
            bbox=word.bbox,
            clean_box=word.bbox,
            polygon=list(word.polygon or []),
            line_polygons=[list(word.polygon or [])] if word.polygon else [],
            symbol_polygons=list(word.symbol_polygons or []),
            line_boxes=[word.bbox],
            line_texts=[word.text],
            color=sample_deterministic_text_color(image, word.bbox, "#111111"),
            font_weight=word.font_weight or 700,
            font_size_estimate=max(8, word.bbox[3] - word.bbox[1]),
            line_height_estimate=max(10, int((word.bbox[3] - word.bbox[1]) * 1.12)),
            align="left",
            surface="overlay",
            translated_text=word.text,
            render_strategy="v5_numeric_bypass",
        )
        source_word_styles = build_v5_polygon_source_word_styles([word], block, image)
        if source_word_styles:
            block = block.model_copy(
                update={
                    "source_word_styles": source_word_styles,
                    "source_style_spans": [
                        {
                            "sourceText": source_word_styles[0]["text"],
                            "semanticRole": "numeric_claim",
                            "sourceWordId": source_word_styles[0]["id"],
                            "style": {
                                **default_typography_style(block),
                                "color": source_word_styles[0]["color"],
                                "backgroundColor": source_word_styles[0].get("backgroundColor"),
                                "hasTextBackground": bool(source_word_styles[0].get("hasTextBackground")),
                                "backgroundContrast": source_word_styles[0].get("backgroundContrast", 0.0),
                                "fontWeight": source_word_styles[0]["fontWeight"],
                                "fontSize": source_word_styles[0].get("fontSize", block.font_size_estimate),
                                "lineHeight": source_word_styles[0].get("lineHeight", block.line_height_estimate),
                                "fontCategory": source_word_styles[0].get("fontCategory", "sans-serif"),
                                "casing": "uppercase" if source_word_styles[0].get("isUppercase") else "mixed",
                                "isItalic": bool(source_word_styles[0].get("isItalic")),
                                "isUnderlined": bool(source_word_styles[0].get("isUnderlined")),
                                "isStrikethrough": bool(source_word_styles[0].get("isStrikethrough")),
                            },
                            "color": source_word_styles[0]["color"],
                            "fontWeight": source_word_styles[0]["fontWeight"],
                            "fontCategory": source_word_styles[0].get("fontCategory", "sans-serif"),
                            "casing": "uppercase" if source_word_styles[0].get("isUppercase") else "mixed",
                            "isItalic": bool(source_word_styles[0].get("isItalic")),
                            "isUnderlined": bool(source_word_styles[0].get("isUnderlined")),
                            "isStrikethrough": bool(source_word_styles[0].get("isStrikethrough")),
                            "forceBreakAfter": False,
                        }
                    ],
                    "color": source_word_styles[0]["color"],
                    "font_weight": int(source_word_styles[0].get("fontWeight") or block.font_weight),
                    "font_size_estimate": int(source_word_styles[0].get("fontSize") or block.font_size_estimate),
                    "line_height_estimate": int(source_word_styles[0].get("lineHeight") or block.line_height_estimate),
                }
            )
        blocks.append(block)
    for index, group in enumerate(groups, start=1):
        group = sorted(group, key=lambda item: (item.bbox[1], item.bbox[0]))
        line_groups: list[list[TextBlock]] = []
        for word in group:
            cy = (word.bbox[1] + word.bbox[3]) / 2
            matched = None
            for line in line_groups:
                line_box = union_bbox([item.bbox for item in line]) or line[0].bbox
                if abs(cy - (line_box[1] + line_box[3]) / 2) <= max(4, (line_box[3] - line_box[1]) * 0.7):
                    matched = line
                    break
            if matched is None:
                line_groups.append([word])
            else:
                matched.append(word)
        line_groups = [sorted(line, key=lambda item: item.bbox[0]) for line in line_groups]
        line_groups.sort(key=lambda line: min(item.bbox[1] for item in line))
        line_texts = [" ".join(item.text for item in line) for line in line_groups]
        block_polygons = [item.polygon for item in group if item.polygon]
        line_boxes = [union_bbox([item.bbox for item in line]) or line[0].bbox for line in line_groups]
        bbox = union_bbox([item.bbox for item in group]) or group[0].bbox
        text = "\n".join(line_texts)
        block_align = infer_v5_polygon_alignment(line_boxes, bbox, image.width)
        block = TextBlock(
            id=f"v5-block-{index}",
            text=text,
            role=classify_text_role(text),
            translate=True,
            bbox=bbox,
            clean_box=bbox,
            polygon=[point for polygon in block_polygons for point in polygon],
            line_polygons=block_polygons,
            line_boxes=line_boxes,
            line_texts=line_texts,
            color=sample_deterministic_text_color(image, bbox, "#111111"),
            font_weight=max((item.font_weight for item in group), default=700),
            font_size_estimate=estimate_font_size_from_bbox(bbox),
            line_height_estimate=estimate_line_height_from_bbox(bbox),
            align=block_align,
            surface="overlay",
        )
        source_word_styles = build_v5_polygon_source_word_styles(group, block, image)
        source_style_spans = [
            {
                "sourceText": word["text"],
                "semanticRole": word["semanticRole"],
                "sourceWordId": word["id"],
                "style": {
                    **default_typography_style(block),
                    "color": word["color"],
                    "backgroundColor": word.get("backgroundColor"),
                    "hasTextBackground": bool(word.get("hasTextBackground")),
                    "backgroundContrast": word.get("backgroundContrast", 0.0),
                    "fontWeight": word["fontWeight"],
                    "fontSize": word.get("fontSize", block.font_size_estimate),
                    "lineHeight": word.get("lineHeight", block.line_height_estimate),
                    "fontCategory": word.get("fontCategory", "sans-serif"),
                    "strokeWidth": word.get("strokeWidth", 0),
                    "strokeFill": word.get("strokeFill"),
                    "fillTransparent": bool(word.get("outline")),
                    "casing": "uppercase" if word.get("isUppercase") else "mixed",
                    "isItalic": bool(word.get("isItalic")),
                    "isUnderlined": bool(word.get("isUnderlined")),
                    "isStrikethrough": bool(word.get("isStrikethrough")),
                },
                "color": word["color"],
                "fontWeight": word["fontWeight"],
                "fontCategory": word.get("fontCategory", "sans-serif"),
                "casing": "uppercase" if word.get("isUppercase") else "mixed",
                "isItalic": bool(word.get("isItalic")),
                "isUnderlined": bool(word.get("isUnderlined")),
                "isStrikethrough": bool(word.get("isStrikethrough")),
                "forceBreakAfter": False,
            }
            for word in source_word_styles
        ]
        block = block.model_copy(
            update={
                "source_word_styles": source_word_styles,
                "source_style_spans": source_style_spans,
                "color": source_word_styles[0]["color"] if source_word_styles else block.color,
                "font_weight": max([int(word.get("fontWeight", 700)) for word in source_word_styles] or [block.font_weight]),
            }
        )
        blocks.append(block)
    return blocks, protected_words, {
        "provider": "google-cloud-vision",
        "mode": "polygon_word_grouping",
        "legacyBboxHeuristics": "disabled",
        "preserveDecision": "polygon_to_product_mask_overlap_only",
        "wordCount": len(words),
        "overlayWordCount": len(overlay_words),
        "numericBypassWordCount": len(numeric_bypass_words),
        "protectedWordCount": len(protected_words),
        "blockCount": len(blocks),
    }


def detect_foreground_bbox(source: Image.Image) -> tuple[int, int, int, int] | None:
    rgb = np.array(source.convert("RGB"))
    if rgb.size == 0:
        return None

    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    border_color = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - border_color.astype(np.float32), axis=2)
    mask = distance > max(18.0, float(np.percentile(distance, 78)))
    ys, xs = np.where(mask)
    if len(xs) < 64 or len(ys) < 64:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def build_sam_or_vision_product_mask(source: Image.Image) -> Image.Image:
    checkpoint = os.getenv("ADAPTIFAI_SAM_CHECKPOINT", "").strip()
    if checkpoint and Path(checkpoint).exists():
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # type: ignore

            model_type = os.getenv("ADAPTIFAI_SAM_MODEL_TYPE", "vit_b").strip() or "vit_b"
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=torch_device())
            generator = SamAutomaticMaskGenerator(sam)
            masks = generator.generate(np.array(source.convert("RGB")))
            full = np.zeros((source.height, source.width), dtype=np.uint8)
            image_area = max(1, source.width * source.height)
            for item in masks:
                seg = item.get("segmentation")
                bbox = item.get("bbox") or [0, 0, 0, 0]
                if seg is None:
                    continue
                x, y, w, h = [int(v) for v in bbox[:4]]
                area = int(item.get("area") or np.count_nonzero(seg))
                center_y = y + h / 2
                if area >= image_area * 0.012 and h >= source.height * 0.12 and center_y >= source.height * 0.28:
                    full[np.asarray(seg, dtype=bool)] = 255
            if int(np.count_nonzero(full)) >= image_area * 0.01:
                return Image.fromarray(full, "L")
        except Exception as exc:
            print(f"[sam] product segmentation unavailable, using vision fallback: {exc}", flush=True)

    import cv2

    rgb = np.array(source.convert("RGB"), dtype=np.uint8)
    if rgb.size == 0:
        return Image.new("L", source.size, 0)
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    border_color = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - border_color.astype(np.float32), axis=2)
    saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    candidate = ((distance > max(22.0, float(np.percentile(distance, 82)))) | (saturation > max(24, int(np.percentile(saturation, 78)))))
    candidate[: int(source.height * 0.18), :] = False
    binary = candidate.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    full = np.zeros_like(binary)
    image_area = max(1, source.width * source.height)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < image_area * 0.006 or h < source.height * 0.09:
            continue
        if y + h / 2 < source.height * 0.30:
            continue
        if w > source.width * 0.88 and h < source.height * 0.18:
            continue
        full[labels == label] = 255
    full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return Image.fromarray(full, "L")


def mask_overlap_fraction(bbox: tuple[int, int, int, int], mask: Image.Image) -> float:
    left, top, right, bottom = bbox
    left = max(0, min(mask.width, left))
    right = max(0, min(mask.width, right))
    top = max(0, min(mask.height, top))
    bottom = max(0, min(mask.height, bottom))
    if right <= left or bottom <= top:
        return 0.0
    crop = np.array(mask.crop((left, top, right, bottom)).convert("L")) > 16
    return float(crop.mean()) if crop.size else 0.0


def bbox_from_luma_mask(mask: Image.Image, min_pixels: int = 64) -> tuple[int, int, int, int] | None:
    arr = np.array(mask.convert("L")) > 16
    ys, xs = np.where(arr)
    if len(xs) < min_pixels or len(ys) < min_pixels:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def component_boxes_from_luma_mask(mask: Image.Image, min_area_ratio: float = 0.004, max_components: int = 8) -> list[tuple[int, int, int, int]]:
    probe = mask.convert("L")
    scale = min(1.0, 620 / max(1, max(probe.size)))
    if scale < 1.0:
        probe = probe.resize((max(1, int(probe.width * scale)), max(1, int(probe.height * scale))), Image.Resampling.NEAREST)
    arr = np.array(probe) > 16
    visited = np.zeros(arr.shape, dtype=bool)
    height, width = arr.shape
    components: list[tuple[int, int, int, int, int]] = []
    min_pixels = max(24, int(width * height * min_area_ratio))
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not arr[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            count = 0
            while stack:
                x, y = stack.pop()
                count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and arr[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if count < min_pixels:
                continue
            components.append((count, min_x, min_y, max_x + 1, max_y + 1))
    components.sort(reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    for _count, left, top, right, bottom in components[:max_components]:
        boxes.append(
            (
                max(0, min(mask.width, int(left / max(scale, 1e-6)))),
                max(0, min(mask.height, int(top / max(scale, 1e-6)))),
                max(0, min(mask.width, int(right / max(scale, 1e-6)))),
                max(0, min(mask.height, int(bottom / max(scale, 1e-6)))),
            )
        )
    return boxes


def detect_protected_region_mask(
    source: Image.Image,
    exclusion_mask: Image.Image | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    import cv2

    rgb = np.array(source.convert("RGB"))
    if rgb.size == 0:
        empty = Image.new("L", source.size, 0)
        return empty, {"rawForegroundBoxes": [], "protectedMaskRefinementMethod": "empty", "refinementUncertain": True}

    raw_bbox = detect_foreground_bbox(source)
    if raw_bbox is None:
        empty = Image.new("L", source.size, 0)
        return empty, {"rawForegroundBoxes": [], "protectedMaskRefinementMethod": "no_bbox", "refinementUncertain": True}

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=5)
    sq_blurred = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=5)
    local_variance = np.maximum(0.0, sq_blurred - blurred * blurred)

    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    border_color = np.median(border, axis=0).astype(np.float32)
    distance = np.linalg.norm(rgb.astype(np.float32) - border_color[None, None, :], axis=2)
    smooth_background = np.logical_or(
        distance < max(18.0, float(np.percentile(distance, 52))),
        local_variance < max(26.0, float(np.percentile(local_variance, 42))),
    )

    mask = np.full(rgb.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    rect = (
        max(0, raw_bbox[0]),
        max(0, raw_bbox[1]),
        max(1, raw_bbox[2] - raw_bbox[0]),
        max(1, raw_bbox[3] - raw_bbox[1]),
    )
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    refinement_uncertain = False
    method = "grabcut+refine"
    exclusion_array = np.zeros(rgb.shape[:2], dtype=bool)
    exclusion_carve = np.zeros(rgb.shape[:2], dtype=bool)
    if exclusion_mask is not None:
        exclusion_array = np.array(exclusion_mask.resize(source.size).convert("L")) > 16
        exclusion_carve = np.logical_and(
            exclusion_array,
            smooth_background,
        )
        if np.any(exclusion_carve):
            carve_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            exclusion_carve = cv2.dilate(exclusion_carve.astype(np.uint8), carve_kernel, iterations=1).astype(bool)
            mask[exclusion_carve] = cv2.GC_BGD
            method += "+text-background-carve"
    try:
        cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        refinement_uncertain = True
        method = "bbox-threshold-fallback"
        x1, y1, x2, y2 = raw_bbox
        protected = np.zeros(rgb.shape[:2], dtype=np.uint8)
        protected[y1:y2, x1:x2] = 255
        return Image.fromarray(protected, "L"), {
            "rawForegroundBoxes": [list(raw_bbox)],
            "protectedMaskRefinementMethod": method,
            "refinementUncertain": refinement_uncertain,
        }

    protected = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    protected = cv2.morphologyEx(protected, cv2.MORPH_OPEN, kernel)
    protected = cv2.morphologyEx(protected, cv2.MORPH_CLOSE, kernel, iterations=2)
    background_like = distance < max(14.0, float(np.percentile(distance, 42)))
    protected[background_like] = 0
    if np.any(exclusion_carve):
        protected[exclusion_carve] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((protected > 16).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(protected)
    image_area = protected.shape[0] * protected.shape[1]
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, w * h)
        fill_ratio = area / bbox_area
        component_mask = labels == label
        component_texture = float(local_variance[component_mask].mean()) if np.any(component_mask) else 0.0
        if area < max(220, int(image_area * 0.00045)) and fill_ratio < 0.42:
            continue
        if component_texture < max(16.0, float(np.percentile(local_variance, 28))) and area < int(image_area * 0.01):
            continue
        if exclusion_mask is not None:
            overlap_ratio = float(np.logical_and(component_mask, exclusion_array).sum()) / max(1, area)
            pad = 8
            ring_x1 = max(0, x - pad)
            ring_y1 = max(0, y - pad)
            ring_x2 = min(protected.shape[1], x + w + pad)
            ring_y2 = min(protected.shape[0], y + h + pad)
            ring_region = np.zeros((ring_y2 - ring_y1, ring_x2 - ring_x1), dtype=bool)
            ring_region[(y - ring_y1):(y - ring_y1 + h), (x - ring_x1):(x - ring_x1 + w)] = component_mask[y:y+h, x:x+w]
            ring_mask = np.logical_and(
                np.ones_like(ring_region, dtype=bool),
                ~ring_region,
            )
            smooth_ratio = float(smooth_background[ring_y1:ring_y2, ring_x1:ring_x2][ring_mask].mean()) if np.any(ring_mask) else 0.0
            if overlap_ratio > 0.22 and smooth_ratio > 0.52 and area < int(image_area * 0.035):
                continue
        filtered[component_mask] = 255
    protected = filtered

    x1, y1, x2, y2 = raw_bbox
    if protected[y1:y2, x1:x2].sum() < ((y2 - y1) * (x2 - x1) * 255 * 0.04):
        refinement_uncertain = True
        method = "grabcut-uncertain"

    protected_image = Image.fromarray(protected, "L").filter(ImageFilter.GaussianBlur(radius=1.2))
    return protected_image, {
        "rawForegroundBoxes": [list(raw_bbox)],
        "protectedMaskRefinementMethod": method,
        "refinementUncertain": refinement_uncertain,
    }


def build_resize_focus_bbox(source: Image.Image) -> tuple[int, int, int, int]:
    detector_blocks: list[TextBlock] = []
    if env_flag("ADAPTIFAI_RESIZE_FOCUS_OCR", "0"):
        temp_path = temp_root() / f"{uuid4().hex}.png"
        try:
            source.save(temp_path, "PNG")
            detector_blocks = marketing_filter(run_trocr_ocr_on_image(temp_path))
        except Exception:
            detector_blocks = []
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        detector_blocks = []
    text_bbox = union_bbox([block.bbox for block in detector_blocks if (block.bbox[2] - block.bbox[0]) > 18 and (block.bbox[3] - block.bbox[1]) > 12])
    foreground_bbox = detect_foreground_bbox(source)
    focus = union_bbox([box for box in [text_bbox, foreground_bbox] if box is not None]) or (0, 0, source.width, source.height)
    left, top, right, bottom = focus
    pad_x = max(24, int((right - left) * 0.12))
    pad_y = max(24, int((bottom - top) * 0.12))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(source.width, right + pad_x),
        min(source.height, bottom + pad_y),
    )


def _detect_cta_button_bbox(
    source: Image.Image,
    text_blocks: list[TextBlock],
) -> tuple[int, int, int, int] | None:
    """
    Detect a CTA button region in a source image.
    Strategy: find text blocks with role='cta', then look for a solid-colour
    rectangular region immediately around them (the button background).
    Returns the expanded button bounding box or None if not found.
    """
    src_w, src_h = source.width, source.height
    cta_blocks = [b for b in text_blocks if b.role == "cta" or (b.translate and classify_text_role(b.text) == "cta")]
    if not cta_blocks:
        return None

    # Pick the most prominent CTA block (largest area among those with short text)
    best: TextBlock | None = None
    for b in cta_blocks:
        bw = b.bbox[2] - b.bbox[0]
        bh = b.bbox[3] - b.bbox[1]
        # Buttons are typically short-text, horizontally wide relative to height
        if bw > bh and len(b.text.split()) <= 6:
            if best is None or bw * bh > (best.bbox[2] - best.bbox[0]) * (best.bbox[3] - best.bbox[1]):
                best = b
    if best is None:
        best = cta_blocks[0]

    tx0, ty0, tx1, ty1 = best.bbox
    # Expand outward to include button padding (typically 30-50% of text height on each side)
    pad_h = max(8, int((ty1 - ty0) * 0.55))
    pad_w = max(16, int((tx1 - tx0) * 0.22))

    # Try to detect the button background by colour uniformity in an expanded region
    bx0 = max(0, tx0 - pad_w * 2)
    by0 = max(0, ty0 - pad_h * 2)
    bx1 = min(src_w, tx1 + pad_w * 2)
    by1 = min(src_h, ty1 + pad_h * 2)

    try:
        region = np.array(source.crop((bx0, by0, bx1, by1)).convert("RGB"), dtype=np.float32)
        # Compute per-pixel distance from region median colour
        median_color = np.median(region.reshape(-1, 3), axis=0)
        dist = np.linalg.norm(region - median_color, axis=2)
        # If >55% of pixels are close to the median, it's a uniform background (button)
        uniform_fraction = float(np.mean(dist < 40))
        if uniform_fraction > 0.55:
            return (bx0, by0, bx1, by1)
    except Exception:
        pass

    # Fallback: just return the padded text bbox
    return (
        max(0, tx0 - pad_w),
        max(0, ty0 - pad_h),
        min(src_w, tx1 + pad_w),
        min(src_h, ty1 + pad_h),
    )


def _suppress_cta_button(
    image: Image.Image,
    button_bbox: tuple[int, int, int, int],
) -> Image.Image:
    """
    Remove a CTA button from the image by filling it with the surrounding
    background colour (sampled from just outside the button edges).
    """
    bx0, by0, bx1, by1 = button_bbox
    img_w, img_h = image.size
    arr = np.array(image.convert("RGB"))

    # Sample background from a thin strip outside the button on each side
    samples: list[np.ndarray] = []
    strip = max(4, min(20, (by1 - by0) // 4))
    if by0 - strip >= 0:
        samples.append(arr[max(0, by0 - strip):by0, bx0:bx1].reshape(-1, 3))
    if by1 + strip <= img_h:
        samples.append(arr[by1:min(img_h, by1 + strip), bx0:bx1].reshape(-1, 3))
    if bx0 - strip >= 0:
        samples.append(arr[by0:by1, max(0, bx0 - strip):bx0].reshape(-1, 3))
    if bx1 + strip <= img_w:
        samples.append(arr[by0:by1, bx1:min(img_w, bx1 + strip)].reshape(-1, 3))

    if samples:
        all_samples = np.concatenate(samples, axis=0)
        fill_color = tuple(int(v) for v in np.median(all_samples, axis=0))
    else:
        fill_color = _sample_edge_color(image)

    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((bx0, by0, bx1, by1), fill=fill_color)  # type: ignore[arg-type]

    # Soften the fill boundary with a small blur patch
    try:
        import cv2  # type: ignore[import]
        result_arr = np.array(result)
        feather = max(2, (by1 - by0) // 6)
        y0f = max(0, by0 - feather)
        y1f = min(img_h, by1 + feather)
        x0f = max(0, bx0 - feather)
        x1f = min(img_w, bx1 + feather)
        patch = result_arr[y0f:y1f, x0f:x1f]
        blurred = cv2.GaussianBlur(patch, (feather * 2 + 1, feather * 2 + 1), feather / 2)
        # Only apply blur to the interior/border of the button, not the full patch
        result_arr[by0:by1, bx0:bx1] = blurred[by0 - y0f:by1 - y0f, bx0 - x0f:bx1 - x0f]
        result = Image.fromarray(result_arr)
    except Exception:
        pass

    return result


# Meta and TikTok placement prefixes â€” these platforms render their own CTA buttons
_NATIVE_CTA_PLATFORMS = {"meta_", "facebook_", "instagram_", "tiktok_"}


def _placement_has_native_cta(placement_id: str) -> bool:
    """Return True if the platform provides its own CTA button natively."""
    pid = (placement_id or "").lower()
    return any(pid.startswith(prefix) for prefix in _NATIVE_CTA_PLATFORMS)


def smart_resize_image(
    source: Image.Image,
    width: int,
    height: int,
    placement_id: str = "",
) -> Image.Image:
    """
    Smart resize pipeline:
    1. OCR to find text blocks + detect CTA button
    2. If META/TIKTOK placement: inpaint/fill the CTA button from source
    3. Detect visual foreground subject bounding box
    4. Build clean background (text removed if possible), scale to COVER target canvas
    5. Scale and reposition the foreground subject onto the canvas, preserving
       its relative anchor position (works for ALL aspect ratio changes)
    6. Re-render translated text at proportionally scaled bounding boxes

    Falls back to focus-aware cover crop on any unhandled exception.
    """
    try:
        src_w, src_h = source.width, source.height
        tgt_w, tgt_h = width, height

        # â”€â”€ Step 1: OCR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Use raw EasyOCR (not the full build_localize_blocks pipeline) to get
        # lightweight text-region bboxes for inpainting.  The full pipeline
        # can fail silently or be slow; raw OCR is fast and reliable on CPU.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        source.save(tmp_path, "PNG")
        try:
            text_blocks = build_localize_blocks(tmp_path, source)
        except Exception as _ocr_exc:
            print(f"[smart_resize_image] build_localize_blocks failed: {_ocr_exc}", flush=True)
            text_blocks = []
        finally:
            tmp_path.unlink(missing_ok=True)

        # Fallback: use raw EasyOCR bboxes if the full pipeline returned nothing
        raw_text_bboxes: list[tuple[int, int, int, int]] = []
        if not text_blocks:
            try:
                detector = load_ocr_detector()
                ocr_image, ocr_scale = fit_for_ocr(source)
                raw_results = detector.readtext(
                    np.array(ocr_image),
                    detail=1,
                    paragraph=False,
                    batch_size=1,
                    decoder="greedy",
                )
                for points, _txt, conf in raw_results:
                    if conf < 0.20:
                        continue
                    xs = [int(p[0] / ocr_scale) for p in points]
                    ys = [int(p[1] / ocr_scale) for p in points]
                    raw_text_bboxes.append((
                        max(0, min(xs)),
                        max(0, min(ys)),
                        min(source.width, max(xs)),
                        min(source.height, max(ys)),
                    ))
                print(f"[smart_resize_image] raw OCR fallback: {len(raw_text_bboxes)} bboxes", flush=True)
            except Exception as _raw_ocr_exc:
                print(f"[smart_resize_image] raw OCR fallback failed: {_raw_ocr_exc}", flush=True)

        has_translatable = any(b.translate for b in text_blocks)

        # â”€â”€ Step 2: CTA suppression for native-CTA platforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        working_source = source
        if _placement_has_native_cta(placement_id) and text_blocks:
            cta_bbox = _detect_cta_button_bbox(source, text_blocks)
            if cta_bbox is not None:
                working_source = _suppress_cta_button(source, cta_bbox)
                # Also remove the CTA block from text_blocks so it isn't re-rendered
                text_blocks = [b for b in text_blocks if b.role != "cta" and classify_text_role(b.text) != "cta"]
                has_translatable = any(b.translate for b in text_blocks)

        # â”€â”€ Step 3: Detect foreground / subject region â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Use the visual-subject-only bbox (colour distance from border).
        # build_resize_focus_bbox includes the union of text + subject, which
        # causes the entire image to be treated as "foreground" in landscape ads
        # with text on the right â€” and that text ends up visible in the result.
        subject_bbox = detect_foreground_bbox(working_source)
        if subject_bbox is None:
            subject_bbox = (0, 0, working_source.width, working_source.height)
        # Pad the subject bbox modestly (already done in build_resize_focus_bbox
        # but we skip that here; add a smaller padding to avoid clipping edges).
        _sbx0, _sby0, _sbx1, _sby1 = subject_bbox
        _pad_x = max(16, int((_sbx1 - _sbx0) * 0.08))
        _pad_y = max(16, int((_sby1 - _sby0) * 0.08))
        subject_bbox = (
            max(0, _sbx0 - _pad_x),
            max(0, _sby0 - _pad_y),
            min(working_source.width, _sbx1 + _pad_x),
            min(working_source.height, _sby1 + _pad_y),
        )
        focus_bbox = subject_bbox
        fx0, fy0, fx1, fy1 = focus_bbox
        fg_w = fx1 - fx0
        fg_h = fy1 - fy0

        # â”€â”€ Step 4: Build clean background (text removed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if has_translatable:
            try:
                cleaned, cleanup_debug = build_clean_background(working_source, text_blocks, cleanup_strength=100, return_debug=True)  # type: ignore[misc]
                gated, _ = enforce_localize_cleanup_gate(working_source, cleaned, text_blocks, cleanup_debug)
                bg_base = gated if gated is not None else cleaned
            except Exception:
                bg_base = working_source
        else:
            bg_base = working_source

        # â”€â”€ Step 4a: Fast OpenCV inpaint â€” remove text from background â”€â”€â”€â”€â”€â”€â”€
        # Run regardless of whether the full cleanup pipeline succeeded, using
        # whichever text source is available (full blocks or raw bboxes).
        # This ensures background text is gone before the cover-scale step.
        all_bboxes: list[tuple[int, int, int, int]] = (
            [b.bbox for b in text_blocks] if text_blocks else raw_text_bboxes
        )
        if all_bboxes:
            try:
                import cv2
                arr = np.array(bg_base.convert("RGB"))
                mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
                for bx0, by0, bx1, by1 in all_bboxes:
                    pad = max(3, int(min(bx1 - bx0, by1 - by0) * 0.06))
                    mask[
                        max(0, by0 - pad):min(arr.shape[0], by1 + pad),
                        max(0, bx0 - pad):min(arr.shape[1], bx1 + pad),
                    ] = 255
                if mask.any():
                    inpainted = cv2.inpaint(arr, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)
                    bg_base = Image.fromarray(inpainted)
                    print(f"[smart_resize_image] inpaint: masked {mask.any(1).sum()} rows from {len(all_bboxes)} bboxes", flush=True)
            except Exception as _e:
                print(f"[smart_resize_image] inpaint step failed: {_e}", flush=True)

        # â”€â”€ Step 4b: Scale background to COVER the target canvas â”€â”€â”€â”€â”€â”€â”€â”€â”€
        bg_color = _sample_edge_color(bg_base)
        canvas = Image.new("RGB", (tgt_w, tgt_h), bg_color)

        bg_scale = max(tgt_w / src_w, tgt_h / src_h)
        bg_resized_w = max(1, int(src_w * bg_scale))
        bg_resized_h = max(1, int(src_h * bg_scale))
        bg_resized = bg_base.resize((bg_resized_w, bg_resized_h), Image.Resampling.LANCZOS)

        # Focus-aware crop for background: keep the focal region visible
        focus_cx = (fx0 + fx1) / 2 / src_w  # 0..1
        focus_cy = (fy0 + fy1) / 2 / src_h
        bg_cx = int(focus_cx * bg_resized_w)
        bg_cy = int(focus_cy * bg_resized_h)
        bg_crop_x = max(0, min(bg_resized_w - tgt_w, bg_cx - tgt_w // 2))
        bg_crop_y = max(0, min(bg_resized_h - tgt_h, bg_cy - tgt_h // 2))
        canvas.paste(bg_resized.crop((bg_crop_x, bg_crop_y, bg_crop_x + tgt_w, bg_crop_y + tgt_h)), (0, 0))

        # â”€â”€ Step 5: Scale and reposition foreground subject â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        #
        # Goal: the foreground keeps roughly the same visual size proportion
        # relative to the new canvas, and its relative anchor (left/center/right,
        # top/center/bottom) is preserved.
        #
        # For extreme ratio changes (e.g. 4:1 â†’ 9:16) the foreground is
        # allowed to grow significantly to fill the taller canvas naturally.
        #
        tgt_fg_max_w = int(tgt_w * 0.88)
        tgt_fg_max_h = int(tgt_h * 0.88)

        # Base scale: fit the foreground within tgt_fg_max
        fg_scale_fit = min(tgt_fg_max_w / max(1, fg_w), tgt_fg_max_h / max(1, fg_h))
        # Cover scale: the scale at which the whole source would cover the target
        cover_scale = max(tgt_w / src_w, tgt_h / src_h)
        # Use whichever is larger so the subject fills the canvas better
        fg_scale = max(fg_scale_fit, cover_scale * 0.9)
        # Hard cap: don't upscale beyond 2.2Ã— (avoids extreme pixellation)
        fg_scale = min(fg_scale, 2.2)

        # Crop from bg_base (text removed) rather than working_source so that
        # text in the foreground area doesn't reappear after the paste.
        fg_crop = bg_base.crop(focus_bbox)
        new_fg_w = max(1, int(fg_w * fg_scale))
        new_fg_h = max(1, int(fg_h * fg_scale))
        fg_resized = fg_crop.resize((new_fg_w, new_fg_h), Image.Resampling.LANCZOS)

        # Anchor: use the relative center of the foreground in the source
        rel_cx = (fx0 + fx1) / 2 / src_w
        rel_cy = (fy0 + fy1) / 2 / src_h
        target_cx = int(rel_cx * tgt_w)
        target_cy = int(rel_cy * tgt_h)

        # For wideâ†’tall: bias subject slightly downward so it feels grounded
        src_ratio = src_w / max(1, src_h)
        tgt_ratio = tgt_w / max(1, tgt_h)
        if src_ratio > tgt_ratio * 1.4:
            target_cy = int(min(0.65, rel_cy + 0.08) * tgt_h)

        paste_x = max(0, min(tgt_w - new_fg_w, target_cx - new_fg_w // 2))
        paste_y = max(0, min(tgt_h - new_fg_h, target_cy - new_fg_h // 2))
        canvas.paste(fg_resized, (paste_x, paste_y))

        # â”€â”€ Step 6: Re-render text at proportionally scaled positions â”€â”€â”€â”€â”€
        if has_translatable and text_blocks:
            try:
                x_scale = tgt_w / src_w
                y_scale = tgt_h / src_h
                font_scale = min(x_scale, y_scale)
                scaled_blocks = []
                for block in text_blocks:
                    if not block.translate:
                        continue
                    bx0s, by0s, bx1s, by1s = block.bbox
                    scaled_block = block.model_copy(update={
                        "bbox": (
                            int(bx0s * x_scale),
                            int(by0s * y_scale),
                            int(bx1s * x_scale),
                            int(by1s * y_scale),
                        ),
                        "font_size_estimate": max(8, int((block.font_size_estimate or 14) * font_scale)),
                        "line_height_estimate": max(9, int((block.line_height_estimate or 16) * font_scale)),
                    })
                    scaled_blocks.append(scaled_block)
                if scaled_blocks:
                    canvas = render_translated_text(canvas, scaled_blocks, render_plan=None)
            except Exception:
                pass

        return canvas

    except Exception as _smart_resize_exc:
        import traceback as _tb
        print(f"[smart_resize_image] fallback triggered ({width}x{height}): {_smart_resize_exc}\n{_tb.format_exc()}", flush=True)
        focus = build_resize_focus_bbox(source)
        return render_resize_image(source, width, height, fit="cover", focus_bbox=focus)


def _sample_edge_color(image: Image.Image) -> tuple[int, int, int]:
    """Sample the average color from a thin border around the image edges."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    border = max(1, min(8, h // 20, w // 20))
    top = arr[:border, :, :]
    bot = arr[h - border:, :, :]
    left = arr[:, :border, :]
    right = arr[:, w - border:, :]
    sample = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3),
                              left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    mean = sample.mean(axis=0).astype(int)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def render_blurred_fit_resize(source: Image.Image, width: int, height: int) -> Image.Image:
    """
    Production resize for ad creatives.

    Preserve the full source creative without stretching or artificial subject
    recomposition. Fill any ratio mismatch with a blurred cover version of the
    same image, then place the source on top with contain fit.
    """
    source = source.convert("RGB")
    if source.width <= 0 or source.height <= 0:
        return Image.new("RGB", (width, height), (250, 249, 245))

    cover_scale = max(width / source.width, height / source.height)
    cover_size = (
        max(1, int(round(source.width * cover_scale))),
        max(1, int(round(source.height * cover_scale))),
    )
    cover = source.resize(cover_size, Image.Resampling.LANCZOS)
    crop_left = max(0, (cover.width - width) // 2)
    crop_top = max(0, (cover.height - height) // 2)
    background = cover.crop((crop_left, crop_top, crop_left + width, crop_top + height))
    blur_radius = max(12, min(width, height) // 28)
    background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    background = ImageEnhance.Brightness(background).enhance(0.92)
    background = ImageEnhance.Contrast(background).enhance(0.88)

    contain_scale = min(width / source.width, height / source.height)
    foreground_size = (
        max(1, int(round(source.width * contain_scale))),
        max(1, int(round(source.height * contain_scale))),
    )
    foreground = source.resize(foreground_size, Image.Resampling.LANCZOS)
    paste_x = (width - foreground.width) // 2
    paste_y = (height - foreground.height) // 2
    background.paste(foreground, (paste_x, paste_y))
    return background


def render_nonblur_contain_placeholder(source: Image.Image, width: int, height: int) -> Image.Image:
    source = source.convert("RGB")
    edge_color = _sample_edge_color(source)
    canvas = build_clean_gradient_panel((width, height), Image.new("RGB", (8, 8), edge_color))
    contain_scale = min(width / source.width, height / source.height)
    foreground_size = (
        max(1, int(round(source.width * contain_scale))),
        max(1, int(round(source.height * contain_scale))),
    )
    foreground = source.resize(foreground_size, Image.Resampling.LANCZOS)
    paste_x = (width - foreground.width) // 2
    paste_y = (height - foreground.height) // 2
    canvas.paste(foreground, (paste_x, paste_y))
    return canvas


def bbox1000_from_pixel_box(box: tuple[int, int, int, int], image_width: int, image_height: int) -> BBox1000:
    left, top, right, bottom = box
    return BBox1000(
        ymin=max(0, min(1000, round(top / max(1, image_height) * 1000))),
        xmin=max(0, min(1000, round(left / max(1, image_width) * 1000))),
        ymax=max(0, min(1000, round(bottom / max(1, image_height) * 1000))),
        xmax=max(0, min(1000, round(right / max(1, image_width) * 1000))),
    )


def estimate_resize_texture_complexity(source: Image.Image) -> float:
    gray = source.convert("L")
    if gray.width > 512 or gray.height > 512:
        scale = 512 / max(gray.width, gray.height)
        gray = gray.resize((max(1, int(gray.width * scale)), max(1, int(gray.height * scale))), Image.Resampling.LANCZOS)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.array(edges, dtype=np.float32) / 255.0
    return float(max(0.0, min(1.0, arr.mean() * 4.0)))


def classify_resize_background(source: Image.Image) -> BackgroundStyle:
    texture = estimate_resize_texture_complexity(source)
    edge_color = _sample_edge_color(source)
    if texture < 0.16:
        bg_type = BackgroundType.SOLID
    elif texture < 0.30:
        bg_type = BackgroundType.GRADIENT
    elif texture < 0.48:
        bg_type = BackgroundType.SOFT_BLUR
    else:
        bg_type = BackgroundType.PHOTOGRAPHIC
    return BackgroundStyle(
        type=bg_type,
        dominant_color_rgb=RGBColor(r=edge_color[0], g=edge_color[1], b=edge_color[2]),
        is_gradient=bg_type == BackgroundType.GRADIENT,
        texture_complexity=texture,
        can_extend_without_ai=texture < 0.34,
    )


def sample_marketing_text_color(source: Image.Image, bbox: tuple[int, int, int, int], fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    left, top, right, bottom = bbox
    crop = source.crop((max(0, left), max(0, top), min(source.width, right), min(source.height, bottom))).convert("RGB")
    if crop.width <= 0 or crop.height <= 0:
        return fallback
    arr = np.array(crop, dtype=np.float32).reshape(-1, 3)
    luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    chroma = arr.max(axis=1) - arr.min(axis=1)
    dark_threshold = min(150.0, float(np.percentile(luma, 38)))
    mask = (luma <= dark_threshold) & (chroma >= 18)
    if int(mask.sum()) < 16:
        mask = luma <= dark_threshold
    if int(mask.sum()) < 16:
        return fallback
    color = np.median(arr[mask], axis=0)
    return tuple(int(max(0, min(255, value))) for value in color)


def build_resize_text_exclusion_mask(source: Image.Image, blocks: list[TextBlock]) -> Image.Image:
    mask = Image.new("L", source.size, 0)
    draw = ImageDraw.Draw(mask)
    for block in blocks:
        if block.surface in {"packaging", "product"}:
            continue
        left, top, right, bottom = block.bbox
        if right <= left or bottom <= top:
            continue
        pad_x = max(4, min(24, int((right - left) * 0.08)))
        pad_y = max(3, min(18, int((bottom - top) * 0.18)))
        draw.rectangle(
            (
                max(0, left - pad_x),
                max(0, top - pad_y),
                min(source.width, right + pad_x),
                min(source.height, bottom + pad_y),
            ),
            fill=255,
        )
    return mask.filter(ImageFilter.GaussianBlur(radius=0.8))


def extract_decorative_resize_layers(
    source: Image.Image,
    product_box: tuple[int, int, int, int],
    text_blocks: list[TextBlock],
    component_boxes: list[tuple[int, int, int, int]] | None = None,
    limit: int = 3,
) -> list[VisualLayer]:
    layers: list[VisualLayer] = []
    text_boxes = [block.bbox for block in text_blocks if block.translate]
    for component_box in component_boxes or []:
        if overlap_fraction(component_box, product_box) > 0.20:
            continue
        if any(overlap_fraction(component_box, text_box) > 0.18 for text_box in text_boxes):
            continue
        layers.append(
            VisualLayer(
                id=f"decorative-{len(layers) + 1}",
                role=LayerRole.DECORATIVE,
                bbox=bbox1000_from_pixel_box(component_box, source.width, source.height),
                confidence=0.50,
                saliency=max(0.30, min(0.72, ((component_box[2] - component_box[0]) * (component_box[3] - component_box[1])) / max(1, source.width * source.height) * 4)),
                protected=False,
                notes="protected_mask_side_component importance:secondary visibility:partial theme_element:true",
            )
        )
        if len(layers) >= limit:
            return layers

    probe = source.convert("RGB")
    scale = min(1.0, 520 / max(1, max(probe.size)))
    if scale < 1.0:
        probe = probe.resize((max(1, int(probe.width * scale)), max(1, int(probe.height * scale))), Image.Resampling.LANCZOS)
    arr = np.array(probe, dtype=np.float32)
    if arr.size == 0:
        return []
    gray = np.array(probe.convert("L"), dtype=np.float32)
    edges = np.array(probe.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    edge_threshold = max(20.0, float(np.percentile(edges, 86)))
    chroma_threshold = max(18.0, float(np.percentile(chroma, 72)))
    candidate = (edges >= edge_threshold) | (chroma >= chroma_threshold)
    candidate &= gray > 18

    def scaled_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            max(0, min(probe.width, int(box[0] * scale))),
            max(0, min(probe.height, int(box[1] * scale))),
            max(0, min(probe.width, int(box[2] * scale))),
            max(0, min(probe.height, int(box[3] * scale))),
        )

    protected_boxes = [scaled_box(product_box)] + [scaled_box(block.bbox) for block in text_blocks if block.translate]
    for left, top, right, bottom in protected_boxes:
        pad_x = max(2, int((right - left) * 0.08))
        pad_y = max(2, int((bottom - top) * 0.08))
        candidate[max(0, top - pad_y): min(probe.height, bottom + pad_y), max(0, left - pad_x): min(probe.width, right + pad_x)] = False

    visited = np.zeros(candidate.shape, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []
    height, width = candidate.shape
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x] or not candidate[start_y, start_x]:
                continue
            stack = [(start_x, start_y)]
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            count = 0
            while stack:
                x, y = stack.pop()
                count += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx] and candidate[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            area_ratio = count / max(1, width * height)
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1
            if area_ratio < 0.004 or area_ratio > 0.30 or box_w < 12 or box_h < 12:
                continue
            components.append((count, min_x, min_y, max_x + 1, max_y + 1))

    components.sort(reverse=True)
    for index, (count, left, top, right, bottom) in enumerate(components[: limit * 2]):
        if len(layers) >= limit:
            break
        source_box = (
            max(0, min(source.width, int(left / max(scale, 1e-6)))),
            max(0, min(source.height, int(top / max(scale, 1e-6)))),
            max(0, min(source.width, int(right / max(scale, 1e-6)))),
            max(0, min(source.height, int(bottom / max(scale, 1e-6)))),
        )
        if overlap_fraction(source_box, product_box) > 0.12:
            continue
        layers.append(
            VisualLayer(
                id=f"decorative-{len(layers) + 1}",
                role=LayerRole.DECORATIVE,
                bbox=bbox1000_from_pixel_box(source_box, source.width, source.height),
                confidence=0.42,
                saliency=max(0.25, min(0.68, count / max(1, width * height) * 8)),
                protected=False,
                notes="heuristic_side_or_theme_element importance:secondary visibility:partial theme_element:true",
            )
        )
    return layers


def _smart_reframe_analysis_prompt(target_language: str) -> str:
    return (
        build_visual_analysis_prompt(target_language)
        + " Return the dimensions of the original source image, not the resized analysis image. "
        + "Be conservative: only mark printed product labels/logos as non-translatable protected layers, and keep marketing overlay text separate from product layers. "
        + "For resize, explicitly identify secondary visual theme elements in other_layers so the planner can recompose them around the protected product instead of cropping the source."
    )


def _prepare_smart_reframe_probe(source: Image.Image) -> tuple[Image.Image, str]:
    probe = source.convert("RGB")
    max_side = int(os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_MAX_SIDE", "1600"))
    if max(probe.size) > max_side:
        scale = max_side / max(probe.size)
        probe = probe.resize((max(1, int(probe.width * scale)), max(1, int(probe.height * scale))), Image.Resampling.LANCZOS)
    encoded_io = io.BytesIO()
    probe.save(encoded_io, format="PNG")
    encoded = base64.b64encode(encoded_io.getvalue()).decode("utf-8")
    return probe, encoded


def _finalize_smart_reframe_analysis(analysis: VisualAnalysis, source: Image.Image, provider_name: str) -> VisualAnalysis:
    if not analysis.product_layers and not analysis.logo_layers and not analysis.other_layers:
        raise ValueError(f"{provider_name} analysis returned no actionable visual layers")
    source_ratio = source.width / max(1, source.height)
    if source_ratio > 1.35 and analysis.product_layers:
        usable_products = []
        for layer in analysis.product_layers:
            left, top, right, bottom = layer.bbox.to_pixel_box(source.width, source.height)
            height_ratio = (bottom - top) / max(1, source.height)
            center_y = (top + bottom) / 2
            if height_ratio >= 0.38 and center_y >= source.height * 0.24:
                usable_products.append(layer)
        if not usable_products:
            raise ValueError(f"{provider_name} analysis product boxes are label/top fragments, not full visual subjects")
    if analysis.source_width != source.width or analysis.source_height != source.height:
        analysis = analysis.model_copy(update={"source_width": source.width, "source_height": source.height})
    warnings = [*analysis.quality_warnings, f"analysis_provider:{provider_name}"]
    return analysis.model_copy(update={"quality_warnings": warnings})


def build_openai_smart_reframe_analysis(source: Image.Image, target_language: str = "EN") -> VisualAnalysis | None:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return None

    provider = os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_PROVIDER", "auto").strip().lower()
    if provider in {"heuristic", "local", "off", "disabled"}:
        return None

    last_error = ""
    for attempt in range(2):
        try:
            _probe, encoded = _prepare_smart_reframe_probe(source)
            client = OpenAI(timeout=float(os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_TIMEOUT", "12")))
            feedback = (
                "\nPREVIOUS JSON FAILED BACKEND VALIDATION. Fix exactly this issue before returning JSON: "
                f"{last_error}\n"
            ) if attempt and last_error else ""
            response = client.chat.completions.create(
                model=os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_MODEL", os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o")),
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": _smart_reframe_analysis_prompt(target_language),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Original source image size is {source.width}x{source.height}. "
                                    "Analyze product/person/logo/text/background layers for resizing. "
                                    "Return strict JSON only with complete, non-fragment visual subject boxes."
                                    f"{feedback}"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            analysis = parse_visual_analysis_payload(payload)
            return _finalize_smart_reframe_analysis(analysis, source, "openai")
        except Exception as exc:
            last_error = str(exc)[:320]
            if attempt == 0:
                print(f"[smart_reframe_analysis] openai validation failed, retrying once: {last_error}", flush=True)
                continue
            print(f"[smart_reframe_analysis] openai analysis failed after retry: {last_error}", flush=True)
            return None
    return None


def build_openrouter_smart_reframe_analysis(source: Image.Image, target_language: str = "EN") -> VisualAnalysis | None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    provider = os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_PROVIDER", "auto").strip().lower()
    if provider in {"heuristic", "local", "off", "disabled", "openai", "gemini", "google"}:
        return None

    last_error = ""
    for attempt in range(2):
        try:
            _probe, encoded = _prepare_smart_reframe_probe(source)
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
            model = (
                os.getenv("ADAPTIFAI_OPENROUTER_SMART_REFRAME_MODEL")
                or os.getenv("ADAPTIFAI_OPENROUTER_MODEL")
                or os.getenv("OPENROUTER_MODEL")
                or "google/gemini-2.5-flash"
            ).strip()
            headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://adaptifai.sasmaz.digital"),
                "X-Title": os.getenv("OPENROUTER_APP_TITLE", "AdaptifAI"),
            }
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=float(os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_TIMEOUT", "12")),
                default_headers={key: value for key, value in headers.items() if value},
            )
            feedback = (
                "\nPREVIOUS JSON FAILED BACKEND VALIDATION. Fix exactly this issue before returning JSON: "
                f"{last_error}\n"
            ) if attempt and last_error else ""
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": _smart_reframe_analysis_prompt(target_language),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Original source image size is {source.width}x{source.height}. "
                                    "Analyze product/person/logo/text/background layers for resizing. "
                                    "Return strict JSON only with complete, non-fragment visual subject boxes."
                                    f"{feedback}"
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        ],
                    },
                ],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            analysis = parse_visual_analysis_payload(payload)
            return _finalize_smart_reframe_analysis(analysis, source, "openrouter")
        except Exception as exc:
            last_error = str(exc)[:320]
            if attempt == 0:
                print(f"[smart_reframe_analysis] openrouter validation failed, retrying once: {last_error}", flush=True)
                continue
            print(f"[smart_reframe_analysis] openrouter analysis failed after retry: {last_error}", flush=True)
            return None
    return None


def build_gemini_smart_reframe_analysis(source: Image.Image, target_language: str = "EN") -> VisualAnalysis | None:
    api_key = google_gemini_api_key()
    if not api_key:
        return None

    provider = os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_PROVIDER", "auto").strip().lower()
    if provider in {"heuristic", "local", "off", "disabled", "openai"}:
        return None

    last_error = ""
    for attempt in range(2):
        try:
            _probe, encoded = _prepare_smart_reframe_probe(source)
            model = os.getenv("GEMINI_VISUAL_ANALYSIS_MODEL", os.getenv("ADAPTIFAI_GEMINI_VISUAL_ANALYSIS_MODEL", "gemini-2.5-pro")).strip() or "gemini-2.5-pro"
            feedback = (
                "\nPREVIOUS JSON FAILED BACKEND VALIDATION. Fix exactly this issue before returning JSON: "
                f"{last_error}\n"
            ) if attempt and last_error else ""
            prompt = _smart_reframe_analysis_prompt(target_language) + " Return strict JSON only with complete, non-fragment visual subject boxes." + feedback
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": f"Original source image size is {source.width}x{source.height}. Analyze product/person/logo/text/background layers for resizing.\n\n{prompt}"},
                                {"inline_data": {"mime_type": "image/png", "data": encoded}},
                            ],
                        }
                    ],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
                },
                timeout=int(os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_TIMEOUT", "12")),
            )
            response.raise_for_status()
            payload = response.json()
            candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            content = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
            parsed = extract_json_object(content)
            analysis = parse_visual_analysis_payload(parsed)
            return _finalize_smart_reframe_analysis(analysis, source, "gemini")
        except Exception as exc:
            last_error = str(exc)[:320]
            if attempt == 0:
                print(f"[smart_reframe_analysis] gemini validation failed, retrying once: {last_error}", flush=True)
                continue
            print(f"[smart_reframe_analysis] gemini analysis failed after retry: {last_error}", flush=True)
            return None
    return None


def build_heuristic_smart_reframe_analysis(source: Image.Image, focus_bbox: tuple[int, int, int, int] | None = None, fallback_reason: str = "") -> VisualAnalysis:
    text_layers: list[TextLayer] = []
    detector_blocks: list[TextBlock] = []
    temp_path = temp_root() / f"{uuid4().hex}-resize-analysis.png"
    try:
        source.save(temp_path, "PNG")
        detector_blocks = run_resize_raw_ocr_on_image(temp_path)
        if (
            not detector_blocks
            and os.getenv("ADAPTIFAI_ALLOW_LOCAL_RESIZE_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}
        ):
            detector_blocks = run_trocr_ocr_on_image(temp_path)
        detector_blocks = marketing_filter(detector_blocks)
        for index, block in enumerate(detector_blocks[:16]):
            sampled_color = sample_marketing_text_color(
                source,
                block.bbox,
                hex_to_rgb(block.color if block.color.startswith("#") else "#111111"),
            )
            text_layers.append(
                TextLayer(
                    id=f"text-{index + 1}",
                    bbox=bbox1000_from_pixel_box(block.bbox, source.width, source.height),
                    confidence=0.58,
                    saliency=0.62,
                    protected=True,
                    original_text=block.text,
                    translated_text=block.translated_text or block.text,
                    translate=True,
                    text_style=TextStyle(
                        color_rgb=RGBColor.from_list(sampled_color),
                        is_bold=block.font_weight >= 700,
                        font_type="sans-serif",
                        estimated_font_size=block.font_size_estimate,
                        uppercase=block.text.isupper(),
                        alignment=block.align if block.align in {"left", "center", "right"} else "unknown",
                    ),
                )
            )
    except Exception:
        text_layers = []
    finally:
        temp_path.unlink(missing_ok=True)

    text_exclusion_mask = build_resize_text_exclusion_mask(source, detector_blocks)
    protected_mask, protected_meta = detect_protected_region_mask(source, exclusion_mask=text_exclusion_mask)
    protected_box = bbox_from_luma_mask(protected_mask)
    protected_components = component_boxes_from_luma_mask(protected_mask)
    selected_component: tuple[int, int, int, int] | None = None
    if focus_bbox and protected_components:
        selected_component = max(protected_components, key=lambda box: overlap_fraction(box, focus_bbox))
        if overlap_fraction(selected_component, focus_bbox) < 0.08:
            selected_component = None
    if selected_component is None and protected_components:
        selected_component = max(protected_components, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    foreground_box = selected_component or protected_box or detect_foreground_bbox(source)
    product_box = foreground_box or focus_bbox or (0, 0, source.width, source.height)
    product_layer = ProductLayer(
        id="product-1",
        bbox=bbox1000_from_pixel_box(product_box, source.width, source.height),
        confidence=0.74 if protected_box else 0.50 if foreground_box else 0.38,
        saliency=0.84,
        protected=True,
        mask_quality="rough_mask" if protected_box else "bbox_only",
        needs_shadow=True,
    )
    decorative_layers = extract_decorative_resize_layers(source, product_box, detector_blocks, protected_components)

    return VisualAnalysis(
        source_width=source.width,
        source_height=source.height,
        background=classify_resize_background(source),
        product_layers=[product_layer],
        text_layers=text_layers,
        other_layers=decorative_layers,
        saliency_summary="Heuristic resize analysis: protected foreground subject, marketing OCR boxes, background texture, and secondary decorative/theme components for recompose planning.",
        quality_warnings=[
            item
            for item in [
                "analysis_provider:heuristic",
                f"protected_mask:{protected_meta.get('protectedMaskRefinementMethod', 'unknown')}",
                "protected_mask_uncertain" if protected_meta.get("refinementUncertain") else "",
                f"protected_components:{len(protected_components)}",
                f"decorative_layers:{len(decorative_layers)}",
                fallback_reason,
            ]
            if item
        ],
    )


def build_smart_reframe_analysis(source: Image.Image, focus_bbox: tuple[int, int, int, int] | None = None, target_language: str = "EN") -> VisualAnalysis:
    provider = os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_PROVIDER", "auto").strip().lower()
    openrouter_configured = bool(os.getenv("OPENROUTER_API_KEY", "").strip())
    if provider == "auto" and openrouter_configured:
        analysis = build_openrouter_smart_reframe_analysis(source, target_language)
        if analysis is not None:
            return analysis
        return build_heuristic_smart_reframe_analysis(source, focus_bbox, "openrouter_auto_strict_analysis_failed_fallback")
    if provider in {"auto", "openai"}:
        analysis = build_openai_smart_reframe_analysis(source, target_language)
        if analysis is not None:
            return analysis
        if provider == "openai":
            return build_heuristic_smart_reframe_analysis(source, focus_bbox, "openai_analysis_failed_fallback")
    if provider in {"auto", "openrouter"}:
        analysis = build_openrouter_smart_reframe_analysis(source, target_language)
        if analysis is not None:
            return analysis
        if provider == "openrouter":
            return build_heuristic_smart_reframe_analysis(source, focus_bbox, "openrouter_analysis_failed_fallback")
    if provider in {"auto", "gemini", "google"}:
        analysis = build_gemini_smart_reframe_analysis(source, target_language)
        if analysis is not None:
            return analysis
        if provider in {"gemini", "google"}:
            return build_heuristic_smart_reframe_analysis(source, focus_bbox, "gemini_analysis_failed_fallback")
    return build_heuristic_smart_reframe_analysis(source, focus_bbox)


def render_hybrid_banner_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> Image.Image:
    product_box = relayout_product_box(source, analysis)
    canvas, mapped_product, _crop_meta = build_content_aware_reframe_canvas(source, width, height, product_box)
    copy_width = max(int(width * 0.50), min(int(width * 0.68), mapped_product[0] - max(8, int(width * 0.02))))
    if copy_width <= int(width * 0.34):
        copy_width = int(width * 0.58)
    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=max(3, height // 8)))
    left_sample = np.array(blurred.crop((0, 0, max(1, min(width, copy_width)), height)).convert("RGB"), dtype=np.float32)
    tint = tuple(int(value) for value in np.percentile(left_sample.reshape(-1, 3), 76, axis=0))
    wash = Image.new("RGB", canvas.size, tint)
    softened = Image.blend(blurred, wash, 0.68)
    mask = Image.new("L", canvas.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    fade_end = min(width, copy_width + max(20, int(width * 0.08)))
    for x in range(fade_end):
        alpha = 185 if x <= copy_width else int(185 * (1 - (x - copy_width) / max(1, fade_end - copy_width)) ** 2)
        mask_draw.line((x, 0, x, height), fill=alpha)
    canvas = Image.composite(softened, canvas, mask)
    draw = ImageDraw.Draw(canvas)

    text_candidates = analysis.marketing_text_layers or analysis.text_layers
    text = " ".join(layer.original_text for layer in text_candidates[:2]).strip()
    if text:
        text_area_w = max(1, copy_width - max(10, int(width * 0.05)))
        text_area_h = max(1, height - max(8, int(height * 0.14)))
        fill = (18, 28, 45)
        first_style = text_candidates[0].text_style if text_candidates else None
        if first_style:
            fill = first_style.color_rgb.as_tuple()
        display_text = text.upper() if first_style and first_style.uppercase else text
        lines, font_size, line_height, _overflow = fit_plain_text_to_box(
            draw,
            display_text,
            text_area_w,
            text_area_h,
            max_font_size=max(10, min(38, int(height * 0.52))),
            min_font_size=8,
            bold=True if first_style is None else first_style.is_bold,
        )
        font = get_font(font_size, bold=True if first_style is None else first_style.is_bold)
        total_h = max(1, len(lines) * line_height)
        y = max(2, (height - total_h) // 2)
        x = max(6, int(width * 0.035))
        for line in lines:
            draw.text((x, y), line, fill=fill, font=font, stroke_width=1, stroke_fill=(232, 238, 244))
            y += line_height

    return canvas


def wrap_plain_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    paragraphs = [part.strip() for part in text.replace("\r", "\n").split("\n")]
    for paragraph in paragraphs:
        if not paragraph:
            continue
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if text_width(draw, candidate, font) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if text_width(draw, word, font) <= max_width:
                current = word
                continue
            chunk = ""
            for char in word:
                candidate_chunk = f"{chunk}{char}"
                if chunk and text_width(draw, candidate_chunk, font) > max_width:
                    lines.append(chunk)
                    chunk = char
                else:
                    chunk = candidate_chunk
            current = chunk
        if current:
            lines.append(current)
    return lines


def fit_plain_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    max_height: int,
    *,
    max_font_size: int,
    min_font_size: int = 8,
    bold: bool = True,
) -> tuple[list[str], int, int, bool]:
    text = " ".join(text.split())
    if not text:
        return [], min_font_size, int(min_font_size * 1.12), False

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = get_font(font_size, bold=bold)
        line_height = max(9, int(font_size * 1.12))
        words_fit = all(text_width(draw, word, font) <= max_width for word in text.split())
        lines = wrap_plain_text_to_width(draw, text, font, max_width)
        if lines and words_fit and len(lines) * line_height <= max_height:
            return lines, font_size, line_height, False

    font = get_font(min_font_size, bold=bold)
    line_height = max(9, int(min_font_size * 1.12))
    lines = wrap_plain_text_to_width(draw, text, font, max_width)
    max_lines = max(1, max_height // line_height)
    overflow = len(lines) > max_lines
    lines = lines[:max_lines]
    if overflow and lines:
        suffix = "..."
        while lines[-1] and text_width(draw, f"{lines[-1]}{suffix}", font) > max_width:
            lines[-1] = lines[-1][:-1].rstrip()
        lines[-1] = f"{lines[-1]}{suffix}" if lines[-1] else suffix
    return lines, min_font_size, line_height, overflow


def clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(value))))


def build_content_aware_reframe_canvas(
    source: Image.Image,
    width: int,
    height: int,
    product_box: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int], dict[str, Any]]:
    source = source.convert("RGB")
    target_ratio = width / max(1, height)
    source_ratio = source.width / max(1, source.height)
    product_left, product_top, product_right, product_bottom = product_box
    product_center_x = (product_left + product_right) / 2
    product_center_y = (product_top + product_bottom) / 2

    if source_ratio > target_ratio:
        crop_h = source.height
        crop_w = max(1, min(source.width, int(round(crop_h * target_ratio))))
        if target_ratio < 0.55:
            focus_x = product_left * 0.65 + product_center_x * 0.35
        else:
            focus_x = product_center_x
        crop_left = clamp_int(focus_x - crop_w * 0.5, 0, source.width - crop_w)
        crop_top = 0
    else:
        crop_w = source.width
        crop_h = max(1, min(source.height, int(round(crop_w / max(0.01, target_ratio)))))
        if target_ratio > 3:
            focus_y = product_top * 0.65 + product_center_y * 0.35
        else:
            focus_y = product_center_y
        crop_left = 0
        crop_top = clamp_int(focus_y - crop_h * 0.5, 0, source.height - crop_h)

    crop_box = (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
    cropped = source.crop(crop_box)
    canvas = cropped.resize((width, height), Image.Resampling.LANCZOS)
    scale_x = width / max(1, crop_w)
    scale_y = height / max(1, crop_h)
    mapped_product = (
        clamp_int((product_left - crop_left) * scale_x, 0, width),
        clamp_int((product_top - crop_top) * scale_y, 0, height),
        clamp_int((product_right - crop_left) * scale_x, 0, width),
        clamp_int((product_bottom - crop_top) * scale_y, 0, height),
    )
    return canvas, mapped_product, {
        "sourceCropBox": list(crop_box),
        "mappedProductBox": list(mapped_product),
        "sourceScaleX": round(scale_x, 4),
        "sourceScaleY": round(scale_y, 4),
    }


def apply_copy_area_feather(
    canvas: Image.Image,
    *,
    copy_bottom: int,
    strength: int = 205,
    blur_radius: int = 12,
) -> Image.Image:
    def cool_ad_tint(values: np.ndarray) -> np.ndarray:
        adjusted = values.astype(np.float32).copy()
        adjusted[0] = adjusted[0] * 0.96 + 4
        adjusted[1] = adjusted[1] * 1.02 + 6
        adjusted[2] = adjusted[2] * 1.08 + 10
        return np.clip(adjusted, 0, 255)

    copy_bottom = max(1, min(canvas.height, copy_bottom))
    blurred = canvas.filter(ImageFilter.GaussianBlur(radius=max(blur_radius, 18)))
    top_sample = np.array(blurred.crop((0, 0, canvas.width, max(1, min(canvas.height, copy_bottom)))).convert("RGB"), dtype=np.float32)
    top_tint = cool_ad_tint(np.percentile(top_sample.reshape(-1, 3), 82, axis=0))
    bottom_start = max(0, copy_bottom - max(8, canvas.height // 14))
    bottom_sample = np.array(blurred.crop((0, bottom_start, canvas.width, copy_bottom)).convert("RGB"), dtype=np.float32)
    bottom_tint = cool_ad_tint(np.percentile(bottom_sample.reshape(-1, 3), 76, axis=0))
    wash = Image.new("RGB", canvas.size, tuple(int(value) for value in top_tint))
    wash_draw = ImageDraw.Draw(wash)
    for y in range(canvas.height):
        t = min(1.0, y / max(1, copy_bottom))
        color = tuple(int(top_tint[index] * (1 - t) + bottom_tint[index] * t) for index in range(3))
        wash_draw.line((0, y, canvas.width, y), fill=color)
    softened = wash
    mask = Image.new("L", canvas.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    fade_end = min(canvas.height, copy_bottom + max(32, int(canvas.height * 0.08)))
    for y in range(fade_end):
        if y <= copy_bottom:
            alpha = 255
        else:
            t = (y - copy_bottom) / max(1, fade_end - copy_bottom)
            alpha = int(strength * (1 - t) ** 2)
        draw_mask.line((0, y, canvas.width, y), fill=max(0, min(255, alpha)))
    return Image.composite(softened, canvas, mask)


def relayout_product_box(source: Image.Image, analysis: VisualAnalysis) -> tuple[int, int, int, int]:
    product_layer = analysis.product_layers[0] if analysis.product_layers else None
    product_box = product_layer.bbox.to_pixel_box(source.width, source.height) if product_layer else (0, 0, source.width, source.height)
    box_area = max(1, product_box[2] - product_box[0]) * max(1, product_box[3] - product_box[1])
    image_area = max(1, source.width * source.height)
    if box_area / image_area <= 0.72:
        return product_box

    text_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.text_layers]
    if text_boxes:
        text_bottom = max(box[3] for box in text_boxes)
        return (
            int(source.width * 0.38),
            min(source.height - 1, text_bottom + int(source.height * 0.045)),
            source.width,
            source.height,
        )
    return (
        int(source.width * 0.20),
        int(source.height * 0.42),
        source.width,
        source.height,
    )


def build_clean_gradient_panel(size: tuple[int, int], sample_image: Image.Image, *, vertical: bool = True) -> Image.Image:
    width, height = size
    sample = np.array(sample_image.convert("RGB"), dtype=np.float32)
    if sample.size == 0:
        top_tint = np.array([220, 230, 244], dtype=np.float32)
        bottom_tint = np.array([198, 214, 232], dtype=np.float32)
    else:
        pixels = sample.reshape(-1, 3)
        top_tint = np.percentile(pixels, 86, axis=0)
        bottom_tint = np.percentile(pixels, 62, axis=0)
    top_tint = np.clip([top_tint[0] * 0.96 + 4, top_tint[1] * 1.02 + 6, top_tint[2] * 1.10 + 12], 0, 255)
    bottom_tint = np.clip([bottom_tint[0] * 0.96 + 4, bottom_tint[1] * 1.02 + 6, bottom_tint[2] * 1.08 + 10], 0, 255)
    panel = Image.new("RGB", (width, height), tuple(int(value) for value in top_tint))
    draw_panel = ImageDraw.Draw(panel)
    steps = height if vertical else width
    for index in range(max(1, steps)):
        t = index / max(1, steps - 1)
        color = tuple(int(top_tint[channel] * (1 - t) + bottom_tint[channel] * t) for channel in range(3))
        if vertical:
            draw_panel.line((0, index, width, index), fill=color)
        else:
            draw_panel.line((index, 0, index, height), fill=color)
    return panel


def build_bottom_locked_hero(
    source: Image.Image,
    width: int,
    height: int,
    hero_top: int,
    product_box: tuple[int, int, int, int],
    min_source_y: int = 0,
    min_source_x: int = 0,
) -> tuple[Image.Image, dict[str, Any]]:
    hero_h = max(1, height - hero_top)
    target_ratio = width / max(1, hero_h)
    product_left, product_top, product_right, product_bottom = product_box
    product_center_x = product_left * 0.55 + ((product_left + product_right) / 2) * 0.45
    min_source_y = max(0, min(source.height - 1, min_source_y))
    available_h = max(1, source.height - min_source_y)
    crop_h = available_h
    crop_w = max(1, int(round(crop_h * target_ratio)))
    if crop_w > source.width:
        crop_w = source.width
        crop_h = max(1, min(source.height, int(round(crop_w / max(0.01, target_ratio)))))
    min_source_x = max(0, min(source.width - crop_w, min_source_x))
    crop_left = clamp_int(product_center_x - crop_w * 0.5, min_source_x, source.width - crop_w)
    desired_top = max(min_source_y, product_top - crop_h * 0.22)
    crop_top = clamp_int(desired_top, min_source_y, source.height - crop_h)
    if product_bottom > crop_top + crop_h:
        crop_top = clamp_int(product_bottom - crop_h, min_source_y, source.height - crop_h)
    crop_box = (crop_left, crop_top, crop_left + crop_w, crop_top + crop_h)
    hero = source.crop(crop_box).resize((width, hero_h), Image.Resampling.LANCZOS)
    return hero, {
        "heroTop": hero_top,
        "heroCropBox": list(crop_box),
        "heroHeight": hero_h,
        "heroMinSourceY": min_source_y,
        "heroMinSourceX": min_source_x,
    }


def build_integrated_vertical_reframe(
    source: Image.Image,
    width: int,
    height: int,
    product_box: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[int, int, int, int], dict[str, Any]]:
    target_ratio = width / max(1, height)
    crop_h = source.height
    crop_w = max(1, min(source.width, int(round(crop_h * target_ratio))))
    product_left, product_top, product_right, product_bottom = product_box
    product_center_x = (product_left + product_right) / 2
    focus_x = product_left * 0.42 + product_center_x * 0.58
    crop_left = clamp_int(focus_x - crop_w * 0.5, 0, source.width - crop_w)
    crop_box = (crop_left, 0, crop_left + crop_w, source.height)
    canvas = source.crop(crop_box).resize((width, height), Image.Resampling.LANCZOS)
    scale_x = width / max(1, crop_w)
    scale_y = height / max(1, crop_h)
    mapped_product = (
        clamp_int((product_left - crop_left) * scale_x, 0, width),
        clamp_int(product_top * scale_y, 0, height),
        clamp_int((product_right - crop_left) * scale_x, 0, width),
        clamp_int(product_bottom * scale_y, 0, height),
    )
    return canvas, mapped_product, crop_box, {
        "sourceCropBox": list(crop_box),
        "mappedProductBox": list(mapped_product),
        "sourceScaleX": round(scale_x, 4),
        "sourceScaleY": round(scale_y, 4),
    }


def build_clean_source_texture_fill(
    source: Image.Image,
    width: int,
    height: int,
    text_source_boxes: list[tuple[int, int, int, int]],
) -> tuple[Image.Image, dict[str, Any]]:
    text_bottom = max((box[3] for box in text_source_boxes), default=int(source.height * 0.38))
    text_right = max((box[2] for box in text_source_boxes), default=0)
    sample_top = 0
    sample_bottom = min(source.height, max(int(source.height * 0.38), text_bottom + int(source.height * 0.08)))
    sample_left = min(source.width - 1, text_right + int(source.width * 0.035))
    if source.width - sample_left < max(80, int(source.width * 0.14)):
        sample_left = int(source.width * 0.70)
    sample_box = (
        max(0, min(source.width - 1, sample_left)),
        sample_top,
        source.width,
        max(sample_top + 1, sample_bottom),
    )
    texture = source.crop(sample_box).convert("RGB")
    if texture.width < 4 or texture.height < 4:
        texture = source.crop((int(source.width * 0.62), 0, source.width, max(1, sample_bottom))).convert("RGB")
        sample_box = (int(source.width * 0.62), 0, source.width, max(1, sample_bottom))
    texture = ImageOps.fit(texture, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    texture = texture.filter(ImageFilter.GaussianBlur(radius=1.2))
    try:
        import cv2

        arr = np.array(texture.convert("RGB"), dtype=np.uint8)
        float_arr = arr.astype(np.float32)
        luma = float_arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        chroma = float_arr.max(axis=2) - float_arr.min(axis=2)
        dark_cutoff = min(150.0, float(np.percentile(luma, 36)))
        mask = ((luma < dark_cutoff) & (chroma > 16)).astype(np.uint8) * 255
        if int(np.count_nonzero(mask)) > max(8, width * height * 0.01):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.dilate(mask, kernel, iterations=2)
            cleaned = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
            texture = Image.fromarray(cleaned, "RGB").filter(ImageFilter.GaussianBlur(radius=0.4))
    except Exception:
        pass
    return texture, {"textureSampleBox": list(sample_box)}


def build_protected_hero_region(source: Image.Image, analysis: VisualAnalysis, product_box: tuple[int, int, int, int], target_width: int) -> tuple[int, int, int, int]:
    text_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.text_layers]
    product_left, product_top, product_right, product_bottom = product_box
    product_w = max(1, product_right - product_left)
    product_h = max(1, product_bottom - product_top)
    product_area_ratio = (product_w * product_h) / max(1, source.width * source.height)
    raw_product_area_ratio = max((layer.bbox.area_ratio() for layer in analysis.product_layers), default=product_area_ratio)
    left = product_left - int(product_w * 0.48)
    right = product_right + int(product_w * 0.34)
    top = product_top - int(product_h * 0.28)
    if raw_product_area_ratio > 0.68 and not text_boxes:
        return (
            max(0, min(source.width - 2, int(source.width * 0.44))),
            max(0, min(source.height - 2, int(source.height * 0.44))),
            source.width,
            source.height,
        )
    bottom = source.height
    if target_width <= 180:
        desired_width = int((bottom - top) * 0.54)
        center = int((product_box[0] + product_box[2]) * 0.5)
        left = clamp_int(center - desired_width * 0.60, max(0, product_left - desired_width), max(0, source.width - desired_width))
        right = min(source.width, left + desired_width)
    min_width = max(product_w + int(source.width * 0.08), int(source.width * (0.24 if target_width <= 180 else 0.30)))
    if right - left < min_width:
        center = (product_left + product_right) // 2
        left = center - min_width // 2
        right = left + min_width
    left = clamp_int(left, 0, max(0, source.width - min_width))
    right = clamp_int(right, left + 2, source.width)
    top = clamp_int(top, int(source.height * 0.32), max(int(source.height * 0.32), source.height - 2))

    crop_box = (left, top, right, bottom)
    for text_box in text_boxes:
        if overlap_fraction(text_box, crop_box) <= 0.08:
            continue
        shift_right = min(source.width - (right - left), text_box[2] + int(source.width * 0.035))
        shifted = (max(0, shift_right), top, max(0, shift_right) + (right - left), bottom)
        if shifted[2] <= source.width and overlap_fraction(text_box, shifted) < overlap_fraction(text_box, crop_box):
            crop_box = shifted
            left, top, right, bottom = crop_box
    return (
        max(0, min(source.width - 2, left)),
        max(0, min(source.height - 2, top)),
        right,
        bottom,
    )


def paste_contained_hero(
    canvas: Image.Image,
    source: Image.Image,
    hero_region: tuple[int, int, int, int],
    hero_top: int,
    *,
    cleanup_overlay_text: bool = True,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    hero_h = max(1, canvas.height - hero_top)
    hero_w = canvas.width
    hero_source = source.crop(hero_region).convert("RGB")
    if cleanup_overlay_text:
        hero_source, cleanup_meta = suppress_large_overlay_text_in_hero_crop(hero_source)
    else:
        cleanup_meta = {"heroTextCleanup": "skipped_clean_crop"}
    scale = min(hero_w / max(1, hero_source.width), hero_h / max(1, hero_source.height))
    hero_size = (
        max(1, int(round(hero_source.width * scale))),
        max(1, int(round(hero_source.height * scale))),
    )
    hero = hero_source.resize(hero_size, Image.Resampling.LANCZOS)
    hero_rgba = hero.convert("RGBA")
    alpha = Image.new("L", hero.size, 255)
    alpha_draw = ImageDraw.Draw(alpha)
    fade_h = min(hero.height, max(10, int(canvas.height * 0.035)))
    for y in range(fade_h):
        alpha_draw.line((0, y, hero.width, y), fill=int(255 * y / max(1, fade_h - 1)))
    hero_rgba.putalpha(alpha)
    paste_x = (hero_w - hero.width) // 2
    paste_y = hero_top + max(0, hero_h - hero.height)
    canvas.paste(hero_rgba.convert("RGB"), (paste_x, paste_y), hero_rgba)
    return (paste_x, paste_y, paste_x + hero.width, paste_y + hero.height), {
        "heroRegion": list(hero_region),
        "heroPasteBox": [paste_x, paste_y, paste_x + hero.width, paste_y + hero.height],
        "heroScale": round(scale, 4),
        "heroTop": hero_top,
        **cleanup_meta,
    }


def suppress_large_overlay_text_in_hero_crop(hero_source: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    try:
        import cv2

        arr = np.array(hero_source.convert("RGB"), dtype=np.uint8)
        height, width = arr.shape[:2]
        if width < 40 or height < 40:
            return hero_source, {"heroTextCleanup": "skipped_small_crop"}
        float_arr = arr.astype(np.float32)
        luma = float_arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        chroma = float_arr.max(axis=2) - float_arr.min(axis=2)
        upper_band = np.zeros((height, width), dtype=bool)
        upper_band[: int(height * 0.66), :] = True
        dark_text = (luma < 118) & (chroma > 24) & upper_band
        mask = np.zeros((height, width), dtype=np.uint8)
        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark_text.astype(np.uint8), 8)
        kept_components = 0
        min_area = max(28, int(width * height * 0.00045))
        for label in range(1, _count):
            x, y, comp_w, comp_h, area = stats[label]
            if area < min_area:
                continue
            if comp_w < max(10, int(width * 0.025)) or comp_h < max(6, int(height * 0.018)):
                continue
            if comp_h > int(height * 0.34) and comp_w < int(width * 0.12):
                continue
            mask[labels == label] = 255
            kept_components += 1
        if kept_components == 0:
            return hero_source, {"heroTextCleanup": "no_large_overlay_components"}
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask_ratio = int(np.count_nonzero(mask)) / max(1, width * height)
        if mask_ratio > 0.08:
            return hero_source, {
                "heroTextCleanup": "skipped_large_mask",
                "heroTextCleanupComponents": kept_components,
                "heroTextCleanupMaskRatio": round(mask_ratio, 4),
            }
        cleaned = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)
        return Image.fromarray(cleaned, "RGB"), {
            "heroTextCleanup": "large_overlay_text_inpaint",
            "heroTextCleanupComponents": kept_components,
            "heroTextCleanupMaskPixels": int(np.count_nonzero(mask)),
        }
    except Exception as exc:
        return hero_source, {"heroTextCleanup": "failed", "heroTextCleanupError": str(exc)[:160]}


def render_large_rectangle_relayout(source: Image.Image, width: int, height: int, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    margin_x = max(10, int(width * 0.08))
    top_margin = max(10, int(height * 0.035))
    product_box = relayout_product_box(source, analysis)
    text_source_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.text_layers]
    copy_fill_bottom = clamp_int(height * (0.42 if width <= 180 else 0.44), int(height * 0.36), int(height * 0.48))
    hero_top = copy_fill_bottom + max(8, int(height * 0.018))
    canvas, texture_meta = build_clean_source_texture_fill(source, width, height, text_source_boxes)
    hero_region = build_protected_hero_region(source, analysis, product_box, width)
    raw_product_area_ratio = max((layer.bbox.area_ratio() for layer in analysis.product_layers), default=0.0)
    cleanup_hero_text = bool(analysis.text_layers) or raw_product_area_ratio <= 0.68
    hero_paste_box, hero_meta = paste_contained_hero(canvas, source, hero_region, hero_top, cleanup_overlay_text=cleanup_hero_text)

    text_candidates = analysis.marketing_text_layers or analysis.text_layers
    text = " ".join(layer.original_text for layer in text_candidates[:3]).strip()
    text_meta = {"textOverflow": False, "lineCount": 0, "fontSize": 0}
    if text:
        draw = ImageDraw.Draw(canvas)
        text_top = top_margin
        text_bottom = copy_fill_bottom - max(8, int(height * 0.018))
        text_area_w = max(1, width - margin_x * 2)
        text_area_h = max(1, text_bottom - text_top)
        max_font = max(12, min(34, int(width * 0.17), int(text_area_h * 0.24)))
        first_style = text_candidates[0].text_style if text_candidates else None
        fill = first_style.color_rgb.as_tuple() if first_style else (18, 28, 45)
        is_bold = True if first_style is None else first_style.is_bold
        display_text = text.upper() if first_style and first_style.uppercase else text
        lines, font_size, line_height, overflow = fit_plain_text_to_box(
            draw,
            display_text,
            text_area_w,
            text_area_h,
            max_font_size=max_font,
            min_font_size=8,
            bold=is_bold,
        )
        font = get_font(font_size, bold=is_bold)
        total_h = len(lines) * line_height
        y = text_top + max(0, (text_area_h - total_h) // 2)
        for line in lines:
            line_w = text_width(draw, line, font)
            x = margin_x + max(0, int((text_area_w - line_w) / 2))
            draw.text((x, y), line, fill=fill, font=font, stroke_width=1, stroke_fill=(232, 238, 244))
            y += line_height
        text_meta = {"textOverflow": overflow, "lineCount": len(lines), "fontSize": font_size}

    return canvas, {
        "provider": "local",
        "strategy": "large_rectangle_relayout",
        "backgroundMode": "protected_contain_hero_texture_fill",
        "copyFillBottom": copy_fill_bottom,
        **hero_meta,
        **texture_meta,
        **text_meta,
    }


def should_outpaint_uncertain_full_subject(analysis: VisualAnalysis) -> bool:
    raw_product_area_ratio = max((layer.bbox.area_ratio() for layer in analysis.product_layers), default=0.0)
    return (
        raw_product_area_ratio > 0.85
        and not analysis.text_layers
        and any("protected_mask_uncertain" in warning for warning in analysis.quality_warnings)
    )


def build_outpaint_seed_and_mask(source: Image.Image, width: int, height: int, plan: Any) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    source_rgba = source.convert("RGBA")
    if source_rgba.size == (width, height) and source_rgba.getchannel("A").getbbox():
        alpha = source_rgba.getchannel("A")
        alpha_values = np.array(alpha, dtype=np.uint8)
        if int(np.count_nonzero(alpha_values < 250)) > 0:
            mask = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            protected = Image.new("RGBA", (width, height), (255, 255, 255, 255))
            protected.putalpha(alpha)
            mask.alpha_composite(protected, (0, 0))
            return source_rgba, mask, {
                "sourcePasteBox": [0, 0, width, height],
                "sourceScale": 1.0,
                "layoutSeedAlphaProtectedPixels": int(np.count_nonzero(alpha_values >= 250)),
                "layoutSeedTransparentPixels": int(np.count_nonzero(alpha_values < 250)),
            }
    seed = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 0))

    scale = min(width / source.width, height / source.height)
    resized_size = (
        max(1, int(round(source.width * scale))),
        max(1, int(round(source.height * scale))),
    )
    resized = source_rgba.resize(resized_size, Image.Resampling.LANCZOS)
    paste_x = (width - resized.width) // 2
    paste_y = (height - resized.height) // 2
    if plan.logic_bucket == LogicBucket.VERTICAL_SQUARE and height > width:
        paste_y = max(0, min(height - resized.height, int(height * 0.30)))
    elif plan.logic_bucket == LogicBucket.LANDSCAPE_WIDE and width > height:
        paste_x = max(0, min(width - resized.width, int(width * 0.50 - resized.width * 0.50)))

    seed.paste(resized, (paste_x, paste_y), resized)
    keep = Image.new("RGBA", resized.size, (255, 255, 255, 255))
    mask.paste(keep, (paste_x, paste_y), keep)
    return seed, mask, {
        "sourcePasteBox": [paste_x, paste_y, paste_x + resized.width, paste_y + resized.height],
        "sourceScale": round(scale, 4),
    }


def build_compositor_background_outpaint_prompt(plan: Any, analysis: VisualAnalysis) -> str:
    theme_notes = "; ".join(
        f"{layer.id}:{layer.notes or layer.role.value}"
        for layer in analysis.other_layers[:4]
    ) or analysis.saliency_summary or "the original campaign background style"
    return (
        "Extend this advertising creative only into the transparent or masked canvas area. "
        "The opaque seed pixels are protected and must remain visually unchanged: preserve existing product shape, product label, logo, lighting, shadows, and background exactly where they already exist. "
        "Do not redesign the ad. Do not add frames, cards, black bars, torn-paper edges, borders, panels, UI, stickers, or decorative graphic systems. "
        "Do not create any new product, bottle, package, cap, pump, person, badge, icon, label, or logo. "
        "Do not alter, rewrite, regenerate, translate, or approximate any visible product-label text or brand marks in the protected seed. "
        "Marketing text may have been removed from the seed; do not recreate it during outpaint because deterministic code will composite final typography later. "
        "Fill only missing canvas with a natural continuation of the existing scene, matching lighting, color, grain, texture, perspective, and campaign theme. "
        "If an existing protected product or material is visibly cut by the canvas boundary, continue only that same visible form naturally; do not invent additional products, props, panels, UI, labels, packaging, people, or decorative objects. "
        "Do not generate any letters, words, numbers, symbols, pseudo-text, captions, labels, logos, watermarks, or readable marks anywhere in the newly generated area. "
        f"Theme reference: {theme_notes}. "
        "Return a clean completed creative canvas with no new text and no new objects."
    )


def render_openai_compositor_background_outpaint(source: Image.Image, width: int, height: int, plan: Any, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI compositor background outpaint.")

    model = os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
    quality = os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium").strip().lower() or "medium"
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "medium"

    seed, mask, seed_meta = build_outpaint_seed_and_mask(source, width, height, plan)
    image_file = io.BytesIO()
    seed.save(image_file, format="PNG")
    image_file.seek(0)
    mask_file = io.BytesIO()
    mask.save(mask_file, format="PNG")
    mask_file.seek(0)

    client = OpenAI(api_key=api_key)
    response, api_variant = create_openai_image_edit_with_fallback(
        client,
        {
            "model": model,
            "image": ("compositor-background-seed.png", image_file, "image/png"),
            "mask": ("compositor-background-mask.png", mask_file, "image/png"),
            "prompt": build_compositor_background_outpaint_prompt(plan, analysis),
            "quality": quality,
        },
        [image_file, mask_file],
        output_format="png",
        size="auto",
        width=width,
        height=height,
    )
    rendered = decode_openai_image_response(response)
    if rendered.size != (width, height):
        rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
    return rendered, {
        "provider": "openai",
        "model": model,
        "quality": quality,
        "strategy": "openai_compositor_background_outpaint",
        "apiVariant": api_variant,
        **seed_meta,
    }


def build_vertex_outpaint_base_and_mask(seed: Image.Image, seed_meta: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    paste_box = seed_meta.get("sourcePasteBox") or [0, 0, seed.width, seed.height]
    try:
        left, top, right, bottom = [int(value) for value in paste_box]
    except Exception:
        left, top, right, bottom = 0, 0, seed.width, seed.height
    rgba_seed = seed.convert("RGBA")
    alpha = np.array(rgba_seed.getchannel("A"), dtype=np.uint8)
    rgb_seed = rgba_seed.convert("RGB")
    if seed_meta.get("layoutSeedTransparentPixels"):
        visible = np.array(rgb_seed, dtype=np.uint8)[alpha > 0]
        fill_color = tuple(int(value) for value in (np.median(visible, axis=0) if visible.size else np.array([245, 245, 245])))
        base = Image.new("RGB", seed.size, fill_color)
        base.paste(rgb_seed, (0, 0), rgba_seed.getchannel("A"))
    elif int(np.count_nonzero(alpha)) < seed.width * seed.height:
        visible = np.array(rgb_seed, dtype=np.uint8)[alpha > 0]
        fill_color = tuple(int(value) for value in (visible.mean(axis=0) if visible.size else np.array([245, 245, 245])))
        base = Image.new("RGB", seed.size, fill_color)
        base.paste(rgb_seed, (0, 0), rgba_seed.getchannel("A"))
    else:
        base = rgb_seed

    if seed_meta.get("layoutSeedTransparentPixels"):
        mask_arr = np.where(alpha >= 250, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, "L")
    else:
        mask = Image.new("L", seed.size, 255)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            (
                max(0, min(seed.width, left)),
                max(0, min(seed.height, top)),
                max(0, min(seed.width, right)),
                max(0, min(seed.height, bottom)),
            ),
            fill=0,
        )
    return base, mask.convert("RGB")


def render_vertex_compositor_background_outpaint(source: Image.Image, width: int, height: int, plan: Any, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    if not vertex_available():
        raise RuntimeError("Vertex service account is not configured.")
    seed, _openai_mask, seed_meta = build_outpaint_seed_and_mask(source, width, height, plan)
    base_image, mask_image = build_vertex_outpaint_base_and_mask(seed, seed_meta)
    project_id = vertex_project_id()
    location = vertex_location()
    model = vertex_imagen_edit_model()
    endpoint = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model}:predict"
    )
    response = vertex_authorized_session().post(
        endpoint,
        json={
            "instances": [
                {
                    "prompt": build_compositor_background_outpaint_prompt(plan, analysis),
                    "referenceImages": [
                        {
                            "referenceType": "REFERENCE_TYPE_RAW",
                            "referenceId": 1,
                            "referenceImage": {"bytesBase64Encoded": image_to_base64_png(base_image)},
                        },
                        {
                            "referenceType": "REFERENCE_TYPE_MASK",
                            "referenceId": 2,
                            "referenceImage": {"bytesBase64Encoded": image_to_base64_png(mask_image)},
                            "maskImageConfig": {
                                "maskMode": "MASK_MODE_USER_PROVIDED",
                                "dilation": float(os.getenv("VERTEX_IMAGEN_MASK_DILATION", "0.03")),
                            },
                        },
                    ],
                }
            ],
            "parameters": {
                "sampleCount": 1,
                "editMode": "EDIT_MODE_OUTPAINT",
                "editConfig": {
                    "baseSteps": int(os.getenv("VERTEX_IMAGEN_EDIT_STEPS", "35")),
                    "outpaintingConfig": {
                        "blendingMode": os.getenv("VERTEX_IMAGEN_OUTPAINT_BLEND_MODE", "alpha-blending"),
                        "blendingFactor": float(os.getenv("VERTEX_IMAGEN_OUTPAINT_BLEND_FACTOR", "0.01")),
                    },
                },
                "safetyFilterLevel": os.getenv("VERTEX_IMAGEN_SAFETY_FILTER_LEVEL", "block_some"),
                "personGeneration": os.getenv("VERTEX_IMAGEN_PERSON_GENERATION", "allow_adult"),
            },
        },
        timeout=int(os.getenv("VERTEX_IMAGEN_TIMEOUT", "35")),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Vertex Imagen compositor background outpaint failed with HTTP {response.status_code}: {response.text[:1200]}") from exc
    payload = response.json()
    predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
    if not predictions:
        raise ValueError("Vertex Imagen compositor background outpaint returned no predictions.")
    rendered = decode_vertex_imagen_prediction(predictions[0])
    if rendered.size != (width, height):
        rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
    return rendered, {
        "provider": "vertex",
        "model": model,
        "location": location,
        "strategy": "vertex_compositor_background_outpaint",
        **seed_meta,
    }


def build_resize_product_completion_prompt(edge_touch: list[str]) -> str:
    missing_parts = ", ".join(edge_touch) if edge_touch else "outer edges"
    return (
        "Complete only the missing outer shape of this isolated product packaging asset. "
        f"The product is visibly truncated at: {missing_parts}. "
        "Preserve the existing opaque product pixels exactly, especially all brand marks, label text, colors, shadows, perspective, and bottle geometry. "
        "Continue the visible closure/cap/body geometry that is already present; do not invent a new closure type. "
        "Do not add screw caps, metal caps, pumps, nozzles, foil, ridges, hands, holders, or any hardware/details that are not clearly implied by the visible product. "
        "Do not create a separate neck, stem, plug, detached cap, or protruding part above/below the existing product. "
        "The completed silhouette must be a smooth continuation of the existing package footprint, not a new component attached to it. "
        "Do not translate, rewrite, redraw, approximate, or invent any label text. "
        "Do not add marketing copy, badges, logos, UI, background scenery, frames, hands, people, or extra products. "
        "Generate only the physically continuous missing cap/body/bottom pixels needed to make the same single product look complete. "
        "Keep the surrounding area plain/transparent-looking and free of text; deterministic code will place the completed product into the final ad layout."
    )


def build_resize_product_completion_seed(product: Image.Image, edge_touch: list[str]) -> tuple[Image.Image, dict[str, Any]]:
    product = product.convert("RGBA")
    pad_x = max(24, int(product.width * 0.18))
    top_pad = max(32, int(product.height * (0.32 if "top" in edge_touch else 0.16)))
    bottom_pad = max(32, int(product.height * (0.32 if "bottom" in edge_touch else 0.16)))
    canvas = Image.new("RGBA", (product.width + pad_x * 2, product.height + top_pad + bottom_pad), (0, 0, 0, 0))
    canvas.alpha_composite(product, (pad_x, top_pad))
    return canvas, {
        "productCompletionSeedPasteBox": [pad_x, top_pad, pad_x + product.width, top_pad + product.height],
        "productCompletionSeedPadding": [pad_x, top_pad, pad_x, bottom_pad],
        "layoutSeedTransparentPixels": True,
    }


def _edge_strip_color_stats(image: Image.Image, edge: str, strip: int) -> tuple[np.ndarray, np.ndarray]:
    rgba = image.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    alpha = arr[:, :, 3]
    if edge == "top":
        region = arr[:strip, :, :]
    elif edge == "bottom":
        region = arr[max(0, arr.shape[0] - strip):, :, :]
    elif edge == "left":
        region = arr[:, :strip, :]
    elif edge == "right":
        region = arr[:, max(0, arr.shape[1] - strip):, :]
    else:
        region = arr
    visible = region[region[:, :, 3] > 24][:, :3]
    if visible.size == 0:
        visible = arr[alpha > 24][:, :3]
    if visible.size == 0:
        return np.array([245.0, 245.0, 245.0]), np.array([0.0, 0.0, 0.0])
    return np.median(visible.astype(np.float32), axis=0), np.std(visible.astype(np.float32), axis=0)


def _rgba_alpha_edge_touch_local(rgba: Image.Image, *, threshold: int = 24) -> list[str]:
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


def _extract_product_completion_from_seed(
    rendered: Image.Image,
    seed: Image.Image,
    product: Image.Image,
    paste_box: list[int],
) -> Image.Image:
    rendered_rgb = rendered.convert("RGB")
    seed_rgba = seed.convert("RGBA")
    product_rgba = product.convert("RGBA")
    rendered_np = np.array(rendered_rgb, dtype=np.uint8)
    seed_alpha = np.array(seed_rgba.getchannel("A"), dtype=np.uint8)
    original_mask = seed_alpha > 24
    transparent_pixels = rendered_np[~original_mask]
    if transparent_pixels.size:
        matte = np.median(transparent_pixels.astype(np.float32), axis=0)
    else:
        corner = max(4, min(rendered_rgb.width, rendered_rgb.height) // 24)
        corners = np.concatenate(
            [
                rendered_np[:corner, :corner].reshape(-1, 3),
                rendered_np[:corner, -corner:].reshape(-1, 3),
                rendered_np[-corner:, :corner].reshape(-1, 3),
                rendered_np[-corner:, -corner:].reshape(-1, 3),
            ],
            axis=0,
        )
        matte = np.median(corners.astype(np.float32), axis=0)

    color_delta = np.linalg.norm(rendered_np.astype(np.float32) - matte.reshape(1, 1, 3), axis=2)
    candidate = (color_delta > 18) | original_mask
    try:
        import cv2

        kernel_size = max(3, min(9, int(min(seed.size) * 0.012) | 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        candidate_u8 = (candidate.astype(np.uint8) * 255)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats((candidate_u8 > 0).astype(np.uint8), 8)
        keep = np.zeros(candidate_u8.shape, dtype=np.uint8)
        for label in range(1, count):
            component = labels == label
            if not np.any(component & original_mask):
                continue
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(16, int(np.count_nonzero(original_mask) * 0.002)):
                continue
            keep[component] = 255
        if keep.max() == 0:
            keep = (original_mask.astype(np.uint8) * 255)
        keep = cv2.dilate(keep, kernel, iterations=1)
        keep = cv2.GaussianBlur(keep, (0, 0), sigmaX=0.85, sigmaY=0.85)
        alpha = keep
    except Exception:
        alpha = Image.fromarray((candidate.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(radius=0.85))
        alpha = np.array(alpha, dtype=np.uint8)

    completed = rendered_rgb.convert("RGBA")
    completed.putalpha(Image.fromarray(alpha, "L"))
    left, top, _right, _bottom = [int(v) for v in paste_box]
    completed.alpha_composite(product_rgba, (left, top))
    return _prepare_foreground_rgba_crop(completed)


def _validate_completed_product_asset(
    original_product: Image.Image,
    completed_product: Image.Image,
    paste_box: list[int],
    edge_touch: list[str],
) -> tuple[bool, dict[str, Any]]:
    original = original_product.convert("RGBA")
    completed = completed_product.convert("RGBA")
    comp = np.array(completed, dtype=np.uint8)
    alpha = comp[:, :, 3]
    left, top, right, bottom = [int(v) for v in paste_box]
    new_area = np.zeros(alpha.shape, dtype=bool)
    if "top" in edge_touch:
        new_area[:max(0, top), :] = True
    if "bottom" in edge_touch:
        new_area[min(alpha.shape[0], bottom):, :] = True
    if "left" in edge_touch:
        new_area[:, :max(0, left)] = True
    if "right" in edge_touch:
        new_area[:, min(alpha.shape[1], right):] = True
    new_visible = new_area & (alpha > 24)
    new_pixels = comp[new_visible][:, :3].astype(np.float32)
    if new_pixels.size == 0:
        return False, {"productCompletionRejected": "no_visible_completed_pixels"}

    original_alpha_pixels = np.count_nonzero(np.array(original.getchannel("A"), dtype=np.uint8) > 24)
    new_alpha_pixels = int(np.count_nonzero(new_visible))
    configured_max_new_ratio = float(os.getenv("ADAPTIFAI_PRODUCT_COMPLETION_MAX_NEW_ALPHA_RATIO", "0.55"))
    # Completing both top and bottom of a pre-cropped package legitimately adds
    # much more alpha than a single-edge repair. Keep the single-edge gate tight,
    # but allow multi-edge product completion to pass the later color/artifact
    # gates instead of being rejected only for area.
    dynamic_max_new_ratio = 0.55 if len(set(edge_touch)) <= 1 else min(0.90, 0.36 + 0.24 * len(set(edge_touch)))
    max_new_ratio = max(configured_max_new_ratio, dynamic_max_new_ratio)
    new_ratio = new_alpha_pixels / max(1, int(original_alpha_pixels))
    if new_ratio > max_new_ratio:
        return False, {
            "productCompletionRejected": "excessive_new_alpha_area",
            "productCompletionNewAlphaRatio": round(new_ratio, 4),
        }

    strip = max(6, min(32, int(min(original.width, original.height) * 0.08)))
    failures: list[str] = []
    distances: dict[str, float] = {}
    for edge in edge_touch:
        source_median, source_std = _edge_strip_color_stats(original, edge, strip)
        if edge == "top":
            edge_region = comp[:max(0, top), :, :]
        elif edge == "bottom":
            edge_region = comp[min(comp.shape[0], bottom):, :, :]
        elif edge == "left":
            edge_region = comp[:, :max(0, left), :]
        elif edge == "right":
            edge_region = comp[:, min(comp.shape[1], right):, :]
        else:
            continue
        edge_pixels = edge_region[edge_region[:, :, 3] > 24][:, :3].astype(np.float32)
        if edge_pixels.size == 0:
            continue
        generated_median = np.median(edge_pixels, axis=0)
        tolerance = max(52.0, float(np.linalg.norm(source_std)) * 1.8)
        distance = float(np.linalg.norm(generated_median - source_median))
        distances[edge] = round(distance, 2)
        if distance > tolerance:
            failures.append(f"{edge}:color_drift_{distance:.1f}>{tolerance:.1f}")

    dark_ratio = float(np.mean(np.min(new_pixels, axis=1) < 36))
    saturation = new_pixels.max(axis=1) - new_pixels.min(axis=1)
    high_saturation_ratio = float(np.mean(saturation > 128))
    if dark_ratio > 0.18:
        failures.append(f"dark_detail_ratio_{dark_ratio:.2f}")
    if high_saturation_ratio > 0.35:
        failures.append(f"high_saturation_ratio_{high_saturation_ratio:.2f}")
    if failures:
        return False, {
            "productCompletionRejected": "failed_edge_consistency_gate",
            "productCompletionGateFailures": failures,
            "productCompletionColorDistances": distances,
            "productCompletionNewAlphaRatio": round(new_ratio, 4),
        }
    return True, {
        "productCompletionGate": "passed",
        "productCompletionColorDistances": distances,
        "productCompletionNewAlphaRatio": round(new_ratio, 4),
    }


def _constrain_completed_product_to_source_footprint(
    completed_product: Image.Image,
    original_product: Image.Image,
    paste_box: list[int],
    edge_touch: list[str],
) -> tuple[Image.Image, dict[str, Any]]:
    """Keep generative product completion inside the original product footprint.

    The edit model may complete the surrounding cream/background texture as if it
    were part of the product. That must never enter the frame-ready product
    asset. We preserve the generated product shape near truncated edges, but
    mathematically clip it to the source product's projected footprint.
    """
    try:
        import cv2

        completed = completed_product.convert("RGBA")
        original = original_product.convert("RGBA")
        comp = np.array(completed, dtype=np.uint8)
        alpha = comp[:, :, 3]
        original_alpha = np.array(original.getchannel("A"), dtype=np.uint8)
        left, top, right, bottom = [int(v) for v in paste_box]
        full_original = np.zeros(alpha.shape, dtype=np.uint8)
        full_original[top:bottom, left:right] = np.maximum(
            full_original[top:bottom, left:right],
            original_alpha,
        )
        ys, xs = np.where(full_original > 24)
        if xs.size == 0 or ys.size == 0:
            return completed, {"productFootprintConstraint": "skipped_empty_original_alpha"}

        col_density = np.count_nonzero(full_original > 24, axis=0)
        row_density = np.count_nonzero(full_original > 24, axis=1)
        dense_cols = np.where(col_density >= max(6, int(float(col_density.max()) * 0.34)))[0]
        dense_rows = np.where(row_density >= max(6, int(float(row_density.max()) * 0.18)))[0]
        if dense_cols.size >= 4:
            x1, x2 = int(dense_cols.min()), int(dense_cols.max()) + 1
        else:
            x1, x2 = int(xs.min()), int(xs.max()) + 1
        if dense_rows.size >= 4:
            y1, y2 = int(dense_rows.min()), int(dense_rows.max()) + 1
        else:
            y1, y2 = int(ys.min()), int(ys.max()) + 1
        footprint_w = max(1, x2 - x1)
        footprint_h = max(1, y2 - y1)
        pad_x = max(8, int(footprint_w * 0.12))
        pad_y = max(10, int(footprint_h * 0.18))

        allowed = np.zeros(alpha.shape, dtype=np.uint8)
        kernel_size = max(9, min(45, min(completed.width, completed.height) // 18) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        allowed = cv2.dilate((full_original > 24).astype(np.uint8) * 255, kernel, iterations=1)

        if "top" in edge_touch:
            allowed[:top, max(0, x1 - pad_x) : min(alpha.shape[1], x2 + pad_x)] = 255
        if "bottom" in edge_touch:
            allowed[bottom:, max(0, x1 - pad_x) : min(alpha.shape[1], x2 + pad_x)] = 255
        if "left" in edge_touch:
            allowed[max(0, y1 - pad_y) : min(alpha.shape[0], y2 + pad_y), :left] = 255
        if "right" in edge_touch:
            allowed[max(0, y1 - pad_y) : min(alpha.shape[0], y2 + pad_y), right:] = 255

        before = int(np.count_nonzero(alpha > 24))
        alpha = np.where(allowed > 0, alpha, 0).astype(np.uint8)

        # Remove saturated/text-like hallucinated strokes in generated regions
        # while preserving the generated silhouette. This is a cleanup pass on
        # the provider result, not a replacement for generative completion.
        new_area = np.zeros(alpha.shape, dtype=bool)
        if "top" in edge_touch:
            new_area[:top, :] = True
        if "bottom" in edge_touch:
            new_area[bottom:, :] = True
        if "left" in edge_touch:
            new_area[:, :left] = True
        if "right" in edge_touch:
            new_area[:, right:] = True
        source_rgb = np.array(original.convert("RGB"), dtype=np.uint8)
        original_visible = original_alpha > 24
        if np.any(original_visible):
            median_rgb = np.median(source_rgb[original_visible], axis=0).astype(np.uint8)
            std_rgb = np.std(source_rgb[original_visible].astype(np.float32), axis=0)
            new_visible = new_area & (alpha > 24)
            rgb = comp[:, :, :3]
            saturation = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
            dark = rgb.min(axis=2) < 42
            distance_from_product = np.linalg.norm(rgb.astype(np.float32) - median_rgb.reshape(1, 1, 3).astype(np.float32), axis=2)
            allowed_distance = max(42.0, float(np.linalg.norm(std_rgb)) * 1.45)
            artifact = new_visible & ((saturation > 96) | dark | (distance_from_product > allowed_distance))
            if np.any(artifact):
                rgb[artifact] = median_rgb
                comp[:, :, :3] = rgb

        comp[:, :, 3] = alpha
        constrained = Image.fromarray(comp, "RGBA")
        after = int(np.count_nonzero(alpha > 24))
        return _prepare_foreground_rgba_crop(constrained), {
            "productFootprintConstraint": "applied",
            "productFootprintAlphaBefore": before,
            "productFootprintAlphaAfter": after,
            "productFootprintAllowedBox": [max(0, x1 - pad_x), max(0, y1 - pad_y), min(alpha.shape[1], x2 + pad_x), min(alpha.shape[0], y2 + pad_y)],
        }
    except Exception as exc:
        return completed_product.convert("RGBA"), {"productFootprintConstraint": "failed", "productFootprintConstraintError": str(exc)[:240]}


def render_resize_product_asset_completion(product: Image.Image, meta: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    edge_touch = [str(item) for item in meta.get("productAlphaEdgeTouch", []) if str(item)]
    if not edge_touch:
        return product.convert("RGBA"), {"productCompletionSkipped": "no_truncated_alpha_edge"}
    cache_file = io.BytesIO()
    product.convert("RGBA").save(cache_file, format="PNG")
    cache_version = b"resize-product-completion-v7-partial-edge-salvage"
    cache_key = hashlib.sha256(cache_file.getvalue() + "|".join(edge_touch).encode("utf-8") + cache_version).hexdigest()
    cached = _RESIZE_PRODUCT_COMPLETION_CACHE.get(cache_key)
    if cached is not None:
        cached_image, cached_meta = cached
        return cached_image.copy(), {**cached_meta, "productCompletionCache": "hit"}
    seed, seed_meta = build_resize_product_completion_seed(product, edge_touch)
    prompt = build_resize_product_completion_prompt(edge_touch)
    openai_enabled = os.getenv("ADAPTIFAI_ENABLE_OPENAI_PRODUCT_COMPLETION", "0").strip().lower() in {"1", "true", "yes", "on"}
    vertex_enabled = env_flag("ADAPTIFAI_ENABLE_VERTEX_PRODUCT_COMPLETION", "1")
    rendered: Image.Image | None = None
    provider_meta: dict[str, Any] = {}

    if vertex_enabled and vertex_available():
        base_image, mask_image = build_vertex_outpaint_base_and_mask(seed, seed_meta)
        project_id = vertex_project_id()
        location = vertex_location()
        model = vertex_imagen_edit_model()
        endpoint = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
            f"locations/{location}/publishers/google/models/{model}:predict"
        )
        response = vertex_authorized_session().post(
            endpoint,
            json={
                "instances": [
                    {
                        "prompt": prompt,
                        "referenceImages": [
                            {
                                "referenceType": "REFERENCE_TYPE_RAW",
                                "referenceId": 1,
                                "referenceImage": {"bytesBase64Encoded": image_to_base64_png(base_image)},
                            },
                            {
                                "referenceType": "REFERENCE_TYPE_MASK",
                                "referenceId": 2,
                                "referenceImage": {"bytesBase64Encoded": image_to_base64_png(mask_image)},
                                "maskImageConfig": {
                                    "maskMode": "MASK_MODE_USER_PROVIDED",
                                    "dilation": float(os.getenv("VERTEX_IMAGEN_PRODUCT_MASK_DILATION", "0.01")),
                                },
                            },
                        ],
                    }
                ],
                "parameters": {
                    "sampleCount": 1,
                    "editMode": "EDIT_MODE_OUTPAINT",
                    "editConfig": {
                        "baseSteps": int(os.getenv("VERTEX_IMAGEN_PRODUCT_EDIT_STEPS", "35")),
                        "outpaintingConfig": {
                            "blendingMode": os.getenv("VERTEX_IMAGEN_OUTPAINT_BLEND_MODE", "alpha-blending"),
                            "blendingFactor": float(os.getenv("VERTEX_IMAGEN_PRODUCT_BLEND_FACTOR", "0.01")),
                        },
                    },
                    "safetyFilterLevel": os.getenv("VERTEX_IMAGEN_SAFETY_FILTER_LEVEL", "block_some"),
                    "personGeneration": os.getenv("VERTEX_IMAGEN_PERSON_GENERATION", "allow_adult"),
                },
            },
            timeout=int(os.getenv("VERTEX_IMAGEN_TIMEOUT", "35")),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Vertex Imagen product completion failed with HTTP {response.status_code}: {response.text[:1200]}") from exc
        payload = response.json()
        predictions = payload.get("predictions", []) if isinstance(payload, dict) else []
        if not predictions:
            raise ValueError("Vertex Imagen product completion returned no predictions.")
        rendered = decode_vertex_imagen_prediction(predictions[0])
        provider_meta = {"provider": "vertex", "model": model, "location": location, "strategy": "vertex_resize_product_asset_completion"}
    elif openai_enabled:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI product completion.")
        mask_arr = np.where(np.array(seed.getchannel("A"), dtype=np.uint8) >= 250, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_arr, "L")
        image_file = io.BytesIO()
        seed.save(image_file, format="PNG")
        image_file.seek(0)
        mask_file = io.BytesIO()
        mask.save(mask_file, format="PNG")
        mask_file.seek(0)
        model = os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
        client = OpenAI(api_key=api_key)
        response, api_variant = create_openai_image_edit_with_fallback(
            client,
            {
                "model": model,
                "image": ("product-completion-seed.png", image_file, "image/png"),
                "mask": ("product-completion-mask.png", mask_file, "image/png"),
                "prompt": prompt,
                "quality": os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium").strip().lower() or "medium",
            },
            [image_file, mask_file],
            output_format="png",
            size="auto",
            width=seed.width,
            height=seed.height,
        )
        rendered = decode_openai_image_response(response)
        provider_meta = {"provider": "openai", "model": model, "strategy": "openai_resize_product_asset_completion", "apiVariant": api_variant}
    else:
        raise RuntimeError("No product completion provider is enabled.")

    if rendered.size != seed.size:
        rendered = rendered.resize(seed.size, Image.Resampling.LANCZOS)
    paste_box = seed_meta["productCompletionSeedPasteBox"]
    completed = _extract_product_completion_from_seed(rendered, seed, product, paste_box)
    completed, footprint_meta = _constrain_completed_product_to_source_footprint(completed, product, paste_box, edge_touch)
    remaining_edges = _rgba_alpha_edge_touch_local(completed)
    unresolved_edges = [edge for edge in edge_touch if edge in remaining_edges]
    if unresolved_edges:
        resolved_edges = [edge for edge in edge_touch if edge not in unresolved_edges]
        if resolved_edges:
            comp = np.array(completed.convert("RGBA"), dtype=np.uint8)
            left, top, right, bottom = [int(v) for v in paste_box]
            if "top" in unresolved_edges:
                comp[:max(0, top), :, 3] = 0
            if "bottom" in unresolved_edges:
                comp[min(comp.shape[0], bottom):, :, 3] = 0
            if "left" in unresolved_edges:
                comp[:, :max(0, left), 3] = 0
            if "right" in unresolved_edges:
                comp[:, min(comp.shape[1], right):, 3] = 0
            completed = Image.fromarray(comp, "RGBA")
            footprint_meta = {
                **footprint_meta,
                "productCompletionPartialApplied": True,
                "productCompletionRemovedUnresolvedEdges": unresolved_edges,
                "productCompletionResolvedEdges": resolved_edges,
            }
        else:
            result_meta = {
                **provider_meta,
                **seed_meta,
                **footprint_meta,
                "productCompletionInputEdgeTouch": edge_touch,
                "productCompletionOutputSize": list(completed.size),
                "productCompletionCache": "miss",
                "productCompletionRejected": "unresolved_truncated_edges_after_completion",
                "productCompletionRemainingEdgeTouch": remaining_edges,
                "productCompletionUnresolvedEdges": unresolved_edges,
            }
            return product.convert("RGBA"), result_meta
    accepted, gate_meta = _validate_completed_product_asset(product, completed, paste_box, edge_touch)
    result_meta = {
        **provider_meta,
        **seed_meta,
        **footprint_meta,
        **gate_meta,
        "productCompletionInputEdgeTouch": edge_touch,
        "productCompletionOutputSize": list(completed.size),
        "productCompletionCache": "miss",
    }
    if not accepted:
        result = product.convert("RGBA")
        return result, result_meta
    if len(_RESIZE_PRODUCT_COMPLETION_CACHE) > 8:
        _RESIZE_PRODUCT_COMPLETION_CACHE.clear()
    _RESIZE_PRODUCT_COMPLETION_CACHE[cache_key] = (completed.copy(), result_meta.copy())
    return completed, result_meta


def smart_reframe_text_style_to_block_color(style: TextStyle | None) -> str:
    if style is None:
        return "#111111"
    return rgb_to_hex(style.color_rgb.as_tuple())


def sample_resize_text_background_color(source: Image.Image, source_box: tuple[int, int, int, int], text_color: str) -> tuple[bool, str | None]:
    try:
        fg = parse_hex_color(text_color, fallback=(17, 17, 17))
        fg_luma = 0.2126 * fg[0] + 0.7152 * fg[1] + 0.0722 * fg[2]
        left, top, right, bottom = source_box
        pad_x = max(8, int((right - left) * 0.28))
        pad_y = max(5, int((bottom - top) * 0.40))
        crop_box = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(source.width, right + pad_x),
            min(source.height, bottom + pad_y),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return False, None
        arr = np.array(source.crop(crop_box).convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        if arr.size == 0:
            return False, None
        fg_arr = np.array(fg, dtype=np.float32)
        dist = np.linalg.norm(arr.astype(np.float32) - fg_arr, axis=1)
        candidates = arr[dist > 52]
        if candidates.size == 0:
            return False, None
        saturated = np.empty((0, 3), dtype=np.uint8)
        try:
            import cv2

            hsv = cv2.cvtColor(candidates.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
            saturated = candidates[(hsv[:, 1] > 48) & (hsv[:, 2] > 80)]
            if saturated.size:
                candidates = saturated
        except Exception:
            pass
        if fg_luma < 210:
            return False, None
        bg = np.median(candidates.reshape(-1, 3), axis=0).astype(int)
        bg_tuple = (int(bg[0]), int(bg[1]), int(bg[2]))
        if saturated.size == 0 and contrast_ratio(bg_tuple, fg) < 2.1:
            return False, None
        return True, rgb_to_hex(bg_tuple)
    except Exception:
        return False, None


def sample_resize_line_style(
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    fallback_color: str,
) -> tuple[str, bool, str | None]:
    """Sample a single source text line so resize redraw keeps per-line style.

    Visual providers often group several marketing lines into one semantic layer.
    Rendering must still preserve the source line styling: blue headline lines stay
    blue, highlighted ribbon lines stay white-on-ribbon, and RTB lines keep their
    own color.
    """
    box = (
        max(0, min(source.width, int(source_box[0]))),
        max(0, min(source.height, int(source_box[1]))),
        max(0, min(source.width, int(source_box[2]))),
        max(0, min(source.height, int(source_box[3]))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return fallback_color, False, None
    def dominant_highlight_in_box(candidate_box: tuple[int, int, int, int]) -> tuple[bool, str | None]:
        crop_arr = np.array(source.crop(candidate_box).convert("RGB"), dtype=np.uint8)
        if crop_arr.size == 0:
            return False, None
        pixels_arr = crop_arr.reshape(-1, 3)
        brightness_arr = pixels_arr.mean(axis=1).astype(np.float32)
        chroma_arr = (pixels_arr.max(axis=1) - pixels_arr.min(axis=1)).astype(np.float32)
        saturated_arr = pixels_arr[(chroma_arr >= 54) & (brightness_arr >= 80) & (brightness_arr <= 245)]
        saturated_ratio_arr = len(saturated_arr) / max(1, len(pixels_arr))
        if len(saturated_arr) < 24 or saturated_ratio_arr < 0.07:
            return False, None
        quantized_arr = (saturated_arr.astype(np.uint16) // 24).astype(np.uint16)
        packed_arr = quantized_arr[:, 0] * 256 + quantized_arr[:, 1] * 16 + quantized_arr[:, 2]
        values_arr, counts_arr = np.unique(packed_arr, return_counts=True)
        dominant_arr = saturated_arr[packed_arr == values_arr[int(np.argmax(counts_arr))]]
        bg_arr = np.median(dominant_arr.reshape(-1, 3), axis=0).astype(int)
        bg_tuple_arr = (int(bg_arr[0]), int(bg_arr[1]), int(bg_arr[2]))
        if max(bg_tuple_arr) - min(bg_tuple_arr) < 42:
            return False, None
        # Avoid mistaking ordinary blue letters for a ribbon. A valid ribbon is
        # a visible filled band, so the dominant saturated cluster must occupy a
        # meaningful share of the local row window.
        dominant_ratio = len(dominant_arr) / max(1, len(pixels_arr))
        if dominant_ratio < 0.035:
            return False, None
        return True, rgb_to_hex(bg_tuple_arr)

    try:
        search_pad_x = max(6, int(round((box[2] - box[0]) * 0.10)))
        search_pad_y = max(5, int(round((box[3] - box[1]) * 0.85)))
        search_box = (
            max(0, box[0] - search_pad_x),
            max(0, box[1] - search_pad_y),
            min(source.width, box[2] + search_pad_x),
            min(source.height, box[3] + search_pad_y),
        )
        has_highlight, highlight_color = dominant_highlight_in_box(search_box)
        if has_highlight and highlight_color:
            return "#ffffff", True, highlight_color

        crop = np.array(source.crop(box).convert("RGB"), dtype=np.uint8)
        if crop.size == 0:
            return fallback_color, False, None
        sampled = sample_deterministic_text_color(source, box, fallback_color)
        try:
            sr, sg, sb = hex_to_rgb(sampled)
            sampled_luma = 0.2126 * sr + 0.7152 * sg + 0.0722 * sb
            sampled_chroma = max(sr, sg, sb) - min(sr, sg, sb)
            if sampled_luma >= 165 and sampled_chroma <= 28:
                widened = sample_deterministic_text_color(source, search_box, fallback_color)
                if widened.lower() != "#111111":
                    sampled = widened
        except Exception:
            pass
        if sampled.lower() == "#111111" and fallback_color and fallback_color.lower() != "#111111":
            # LLM/vision global style sometimes collapses to black for grouped
            # layers. Prefer a non-black inherited fallback when deterministic
            # sampling cannot isolate a better foreground.
            return fallback_color, False, None
        return sampled, False, None
    except Exception:
        return fallback_color, False, None


def split_resize_box_into_line_boxes(
    source_box: tuple[int, int, int, int],
    line_count: int,
) -> list[tuple[int, int, int, int]]:
    line_count = max(1, int(line_count))
    left, top, right, bottom = source_box
    height = max(1, bottom - top)
    line_boxes: list[tuple[int, int, int, int]] = []
    for index in range(line_count):
        y1 = top + int(round(index * height / line_count))
        y2 = top + int(round((index + 1) * height / line_count))
        if y2 <= y1:
            y2 = y1 + 1
        line_boxes.append((left, y1, right, bottom if index == line_count - 1 else y2))
    return line_boxes


_RESIZE_TURKISH_OCR_TOKEN_FIXES = {
    "TUM": "TÜM",
    "CILT": "CİLT",
    "CILTLER": "CİLTLER",
    "TIPLERI": "TİPLERİ",
    "TIPLERINE": "TİPLERİNE",
    "ICIN": "İÇİN",
    "COK": "ÇOK",
    "YUKSEK": "YÜKSEK",
    "GUNES": "GÜNEŞ",
    "KORUMASI": "KORUMASI",
    "DAHIL": "DAHİL",
    "HASSAS": "HASSAS",
    "AKISKAN": "AKIŞKAN",
    "HIZLI": "HIZLI",
    "EMILEN": "EMİLEN",
    "DOKU": "DOKU",
    "SIVILCE": "SİVİLCE",
    "YAPMAYAN": "YAPMAYAN",
    "FORMUL": "FORMÜL",
    "FORMOL": "FORMÜL",
    "FORMU": "FORMÜL",
    "GUNEŞ": "GÜNEŞ",
    "YUKSEK": "YÜKSEK",
    "KORUMA": "KORUMA",
    "DERMATOLOGLARIN": "DERMATOLOGLARIN",
    "TAVSIYE": "TAVSİYE",
    "ETTIGI": "ETTİĞİ",
    "MARKA": "MARKA",
}


def normalize_resize_ocr_copy(text: str) -> str:
    """Repair conservative OCR casing/diacritic loss before deterministic resize redraw."""
    if not text:
        return text

    def fix_token(match: re.Match[str]) -> str:
        token = match.group(0)
        normalized = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii")
        key = re.sub(r"[^A-Za-z0-9]+", "", normalized).upper()
        fixed = _RESIZE_TURKISH_OCR_TOKEN_FIXES.get(key)
        if not fixed:
            return token
        return fixed if token.upper() == token or any(char.isupper() for char in token) else fixed.lower()

    fixed_lines = []
    for line in text.splitlines():
        repaired = re.sub(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", fix_token, line)
        repaired = re.sub(r"\s+", " ", repaired).strip()
        fixed_lines.append(repaired)
    return "\n".join(line for line in fixed_lines if line)


def build_resize_compositor_text_blocks(
    source: Image.Image,
    width: int,
    height: int,
    plan: Any,
    analysis: VisualAnalysis,
) -> list[TextBlock]:
    layers_by_id = {layer.id: layer for layer in analysis.marketing_text_layers}
    blocks: list[TextBlock] = []
    source_area = max(1, source.width * source.height)
    target_area = max(1, width * height)
    # Resize typography follows the actual requested canvas area, not the
    # narrowest axis. A larger target should let marketing copy grow; a smaller
    # target should shrink copy while preserving the original style hierarchy.
    area_scale = max(0.42, min(2.35, (target_area / source_area) ** 0.5))
    text_source_boxes = [layer.bbox.to_pixel_box(source.width, source.height) for layer in analysis.marketing_text_layers]
    if text_source_boxes:
        text_source_union = (
            min(box[0] for box in text_source_boxes),
            min(box[1] for box in text_source_boxes),
            max(box[2] for box in text_source_boxes),
            max(box[3] for box in text_source_boxes),
        )
    else:
        text_source_union = (0, 0, source.width, max(1, int(source.height * 0.35)))
    for placement in sorted(plan.placements, key=lambda item: item.z_index):
        if placement.role not in {LayerRole.MARKETING_TEXT, LayerRole.CTA}:
            continue
        layer = layers_by_id.get(placement.layer_id)
        if layer is None or not layer.original_text.strip():
            continue
        resize_text = normalize_resize_ocr_copy(repair_mojibake(layer.original_text).strip())
        source_box = layer.bbox.to_pixel_box(source.width, source.height)
        placement_box = placement.target_bbox.to_pixel_box(width, height)
        union_w = max(1, text_source_union[2] - text_source_union[0])
        union_h = max(1, text_source_union[3] - text_source_union[1])
        rel_left = (source_box[0] - text_source_union[0]) / union_w
        rel_top = (source_box[1] - text_source_union[1]) / union_h
        rel_right = (source_box[2] - text_source_union[0]) / union_w
        rel_bottom = (source_box[3] - text_source_union[1]) / union_h
        placement_w = max(1, placement_box[2] - placement_box[0])
        placement_h = max(1, placement_box[3] - placement_box[1])
        target_box = (
            placement_box[0] + int(round(rel_left * placement_w)),
            placement_box[1] + int(round(rel_top * placement_h)),
            placement_box[0] + int(round(rel_right * placement_w)),
            placement_box[1] + int(round(rel_bottom * placement_h)),
        )
        min_box_h = max(18, int((source_box[3] - source_box[1]) * area_scale * 0.75))
        if target_box[3] - target_box[1] < min_box_h:
            target_box = (target_box[0], target_box[1], target_box[2], min(placement_box[3], target_box[1] + min_box_h))
        if target_box[2] - target_box[0] < max(40, int((source_box[2] - source_box[0]) * area_scale * 0.55)):
            target_box = (target_box[0], target_box[1], min(placement_box[2], target_box[0] + max(40, int((source_box[2] - source_box[0]) * area_scale * 0.55))), target_box[3])
        style = layer.text_style
        source_height = max(1, source_box[3] - source_box[1])
        source_lines = [line.strip() for line in resize_text.splitlines() if line.strip()]
        line_count = max(1, len(source_lines))
        source_font_size = int(style.estimated_font_size) if style and style.estimated_font_size else max(8, int(source_height / line_count * 0.72))
        target_line_capacity = max(8, int(max(1, target_box[3] - target_box[1]) / line_count * 0.76))
        font_size = max(8, min(180, int(round(min(source_font_size * area_scale, target_line_capacity)))))
        color = smart_reframe_text_style_to_block_color(style)
        font_weight = 700 if style is None or style.is_bold else 400
        align = style.alignment if style and style.alignment != "unknown" else "center"
        font_category = (style.font_type if style and style.font_type != "unknown" else "sans-serif").replace("script", "handwriting")
        source_lines = [line.strip() for line in resize_text.splitlines() if line.strip()]
        if not source_lines:
            source_lines = [resize_text]
        # Resize is not localization: keep the source creative language exactly.
        # Analyzer translated_text may be populated by layout providers, but using
        # it here changes the user's uploaded creative language in Resize output.
        translated_lines = source_lines
        line_count_for_style = max(len(source_lines), 1)
        line_source_boxes = split_resize_box_into_line_boxes(source_box, line_count_for_style)

        source_word_styles: list[dict[str, Any]] = []
        spans: list[dict[str, Any]] = []
        for line_index, translated_line in enumerate(translated_lines):
            source_line = source_lines[min(line_index, len(source_lines) - 1)] if source_lines else translated_line
            line_source_box = line_source_boxes[min(line_index, len(line_source_boxes) - 1)]
            line_color, has_text_background, text_background_color = sample_resize_line_style(source, line_source_box, color)
            line_words = [word for word in translated_line.split() if word]
            line_word_styles: list[dict[str, Any]] = []
            line_target_top = target_box[1] + int(round(line_index * max(1, target_box[3] - target_box[1]) / max(1, len(translated_lines))))
            for word_index, word in enumerate(line_words):
                style_item = {
                    "id": f"{layer.id}-l{line_index}-w{word_index}",
                    "text": word,
                    "bbox": [
                        target_box[0],
                        line_target_top,
                        target_box[0] + max(1, font_size),
                        line_target_top + font_size,
                    ],
                    "fontSize": font_size,
                    "peerRowFontSize": font_size,
                    "fontWeight": font_weight,
                    "fontCategory": font_category,
                    "color": line_color,
                    "hasTextBackground": has_text_background,
                    "backgroundColor": text_background_color,
                    "lineIndex": line_index,
                }
                line_word_styles.append(style_item)
                source_word_styles.append(style_item)
            spans.append(
                {
                    "translatedText": translated_line,
                    "sourceText": source_line,
                    "semanticRole": layer.role.value,
                    "matchedSourceRole": layer.role.value,
                    "forceBreakAfter": True,
                    "sourceWordStyles": line_word_styles,
                    "sourceWordIds": [item["id"] for item in line_word_styles],
                    "style": {
                        "fontFamily": "DejaVu Sans",
                        "fontCategory": font_category,
                        "fontWeight": font_weight,
                        "fontSize": font_size,
                        "color": line_color,
                        "hasTextBackground": has_text_background,
                        "backgroundColor": text_background_color,
                        "opacity": 1.0,
                        "letterSpacing": 0.0,
                        "lineHeight": int(round(font_size * 1.18)),
                        "alignment": align,
                        "casing": "uppercase" if style and style.uppercase else "mixed",
                        "strokeWidth": 0,
                        "strokeFill": None,
                    },
                }
            )
        block_text = "\n".join(translated_lines)
        block = TextBlock(
            id=f"v5-resize-{layer.id}",
            text=block_text,
            translated_text=block_text,
            role=layer.role.value,
            translate=True,
            bbox=target_box,
            clean_box=target_box,
            font_weight=font_weight,
            font_size_estimate=font_size,
            line_height_estimate=int(round(font_size * 1.18)),
            color=color,
            align=align,
            surface="overlay",
            line_boxes=[target_box],
            line_texts=translated_lines,
            resize_source_box=source_box,
            source_word_styles=source_word_styles,
            translated_style_spans=spans,
            render_strategy="resize_compositor_redraw",
        )
        blocks.append(block)
    return blocks


def render_clean_base_outpaint_for_compositor(source: Image.Image, width: int, height: int, plan: Any, analysis: VisualAnalysis) -> tuple[Image.Image, dict[str, Any]]:
    openai_outpaint_enabled = os.getenv("ADAPTIFAI_ENABLE_OPENAI_OUTPAINT", "0").strip().lower() in {"1", "true", "yes", "on"}
    vertex_outpaint_enabled = env_flag("ADAPTIFAI_ENABLE_VERTEX_OUTPAINT", "1")
    if openai_outpaint_enabled:
        try:
            rendered, meta = render_openai_compositor_background_outpaint(source, width, height, plan, analysis)
            return rendered, {**meta, "strategy": "openai_outpaint_clean_base"}
        except Exception as exc:
            if vertex_outpaint_enabled and vertex_available():
                rendered, meta = render_vertex_compositor_background_outpaint(source, width, height, plan, analysis)
                return rendered, {**meta, "strategy": "vertex_outpaint_clean_base_after_openai_failed", "openaiFallbackReason": str(exc)}
            raise
    if vertex_outpaint_enabled and vertex_available():
        rendered, meta = render_vertex_compositor_background_outpaint(source, width, height, plan, analysis)
        return rendered, {**meta, "strategy": "vertex_outpaint_clean_base"}
    raise RuntimeError("No outpaint provider is enabled for deterministic compositor.")


def render_smart_reframe_image(
    source: Image.Image,
    width: int,
    height: int,
    plan: Any,
    analysis: VisualAnalysis,
    *,
    allow_provider_outpaint: bool = True,
    allow_product_completion: bool = True,
) -> tuple[Image.Image, dict[str, Any]]:
    text_blocks = build_resize_compositor_text_blocks(source, width, height, plan, analysis)
    return render_deterministic_compositor(
        source,
        width,
        height,
        plan,
        analysis,
        text_blocks=text_blocks,
        draw_text=draw_fitted_localize_v2_text,
        outpaint_renderer=render_clean_base_outpaint_for_compositor if allow_provider_outpaint else None,
        fallback_renderer=render_nonblur_contain_placeholder,
        product_completion_renderer=render_resize_product_asset_completion if allow_product_completion else None,
    )


def crop_to_ratio(image: Image.Image, focus_bbox: tuple[int, int, int, int], target_ratio: float, offset_x: int = 0, offset_y: int = 0) -> Image.Image:
    image_ratio = image.width / max(1, image.height)
    if abs(image_ratio - target_ratio) < 0.015:
        return image

    left, top, right, bottom = focus_bbox
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2

    crop_width = image.width
    crop_height = image.height
    if image_ratio > target_ratio:
        crop_width = max(int((bottom - top) * target_ratio), min(image.width, int(image.height * target_ratio)))
        crop_height = image.height
    else:
        crop_width = image.width
        crop_height = max(int((right - left) / target_ratio), min(image.height, int(image.width / target_ratio)))

    crop_width = min(image.width, max(1, crop_width))
    crop_height = min(image.height, max(1, crop_height))
    x0 = int(round(center_x - crop_width / 2 + (image.width - crop_width) * (offset_x / 100)))
    y0 = int(round(center_y - crop_height / 2 + (image.height - crop_height) * (offset_y / 100)))
    x0 = min(max(0, x0), image.width - crop_width)
    y0 = min(max(0, y0), image.height - crop_height)
    return image.crop((x0, y0, x0 + crop_width, y0 + crop_height))


def render_resize_image(source: Image.Image, width: int, height: int, fit: str = "cover", scale: int = 100, offset_x: int = 0, offset_y: int = 0, focus_bbox: tuple[int, int, int, int] | None = None) -> Image.Image:
    source = source.convert("RGB")
    background = Image.new("RGB", (width, height), tuple(int(value) for value in os.getenv("ADAPTIFAI_RESIZE_BG", "250,249,245").split(",")[:3]))
    if fit == "fill":
        return source.resize((width, height), Image.Resampling.LANCZOS)

    smart_source = source
    if fit == "cover":
        focus = focus_bbox or build_resize_focus_bbox(source)
        smart_source = crop_to_ratio(source, focus, width / max(1, height), offset_x, offset_y)
        offset_x = 0
        offset_y = 0

    if fit == "contain":
        base_scale = min(width / source.width, height / source.height)
    else:
        base_scale = max(width / smart_source.width, height / smart_source.height)

    factor = max(0.25, scale / 100)
    resized_source = source if fit == "contain" else smart_source
    target_width = max(1, int(resized_source.width * base_scale * factor))
    target_height = max(1, int(resized_source.height * base_scale * factor))
    resized = resized_source.resize((target_width, target_height), Image.Resampling.LANCZOS)

    room_x = width - target_width
    room_y = height - target_height
    pos_x = int(room_x / 2 + (room_x / 2) * (offset_x / 24))
    pos_y = int(room_y / 2 + (room_y / 2) * (offset_y / 24))

    background.paste(resized, (pos_x, pos_y))
    return background.crop((0, 0, width, height)) if fit == "cover" else background


def safe_zone_warnings_for(placement_id: str) -> list[str]:
    placement_id = canonical_placement_id(placement_id)
    return list(SHARED_PREVIEW_PLACEMENT_MAP.get(placement_id, {}).get("safeArea", {}).get("warnings", []))


PLACEMENT_ID_ALIASES = {
    "meta-feed": "social-feed-square",
    "facebook-feed": "social-feed-square",
    "facebook-marketplace": "social-feed-square",
    "instagram-feed": "social-feed-square",
    "linkedin-single-square": "social-feed-square",
    "meta-marketplace": "social-feed-square",
    "meta-right-column": "wide-landscape",
    "facebook-right-column": "wide-landscape",
    "linkedin-single-wide": "wide-landscape",
    "linkedin-sponsored": "wide-landscape",
    "meta-stories": "story-image",
    "instagram-story": "story-image",
    "snap-top-snap": "story-image",
    "snap-story-ad": "story-image",
    "instagram-reels": "story-image",
    "tiktok-in-feed": "story-image",
    "tiktok-topview": "story-image",
    "tiktok-branded": "story-image",
    "tiktok-branded-content": "story-image",
    "youtube-16x9": "google-responsive-landscape",
    "youtube-instream": "google-responsive-landscape",
    "youtube-shorts": "story-image",
    "native-custom": "custom-display",
}


def canonical_placement_id(placement_id: str) -> str:
    return PLACEMENT_ID_ALIASES.get(placement_id, placement_id)


PREVIEW_TEMPLATE_REGISTRY: dict[str, dict[str, Any]] = {
    placement_id: {
        "platform": item.get("platform", "NATIVE/WEB"),
        "placement": item.get("templateId", placement_id),
        "templateId": item.get("templateId", placement_id.replace("-", "_")),
        "status": item.get("templateStatus", "production"),
        "shell": item.get("templateId", placement_id.replace("-", "_")),
        "reusedShell": False,
        "reusedShellReason": "",
        "supportsCarousel": bool(item.get("carouselSupported", False)),
        "assetPlaceholderBox": item.get("assetPlaceholderBox", [0.05, 0.05, 0.9, 0.9]),
        "safeArea": item.get("safeArea", {"warnings": [], "zones": []}),
        "supportedMetadataFields": item.get("supportedMetadataFields", []),
        "uiElements": item.get("uiElements", []),
        "deviceFrame": bool(item.get("deviceFrame", False)),
        "resizeRules": item.get("resizeRules", {"mode": "cover", "protectText": True, "protectProduct": True}),
        "layoutBoxes": item.get("layoutBoxes", {}),
        "canvasSize": item.get("dimensions", {"width": 1200, "height": 800}),
    }
    for placement_id, item in SHARED_PREVIEW_PLACEMENT_MAP.items()
}

for alias_id, canonical_id in PLACEMENT_ID_ALIASES.items():
    canonical_template = PREVIEW_TEMPLATE_REGISTRY.get(canonical_id)
    if not canonical_template:
        continue
    PREVIEW_TEMPLATE_REGISTRY[alias_id] = {
        **canonical_template,
        "reusedShell": True,
        "reusedShellReason": f"legacy alias to {canonical_id}",
    }


def placement_preview_metadata(placement_id: str, source_name: str, translated_text: str | None, creative_mode: str = "single") -> dict[str, Any]:
    placement_id = canonical_placement_id(placement_id)
    base_name = Path(source_name).stem.replace("-", " ").replace("_", " ").strip() or "AdaptifAI"
    lines = [line.strip() for line in (translated_text or "").splitlines() if line.strip()]
    headline = " ".join(lines[:2]).strip() or "Localized campaign headline"
    description = " ".join(lines[2:]).strip() or "Adapted creative placed inside a native ad context."
    cta_map = {
        "facebook-feed": "Shop Now",
        "facebook-marketplace": "Shop Now",
        "facebook-right-column": "Learn More",
        "instagram-feed": "Shop Now",
        "instagram-story": "Learn More",
        "instagram-reels": "Learn More",
        "tiktok-in-feed": "Shop Now",
        "tiktok-topview": "Learn More",
        "tiktok-branded-content": "Learn More",
        "snap-top-snap": "Swipe Up",
        "snap-story-ad": "Swipe Up",
        "linkedin-single-wide": "Visit Website",
        "linkedin-single-square": "Visit Website",
        "linkedin-sponsored": "Visit Website",
        "youtube-instream": "Learn More",
        "youtube-shorts": "Shop Now",
        "custom-display": "Read More",
    }
    return {
        "brandName": base_name.title(),
        "accountName": "brand.co",
        "sponsorLabel": "Promoted" if placement_id.startswith("linkedin") else "Sponsored",
        "headline": headline,
        "description": description,
        "ctaText": cta_map.get(placement_id, "Learn More"),
        "caption": description,
        "price": "â‚¬39.90",
        "creativeMode": creative_mode,
        "carouselActivationSource": "user_selected" if creative_mode == "carousel" else "forced_single",
        "unusedAssets": [],
        "carouselAssetsProvided": False,
        "carouselAssets": [],
    }


def render_native_placement_preview(
    asset: Image.Image,
    placement_id: str,
    metadata: dict[str, Any],
    carousel_assets: list[Image.Image] | None = None,
    active_slide_index: int = 0,
) -> tuple[Image.Image, dict[str, Any], dict[str, Any], dict[str, Any]]:
    placement_id = canonical_placement_id(placement_id)
    template = PREVIEW_TEMPLATE_REGISTRY.get(placement_id, PREVIEW_TEMPLATE_REGISTRY["custom-display"])
    shell = template["shell"]
    asset_width, asset_height = PLACEMENT_DIMENSIONS.get(placement_id, PLACEMENT_DIMENSIONS["custom-display"])
    if placement_id.startswith("gdn-"):
        canvas_width, canvas_height = asset_width + 24, asset_height + 24
    elif placement_id == "facebook-right-column":
        canvas_width, canvas_height = 440, 560
    elif placement_id in {"linkedin-single-wide", "linkedin-sponsored", "youtube-instream", "custom-display"}:
        canvas_width, canvas_height = 900, 720
    elif placement_id in {"linkedin-single-square", "facebook-feed", "facebook-marketplace", "instagram-feed"}:
        canvas_width, canvas_height = 460, 860
    else:
        canvas_width, canvas_height = 380, 860

    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f4f5f7")
    draw = ImageDraw.Draw(canvas)
    black = "#151515"
    gray = "#667085"
    blue = "#2550a8"
    light = "#f7f8fb"
    white = "#ffffff"
    boxes: dict[str, list[int]] = {}
    ui_elements_rendered: list[str] = []
    warnings = list(safe_zone_warnings_for(placement_id))
    carousel_supported = bool(template.get("supportsCarousel"))
    requested_creative_mode = str(metadata.get("creativeMode") or "single").strip().lower()
    creative_mode = requested_creative_mode if requested_creative_mode in {"single", "carousel"} else "single"
    resolved_carousel_assets = [item for item in (carousel_assets or []) if isinstance(item, Image.Image)]
    carousel_asset_labels = metadata.get("carouselAssetLabels") or []
    carousel_assets_provided = len(resolved_carousel_assets) > 1 if resolved_carousel_assets else bool(metadata.get("carouselAssetsProvided"))
    if not carousel_supported:
        creative_mode = "single"
        carousel_activation_source = "forced_single"
    elif creative_mode == "carousel" and not carousel_assets_provided:
        carousel_activation_source = "invalid_missing_assets"
    else:
        carousel_activation_source = "user_selected"
    carousel_assets_missing = carousel_supported and creative_mode == "carousel" and not carousel_assets_provided
    carousel_slide_count = len(resolved_carousel_assets) if resolved_carousel_assets else 1
    carousel_active_index = max(0, min(active_slide_index, max(0, carousel_slide_count - 1)))
    if carousel_assets_missing:
        warnings.append("carouselAssetsMissing")
    unused_assets = carousel_asset_labels[1:] if creative_mode == "single" and isinstance(carousel_asset_labels, list) and len(carousel_asset_labels) > 1 else []

    def draw_text_block(x: int, y: int, text: str, font_size: int, fill: str, max_width: int, *, bold: bool = False, max_lines: int = 2) -> int:
        font = get_font(font_size, bold=bold)
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if text_width(draw, candidate, font) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        lines = lines[:max_lines]
        cursor_y = y
        for line in lines:
            draw.text((x, cursor_y), line, fill=fill, font=font)
            cursor_y += int(font_size * 1.25)
        return cursor_y

    def draw_carousel_indicators(box: tuple[int, int, int, int], count: int, active: int) -> None:
        if count <= 1:
            return
        dot_size = max(6, min(10, int((box[3] - box[1]) * 0.045)))
        gap = max(6, dot_size // 2)
        total_width = count * dot_size + (count - 1) * gap
        start_x = box[0] + max(12, (box[2] - box[0] - total_width) // 2)
        y = box[3] - dot_size - max(10, dot_size)
        for index in range(count):
            fill = "#ffffff" if index == active else "#ffffff88"
            width = dot_size * 2 if index == active else dot_size
            draw.rounded_rectangle((start_x, y, start_x + width, y + dot_size), radius=dot_size // 2, fill=fill)
            start_x += width + gap
        ui_elements_rendered.append("carousel_dots")

    def paste_asset(box: tuple[int, int, int, int], fit: str = "contain") -> None:
        target_w = max(1, box[2] - box[0])
        target_h = max(1, box[3] - box[1])
        if carousel_supported and creative_mode == "carousel" and carousel_assets_provided and resolved_carousel_assets:
            current_asset = resolved_carousel_assets[carousel_active_index]
            prepared = render_resize_image(current_asset, target_w, target_h, fit=fit)
            canvas.paste(prepared, (box[0], box[1]))
            if len(resolved_carousel_assets) > 1:
                next_index = (carousel_active_index + 1) % len(resolved_carousel_assets)
                next_asset = render_resize_image(resolved_carousel_assets[next_index], max(1, int(target_w * 0.38)), target_h, fit=fit)
                peek_box = (
                    max(box[0] + int(target_w * 0.74), box[0] + 12),
                    box[1] + max(8, target_h // 18),
                    box[2] - max(10, target_w // 30),
                    box[3] - max(8, target_h // 18),
                )
                peek_width = max(1, peek_box[2] - peek_box[0])
                peek_height = max(1, peek_box[3] - peek_box[1])
                next_asset = next_asset.resize((peek_width, peek_height), Image.Resampling.LANCZOS)
                overlay = Image.new("RGBA", (peek_width, peek_height), (255, 255, 255, 0))
                overlay.paste(next_asset.convert("RGBA"), (0, 0))
                ImageDraw.Draw(overlay).rounded_rectangle((0, 0, peek_width - 1, peek_height - 1), radius=max(12, peek_width // 12), outline=(255, 255, 255, 180), width=2)
                canvas.paste(overlay.convert("RGB"), (peek_box[0], peek_box[1]))
                arrow_radius = max(14, min(22, target_w // 14))
                left_center = (box[0] + arrow_radius + 12, box[1] + target_h // 2)
                right_center = (box[2] - arrow_radius - 12, box[1] + target_h // 2)
                draw.ellipse((left_center[0] - arrow_radius, left_center[1] - arrow_radius, left_center[0] + arrow_radius, left_center[1] + arrow_radius), fill="#10111499")
                draw.ellipse((right_center[0] - arrow_radius, right_center[1] - arrow_radius, right_center[0] + arrow_radius, right_center[1] + arrow_radius), fill="#10111499")
                draw.text((left_center[0], left_center[1] - 2), "â€¹", fill="white", font=get_font(max(16, arrow_radius), bold=True), anchor="mm")
                draw.text((right_center[0], right_center[1] - 2), "â€º", fill="white", font=get_font(max(16, arrow_radius), bold=True), anchor="mm")
                ui_elements_rendered.extend(["carousel_arrows", "carousel_peek"])
            draw_carousel_indicators(box, len(resolved_carousel_assets), carousel_active_index)
        else:
            prepared = render_resize_image(asset, target_w, target_h, fit=fit)
            canvas.paste(prepared, (box[0], box[1]))
        boxes["asset"] = [box[0], box[1], box[2], box[3]]

    def draw_phone_shell(fill: str = "#111111") -> tuple[int, int, int, int]:
        outer = (24, 20, canvas_width - 24, canvas_height - 20)
        draw.rounded_rectangle(outer, radius=36, fill=fill)
        draw.rounded_rectangle((canvas_width // 2 - 58, 28, canvas_width // 2 + 58, 42), radius=8, fill="#0a0a0a")
        inner = (38, 34, canvas_width - 38, canvas_height - 34)
        boxes["deviceFrame"] = list(outer)
        ui_elements_rendered.append("device_frame")
        return inner

    def draw_progress(inner: tuple[int, int, int, int], count: int = 5, active: int = 1) -> None:
        progress_height = 4
        total_gap = 4 * (count - 1)
        segment_w = (inner[2] - inner[0] - 24 - total_gap) // count
        x = inner[0] + 12
        for index in range(count):
            fill = "#ffffff" if index < active else "#ffffff66"
            draw.rounded_rectangle((x, inner[1] + 12, x + segment_w, inner[1] + 12 + progress_height), radius=2, fill=fill)
            x += segment_w + 4
        ui_elements_rendered.append("progress_bar")

    def draw_action_rail(inner: tuple[int, int, int, int], labels: list[str], *, start_y: int) -> None:
        x = inner[2] - 42
        for index, label in enumerate(labels):
            y = start_y + index * 54
            draw.rounded_rectangle((x, y, x + 32, y + 32), radius=16, fill="#0f1014")
            draw.text((x + 7, y + 8), label, fill="white", font=get_font(11, bold=True))
        ui_elements_rendered.append("action_rail")

    if shell == "facebook_feed":
        card = (18, 18, canvas_width - 18, canvas_height - 18)
        draw.rounded_rectangle(card, radius=24, fill=white, outline="#d8dce6")
        header = (card[0] + 18, card[1] + 16, card[2] - 18, card[1] + 72)
        draw.ellipse((header[0], header[1], header[0] + 36, header[1] + 36), fill=blue)
        draw_text_block(header[0] + 48, header[1] + 2, metadata["accountName"], 17, black, 180, bold=True, max_lines=1)
        draw.text((header[0] + 48, header[1] + 24), metadata["sponsorLabel"], fill=gray, font=get_font(12))
        draw.text((header[2] - 10, header[1] + 8), "...", fill=gray, font=get_font(16, bold=True), anchor="ra")
        asset_box = (card[0] + 12, header[3] + 8, card[2] - 12, card[1] + 560)
        paste_asset(asset_box, fit="contain")
        actions_y = asset_box[3] + 10
        draw.text((card[0] + 18, actions_y), "Like   Comment   Share", fill=gray, font=get_font(12, bold=True))
        draw.text((card[0] + 18, actions_y + 28), "1,234 likes", fill=black, font=get_font(12, bold=True))
        draw_text_block(card[0] + 18, actions_y + 50, metadata["caption"], 12, black, card[2] - card[0] - 36, max_lines=3)
        draw.text((card[0] + 18, card[3] - 44), "View all 12 comments", fill=gray, font=get_font(11))
        ui_elements_rendered.extend(["profile_image", "account_name", "sponsored_label", "menu", "likes", "caption", "comments_preview", "social_actions"])
    elif shell == "facebook_marketplace":
        draw.rectangle((0, 0, canvas_width, 88), fill=white)
        draw.text((22, 20), "Marketplace", fill=black, font=get_font(21, bold=True))
        draw.text((canvas_width - 32, 20), "Search", fill=gray, font=get_font(11), anchor="ra")
        card = (24, 108, canvas_width - 24, canvas_height - 28)
        draw.rounded_rectangle(card, radius=24, fill=white, outline="#d9dee8")
        asset_box = (card[0] + 14, card[1] + 14, card[2] - 14, card[1] + 390)
        paste_asset(asset_box, fit="contain")
        draw.text((card[0] + 18, asset_box[3] + 16), metadata["price"], fill=black, font=get_font(21, bold=True))
        draw_text_block(card[0] + 18, asset_box[3] + 52, metadata["headline"], 15, black, card[2] - card[0] - 150, bold=True, max_lines=2)
        draw_text_block(card[0] + 18, asset_box[3] + 94, metadata["description"], 12, gray, card[2] - card[0] - 36, max_lines=2)
        draw.rounded_rectangle((card[2] - 146, card[3] - 52, card[2] - 18, card[3] - 16), radius=16, fill=blue)
        draw.text((card[2] - 82, card[3] - 42), metadata["ctaText"], fill=white, font=get_font(12, bold=True), anchor="ma")
        draw.rounded_rectangle((card[0] + 18, card[3] - 52, card[0] + 120, card[3] - 20), radius=16, fill="#eef4ff")
        draw.text((card[0] + 69, card[3] - 42), metadata["sponsorLabel"], fill=blue, font=get_font(11, bold=True), anchor="ma")
        ui_elements_rendered.extend(["marketplace_header", "listing_card", "price", "sponsored_label", "cta"])
    elif shell == "facebook_right_column":
        draw.rounded_rectangle((24, 18, canvas_width - 24, canvas_height - 18), radius=16, fill=white, outline="#d7dce5")
        draw.text((40, 32), "Sponsored", fill=gray, font=get_font(11, bold=True))
        asset_box = (40, 58, canvas_width - 40, 250)
        paste_asset(asset_box, fit="contain")
        draw_text_block(40, asset_box[3] + 14, metadata["headline"], 15, black, canvas_width - 92, bold=True, max_lines=2)
        draw_text_block(40, asset_box[3] + 54, metadata["description"], 11, gray, canvas_width - 92, max_lines=3)
        draw.rounded_rectangle((40, canvas_height - 62, 164, canvas_height - 26), radius=8, fill=blue)
        draw.text((102, canvas_height - 51), metadata["ctaText"], fill=white, font=get_font(11, bold=True), anchor="ma")
        ui_elements_rendered.extend(["desktop_ad_label", "headline", "description", "cta"])
    elif shell == "instagram_feed":
        inner = draw_phone_shell()
        header = (inner[0] + 8, inner[1] + 8, inner[2] - 8, inner[1] + 54)
        draw.ellipse((header[0], header[1], header[0] + 32, header[1] + 32), fill="#1d1d1f")
        draw_text_block(header[0] + 42, header[1], metadata["accountName"], 13, black, 120, bold=True, max_lines=1)
        draw.text((header[0] + 42, header[1] + 20), metadata["sponsorLabel"], fill=gray, font=get_font(10))
        draw.text((header[2] - 8, header[1] + 4), "...", fill=gray, font=get_font(16, bold=True), anchor="ra")
        asset_box = (inner[0], header[3] + 4, inner[2], header[3] + 4 + (inner[2] - inner[0]))
        paste_asset(asset_box, fit="cover")
        draw.text((inner[0] + 12, asset_box[3] + 14), "Like   Comment   Share", fill=black, font=get_font(11, bold=True))
        draw.text((inner[2] - 12, asset_box[3] + 14), "Save", fill=black, font=get_font(11, bold=True), anchor="ra")
        draw.text((inner[0] + 12, asset_box[3] + 36), metadata.get("likesLabel", "1,234 likes"), fill=black, font=get_font(11, bold=True))
        draw_text_block(inner[0] + 12, asset_box[3] + 56, f"{metadata['accountName']} {metadata['headline']}", 11, black, inner[2] - inner[0] - 24, max_lines=2)
        draw_text_block(inner[0] + 12, asset_box[3] + 84, metadata["description"], 11, gray, inner[2] - inner[0] - 24, max_lines=2)
        draw.text((inner[0] + 12, inner[3] - 26), "View all 12 comments", fill=gray, font=get_font(10))
        ui_elements_rendered.extend(["device_frame", "profile_image", "account_name", "sponsored_label", "feed_asset", "social_actions", "caption", "bottom_navigation"])
    elif shell == "instagram_story":
        inner = draw_phone_shell()
        paste_asset(inner, fit="cover")
        draw_progress(inner, count=5, active=1)
        draw.text((inner[0] + 12, inner[1] + 24), metadata["accountName"], fill=white, font=get_font(12, bold=True))
        draw.text((inner[0] + 12, inner[1] + 40), metadata["sponsorLabel"], fill="#f5f5f5", font=get_font(10))
        draw.text((inner[2] - 12, inner[1] + 24), "Share", fill=white, font=get_font(10), anchor="ra")
        cta = (inner[0] + 18, inner[3] - 54, inner[2] - 18, inner[3] - 14)
        draw.rounded_rectangle(cta, radius=20, fill=white)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 10), metadata["ctaText"], fill=black, font=get_font(12, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["progress_bar", "account_name", "sponsored_label", "share_control", "cta"])
    elif shell == "instagram_reels":
        inner = draw_phone_shell()
        paste_asset(inner, fit="cover")
        draw_action_rail(inner, ["Like", "Comment", "Share", "More"], start_y=inner[1] + 220)
        draw.text((inner[0] + 14, inner[3] - 92), f"@{metadata['accountName']}", fill=white, font=get_font(12, bold=True))
        draw_text_block(inner[0] + 14, inner[3] - 72, metadata["headline"], 12, white, inner[2] - inner[0] - 72, bold=True, max_lines=2)
        draw_text_block(inner[0] + 14, inner[3] - 42, metadata["description"], 10, "#ececec", inner[2] - inner[0] - 72, max_lines=2)
        ui_elements_rendered.extend(["device_frame", "action_rail", "caption", "audio_row"])
    elif shell == "tiktok_infeed":
        inner = draw_phone_shell(fill="#0d0f12")
        paste_asset(inner, fit="cover")
        draw.text((inner[0] + 14, inner[1] + 16), metadata["sponsorLabel"], fill=white, font=get_font(11, bold=True))
        draw_action_rail(inner, ["Like", "Comment", "Share"], start_y=inner[1] + 250)
        draw.text((inner[0] + 14, inner[3] - 90), f"@{metadata['accountName']}", fill=white, font=get_font(11, bold=True))
        draw_text_block(inner[0] + 14, inner[3] - 70, metadata["description"], 10, "#e5e5e5", inner[2] - inner[0] - 78, max_lines=2)
        cta = (inner[0] + 14, inner[3] - 36, inner[0] + 148, inner[3] - 8)
        draw.rounded_rectangle(cta, radius=14, fill=white)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 7), metadata["ctaText"], fill=black, font=get_font(10, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["device_frame", "sponsored_label", "action_rail", "caption", "cta"])
    elif shell == "tiktok_topview":
        inner = draw_phone_shell(fill="#0d0f12")
        paste_asset(inner, fit="cover")
        draw.rounded_rectangle((inner[0] + 12, inner[1] + 12, inner[0] + 108, inner[1] + 34), radius=11, fill="#ffffff")
        draw.text((inner[0] + 60, inner[1] + 18), "TopView", fill=black, font=get_font(10, bold=True), anchor="ma")
        draw_action_rail(inner, ["Like", "Comment", "Share"], start_y=inner[1] + 230)
        draw_text_block(inner[0] + 14, inner[3] - 96, metadata["headline"], 13, white, inner[2] - inner[0] - 82, bold=True, max_lines=2)
        draw_text_block(inner[0] + 14, inner[3] - 62, metadata["description"], 10, "#e5e5e5", inner[2] - inner[0] - 82, max_lines=2)
        cta = (inner[0] + 14, inner[3] - 34, inner[0] + 170, inner[3] - 6)
        draw.rounded_rectangle(cta, radius=14, fill="#f8d948")
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 7), metadata["ctaText"], fill=black, font=get_font(10, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["device_frame", "topview_badge", "action_rail", "caption", "cta"])
    elif shell == "tiktok_branded_content":
        inner = draw_phone_shell(fill="#0d0f12")
        paste_asset(inner, fit="cover")
        draw.rounded_rectangle((inner[0] + 12, inner[1] + 12, inner[0] + 152, inner[1] + 36), radius=12, fill="#ffffff33")
        draw.text((inner[0] + 82, inner[1] + 18), "Branded content", fill=white, font=get_font(10, bold=True), anchor="ma")
        draw_action_rail(inner, ["Like", "Comment", "Share"], start_y=inner[1] + 245)
        draw.text((inner[0] + 14, inner[3] - 78), f"Creator x {metadata['brandName']}", fill=white, font=get_font(11, bold=True))
        draw_text_block(inner[0] + 14, inner[3] - 56, metadata["description"], 10, "#e8e8e8", inner[2] - inner[0] - 78, max_lines=2)
        ui_elements_rendered.extend(["device_frame", "branded_label", "action_rail", "creator_context"])
    elif shell == "snap_top_snap":
        inner = draw_phone_shell(fill="#0f1115")
        paste_asset(inner, fit="cover")
        draw.text((inner[0] + 14, inner[1] + 18), metadata["brandName"], fill=white, font=get_font(11, bold=True))
        draw.text((inner[2] - 14, inner[1] + 18), metadata["sponsorLabel"], fill=white, font=get_font(10), anchor="ra")
        cta = (inner[0] + 24, inner[3] - 38, inner[2] - 24, inner[3] - 10)
        draw.rounded_rectangle(cta, radius=14, fill=white)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 7), metadata["ctaText"], fill=black, font=get_font(10, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["device_frame", "brand_label", "sponsored_label", "cta"])
    elif shell == "snap_story_ad":
        inner = draw_phone_shell(fill="#0f1115")
        paste_asset(inner, fit="cover")
        draw_progress(inner, count=4, active=1)
        draw.text((inner[0] + 14, inner[1] + 24), metadata["brandName"], fill=white, font=get_font(11, bold=True))
        cta = (inner[0] + 24, inner[3] - 42, inner[2] - 24, inner[3] - 10)
        draw.rounded_rectangle(cta, radius=16, fill=white)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 8), metadata["ctaText"], fill=black, font=get_font(10, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["device_frame", "progress_bar", "brand_label", "cta"])
    elif shell in {"linkedin_single_image_1200x628", "linkedin_single_image_1080x1080", "linkedin_sponsored_content"}:
        card = (20, 20, canvas_width - 20, canvas_height - 20)
        draw.rounded_rectangle(card, radius=18, fill=white, outline="#d7dce5")
        draw.ellipse((card[0] + 18, card[1] + 16, card[0] + 52, card[1] + 50), fill="#0a66c2")
        draw_text_block(card[0] + 62, card[1] + 14, metadata["brandName"], 16, black, 220, bold=True, max_lines=1)
        draw.text((card[0] + 62, card[1] + 34), metadata["sponsorLabel"], fill=gray, font=get_font(11))
        asset_box = (card[0] + 18, card[1] + 70, card[2] - 18, card[1] + (430 if shell == "linkedin_single_image_1080x1080" else 380))
        paste_asset(asset_box, fit="contain")
        draw_text_block(card[0] + 18, asset_box[3] + 14, metadata["headline"], 16, black, card[2] - card[0] - 36, bold=True, max_lines=2)
        draw_text_block(card[0] + 18, asset_box[3] + 54, metadata["description"], 12, gray, card[2] - card[0] - 36, max_lines=2)
        cta = (card[2] - 168, card[3] - 54, card[2] - 18, card[3] - 20)
        draw.rounded_rectangle(cta, radius=8, fill="#0a66c2")
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 8), metadata["ctaText"], fill=white, font=get_font(11, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["company_header", "promoted_label", "headline", "description", "cta", "social_actions"])
    elif shell == "gdn_300x250":
        draw.rounded_rectangle((12, 12, canvas_width - 12, canvas_height - 12), radius=12, fill=white, outline="#cdd4df")
        draw.text((24, 22), metadata["brandName"], fill=gray, font=get_font(10, bold=True))
        asset_box = (20, 42, canvas_width - 20, canvas_height - 66)
        paste_asset(asset_box, fit="contain")
        draw.rounded_rectangle((canvas_width - 98, canvas_height - 46, canvas_width - 22, canvas_height - 20), radius=10, fill=blue)
        draw.text((canvas_width - 60, canvas_height - 39), metadata["ctaText"], fill=white, font=get_font(10, bold=True), anchor="ma")
        ui_elements_rendered.extend(["banner_container", "brand_label", "cta"])
    elif shell == "gdn_728x90":
        draw.rounded_rectangle((8, 8, canvas_width - 8, canvas_height - 8), radius=10, fill=white, outline="#cdd4df")
        asset_box = (14, 14, int(canvas_width * 0.58), canvas_height - 14)
        paste_asset(asset_box, fit="contain")
        draw_text_block(asset_box[2] + 12, 20, metadata["headline"], 14, black, canvas_width - asset_box[2] - 30, bold=True, max_lines=2)
        draw.rounded_rectangle((canvas_width - 120, canvas_height - 34, canvas_width - 16, canvas_height - 12), radius=10, fill=blue)
        draw.text((canvas_width - 68, canvas_height - 28), metadata["ctaText"], fill=white, font=get_font(10, bold=True), anchor="ma")
        ui_elements_rendered.extend(["banner_container", "headline", "cta"])
    elif shell == "gdn_160x600":
        draw.rounded_rectangle((8, 8, canvas_width - 8, canvas_height - 8), radius=10, fill=white, outline="#cdd4df")
        asset_box = (18, 18, canvas_width - 18, 360)
        paste_asset(asset_box, fit="contain")
        draw_text_block(18, 382, metadata["headline"], 13, black, canvas_width - 36, bold=True, max_lines=3)
        draw_text_block(18, 448, metadata["description"], 10, gray, canvas_width - 36, max_lines=4)
        draw.rounded_rectangle((18, canvas_height - 52, canvas_width - 18, canvas_height - 18), radius=10, fill=blue)
        draw.text((canvas_width // 2, canvas_height - 43), metadata["ctaText"], fill=white, font=get_font(10, bold=True), anchor="ma")
        ui_elements_rendered.extend(["banner_container", "headline", "description", "cta"])
    elif shell == "gdn_320x50":
        draw.rounded_rectangle((8, 8, canvas_width - 8, canvas_height - 8), radius=10, fill=white, outline="#cdd4df")
        asset_box = (14, 14, 102, canvas_height - 14)
        paste_asset(asset_box, fit="contain")
        draw_text_block(112, 16, metadata["headline"], 10, black, canvas_width - 200, bold=True, max_lines=2)
        draw.rounded_rectangle((canvas_width - 88, 14, canvas_width - 16, canvas_height - 14), radius=8, fill=blue)
        draw.text((canvas_width - 52, 20), metadata["ctaText"], fill=white, font=get_font(9, bold=True), anchor="ma")
        ui_elements_rendered.extend(["banner_container", "headline", "cta"])
    elif shell == "gdn_300x600":
        draw.rounded_rectangle((10, 10, canvas_width - 10, canvas_height - 10), radius=10, fill=white, outline="#cdd4df")
        asset_box = (18, 18, canvas_width - 18, 340)
        paste_asset(asset_box, fit="contain")
        draw_text_block(18, 362, metadata["headline"], 16, black, canvas_width - 36, bold=True, max_lines=3)
        draw_text_block(18, 440, metadata["description"], 11, gray, canvas_width - 36, max_lines=4)
        draw.rounded_rectangle((18, canvas_height - 54, canvas_width - 18, canvas_height - 18), radius=12, fill=blue)
        draw.text((canvas_width // 2, canvas_height - 44), metadata["ctaText"], fill=white, font=get_font(11, bold=True), anchor="ma")
        ui_elements_rendered.extend(["banner_container", "headline", "description", "cta"])
    elif shell == "youtube_instream":
        draw.rounded_rectangle((20, 20, canvas_width - 20, canvas_height - 20), radius=18, fill="#121212")
        player = (34, 34, canvas_width - 34, 474)
        paste_asset(player, fit="cover")
        draw.rectangle((player[0], player[3] - 8, player[2], player[3]), fill="#2b2c31")
        draw.rectangle((player[0], player[3] - 8, player[0] + int((player[2] - player[0]) * 0.24), player[3]), fill="#ff0033")
        draw.rounded_rectangle((player[0] + 12, player[1] + 12, player[0] + 70, player[1] + 32), radius=8, fill="#00000099")
        draw.text((player[0] + 41, player[1] + 17), "Ad", fill=white, font=get_font(10, bold=True), anchor="ma")
        draw_text_block(40, 500, metadata["brandName"], 16, white, 260, bold=True, max_lines=1)
        draw_text_block(40, 526, metadata["headline"], 14, white, canvas_width - 220, bold=True, max_lines=2)
        draw_text_block(40, 564, metadata["description"], 11, "#d0d5dd", canvas_width - 220, max_lines=2)
        cta = (canvas_width - 170, 520, canvas_width - 36, 556)
        draw.rounded_rectangle(cta, radius=16, fill=white)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 10), metadata["ctaText"], fill=black, font=get_font(11, bold=True), anchor="ma")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["player_context", "ad_label", "progress_bar", "cta"])
    elif shell == "youtube_shorts":
        inner = draw_phone_shell(fill="#0d0f12")
        paste_asset(inner, fit="cover")
        draw_action_rail(inner, ["Like", "Comment", "Share"], start_y=inner[1] + 238)
        draw_text_block(inner[0] + 14, inner[3] - 86, metadata["headline"], 12, white, inner[2] - inner[0] - 78, bold=True, max_lines=2)
        draw_text_block(inner[0] + 14, inner[3] - 54, metadata["description"], 10, "#ececec", inner[2] - inner[0] - 78, max_lines=2)
        ui_elements_rendered.extend(["device_frame", "shorts_action_rail", "caption"])
    else:
        draw.rounded_rectangle((20, 20, canvas_width - 20, canvas_height - 20), radius=18, fill=white, outline="#d8dce6")
        draw.rectangle((20, 20, canvas_width - 20, 52), fill=light)
        draw.text((38, 30), "Publisher", fill="#0f766e", font=get_font(12, bold=True))
        draw.text((canvas_width - 38, 30), "Ad", fill=gray, font=get_font(10, bold=True), anchor="ra")
        draw_text_block(38, 84, metadata["headline"], 20, black, canvas_width // 2 - 64, bold=True, max_lines=3)
        draw_text_block(38, 170, metadata["description"], 12, gray, canvas_width // 2 - 64, max_lines=4)
        cta = (38, canvas_height - 68, 180, canvas_height - 28)
        draw.rounded_rectangle(cta, radius=18, fill=black)
        draw.text(((cta[0] + cta[2]) // 2, cta[1] + 11), metadata["ctaText"], fill=white, font=get_font(11, bold=True), anchor="ma")
        asset_box = (canvas_width // 2 + 20, 94, canvas_width - 36, canvas_height - 42)
        paste_asset(asset_box, fit="contain")
        boxes["cta"] = list(cta)
        ui_elements_rendered.extend(["publisher_header", "headline", "description", "cta"])

    safe_area = {"warnings": warnings}
    preview_template_used = {
        "platform": template["platform"],
        "placement": placement_id,
        "templateId": template["templateId"],
        "templateStatus": template["status"],
        "sharedTemplateSchemaVersion": SHARED_TEMPLATE_SCHEMA_VERSION,
        "creativeMode": creative_mode,
        "carouselActivationSource": carousel_activation_source,
        "unusedAssets": unused_assets,
        "canvasSize": [canvas_width, canvas_height],
        "assetPlaceholderBox": template.get("assetPlaceholderBox", boxes.get("asset")),
        "safeArea": safe_area,
        "uiElementsRendered": ui_elements_rendered,
        "reusedShell": bool(template.get("reusedShell")),
        "reusedShellReason": template.get("reusedShellReason", ""),
        "carouselSupported": carousel_supported,
        "carouselAssetsProvided": carousel_assets_provided,
        "carouselPreviewGenerated": carousel_supported and carousel_assets_provided,
        "carouselAssetsMissing": carousel_assets_missing,
    }
    placement_render_log = {
        "brandName": metadata.get("brandName"),
        "accountName": metadata.get("accountName"),
        "sponsorLabel": metadata.get("sponsorLabel"),
        "headline": metadata.get("headline"),
        "description": metadata.get("description"),
        "CTA": metadata.get("ctaText"),
        "caption": metadata.get("caption"),
        "price": metadata.get("price"),
        "creativeMode": creative_mode,
        "carouselActivationSource": carousel_activation_source,
        "unusedAssets": unused_assets,
        "carouselMetadata": {
            "supported": carousel_supported,
            "assetsProvided": carousel_assets_provided,
            "assetsMissing": carousel_assets_missing,
            "count": len(resolved_carousel_assets),
            "activeSlideIndex": carousel_active_index,
        },
        "warnings": warnings,
        "layoutBoxes": boxes,
        "placement": placement_id,
        "templateId": template["templateId"],
        "templateStatus": template["status"],
    }
    carousel_render_log = {
        "placement": placement_id,
        "templateId": template["templateId"],
        "supported": carousel_supported,
        "creativeMode": creative_mode,
        "carouselActivationSource": carousel_activation_source,
        "unusedAssets": unused_assets,
        "assetsProvided": carousel_assets_provided,
        "assetsMissing": carousel_assets_missing,
        "activeSlideIndex": carousel_active_index,
        "slideCount": len(resolved_carousel_assets),
        "assetLabels": carousel_asset_labels[: len(resolved_carousel_assets)] if isinstance(carousel_asset_labels, list) else [],
        "warnings": [warning for warning in warnings if "carousel" in warning.lower()],
        "uiElementsRendered": [item for item in ui_elements_rendered if item.startswith("carousel_")],
    }
    return canvas, preview_template_used, placement_render_log, carousel_render_log

def sanitize_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in path.stem.lower()).strip("-") or "creative"


def sanitize_debug_token(value: Any) -> str:
    token = str(value or "unknown").strip().lower().replace("_", "-")
    return "".join(char if char.isalnum() or char == "-" else "-" for char in token).strip("-") or "unknown"


def debug_strategy_filenames_enabled() -> bool:
    return os.getenv("ADAPTIFAI_DEBUG_STRATEGY_FILENAMES", "0").strip().lower() in {"1", "true", "yes", "on"}


def source_suffix(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return digest


def localize_filename(source_path: Path, language: str, output_format: str) -> str:
    extension = normalize_output_format(output_format, source_path)
    return f"{sanitize_stem(source_path)}-{language.lower()}.{extension}"


def resize_filename(source_path: Path, placement_id: str, output_format: str) -> str:
    extension = normalize_output_format(output_format, source_path)
    return f"{sanitize_stem(source_path)}-{placement_id}.{extension}"


def localization_v2_enabled() -> bool:
    return True


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "bold", "uppercase", "all_caps"}


def plain_text_from_segments(segments: Any) -> str:
    if not isinstance(segments, list):
        return ""
    parts: list[str] = []
    for raw in segments:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or raw.get("translatedText") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def plain_text_from_translation_item(item: dict[str, Any]) -> str:
    lines = item.get("lines")
    if isinstance(lines, list):
        line_texts: list[str] = []
        for line in lines:
            if not isinstance(line, dict):
                continue
            text = plain_text_from_segments(line.get("segments"))
            if text:
                line_texts.append(text)
        if line_texts:
            return "\n".join(line_texts).strip()
    return plain_text_from_segments(item.get("segments"))


def normalize_rich_text_segments(raw_segments: Any, *, block: TextBlock, translated: bool) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []
    normalized: list[dict[str, Any]] = []
    base_style = default_typography_style(block)
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        text = repair_mojibake(str(raw.get("text") or raw.get("translatedText") or raw.get("sourceText") or "")).strip()
        if not text:
            continue
        color = str(raw.get("color") or base_style["color"]).strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = base_style["color"]
        is_bold = normalize_bool(raw.get("is_bold", raw.get("bold", base_style["fontWeight"] >= 700)))
        is_uppercase = normalize_bool(raw.get("is_uppercase", raw.get("all_caps", text.isupper())))
        role = str(raw.get("semantic_role") or raw.get("role") or classify_semantic_role(text)).strip() or "benefit"
        source_word_hint = str(raw.get("source_word_id") or raw.get("sourceWordId") or "").strip()
        style = dict(base_style)
        style.update(
            {
                "fontWeight": 700 if is_bold else 400,
                "fontSize": max(8, int(round(float(raw.get("font_size") or raw.get("fontSize") or base_style.get("fontSize") or 16)))),
                "lineHeight": max(10, int(round(float(raw.get("line_height") or raw.get("lineHeight") or base_style.get("lineHeight") or 18)))),
                "color": color,
                "casing": "uppercase" if is_uppercase else "mixed",
                "fontCategory": normalize_font_category(raw.get("font_category") or raw.get("fontCategory") or base_style.get("fontCategory")),
                "strokeWidth": int(raw.get("stroke_width") or raw.get("strokeWidth") or 0),
                "strokeFill": raw.get("stroke_fill") or raw.get("strokeFill"),
                "isItalic": normalize_bool(raw.get("is_italic", raw.get("isItalic", raw.get("italic", False)))),
                "isUnderlined": normalize_bool(raw.get("is_underlined", raw.get("isUnderlined", raw.get("underline", raw.get("underlined", False))))),
                "isStrikethrough": normalize_bool(raw.get("is_strikethrough", raw.get("isStrikethrough", raw.get("strikethrough", raw.get("strike", False))))),
            }
        )
        segment_text = text.upper() if is_uppercase else text
        normalized.append(
            {
                "translatedText" if translated else "sourceText": segment_text,
                "matchedSourceRole" if translated else "semanticRole": role,
                "style": style,
                "sourceSegmentHint": str(raw.get("source_segment_hint") or raw.get("sourceText") or raw.get("source_text") or "").strip(),
                "sourceWordId": source_word_hint,
                "forceBreakAfter": bool(raw.get("forceBreakAfter", raw.get("force_break_after", False))),
                "color": color,
                "fontWeight": style["fontWeight"],
                "casing": style["casing"],
                "semanticStyleKey": f"{role}:{normalize_ocr_text(str(raw.get('source_segment_hint') or text))}",
                "styleTransferMode": "model_word_level_semantic_mapping",
            }
        )
    return normalized


def normalize_cross_line_rich_text_segments(item: dict[str, Any], *, block: TextBlock) -> list[dict[str, Any]]:
    base_style = default_typography_style(block)
    lookup = source_word_style_lookup(block)
    raw_lines = item.get("lines")
    flattened: list[tuple[dict[str, Any], bool]] = []
    if isinstance(raw_lines, list) and raw_lines:
        for line in raw_lines:
            if not isinstance(line, dict):
                continue
            segments = [segment for segment in line.get("segments", []) if isinstance(segment, dict)]
            for segment_index, segment in enumerate(segments):
                flattened.append((segment, segment_index == len(segments) - 1))
    if not flattened:
        raw_segments = item.get("segments")
        if isinstance(raw_segments, list):
            flattened = [(segment, bool(segment.get("forceBreakAfter", segment.get("force_break_after", False)))) for segment in raw_segments if isinstance(segment, dict)]
    normalized: list[dict[str, Any]] = []
    for raw, force_break in flattened:
        text = repair_mojibake(str(raw.get("text") or raw.get("translatedText") or "").strip())
        text = text.replace("[BOLD]", "").replace("[/BOLD]", "").strip()
        if not text:
            continue
        source_id = str(raw.get("source_word_id") or raw.get("sourceWordId") or "").strip()
        hint = str(raw.get("source_segment_hint") or raw.get("sourceText") or raw.get("source_text") or "").strip()
        source_styles = source_styles_for_segment(raw, lookup, text, hint)
        inherited_style = dominant_source_style(source_styles, base_style)
        source_ids = [str(style.get("id") or "") for style in source_styles if style.get("id")]
        if not source_id and source_ids:
            source_id = source_ids[0]
        color = str(inherited_style.get("color") or base_style["color"])
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = base_style["color"]
        font_weight = int(inherited_style.get("fontWeight") or base_style["fontWeight"])
        is_uppercase = inherited_style.get("casing") == "uppercase" or normalize_bool(raw.get("is_uppercase", raw.get("all_caps", False)))
        role = str(raw.get("semantic_role") or raw.get("role") or (source_styles[0] if source_styles else {}).get("semanticRole") or classify_semantic_role(text)).strip() or "benefit"
        style = dict(base_style)
        style.update(
            {
                "fontWeight": 700 if font_weight >= 700 else 400,
                "fontSize": max(8, int(round(float(inherited_style.get("fontSize") or base_style.get("fontSize") or 16)))),
                "lineHeight": max(10, int(round(float(inherited_style.get("lineHeight") or base_style.get("lineHeight") or 18)))),
                "color": color,
                "backgroundColor": inherited_style.get("backgroundColor"),
                "hasTextBackground": bool(inherited_style.get("hasTextBackground")),
                "backgroundContrast": float(inherited_style.get("backgroundContrast") or 0.0),
                "casing": "uppercase" if is_uppercase else "mixed",
                "fontCategory": normalize_font_category(raw.get("font_category") or raw.get("fontCategory") or inherited_style.get("fontCategory") or base_style.get("fontCategory")),
                "strokeWidth": int(raw.get("stroke_width") or raw.get("strokeWidth") or inherited_style.get("strokeWidth") or 0),
                "strokeFill": raw.get("stroke_fill") or raw.get("strokeFill") or inherited_style.get("strokeFill"),
                "fillTransparent": bool(raw.get("fill_transparent") or raw.get("fillTransparent") or inherited_style.get("fillTransparent")),
                "isItalic": bool(raw.get("is_italic") or raw.get("isItalic") or raw.get("italic") or inherited_style.get("isItalic")),
                "isUnderlined": bool(raw.get("is_underlined") or raw.get("isUnderlined") or raw.get("underline") or raw.get("underlined") or inherited_style.get("isUnderlined")),
                "isStrikethrough": bool(raw.get("is_strikethrough") or raw.get("isStrikethrough") or raw.get("strikethrough") or raw.get("strike") or inherited_style.get("isStrikethrough")),
            }
        )
        segment_text = text.upper() if is_uppercase else text
        normalized.append(
            {
                "translatedText": segment_text,
                "matchedSourceRole": role,
                "style": style,
                "sourceSegmentHint": hint or " ".join(str(source.get("text") or "") for source in source_styles).strip(),
                "sourceWordId": source_id,
                "sourceWordIds": source_ids or ([source_id] if source_id else []),
                "sourceWordStyles": source_styles,
                "forceBreakAfter": force_break,
                "color": color,
                "fontWeight": style["fontWeight"],
                "casing": style["casing"],
                "semanticStyleKey": f"{role}:{normalize_ocr_text(source_id or hint or text)}",
                "styleTransferMode": "deterministic_nm_source_word_foreground_style_cross_line",
            }
        )
    if len([line for line in (block.line_texts or []) if str(line).strip()]) > 1:
        for index in range(len(normalized) - 1):
            current_lines = {
                int(style.get("lineIndex"))
                for style in normalized[index].get("sourceWordStyles", [])
                if isinstance(style, dict) and style.get("lineIndex") is not None
            }
            next_lines = {
                int(style.get("lineIndex"))
                for style in normalized[index + 1].get("sourceWordStyles", [])
                if isinstance(style, dict) and style.get("lineIndex") is not None
            }
            if current_lines and next_lines and current_lines != next_lines:
                normalized[index]["forceBreakAfter"] = True
    return preserve_visual_heading_span_order(block, normalized)


def is_decorative_or_numeric_only(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and bool(re.fullmatch(r"[\d\s.,:/#%+-]+", cleaned))


def should_translate_ocr_overlay_block(block: TextBlock, image_size: tuple[int, int]) -> bool:
    text = block.text.strip()
    if not text or is_decorative_or_numeric_only(text):
        return False
    normalized = text.lower()
    box_width = max(1, block.bbox[2] - block.bbox[0])
    box_height = max(1, block.bbox[3] - block.bbox[1])
    image_width, image_height = image_size
    area_ratio = (box_width * box_height) / max(1, image_width * image_height)
    instructional_context = any(
        term in normalized
        for term in (
            "with", "apply", "use", "for", "remove", "rinse", "cleanse", "soak",
            "spray", "shoe", "shoes", "wear", "wipe", "dry", "shake",
            "gebrauch", "schutteln", "schütteln", "tragen", "schuhe", "spruhen", "sprühen", "trocknen", "abwischen", "gelangt",
            "ile", "uygula", "kullan", "durula", "temizle",
        )
    )
    if block.bbox[1] >= image_height * 0.45 and box_height <= image_height * 0.09 and area_ratio <= 0.014 and not instructional_context:
        return False
    if has_packaging_cues(text) and not instructional_context:
        return False
    if any(token in normalized for token in ("www.", ".com", ".de", "http", "qr", "ask.naos")):
        return False
    if box_height <= max(10, image_height * 0.018) and len(text) >= 22:
        return False
    if area_ratio <= 0.00018 and len(text) >= 16:
        return False
    return True


def is_instructional_context_text(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    return any(
        term in normalized
        for term in (
            "with", "apply", "use", "for", "remove", "rinse", "cleanse", "soak",
            "spray", "shoe", "shoes", "wear", "wipe", "dry", "shake",
            "gebrauch", "schutteln", "schütteln", "tragen", "schuhe", "spruhen", "sprühen", "trocknen", "abwischen", "gelangt",
            "ile", "uygula", "kullan", "durula", "temizle",
        )
    )


def is_short_product_context_token(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized or is_decorative_or_numeric_only(text):
        return False
    if any(token in normalized for token in ("www", "http", "com", "qr", "ingredients", "warning")):
        return False
    words = [part for part in re.split(r"\s+", normalized) if part]
    if len(words) > 4 or len(normalized) > 36:
        return False
    has_digit_or_model_suffix = bool(re.search(r"\d|[a-z][a-z]*[0o][a-z0-9]?$", normalized))
    has_brand_like_case = any(char.isupper() for char in text) and any(char.islower() for char in text)
    has_product_noun = any(token in normalized for token in ("serum", "h2o", "hzo", "spf", "cream", "skincare", "lotion", "gel", "spray"))
    return has_digit_or_model_suffix or has_brand_like_case or has_product_noun


def should_preserve_ocr_structure(block: TextBlock, image_size: tuple[int, int]) -> bool:
    text = block.text.strip()
    if not text:
        return False
    if is_decorative_or_numeric_only(text):
        return True
    if has_packaging_cues(text) and not is_instructional_context_text(text):
        return True
    return not should_translate_ocr_overlay_block(block, image_size)


def normalize_v212_translation_items(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        block_id = str(item.get("id") or "").strip()
        if not block_id:
            continue
        mapped[block_id] = item
    return mapped


def fallback_segments_from_translation(block: TextBlock, translated_text: str) -> list[dict[str, Any]]:
    source_words = [word for word in block.source_word_styles if isinstance(word, dict)]
    if not source_words:
        source_words = [{"id": "", "text": block.text, "semanticRole": classify_semantic_role(block.text)}]
    lines = [line.strip() for line in translated_text.splitlines() if line.strip()] or [translated_text.strip()]
    output_lines: list[dict[str, Any]] = []
    cursor = 0
    for line in lines:
        segments: list[dict[str, Any]] = []
        words = [word for word in line.split() if word]
        for target_word in words:
            source_style = source_words[min(cursor, len(source_words) - 1)]
            segments.append(
                {
                    "text": target_word,
                    "source_word_id": source_style.get("id", ""),
                    "source_word_ids": [source_style.get("id", "")] if source_style.get("id") else [],
                    "source_segment_hint": source_style.get("text", ""),
                    "semantic_role": source_style.get("semanticRole", classify_semantic_role(target_word)),
                }
            )
            cursor += 1
        if segments:
            output_lines.append({"segments": segments})
    return output_lines


def ensure_v212_translation_coverage(payload: dict[str, Any], candidates: list[TextBlock], target_language: str) -> dict[str, Any]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        items = []
    mapped = {
        str(item.get("id") or "").strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    missing: list[TextBlock] = []
    for block in candidates:
        item = mapped.get(block.id or "")
        if not item or item.get("translate") is False:
            missing.append(block)
            continue
        text = repair_mojibake(str(item.get("translated_text") or plain_text_from_translation_item(item) or "").strip())
        if target_language.upper() != "EN" and normalize_ocr_text(text) == normalize_ocr_text(block.text) and not has_packaging_cues(block.text):
            missing.append(block)
    if not missing:
        return {**payload, "blocks": items}
    translations = translate_with_gpt4o(missing, [target_language]).get(target_language, [block.text for block in missing])
    for block, translated_text in zip(missing, translations, strict=False):
        translated_text = preserve_source_metric_tokens(block.text, repair_mojibake(str(translated_text or block.text).strip()))
        if block.line_boxes and len([line for line in translated_text.splitlines() if line.strip()]) < len(block.line_boxes):
            translated_text = split_text_across_lines(translated_text, len(block.line_boxes))
        mapped[block.id or ""] = {
            "id": block.id,
            "translate": True,
            "translated_text": translated_text,
            "lines": fallback_segments_from_translation(block, translated_text),
            "coverage_source": "backend_translation_fallback",
        }
    ordered: list[dict[str, Any]] = []
    for block in candidates:
        item = mapped.get(block.id or "")
        if item:
            ordered.append(item)
    return {**payload, "blocks": ordered, "coverageFallbackApplied": True}


def expressive_punctuation(text: str) -> list[str]:
    marks: list[str] = []
    for mark in re.findall(r"[!?]", text or ""):
        if mark not in marks:
            marks.append(mark)
    return marks


def missing_expressive_punctuation(payload: dict[str, Any], candidates: list[TextBlock]) -> list[dict[str, Any]]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return []
    mapped = {
        str(item.get("id") or "").strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    missing: list[dict[str, Any]] = []
    for block in candidates:
        expected = expressive_punctuation(block.text)
        if not expected:
            continue
        item = mapped.get(block.id or "")
        translated = plain_text_from_translation_item(item or {}) if item else ""
        for mark in expected:
            if mark not in translated:
                missing.append({"id": block.id, "mark": mark, "sourceText": block.text, "translatedText": translated})
    return missing


def natural_language_violations(payload: dict[str, Any], candidates: list[TextBlock], target_language: str) -> list[dict[str, Any]]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return []
    mapped = {
        str(item.get("id") or "").strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    violations: list[dict[str, Any]] = []
    for block in candidates:
        item = mapped.get(block.id or "")
        translated = plain_text_from_translation_item(item or {}) if item else ""
        translated_norm = normalize_ocr_text(translated)
        source = f" {normalize_ocr_text(block.text)} "
        if target_language.upper() == "TR" and re.search(r"\ba\b", source) and re.search(r"\b[Bb]ir\s+\S+", translated):
            violations.append(
                {
                    "id": block.id,
                    "sourceText": block.text,
                    "translatedText": translated,
                    "reason": "literal_turkish_indefinite_article",
                }
            )
        if (
            target_language.upper() == "TR"
            and "skin" in source
            and re.search(r"\b(ciltleri|cildi|cilti)\b", translated, flags=re.IGNORECASE)
        ):
            violations.append(
                {
                    "id": block.id,
                    "sourceText": block.text,
                    "translatedText": translated,
                    "reason": "literal_turkish_skin_possessive_label",
                }
            )
        if target_language.upper() == "TR" and ("&" in translated or re.search(r"\s/\s", translated)):
            violations.append(
                {
                    "id": block.id,
                    "sourceText": block.text,
                    "translatedText": translated,
                    "reason": "literal_symbol_punctuation_in_turkish_copy",
                }
            )
        for word in block.source_word_styles or []:
            if not isinstance(word, dict):
                continue
            source_token = str(word.get("text") or "").strip()
            source_norm = normalize_ocr_text(source_token)
            if not source_norm or not has_packaging_cues(source_token):
                continue
            if source_norm not in translated_norm:
                violations.append(
                    {
                        "id": block.id,
                        "sourceText": block.text,
                        "translatedText": translated,
                        "missingToken": source_token,
                        "reason": "missing_brand_or_product_token",
                    }
                )
    return violations


def repair_turkish_skin_label_text(text: str, source_text: str) -> str:
    if "skin" not in normalize_ocr_text(source_text):
        return text

    def keep_case(match: re.Match[str], replacement_upper: str, replacement_title: str, replacement_lower: str) -> str:
        value = match.group(0)
        if value.upper() == value:
            return replacement_upper
        if value[:1].upper() == value[:1]:
            return replacement_title
        return replacement_lower

    repaired = re.sub(
        r"\b(ciltleri|ciltileri)\b",
        lambda match: keep_case(match, "CİLTLER", "Ciltler", "ciltler"),
        text,
        flags=re.IGNORECASE,
    )
    repaired = re.sub(
        r"\b(cildi|cilti)\b",
        lambda match: keep_case(match, "CİLT", "Cilt", "cilt"),
        repaired,
        flags=re.IGNORECASE,
    )
    return repaired


def repair_turkish_literal_article_text(text: str, source_text: str) -> str:
    source = f" {normalize_ocr_text(source_text)} "
    if not re.search(r"\ba\b", source):
        return text
    return re.sub(r"^\s*[Bb]ir\s+", "", text or "")


def normalize_translated_punctuation_spacing(text: str) -> str:
    repaired = re.sub(r"\s+([!?.,;:%])", r"\1", text or "")
    repaired = re.sub(r"([¿¡])\s+", r"\1", repaired)
    return repaired


def normalize_payload_punctuation_spacing(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return payload
    changed = False
    normalized_items: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            normalized_items.append(item)
            continue
        updated = dict(item)
        for key in ("translated_text", "text", "translatedText"):
            if key in updated and isinstance(updated.get(key), str):
                value = normalize_translated_punctuation_spacing(str(updated.get(key) or ""))
                if value != updated.get(key):
                    updated[key] = value
                    changed = True
        lines = updated.get("lines")
        if isinstance(lines, list):
            normalized_lines: list[Any] = []
            for line in lines:
                if not isinstance(line, dict):
                    normalized_lines.append(line)
                    continue
                line_copy = dict(line)
                segments = line_copy.get("segments")
                if isinstance(segments, list):
                    normalized_segments: list[Any] = []
                    for segment in segments:
                        if not isinstance(segment, dict):
                            normalized_segments.append(segment)
                            continue
                        segment_copy = dict(segment)
                        for key in ("text", "translatedText"):
                            if key in segment_copy and isinstance(segment_copy.get(key), str):
                                value = normalize_translated_punctuation_spacing(str(segment_copy.get(key) or ""))
                                if value != segment_copy.get(key):
                                    segment_copy[key] = value
                                    changed = True
                        normalized_segments.append(segment_copy)
                    line_copy["segments"] = normalized_segments
                normalized_lines.append(line_copy)
            updated["lines"] = normalized_lines
        normalized_items.append(updated)
    if changed:
        return {**payload, "blocks": normalized_items, "punctuationSpacingNormalized": True}
    return payload


def repair_target_language_morphology(payload: dict[str, Any], candidates: list[TextBlock], target_language: str) -> dict[str, Any]:
    if target_language.upper() != "TR":
        return payload
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return payload
    by_id = {str(block.id or ""): block for block in candidates}
    repaired_any = False
    repaired_items: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            repaired_items.append(item)
            continue
        block = by_id.get(str(item.get("id") or ""))
        if not block:
            repaired_items.append(item)
            continue
        updated = dict(item)
        original_text = str(updated.get("translated_text") or plain_text_from_translation_item(updated) or "")
        repaired_text = repair_turkish_literal_article_text(repair_turkish_skin_label_text(original_text, block.text), block.text)
        if repaired_text != original_text:
            updated["translated_text"] = repaired_text
            repaired_any = True
        lines = updated.get("lines")
        if isinstance(lines, list):
            repaired_lines: list[Any] = []
            for line in lines:
                if not isinstance(line, dict):
                    repaired_lines.append(line)
                    continue
                line_copy = dict(line)
                segments = line_copy.get("segments")
                if isinstance(segments, list):
                    repaired_segments: list[Any] = []
                    for segment in segments:
                        if not isinstance(segment, dict):
                            repaired_segments.append(segment)
                            continue
                        segment_copy = dict(segment)
                        for key in ("text", "translatedText"):
                            if key in segment_copy:
                                segment_original = str(segment_copy.get(key) or "")
                                segment_repaired = repair_turkish_literal_article_text(repair_turkish_skin_label_text(segment_original, block.text), block.text)
                                if segment_repaired != segment_original:
                                    segment_copy[key] = segment_repaired
                                    repaired_any = True
                        repaired_segments.append(segment_copy)
                    line_copy["segments"] = repaired_segments
                repaired_lines.append(line_copy)
            updated["lines"] = repaired_lines
        repaired_items.append(updated)
    if not repaired_any:
        return payload
    return {**payload, "blocks": repaired_items, "morphologyRepairApplied": True}


def finalize_v212_translation_payload(payload: dict[str, Any], candidates: list[TextBlock], target_language: str) -> dict[str, Any]:
    punctuated = repair_missing_expressive_punctuation(payload, candidates)
    spaced = normalize_payload_punctuation_spacing(punctuated)
    return repair_target_language_morphology(spaced, candidates, target_language)


def repair_missing_expressive_punctuation(payload: dict[str, Any], candidates: list[TextBlock]) -> dict[str, Any]:
    items = payload.get("blocks", [])
    if not isinstance(items, list):
        return payload
    by_id = {str(block.id or ""): block for block in candidates}
    repaired = False
    for item in items:
        if not isinstance(item, dict):
            continue
        block = by_id.get(str(item.get("id") or ""))
        if not block:
            continue
        translated = plain_text_from_translation_item(item)
        for mark in expressive_punctuation(block.text):
            if mark in translated:
                continue
            lines = item.get("lines")
            if isinstance(lines, list) and lines:
                last_line = next((line for line in reversed(lines) if isinstance(line, dict)), None)
                segments = last_line.get("segments") if isinstance(last_line, dict) else None
                last_segment = next((segment for segment in reversed(segments or []) if isinstance(segment, dict)), None)
                if last_segment is not None:
                    last_segment["text"] = f"{str(last_segment.get('text') or '').rstrip()}{mark}"
                    repaired = True
                    continue
            item["translated_text"] = f"{str(item.get('translated_text') or translated).rstrip()}{mark}"
            repaired = True
    if repaired:
        return {**payload, "punctuationRepairApplied": True, "blocks": items}
    return payload


def analyze_localize_v212_ocr_translations(blocks: list[TextBlock], target_language: str) -> dict[str, Any]:
    candidates = [block for block in blocks if block.translate]
    prompt = {
        "task": "V5 polygon localize: translate Google Vision boundingPoly overlay text as semantic marketing/instruction blocks and return rich-text N:M style-mapped segments. Do not infer or modify coordinates.",
        "targetLanguage": LANGUAGE_NAMES.get(target_language.upper(), target_language),
        "schema": LOCALIZE_V212_TRANSLATION_SCHEMA,
        "input_blocks": [
            {
                "id": block.id,
                "source_text": block.text,
                "source_words": [
                    {
                        "id": word.get("id"),
                        "text": word.get("text"),
                        "line_index": word.get("lineIndex"),
                        "word_index": word.get("wordIndex"),
                        "hex_color_from_foreground_pixels": word.get("color"),
                        "is_bold": bool(word.get("isBold")),
                        "font_weight": word.get("fontWeight"),
                        "font_category": word.get("fontCategory", "sans-serif"),
                        "is_uppercase": bool(word.get("isUppercase")),
                        "semantic_role": word.get("semanticRole"),
                    }
                    for word in (block.source_word_styles or [])
                    if isinstance(word, dict)
                ],
                "style_hint": {
                    "color": block.color,
                    "is_bold": block.font_weight >= 700,
                    "is_uppercase": block.text.isupper(),
                    "semantic_role": classify_semantic_role(block.text),
                },
            }
            for block in candidates
        ],
        "rules": [
            "Return compact JSON only.",
            "Use exactly the ids from input_blocks; do not create new ids.",
            "Set translate=false only if the input is not marketing/instructional overlay copy.",
            "Never return x, y, w, h, bbox, or any coordinate.",
            "HARD RULE: Translate each input block as one semantic unit, never as independent source lines or word-by-word fragments.",
            "HARD RULE: First decide how the marketing or instruction message is naturally said in the target language. Do not literal-translate source-language articles, filler words, or word order when the target language would omit or move them. Example for Turkish: 'Soak a cotton pad with SÃ©bium H2O' must become 'Pamuk pedi SÃ©bium H2O ile Ä±slatÄ±n', not 'Bir pamuk pedi SÃ©bium H2O ile Ä±slatÄ±n'.",
            "HARD RULE: For Turkish skincare condition labels, do not create possessive forms such as 'ciltleri' or 'cildi' unless the source explicitly means 'their skin'. Translate standalone 'acne-prone adult skin' style labels as natural label copy such as 'akneye eÄŸilimli yetiÅŸkin ciltler/cilt'.",
            "HARD RULE: Brand/product tokens such as SÃ©bium H2O that are in overlay copy are movable semantic tokens. Place them where the target-language grammar requires; never lock them to their original source line or coordinate.",
            "HARD RULE: Numeric marketing claims inside overlay copy are translatable/localizable text, not fixed step markers. Localize number, percent, decimal, duration, and unit formatting for the target language, e.g. English '90%' may become Turkish '%90', and '24h' may become the target-language natural short duration form.",
            "HARD RULE: Pure numeric step/bullet markers are not included in input_blocks. If a number is present here, it is part of marketing or instructional copy and must be localized with the surrounding sentence.",
            "HARD RULE: Do not preserve source line order when it harms target-language grammar. Decide target lines only after semantic translation.",
            "HARD RULE: Preserve expressive punctuation exactly or with a target-language equivalent. Every source ! must remain ! in the target, every source ? must remain ?. Validate before returning JSON.",
            "HARD RULE: Do not preserve connector symbols literally when they are unnatural in the target language. For Turkish, translate '&' as 've' when it means 'and', and convert slash alternatives into natural wording unless the slash is a required product/SKU token.",
            "Return target-language copy as lines[].segments[]. Target lines may differ from source lines when grammar requires it.",
            "HARD RULE: For every target segment, set source_word_ids to one or more source_words ids whose meaning/emphasis the segment inherited.",
            "HARD RULE: source_word_ids must be the exact semantic counterpart of the target segment, not the entire source line. Example: if source is 'Soak a cotton pad with SÃ©bium H2O', Turkish target 'Pamuk pedi' inherits the grey/black source words 'a cotton pad', target 'SÃ©bium H2O ile' inherits the product/with source words, and target 'Ä±slatÄ±n' inherits only 'Soak'.",
            "HARD RULE: Style inheritance is local to the current input block. Never copy color, boldness, casing, or source_word_ids from any other block.",
            "HARD RULE: For 1-to-N mapping, repeat the same source word id across all target segments that inherit that source word style.",
            "HARD RULE: For N-to-1 mapping, put all contributing source word ids in source_word_ids; the backend will apply dominant style.",
            "Do not invent colors or font weights. The backend will copy exact hex_color_from_foreground_pixels and font_weight from source_word_ids.",
            "HARD RULE: Preserve the original visible line count only after the natural target-language wording is decided and only if it can fit without changing word order, grammar, or meaning.",
            *LOCALIZE_V2_RICH_TEXT_PROMPT_RULES,
        ],
    }
    if not candidates:
        return {"analysis_provider": "deterministic-ocr", "blocks": []}
    try:
        if vertex_available():
            parsed = generate_vertex_gemini_json(prompt, timeout=int(os.getenv("VERTEX_GEMINI_TIMEOUT", "55")))
            if isinstance(parsed.get("blocks"), list):
                parsed["analysis_provider"] = "vertex-gemini-v5-polygon-nm-style"
                parsed = ensure_v212_translation_coverage(parsed, candidates, target_language)
                if missing_expressive_punctuation(parsed, candidates):
                    retry_prompt = {**prompt, "retry_reason": "Previous response violated HARD punctuation preservation. Retry and preserve every source ! and ? in the target JSON."}
                    retry = generate_vertex_gemini_json(retry_prompt, timeout=int(os.getenv("VERTEX_GEMINI_TIMEOUT", "55")))
                    if isinstance(retry.get("blocks"), list):
                        retry["analysis_provider"] = "vertex-gemini-v5-polygon-nm-style-retry-punctuation"
                        parsed = ensure_v212_translation_coverage(retry, candidates, target_language)
                if natural_language_violations(parsed, candidates, target_language):
                    retry_prompt = {
                        **prompt,
                        "retry_reason": "Previous response violated natural target-language wording. For Turkish, do not carry English articles such as 'a/an' into literal 'Bir ...' unless the meaning is numeric one, and do not keep '&' or spaced slash connectors when Turkish natural copy should use words. Rewrite naturally while preserving source_word_ids style mapping.",
                        "previous_response": parsed,
                    }
                    retry = generate_vertex_gemini_json(retry_prompt, timeout=int(os.getenv("VERTEX_GEMINI_TIMEOUT", "55")))
                    if isinstance(retry.get("blocks"), list):
                        retry["analysis_provider"] = "vertex-gemini-v5-polygon-nm-style-retry-natural-language"
                        parsed = ensure_v212_translation_coverage(retry, candidates, target_language)
                return finalize_v212_translation_payload(parsed, candidates, target_language)
    except Exception as exc:
        print(f"[localize-v2.1.2] Vertex Gemini OCR translation failed: {exc}", flush=True)
    try:
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI()
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
                temperature=0,
                messages=[
                    {"role": "system", "content": "You localize OCR text lines and return compact JSON only. Never output coordinates."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            parsed = extract_json_object(response.choices[0].message.content or "{}")
            if isinstance(parsed.get("blocks"), list):
                parsed["analysis_provider"] = "openai-v5-polygon-nm-style-fallback"
                parsed = ensure_v212_translation_coverage(parsed, candidates, target_language)
                if missing_expressive_punctuation(parsed, candidates):
                    response = client.chat.completions.create(
                        model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You localize OCR text and return compact JSON only. This is a retry: preserve every source ! and ? in the target."},
                            {"role": "user", "content": json.dumps({**prompt, "retry_reason": "Dropped required expressive punctuation."}, ensure_ascii=False)},
                        ],
                        response_format={"type": "json_object"},
                    )
                    retry = extract_json_object(response.choices[0].message.content or "{}")
                    if isinstance(retry.get("blocks"), list):
                        retry["analysis_provider"] = "openai-text-ocr-layout-retry-punctuation"
                        parsed = ensure_v212_translation_coverage(retry, candidates, target_language)
                if natural_language_violations(parsed, candidates, target_language):
                    response = client.chat.completions.create(
                        model=os.getenv("OPENAI_TRANSLATION_MODEL", "gpt-4o"),
                        temperature=0,
                        messages=[
                            {"role": "system", "content": "You localize OCR text and return compact JSON only. This is a retry: fix literal target-language wording while preserving source_word_ids style mapping."},
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        **prompt,
                                        "retry_reason": "Previous response used literal Turkish wording such as 'Bir ...' for English 'a/an'. Rewrite as natural Turkish; omit 'Bir' unless the meaning is numeric one.",
                                        "previous_response": parsed,
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        response_format={"type": "json_object"},
                    )
                    retry = extract_json_object(response.choices[0].message.content or "{}")
                    if isinstance(retry.get("blocks"), list):
                        retry["analysis_provider"] = "openai-text-ocr-layout-retry-natural-language"
                        parsed = ensure_v212_translation_coverage(retry, candidates, target_language)
                return finalize_v212_translation_payload(parsed, candidates, target_language)
    except Exception as exc:
        print(f"[localize-v2.1.2] OpenAI OCR translation failed: {exc}", flush=True)
    return finalize_v212_translation_payload(
        ensure_v212_translation_coverage({"analysis_provider": "deterministic-ocr-fallback", "blocks": []}, candidates, target_language),
        candidates,
        target_language,
    )


def apply_v212_translations(blocks: list[TextBlock], payload: dict[str, Any]) -> list[TextBlock]:
    translations = normalize_v212_translation_items(payload)
    updated: list[TextBlock] = []
    for block in blocks:
        if not block.translate:
            updated.append(block)
            continue
        item = translations.get(block.id or "")
        if not item or item.get("translate") is False:
            updated.append(block.model_copy(update={"translate": False, "line_boxes": [], "line_texts": []}))
            continue
        translated_text = repair_mojibake(str(item.get("translated_text") or "").strip())
        if not translated_text:
            translated_text = repair_mojibake(plain_text_from_translation_item(item)) or block.text
        translated_style_spans = normalize_cross_line_rich_text_segments(item, block=block)
        source_style_spans = normalize_rich_text_segments(item.get("source_style_segments"), block=block, translated=False)
        if not translated_style_spans:
            source_style_spans = source_style_spans or block.source_style_spans or infer_source_style_spans(block.text, block)
            translated_style_spans = infer_translated_style_spans(block.text, translated_text, source_style_spans, block)
        elif not source_style_spans:
            source_style_spans = block.source_style_spans or infer_source_style_spans(block.text, block)
        updated.append(
            block.model_copy(
                update={
                    "translated_text": translated_text,
                    "source_style_spans": source_style_spans,
                    "translated_style_spans": translated_style_spans,
                }
            )
        )
    return updated


def strict_mask_dilation_px(image_size: tuple[int, int]) -> int:
    configured = os.getenv("ADAPTIFAI_STRICT_MASK_DILATION_PX", "").strip()
    if configured:
        try:
            return max(0, min(3, int(configured)))
        except ValueError:
            pass
    return 3


def strict_mask_feather_px(image_size: tuple[int, int]) -> float:
    configured = os.getenv("ADAPTIFAI_STRICT_MASK_FEATHER_PX", "").strip()
    if configured:
        try:
            return max(0.6, min(3.0, float(configured)))
        except ValueError:
            pass
    return max(0.8, min(2.0, max(image_size) * 0.0012))


def dilate_mask(mask: Image.Image, dilation_px: int) -> Image.Image:
    if dilation_px <= 0:
        return mask.convert("L")
    import cv2

    mask_np = np.array(mask.convert("L"))
    kernel_size = max(3, dilation_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    return Image.fromarray(dilated, mode="L")


def confinement_mask_for_boxes(image_size: tuple[int, int], boxes: list[tuple[int, int, int, int]], pad_px: int = 2) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image_size
    for left, top, right, bottom in boxes:
        if right <= left or bottom <= top:
            continue
        draw.rectangle(
            (
                max(0, left - pad_px),
                max(0, top - pad_px),
                min(width, right + pad_px),
                min(height, bottom + pad_px),
            ),
            fill=255,
        )
    return mask


def should_use_rectangular_strict_mask(block: TextBlock, image_size: tuple[int, int]) -> bool:
    if os.getenv("ADAPTIFAI_LOCALIZE_V212_CONTOUR_MASK", "1").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    width, height = image_size
    line_boxes = block.line_boxes or [block.bbox]
    if len(line_boxes) >= 2:
        return True
    for box in line_boxes:
        box_width = max(1, box[2] - box[0])
        box_height = max(1, box[3] - box[1])
        if block.role == "cta" and box_width >= width * 0.08 and box_height >= height * 0.045:
            return True
        if len(normalize_ocr_text(block.text)) <= 5 and block.text.strip().upper() == block.text.strip() and box_height >= height * 0.045:
            return True
        if box_width >= width * 0.30 and box_height >= height * 0.035:
            return True
        if (box_width * box_height) / max(1, width * height) >= 0.018:
            return True
    return False


def parse_hex_color(value: str, fallback: tuple[int, int, int] = (17, 17, 17)) -> tuple[int, int, int]:
    cleaned = str(value or "").strip().lstrip("#")
    if len(cleaned) != 6:
        return fallback
    try:
        return tuple(int(cleaned[index:index + 2], 16) for index in range(0, 6, 2))
    except ValueError:
        return fallback


def is_v5_marketing_numeric_source_word(word: dict[str, Any]) -> bool:
    text = str(word.get("text") or "").strip()
    normalized = normalize_ocr_text(text)
    role = str(word.get("semanticRole") or "").strip().lower()
    return bool(
        re.search(r"\d", normalized)
        or "%" in text
        or role in {"percentage", "numeric_claim", "duration", "measurement"}
    )


def is_v5_large_heading_source_word(word: dict[str, Any], image_size: tuple[int, int]) -> bool:
    text = str(word.get("text") or "").strip()
    if not text or is_decorative_or_numeric_only(text):
        return False
    box = v5_word_box(word)
    if box is None:
        return False
    height = max(1, box[3] - box[1])
    width = max(1, box[2] - box[0])
    return (
        height >= max(24, int(image_size[1] * 0.035))
        and width >= image_size[0] * 0.12
        and text.upper() == text
        and any(char.isalpha() for char in text)
    )


def build_v5_word_box_support_mask(image_size: tuple[int, int], word: dict[str, Any], pad_px: int) -> Image.Image:
    box = v5_word_box(word)
    if box is None:
        return Image.new("L", image_size, 0)
    x1, y1, x2, y2 = expand_bbox(box, image_size, pad_px)
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x1, y1, x2, y2), fill=255)
    return mask


def build_v5_foreground_pixel_word_mask(image: Image.Image, word: dict[str, Any], dilation_px: int) -> Image.Image:
    import cv2

    box = v5_word_box(word)
    if box is None:
        return Image.new("L", image.size, 0)
    x1, y1, x2, y2 = expand_bbox(box, image.size, 1)
    if x2 <= x1 or y2 <= y1:
        return Image.new("L", image.size, 0)
    crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"), dtype=np.uint8)
    if crop.size == 0:
        return Image.new("L", image.size, 0)

    text_rgb = np.array(parse_hex_color(str(word.get("color") or "#111111")), dtype=np.float32)
    crop_f = crop.astype(np.float32)
    border_pixels = np.concatenate(
        [
            crop[0:1, :, :].reshape(-1, 3),
            crop[-1:, :, :].reshape(-1, 3),
            crop[:, 0:1, :].reshape(-1, 3),
            crop[:, -1:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    bg_rgb = np.median(border_pixels, axis=0)
    dist_to_text = np.linalg.norm(crop_f - text_rgb, axis=2)
    dist_to_bg = np.linalg.norm(crop_f - bg_rgb, axis=2)
    luma = crop_f[:, :, 0] * 0.299 + crop_f[:, :, 1] * 0.587 + crop_f[:, :, 2] * 0.114
    text_luma = float(np.dot(text_rgb, [0.299, 0.587, 0.114]))
    bg_luma = float(np.dot(bg_rgb, [0.299, 0.587, 0.114]))

    foreground = (dist_to_text <= 76.0) & (dist_to_bg >= 18.0)
    if text_luma >= 205 and bg_luma < text_luma - 30:
        foreground |= (luma >= text_luma - 52.0) & (dist_to_bg >= 22.0)
    elif text_luma <= 80 and bg_luma > text_luma + 30:
        foreground |= (luma <= text_luma + 52.0) & (dist_to_bg >= 22.0)

    mask_np = foreground.astype(np.uint8) * 255
    if mask_np.max() == 0:
        return Image.new("L", image.size, 0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    filtered = np.zeros_like(mask_np)
    crop_area = max(1, mask_np.shape[0] * mask_np.shape[1])
    for label in range(1, num_labels):
        _, _, _, _, area = stats[label]
        if area < 2:
            continue
        if area / crop_area > 0.36:
            continue
        filtered[labels == label] = 255
    if filtered.max() == 0:
        filtered = mask_np
    kernel_size = max(1, min(7, int(dilation_px) * 2 + 1))
    if kernel_size >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filtered = cv2.dilate(filtered, kernel, iterations=1)
    full = Image.new("L", image.size, 0)
    full.paste(Image.fromarray(filtered, "L"), (x1, y1))
    return full


def build_v5_polygon_text_mask(image: Image.Image, block: TextBlock) -> tuple[Image.Image, dict[str, Any]]:
    masks: list[Image.Image] = []
    word_reports: list[dict[str, Any]] = []
    source_words = [word for word in (block.source_word_styles or []) if isinstance(word, dict)]
    for word in block.source_word_styles or []:
        if not isinstance(word, dict):
            continue
        if is_v5_isolated_step_marker_word(word, source_words):
            word_reports.append(
                {
                    "word": word.get("text"),
                    "status": "preserved_step_marker",
                    "polygonCount": 0,
                }
            )
            continue
        raw_polygons = word.get("symbolPolygons") or ([word.get("polygon")] if word.get("polygon") else [])
        polygons: list[list[tuple[int, int]]] = []
        for raw_polygon in raw_polygons:
            if not isinstance(raw_polygon, list):
                continue
            polygon: list[tuple[int, int]] = []
            for point in raw_polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon.append((int(point[0]), int(point[1])))
            if len(polygon) >= 3 and polygon_area(polygon) > 1:
                polygons.append(polygon)
        if not polygons:
            continue
        stroke_width = int(word.get("strokeWidth") or 0)
        has_source_text_background = bool(word.get("hasTextBackground") and word.get("backgroundColor"))
        is_numeric_claim = is_v5_marketing_numeric_source_word(word) and not has_source_text_background
        is_large_heading = is_v5_large_heading_source_word(word, image.size) and not has_source_text_background
        dilation_px = stroke_width + 5 if word.get("outline") or stroke_width > 0 else int(os.getenv("ADAPTIFAI_V5_POLYGON_DILATION_PX", "5"))
        if is_numeric_claim:
            dilation_px = max(dilation_px, int(os.getenv("ADAPTIFAI_V5_NUMERIC_CLAIM_DILATION_PX", "8")))
        if is_large_heading:
            dilation_px = max(dilation_px, int(os.getenv("ADAPTIFAI_V5_HEADING_DILATION_PX", "9")))
        dilation_px = max(3, min(16, dilation_px))
        if has_source_text_background:
            word_mask = build_v5_foreground_pixel_word_mask(image, word, min(3, dilation_px))
            if not word_mask.getbbox():
                word_mask = polygon_mask(image.size, polygons, dilation_px=min(3, dilation_px))
        else:
            word_mask = polygon_mask(image.size, polygons, dilation_px=dilation_px)
        if is_numeric_claim or is_large_heading:
            support_pad = max(2, min(8, dilation_px // 2))
            support_mask = build_v5_word_box_support_mask(image.size, word, support_pad)
            word_mask = ImageChops.lighter(word_mask.convert("L"), support_mask.convert("L"))
        masks.append(word_mask)
        word_reports.append(
            {
                "word": word.get("text"),
                "outline": bool(word.get("outline")),
                "strokeWidth": stroke_width,
                "dilationPx": dilation_px,
                "numericClaimSupport": bool(is_numeric_claim),
                "largeHeadingSupport": bool(is_large_heading),
                "polygonCount": len(polygons),
            }
        )
    if not masks:
        return Image.new("L", image.size, 0), {"strategy": "v5_polygon_mask", "status": "empty", "words": []}
    mask = Image.new("L", image.size, 0)
    for item in masks:
        mask = ImageChops.lighter(mask, item.convert("L"))
    return mask, {
        "strategy": "v5_google_vision_symbol_polygon_mask",
        "status": "passed" if mask.getbbox() else "empty",
        "words": word_reports,
        "whitePixelRatio": float(np.array(mask).mean() / 255.0),
    }


def build_ocr_contour_text_mask(image: Image.Image, block: TextBlock, dilation_px: int) -> Image.Image:
    import cv2

    full = Image.new("L", image.size, 0)
    full_np = np.array(full)
    image_rgb = image.convert("RGB")
    word_styles = [word for word in (block.source_word_styles or []) if isinstance(word, dict) and word.get("bbox")]
    antighost_px = anti_ghost_text_dilation_px(image.size)
    for word in word_styles:
        try:
            word_box = tuple(int(value) for value in word.get("bbox", [])[:4])
        except Exception:
            continue
        if len(word_box) != 4:
            continue
        crop_box = expand_bbox(word_box, image.size, antighost_px + 2)
        x1, y1, x2, y2 = crop_box
        if x2 <= x1 or y2 <= y1:
            continue
        crop = np.array(image_rgb.crop(crop_box), dtype=np.uint8)
        raw_foreground = choose_raw_foreground_mask(crop)
        char_mask_bool, graphic_mask_bool = split_character_and_graphic_contours(raw_foreground)
        foreground = char_mask_bool.astype(np.uint8) * 255
        graphic = graphic_mask_bool.astype(np.uint8) * 255
        if foreground.max() == 0:
            continue
        kernel_size = max(3, antighost_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded_text = cv2.dilate(foreground, kernel, iterations=1)
        graphic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        graphic_protection = cv2.dilate(graphic, graphic_kernel, iterations=1) if graphic.max() else graphic
        foreground = cv2.subtract(expanded_text, graphic_protection)
        estimated_box_values = word.get("estimatedBbox") if isinstance(word, dict) else None
        try:
            estimated_box = tuple(int(value) for value in (estimated_box_values or word_box)[:4])
        except Exception:
            estimated_box = word_box
        word_confinement = np.array(confinement_mask_for_boxes(image.size, [estimated_box], pad_px=antighost_px + 2).crop(crop_box).convert("L"))
        foreground = cv2.bitwise_and(foreground, word_confinement)
        full_np[y1:y2, x1:x2] = np.maximum(full_np[y1:y2, x1:x2], foreground[: y2 - y1, : x2 - x1])
    if word_styles and full_np.max() > 0:
        return Image.fromarray(full_np, "L")
    for box in list(block.line_boxes or []) or [block.bbox]:
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            continue
        crop_box = expand_bbox((left, top, right, bottom), image.size, 2)
        crop = np.array(image_rgb.crop(crop_box))
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
        local_delta = cv2.absdiff(gray, blurred)
        edges = cv2.Canny(gray, 24, 110)
        median = float(np.median(gray))
        rgb = crop.astype(np.int16)
        text_rgb = np.array(parse_hex_color(block.color), dtype=np.int16)
        color_dist = np.linalg.norm(rgb - text_rgb.reshape(1, 1, 3), axis=2)
        saturation = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
        text_brightness = float(np.mean(text_rgb))
        if text_brightness >= 210:
            color_candidate = (color_dist <= 96) & (gray.astype(np.float32) >= median + 4)
        elif text_brightness <= 70:
            color_candidate = (color_dist <= 118) & (gray.astype(np.float32) <= median - 4)
        else:
            color_candidate = (color_dist <= 112) & (saturation >= max(4, int(np.median(saturation) + 2)))
        contrast_candidate = local_delta >= max(5, int(np.percentile(local_delta, 68)))
        edge_candidate = edges > 0
        shadow_candidate = (local_delta >= max(4, int(np.percentile(local_delta, 58)))) & (np.abs(gray.astype(np.float32) - median) >= 3)
        candidate = color_candidate | (edge_candidate & contrast_candidate) | (edge_candidate & shadow_candidate)
        candidate = candidate.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        char_mask_bool, graphic_mask_bool = split_character_and_graphic_contours(candidate)
        candidate = char_mask_bool.astype(np.uint8) * 255
        graphic_mask = graphic_mask_bool.astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        filtered = np.zeros_like(candidate)
        crop_area = max(1, candidate.shape[0] * candidate.shape[1])
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 3:
                continue
            if area / crop_area > 0.42:
                continue
            if h > candidate.shape[0] * 0.96 and w > candidate.shape[1] * 0.55:
                continue
            filtered[labels == label] = 255
        if filtered.max() == 0:
            filtered = candidate
        kernel_size = max(3, antighost_px * 2 + 1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filtered = cv2.dilate(filtered, dilate_kernel, iterations=1)
        if graphic_mask.max():
            graphic_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            filtered = cv2.subtract(filtered, cv2.dilate(graphic_mask, graphic_kernel, iterations=1))
        x1, y1, x2, y2 = crop_box
        full_np[y1:y2, x1:x2] = np.maximum(full_np[y1:y2, x1:x2], filtered[: y2 - y1, : x2 - x1])
    return Image.fromarray(full_np, "L")


def build_strict_text_removal_mask(
    image: Image.Image,
    blocks: list[TextBlock],
    protected_blocks: list[TextBlock] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    raw = Image.new("L", image.size, 0)
    boxes: list[tuple[int, int, int, int]] = []
    mask_reports: list[dict[str, Any]] = []
    dilation_px = strict_mask_dilation_px(image.size)
    feather_px = int(round(strict_mask_feather_px(image.size)))
    for block in blocks:
        if block.render_strategy == "v5_numeric_bypass":
            mask_reports.append({"block": block.id, "strategy": "v5_numeric_bypass", "status": "preserved_step_marker"})
            continue
        has_v5_polygons = any(isinstance(word, dict) and (word.get("symbolPolygons") or word.get("polygon")) for word in (block.source_word_styles or []))
        source_boxes = list(block.line_boxes or []) or [block.bbox]
        for box in source_boxes:
            left, top, right, bottom = box
            if right <= left or bottom <= top:
                continue
            boxes.append((left, top, right, bottom))
        if has_v5_polygons:
            v5_mask, v5_report = build_v5_polygon_text_mask(image, block)
            raw = ImageChops.lighter(raw, v5_mask.convert("L").resize(image.size))
            mask_reports.append({"block": block.id, **v5_report})
            continue
        if should_use_rectangular_strict_mask(block, image.size):
            pass
        try:
            stroke = build_precise_text_stroke_mask(
                image,
                block,
                dilation_px=dilation_px,
                feather_px=max(1, feather_px),
                allow_protected_overlap=False,
            )
            block_mask = stroke.get("inpaintMask_binary") or stroke.get("final")
            contour_mask = build_ocr_contour_text_mask(image, block, dilation_px=dilation_px)
            if isinstance(block_mask, Image.Image):
                block_mask = ImageChops.lighter(block_mask.convert("L").resize(image.size), contour_mask.convert("L").resize(image.size))
            elif contour_mask.getbbox():
                block_mask = contour_mask
            if isinstance(block_mask, Image.Image) and block_mask.getbbox():
                block_mask = ImageChops.multiply(
                    block_mask.convert("L").resize(image.size),
                    confinement_mask_for_boxes(image.size, source_boxes, pad_px=2),
                )
                raw = ImageChops.lighter(raw, block_mask.convert("L").resize(image.size))
                mask_reports.append(
                    {
                        "block": block.id,
                        "status": stroke.get("maskQualityStatus"),
                        "strategy": "precise_text_stroke_contour_with_dilation",
                        "failure": stroke.get("maskFailureReason"),
                        "whitePixelRatio": stroke.get("whitePixelRatio"),
                        "textCoverageEstimate": stroke.get("textCoverageEstimate"),
                        "backgroundLeakageEstimate": stroke.get("backgroundLeakageEstimate"),
                    }
                )
                continue
        except Exception as exc:
            mask_reports.append({"block": block.id, "status": "precise_mask_failed", "failure": str(exc)[:160]})
        contour = build_ocr_contour_text_mask(image, block, dilation_px)
        contour = ImageChops.multiply(contour.convert("L"), confinement_mask_for_boxes(image.size, source_boxes, pad_px=2))
        raw = ImageChops.lighter(raw, contour)
        mask_reports.append(
            {
                "block": block.id,
                "status": "passed" if contour.getbbox() else "empty_contour_mask",
                "strategy": "ocr_text_contour_with_dilation",
                "whitePixelRatio": float(np.array(contour).mean() / 255.0),
                "textCoverageEstimate": 0.86 if contour.getbbox() else 0.0,
                "backgroundLeakageEstimate": 0.04 if contour.getbbox() else 0.0,
            }
        )
    dilated = raw.convert("L")
    protected_boxes: list[tuple[int, int, int, int]] = []
    if protected_blocks:
        protected = Image.new("L", image.size, 0)
        protected_draw = ImageDraw.Draw(protected)
        for block in protected_blocks:
            if block.translate:
                continue
            for box in list(block.line_boxes or []) or [block.bbox]:
                left, top, right, bottom = box
                if right <= left or bottom <= top:
                    continue
                protected_boxes.append((left, top, right, bottom))
                pad = max(2, min(8, strict_mask_dilation_px(image.size) // 2))
                protected_draw.rectangle(
                    (
                        max(0, left - pad),
                        max(0, top - pad),
                        min(image.width, right + pad),
                        min(image.height, bottom + pad),
                    ),
                    fill=255,
                )
        if protected_boxes:
            dilated = ImageChops.multiply(dilated, ImageOps.invert(protected.convert("L")))
    return dilated, {
        "mode": "strict_text_removal_mask",
        "shape": "glyph_stroke_mask_with_dilation",
        "dilationPx": dilation_px,
        "featherPx": strict_mask_feather_px(image.size),
        "rawWhitePixelRatio": float(np.array(raw).mean() / 255.0),
        "dilatedWhitePixelRatio": float(np.array(dilated).mean() / 255.0),
        "boxes": [list(box) for box in boxes],
        "protectedBoxes": [list(box) for box in protected_boxes],
        "blockMaskReports": mask_reports,
        "strictPreservation": "provider output is composited only through this dilated mask; all other pixels remain source-original",
    }


def resolved_replicate_lama_model() -> str:
    return os.getenv(
        "REPLICATE_LAMA_MODEL",
        "twn39/lama:2b91ca2340801c2a5be745612356fac36a17f698354a07f48a62d564d3b3a7a0",
    ).strip()


def replicate_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def run_replicate_lama_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image | None:
    global LAST_REPLICATE_LAMA_ERROR
    LAST_REPLICATE_LAMA_ERROR = ""
    token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    model = resolved_replicate_lama_model()
    if not token or "/" not in model:
        LAST_REPLICATE_LAMA_ERROR = "missing_REPLICATE_API_TOKEN" if not token else "invalid_REPLICATE_LAMA_MODEL"
        return None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Prefer": "wait"}
    input_payload = {
        "image": replicate_data_uri(image.convert("RGB")),
        "mask": replicate_data_uri(mask.convert("L")),
    }
    try:
        if ":" in model:
            endpoint = "https://api.replicate.com/v1/predictions"
            request_payload = {"version": model, "input": input_payload}
        else:
            owner, name = model.split("/", 1)
            endpoint = f"https://api.replicate.com/v1/models/{owner}/{name}/predictions"
            request_payload = {"input": input_payload}
        response = requests.post(
            endpoint,
            headers=headers,
            json=request_payload,
            timeout=int(os.getenv("REPLICATE_TIMEOUT", "90")),
        )
        response.raise_for_status()
        payload = response.json()
        get_url = (payload.get("urls") or {}).get("get")
        status = str(payload.get("status", "")).lower()
        deadline = time.time() + int(os.getenv("REPLICATE_POLL_TIMEOUT", "120"))
        while status not in {"succeeded", "failed", "canceled"} and get_url and time.time() < deadline:
            time.sleep(2)
            poll = requests.get(get_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
            poll.raise_for_status()
            payload = poll.json()
            status = str(payload.get("status", "")).lower()
        if status in {"failed", "canceled"}:
            LAST_REPLICATE_LAMA_ERROR = f"status={status}; error={payload.get('error')}"
            print(f"[localize-v2] Replicate LaMa ended with {LAST_REPLICATE_LAMA_ERROR}", flush=True)
            return None
        output = payload.get("output")
        if isinstance(output, list):
            output = output[0] if output else None
        if isinstance(output, str) and output.startswith("http"):
            image_response = requests.get(output, timeout=45)
            image_response.raise_for_status()
            return Image.open(io.BytesIO(image_response.content)).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    except Exception as exc:
        LAST_REPLICATE_LAMA_ERROR = str(exc)[:500]
        print(f"[localize-v2] Replicate LaMa failed: {exc}", flush=True)
    return None


def run_openai_localize_v2_cleanup(image: Image.Image, mask: Image.Image) -> Image.Image | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        client = OpenAI()
        image_stream = io.BytesIO()
        mask_stream = io.BytesIO()
        image.convert("RGBA").save(image_stream, format="PNG")
        mask_l = mask.convert("L").resize(image.size)
        alpha = ImageOps.invert(mask_l)
        rgba_mask = Image.new("RGBA", image.size, (255, 255, 255, 255))
        rgba_mask.putalpha(alpha)
        rgba_mask.save(mask_stream, format="PNG")
        image_stream.name = "image.png"
        mask_stream.name = "mask.png"
        prompt = (
            "Remove only the visible marketing text inside the transparent/editable masked areas. "
            "Reconstruct the original background, colors, gradients, edges, and texture. "
            "Do not add any text. Do not alter products, logos, packaging, labels, arrows, or unmasked areas."
        )
        request_kwargs = {
            "model": os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-2"),
            "image": image_stream,
            "mask": mask_stream,
            "prompt": prompt,
            "quality": os.getenv("ADAPTIFAI_OPENAI_IMAGE_QUALITY", "medium"),
            "size": nearest_openai_edit_size(image.width, image.height),
        }
        try:
            response = client.images.edit(**request_kwargs, output_format="png", response_format="b64_json")
        except Exception:
            image_stream.seek(0)
            mask_stream.seek(0)
            response = client.images.edit(**request_kwargs)
        data = response.data[0]
        encoded = getattr(data, "b64_json", None)
        if not encoded and hasattr(data, "model_dump"):
            encoded = data.model_dump().get("b64_json")
        if encoded:
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
    except Exception as exc:
        print(f"[localize-v2] OpenAI cleanup failed: {exc}", flush=True)
    return None


def composite_provider_cleanup_over_source(source: Image.Image, provider_image: Image.Image, mask: Image.Image) -> Image.Image:
    provider_resized = provider_image.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    hard_mask = mask.convert("L").resize(source.size)
    if env_flag("ADAPTIFAI_LOCALIZE_V2_FEATHER_PROVIDER_COMPOSITE", "0"):
        feather_px = strict_mask_feather_px(source.size)
        soft_edge = hard_mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        alpha = ImageChops.lighter(hard_mask, soft_edge)
    else:
        alpha = Image.fromarray(np.where(np.array(hard_mask, dtype=np.uint8) > 16, 255, 0).astype(np.uint8), "L")
    if env_flag("ADAPTIFAI_LOCALIZE_V213_SEAMLESS_PROVIDER_BLEND", "1"):
        blended = seamless_clone_local_patch(source.convert("RGB"), provider_resized, hard_mask)
        return Image.composite(blended.convert("RGB"), source.convert("RGB"), alpha)
    return Image.composite(provider_resized, source.convert("RGB"), alpha)


def seamless_clone_local_patch(base: Image.Image, patch: Image.Image, mask: Image.Image) -> Image.Image:
    import cv2

    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    bbox = Image.fromarray(mask_np, "L").getbbox()
    if not bbox:
        return Image.composite(patch.convert("RGB"), base.convert("RGB"), mask.convert("L"))
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 4 or y2 - y1 < 4:
        return Image.composite(patch.convert("RGB"), base.convert("RGB"), mask.convert("L"))
    base_rgb = base.convert("RGB")
    patch_rgb = patch.convert("RGB").resize(base.size, Image.Resampling.BICUBIC)
    source_np = cv2.cvtColor(np.array(base_rgb), cv2.COLOR_RGB2BGR)
    patch_np = cv2.cvtColor(np.array(patch_rgb), cv2.COLOR_RGB2BGR)
    clone_mask = np.where(mask_np > 16, 255, 0).astype(np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    try:
        cloned = cv2.seamlessClone(patch_np, source_np, clone_mask, center, cv2.NORMAL_CLONE)
        return Image.fromarray(cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.composite(patch_rgb, base_rgb, mask.convert("L"))


def dominant_background_fill_color(sample: np.ndarray, local_mask: np.ndarray) -> tuple[int, int, int]:
    if sample.size == 0:
        return (255, 255, 255)
    pixels = sample[~local_mask] if local_mask.shape[:2] == sample.shape[:2] and np.any(~local_mask) else sample.reshape(-1, 3)
    if pixels.size == 0:
        pixels = sample.reshape(-1, 3)
    pixels = pixels.reshape(-1, 3)
    brightness = pixels.mean(axis=1)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    bright_neutral = pixels[(brightness >= 232) & (chroma <= 24)]
    if len(bright_neutral) >= max(24, len(pixels) * 0.08):
        chosen = np.median(bright_neutral, axis=0)
    else:
        chosen = np.median(pixels, axis=0)
    return tuple(int(value) for value in np.clip(chosen, 0, 255))


def fill_remaining_mask_components(result: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    import cv2

    output = result.convert("RGB")
    base = source.convert("RGB")
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    binary = (mask_np > 16).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return output
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or w <= 0 or h <= 0:
            continue
        pad = max(10, min(32, int(round(max(w, h) * 0.45))))
        sample_box = (
            max(0, x - pad),
            max(0, y - pad),
            min(source.width, x + w + pad),
            min(source.height, y + h + pad),
        )
        sample = np.array(base.crop(sample_box))
        local_mask = np.array(mask.crop(sample_box).convert("L")) > 16
        fill_rgb = dominant_background_fill_color(sample, local_mask)
        patch = Image.new("RGB", (sample_box[2] - sample_box[0], sample_box[3] - sample_box[1]), fill_rgb)
        patch_mask_np = np.array(mask.crop(sample_box).convert("L"), dtype=np.uint8)
        hard_patch_mask = Image.fromarray(np.where(patch_mask_np > 16, 255, 0).astype(np.uint8), "L")
        output.paste(patch, sample_box[:2], hard_patch_mask)
    return output


def scrub_provider_residuals_inside_mask(result: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    if not env_flag("ADAPTIFAI_LOCALIZE_V2_PROVIDER_RESIDUAL_SCRUB", "1"):
        return result.convert("RGB")
    return fill_remaining_mask_components(result, source, mask.convert("L"))


def restore_v5_masked_background_by_source_style(result: Image.Image, source: Image.Image, blocks: list[TextBlock]) -> Image.Image:
    output = result.convert("RGB")
    base = source.convert("RGB")
    for block in blocks:
        if block.render_strategy == "v5_numeric_bypass":
            continue
        for word in block.source_word_styles or []:
            if not isinstance(word, dict):
                continue
            if is_v5_isolated_step_marker_word(word, [item for item in block.source_word_styles if isinstance(item, dict)]):
                continue
            box = v5_word_box(word)
            if box is None:
                continue
            has_bg = bool(word.get("hasTextBackground") and word.get("backgroundColor"))
            if has_bg:
                word_mask = build_v5_foreground_pixel_word_mask(source, word, 2)
                fill_rgb = parse_hex_color(str(word.get("backgroundColor") or "#ffffff"), fallback=(255, 255, 255))
            else:
                raw_polygons = word.get("symbolPolygons") or ([word.get("polygon")] if word.get("polygon") else [])
                polygons: list[list[tuple[int, int]]] = []
                for raw_polygon in raw_polygons:
                    if not isinstance(raw_polygon, list):
                        continue
                    polygon = [
                        (int(point[0]), int(point[1]))
                        for point in raw_polygon
                        if isinstance(point, (list, tuple)) and len(point) >= 2
                    ]
                    if len(polygon) >= 3 and polygon_area(polygon) > 1:
                        polygons.append(polygon)
                if not polygons:
                    continue
                word_mask = polygon_mask(source.size, polygons, dilation_px=3)
                sample_box = expand_bbox(box, source.size, max(8, int((box[3] - box[1]) * 0.9)))
                sample = np.array(base.crop(sample_box).convert("RGB"))
                local_mask = np.array(word_mask.crop(sample_box).convert("L")) > 16
                fill_rgb = dominant_background_fill_color(sample, local_mask)
            if not word_mask.getbbox():
                continue
            patch = Image.new("RGB", output.size, fill_rgb)
            hard_mask = Image.fromarray(np.where(np.array(word_mask.convert("L"), dtype=np.uint8) > 16, 255, 0).astype(np.uint8), "L")
            output = Image.composite(patch, output, hard_mask)
    return output


def strict_local_fill_cleanup(source: Image.Image, mask: Image.Image, blocks: list[TextBlock]) -> tuple[Image.Image, dict[str, Any]]:
    base = source.convert("RGB")
    result = base.copy()
    mask_l = mask.convert("L")
    seamless_used = 0
    for block in blocks:
        boxes = list(block.line_boxes or [block.bbox])
        for box in boxes:
            left, top, right, bottom = box
            if right <= left or bottom <= top:
                continue
            pad = max(8, strict_mask_dilation_px(source.size) * 2)
            sample_box = (
                max(0, left - pad),
                max(0, top - pad),
                min(source.width, right + pad),
                min(source.height, bottom + pad),
            )
            sample = np.array(base.crop(sample_box))
            local_mask = np.array(mask_l.crop(sample_box)) > 0
            if sample.size == 0:
                continue
            fill_rgb = dominant_background_fill_color(sample, local_mask)
            patch = Image.new("RGB", (sample_box[2] - sample_box[0], sample_box[3] - sample_box[1]), fill_rgb)
            patch_mask = mask_l.crop(sample_box)
            if env_flag("ADAPTIFAI_LOCALIZE_V213_SEAMLESS_LOCAL_FILL", "0"):
                region_base = result.crop(sample_box)
                patch_candidate = Image.new("RGB", region_base.size, fill_rgb)
                region_pixels = np.array(region_base)
                local_std = float(np.mean(np.std(region_pixels.reshape(-1, 3), axis=0))) if region_pixels.size else 0.0
                if local_std > 6.0 and np.count_nonzero(np.array(patch_mask) > 16) >= 24:
                    blended = seamless_clone_local_patch(region_base, patch_candidate, patch_mask)
                    result.paste(blended, sample_box[:2], patch_mask)
                    seamless_used += 1
                    continue
            result.paste(patch, sample_box[:2], patch_mask)
    result = fill_remaining_mask_components(result, source, mask_l)
    return result, {"provider": "local-fill", "strategy": "median_border_fill_inside_strict_mask_hard_inner_feather_edge", "seamlessClonePatches": seamless_used}


def strict_mask_region_is_flat(source: Image.Image, mask: Image.Image) -> bool:
    mask_np = np.array(mask.convert("L")) > 0
    if not np.any(mask_np):
        return True
    rgb = np.array(source.convert("RGB"))
    pixels = rgb[mask_np]
    if pixels.size == 0:
        return True
    channel_std = float(np.mean(np.std(pixels.reshape(-1, 3), axis=0)))
    return channel_std <= float(os.getenv("ADAPTIFAI_LOCALIZE_V2_FLAT_STD_THRESHOLD", "48"))


def polish_masked_cleanup_with_opencv(cleaned: Image.Image, mask: Image.Image) -> Image.Image:
    import cv2

    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    if int(np.count_nonzero(mask_np > 16)) == 0:
        return cleaned.convert("RGB")
    source = cv2.cvtColor(np.array(cleaned.convert("RGB")), cv2.COLOR_RGB2BGR)
    radius = max(3, int(os.getenv("ADAPTIFAI_LOCALIZE_V212_POLISH_RADIUS", "4")))
    polished = cv2.inpaint(source, mask_np, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(polished, cv2.COLOR_BGR2RGB))


def inpaint_localize_v2_base(image: Image.Image, mask: Image.Image, blocks: list[TextBlock] | None = None) -> tuple[Image.Image, dict[str, Any]]:
    is_v5_polygon_pipeline = any(str(block.id or "").startswith("v5-block-") for block in (blocks or []))
    # Flat graphic/text panels are safer with deterministic fill: generative
    # providers can hallucinate residual letters even when instructed not to.
    if not is_v5_polygon_pipeline and env_flag("ADAPTIFAI_LOCALIZE_V2_LOCAL_FILL_FIRST", "1"):
        return strict_local_fill_cleanup(image, mask, blocks or [])
    split_small_overlay = (not is_v5_polygon_pipeline) and not env_flag("ADAPTIFAI_LOCALIZE_V212_PROVIDER_CLEANUP", "1")
    if blocks and split_small_overlay:
        large_blocks = [block for block in blocks if should_use_rectangular_strict_mask(block, image.size)]
        small_blocks = [block for block in blocks if block not in large_blocks]
        working = image
        small_meta: dict[str, Any] | None = None
        provider_mask = mask
        if small_blocks:
            small_mask, small_mask_meta = build_strict_text_removal_mask(image, small_blocks)
            working, small_meta = strict_local_fill_cleanup(image, small_mask, small_blocks)
            small_meta = {**small_meta, "strictMasking": small_mask_meta}
        if large_blocks:
            provider_mask, _large_mask_meta = build_strict_text_removal_mask(image, large_blocks)
            image = working
            mask = provider_mask
        elif small_meta is not None:
            return working, {"provider": "split-local-fill", "smallOverlayCleanup": small_meta}
    replicate_result = run_replicate_lama_inpaint(image, mask)
    if replicate_result is not None:
        meta: dict[str, Any] = {"provider": "replicate", "model": resolved_replicate_lama_model()}
        if blocks:
            meta["splitCleanup"] = {"provider": "replicate", "maskMode": "contour" if not split_small_overlay else "split"}
        cleaned = composite_provider_cleanup_over_source(image, replicate_result, mask)
        if is_v5_polygon_pipeline:
            restored = restore_v5_masked_background_by_source_style(cleaned, image, blocks or [])
            return restored, {
                **meta,
                "maskedPolish": "disabled_for_v5_polygon_to_prevent_color_bleed",
                "residualScrub": "v5_word_level_source_background_restore",
            }
        scrubbed = scrub_provider_residuals_inside_mask(cleaned, image, mask)
        return polish_masked_cleanup_with_opencv(scrubbed, mask), {**meta, "maskedPolish": "opencv-telea", "residualScrub": "inside-character-mask"}
    if is_v5_polygon_pipeline:
        raise RuntimeError(f"V5 polygon localize requires Replicate LaMa cleanup; replicate failed: {LAST_REPLICATE_LAMA_ERROR or 'unknown'}")
    vertex_prompt = (
        "Remove only masked marketing text and reconstruct the original background faithfully. "
        "Do not add text. Preserve products, packaging, logos, arrows, labels, colors, gradients, and unmasked areas."
    )
    vertex_result = run_vertex_full_image_cleanup(image, mask, vertex_prompt)
    if vertex_result.get("success") and isinstance(vertex_result.get("image"), Image.Image):
        meta = {
            "provider": "vertex",
            "model": vertex_imagen_edit_model(),
        }
        if LAST_REPLICATE_LAMA_ERROR:
            meta["replicateFallbackReason"] = LAST_REPLICATE_LAMA_ERROR
        if blocks:
            meta["splitCleanup"] = {"smallOverlayProvider": "local-fill", "largeOverlayProvider": "vertex"}
        cleaned = composite_provider_cleanup_over_source(image, vertex_result["image"], mask)
        scrubbed = scrub_provider_residuals_inside_mask(cleaned, image, mask)
        return polish_masked_cleanup_with_opencv(scrubbed, mask), {**meta, "maskedPolish": "opencv-telea", "residualScrub": "inside-character-mask"}
    openai_result = run_openai_localize_v2_cleanup(image, mask)
    if openai_result is not None:
        meta = {"provider": "openai", "model": os.getenv("ADAPTIFAI_OPENAI_IMAGE_MODEL", "gpt-image-2")}
        if LAST_REPLICATE_LAMA_ERROR:
            meta["replicateFallbackReason"] = LAST_REPLICATE_LAMA_ERROR
        if blocks:
            meta["splitCleanup"] = {"smallOverlayProvider": "local-fill", "largeOverlayProvider": "openai"}
        cleaned = composite_provider_cleanup_over_source(image, openai_result, mask)
        scrubbed = scrub_provider_residuals_inside_mask(cleaned, image, mask)
        return polish_masked_cleanup_with_opencv(scrubbed, mask), {**meta, "maskedPolish": "opencv-telea", "residualScrub": "inside-character-mask"}
    import cv2

    source = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask_np = np.array(mask.convert("L"))
    radius = int(os.getenv("ADAPTIFAI_LOCALIZE_V2_OPENCV_RADIUS", "5"))
    cleaned = cv2.inpaint(source, mask_np, radius, cv2.INPAINT_TELEA)
    result = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    meta = {"provider": "opencv", "radius": radius}
    if LAST_REPLICATE_LAMA_ERROR:
        meta["replicateFallbackReason"] = LAST_REPLICATE_LAMA_ERROR
    if blocks:
        meta["splitCleanup"] = {"smallOverlayProvider": "local-fill", "largeOverlayProvider": "opencv"}
    return result, meta


def wrap_plain_text_for_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for source_line in [line.strip() for line in text.splitlines() if line.strip()] or [text.strip()]:
        words = source_line.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [text]


def fit_plain_text_for_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    bold: bool,
    preferred_size: int,
) -> tuple[list[str], int, int]:
    max_width = max(8, box[2] - box[0])
    max_height = max(8, box[3] - box[1])
    start = max(8, min(96, preferred_size))
    for size in range(start, 7, -1):
        font = get_font(size, bold=bold)
        line_height = max(size + 2, int(size * 1.08))
        lines = wrap_plain_text_for_box(draw, text, font, max_width)
        widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
        if widest <= max_width and len(lines) * line_height <= max_height:
            return lines, size, line_height
    font = get_font(8, bold=bold)
    return wrap_plain_text_for_box(draw, text, font, max_width), 8, 10


def block_has_rich_text_segments(block: TextBlock) -> bool:
    spans = block.translated_style_spans or []
    if "[BOLD]" in (block.translated_text or "") or "[/BOLD]" in (block.translated_text or ""):
        return True
    if not spans:
        return False
    if is_v5_block(block):
        return True
    if any(span.get("styleTransferMode") == "model_word_level_semantic_mapping" for span in spans):
        return True
    if any(
        (span.get("style") or {}).get("hasTextBackground")
        or (span.get("style") or {}).get("backgroundColor")
        or (span.get("style") or {}).get("strokeWidth")
        or (span.get("style") or {}).get("fillTransparent")
        for span in spans
    ):
        return True
    colors = {str((span.get("style") or {}).get("color") or span.get("color") or "").lower() for span in spans}
    weights = {int((span.get("style") or {}).get("fontWeight") or span.get("fontWeight") or block.font_weight) for span in spans}
    return len({color for color in colors if color}) > 1 or len(weights) > 1


def fit_styled_spans_strict(
    draw: ImageDraw.ImageDraw,
    spans: list[dict[str, Any]],
    box: tuple[int, int, int, int],
    base_typography: dict[str, Any],
    *,
    honor_force_breaks: bool = True,
    allow_vertical_overflow: bool = False,
) -> dict[str, Any]:
    max_width = max(8, box[2] - box[0])
    max_height = max(8, box[3] - box[1])
    line_height_ratio = max(1.12, min(1.42, float(base_typography.get("lineHeight", 18)) / max(1.0, float(base_typography.get("fontSize", 16)))))
    if allow_vertical_overflow:
        lines = build_styled_lines(draw, spans, max_width, 1.0, honor_force_breaks=honor_force_breaks)
        widest, total_height, line_layouts = measure_styled_layout(draw, lines, 1.0, line_height_ratio)
        return {
            "scaleFactor": 1.0,
            "lines": line_layouts,
            "widest": widest,
            "totalHeight": total_height,
            "overflow": max(0, widest - max_width),
            "verticalOverflowAllowed": True,
            "preferredLineCount": None,
            "lineCountDelta": 0,
            "score": 1000 - max(0, widest - max_width) * 10,
        }
    best: dict[str, Any] | None = None
    for scale_percent in range(100, 19, -5):
        scale_factor = scale_percent / 100.0
        lines = build_styled_lines(draw, spans, max_width, scale_factor, honor_force_breaks=honor_force_breaks)
        if not lines:
            continue
        widest, total_height, line_layouts = measure_styled_layout(draw, lines, scale_factor, line_height_ratio)
        overflow = max(0, widest - max_width) + max(0, total_height - max_height)
        candidate = {
            "scaleFactor": scale_factor,
            "lines": line_layouts,
            "widest": widest,
            "totalHeight": total_height,
            "overflow": overflow,
            "preferredLineCount": None,
            "lineCountDelta": 0,
            "score": scale_factor * 1000 - overflow * 10,
        }
        if overflow == 0:
            return candidate
        if best is None or overflow < best.get("overflow", 10**9) or candidate["score"] > best.get("score", -10**9):
            best = candidate
    return best or {"scaleFactor": 1.0, "lines": [], "widest": 0, "totalHeight": 0, "overflow": max_width + max_height}


def v62_source_word_height_map(block: TextBlock) -> dict[str, int]:
    heights: dict[str, int] = {}
    source_words = [word for word in (block.source_word_styles or []) if isinstance(word, dict)]
    for word in block.source_word_styles or []:
        if not isinstance(word, dict):
            continue
        if is_v5_isolated_step_marker_word(word, source_words):
            continue
        source_id = str(word.get("id") or "").strip()
        if not source_id:
            continue
        box = word.get("bbox") or word.get("estimatedBbox") or []
        if word.get("peerRowFontSize"):
            height = max(1, int(word.get("peerRowFontSize") or word.get("fontSize") or block.font_size_estimate or 16))
        elif isinstance(box, list) and len(box) == 4:
            height = max(1, int(box[3]) - int(box[1]))
        else:
            height = max(1, int(word.get("fontSize") or block.font_size_estimate or 16))
        heights[source_id] = height
    return heights


def v62_token_source_height(token: dict[str, Any], source_heights: dict[str, int], fallback_size: int) -> int:
    ids = token.get("sourceWordIds") or []
    matched = [source_heights[source_id] for source_id in ids if source_id in source_heights]
    if matched:
        return max(matched)
    return max(1, int(fallback_size))


def order_v62_spans_for_source_background_slots(block: TextBlock, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_words = [word for word in (block.source_word_styles or []) if isinstance(word, dict)]
    if not any(word.get("hasTextBackground") and word.get("backgroundColor") for word in source_words):
        return spans
    if len(spans) <= 1:
        return spans

    word_line_by_id = {
        str(word.get("id") or ""): int(word.get("lineIndex") or 0)
        for word in source_words
        if str(word.get("id") or "")
    }

    def span_line_index(span: dict[str, Any], fallback: int) -> int:
        source_styles = [item for item in (span.get("sourceWordStyles") or []) if isinstance(item, dict)]
        line_indexes = [int(item.get("lineIndex") or 0) for item in source_styles if item.get("lineIndex") is not None]
        if line_indexes:
            return min(line_indexes)
        ids = []
        raw_ids = span.get("sourceWordIds") or span.get("source_word_ids") or []
        if isinstance(raw_ids, list):
            ids.extend(str(value) for value in raw_ids if str(value))
        source_id = str(span.get("sourceWordId") or span.get("source_word_id") or "")
        if source_id:
            ids.append(source_id)
        matched = [word_line_by_id[source_id] for source_id in ids if source_id in word_line_by_id]
        return min(matched) if matched else fallback

    return [
        span
        for _, _, span in sorted(
            (span_line_index(span, index), index, span)
            for index, span in enumerate(spans)
        )
    ]


def v62_measure_tokens_with_line_font(draw: ImageDraw.ImageDraw, tokens: list[dict[str, Any]], line_font_size: int) -> tuple[int, list[dict[str, Any]], int, int]:
    line_width = 0.0
    metrics: list[dict[str, Any]] = []
    max_ascent = 0
    max_descent = 0
    first_style = dict((tokens[0].get("style") if tokens else {}) or {})
    first_weight = int(first_style.get("fontWeight") or 700)
    line_point_size = v62_calibrated_point_size(
        draw,
        tokens,
        max(1, int(line_font_size)),
        font_weight=first_weight,
        font_category=str(first_style.get("fontCategory") or "sans-serif"),
        bold=first_weight >= 700,
    )

    for index, token in enumerate(tokens):
        style = dict(token.get("style", {}))
        style["fontSize"] = max(1, int(line_point_size))
        style["sourcePixelHeight"] = max(1, int(line_font_size))

        font_weight = int(style.get("fontWeight") or 700)
        font = get_font(
            size=int(line_point_size),
            bold=font_weight >= 700,
            category=str(style.get("fontCategory") or "sans-serif"),
            weight=font_weight,
        )

        try:
            ascent, descent = font.getmetrics()
        except Exception:
            ascent, descent = max(1, int(line_point_size)), max(1, int(line_point_size) // 4)

        token_text = token["text"]

        next_token = tokens[index + 1] if index < len(tokens) - 1 else None
        add_trailing_space = bool(next_token) and not bool(next_token.get("noSpaceBefore"))
        try:
            width = draw.textlength(token_text, font=font) + (draw.textlength(" ", font=font) if add_trailing_space else 0.0)
        except AttributeError:
            width = text_width(draw, token_text, font) + (text_width(draw, " ", font) if add_trailing_space else 0.0)

        line_width += width
        max_ascent = max(max_ascent, ascent)
        max_descent = max(max_descent, descent)

        metrics.append({
            "font": font,
            "size": int(line_point_size),
            "sourcePixelHeight": int(line_font_size),
            "ascent": ascent,
            "descent": descent,
            "style": style,
            "role": token.get("role", "benefit"),
            "text": token["text"],
            "xAdvance": width,
            "noSpaceBefore": bool(token.get("noSpaceBefore")),
        })

    return int(round(line_width)), metrics, max_ascent, max_descent


def fit_v62_geometric_typesetting(
    draw: ImageDraw.ImageDraw,
    block: TextBlock,
    spans: list[dict[str, Any]],
    box: tuple[int, int, int, int],
) -> dict[str, Any]:
    max_width = max(1, box[2] - box[0])
    max_height = max(1, box[3] - box[1])
    source_line_widths = [
        max(1, int(line_box[2] - line_box[0]))
        for line_box in (block.line_boxes or [])
        if isinstance(line_box, (list, tuple)) and len(line_box) >= 4 and int(line_box[2]) > int(line_box[0])
    ]
    if len(source_line_widths) > 1:
        max_width = min(max_width, max(source_line_widths) + max(6, int(max(source_line_widths) * 0.06)))
    source_heights = v62_source_word_height_map(block)
    fallback_size = max(1, int(block.font_size_estimate or default_typography_style(block).get("fontSize", 16)))

    tokens: list[dict[str, Any]] = []
    for span in spans:
        tokens.extend(tokenize_style_span(span))

    if not tokens:
        block_center_x = (box[0] + box[2]) / 2.0
        return {
            "scaleFactor": 1.0,
            "lines": [],
            "widest": 0,
            "totalHeight": 0,
            "overflow": 0,
            "verticalOverflowAllowed": True,
            "blockCenterX": block_center_x,
            "typesetting": "v6.5-calibrated-rich-v5",
            "score": 1000,
        }

    block_center_x = (box[0] + box[2]) / 2.0
    preferred_line_count = max(1, len([line for line in (block.line_texts or []) if str(line).strip()]) or 1)
    best_layout: dict[str, Any] | None = None
    best_preferred_layout: dict[str, Any] | None = None
    has_text_background = any(isinstance(word, dict) and word.get("hasTextBackground") for word in (block.source_word_styles or []))
    min_scale = float(os.getenv("ADAPTIFAI_V5_BG_RENDER_MIN_SCALE", "0.28")) if has_text_background else float(os.getenv("ADAPTIFAI_V5_RENDER_MIN_SCALE", "0.45"))
    for scale_factor in np.linspace(1.0, max(0.38, min_scale), 20):
        scaled_height = lambda token: max(1, int(round(v62_token_source_height(token, source_heights, fallback_size) * float(scale_factor))))
        full_line_font_size = max(scaled_height(token) for token in tokens)
        full_line_width, _, _, _ = v62_measure_tokens_with_line_font(draw, tokens, full_line_font_size)
        honor_force_breaks = full_line_width > max_width

        lines: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for token in tokens:
            candidate = current + [token]
            candidate_font_size = max(scaled_height(item) for item in candidate)
            candidate_width, _, _, _ = v62_measure_tokens_with_line_font(draw, candidate, candidate_font_size)
            if current and candidate_width > max_width:
                lines.append(current)
                current = [token]
            else:
                current = candidate
            if honor_force_breaks and token.get("forceBreakAfter") and current:
                lines.append(current)
                current = []
        if current:
            lines.append(current)

        line_layouts: list[dict[str, Any]] = []
        widest = 0
        total_height = 0
        for line in lines:
            line_font_size = max(scaled_height(token) for token in line)
            line_width, metrics, max_ascent, max_descent = v62_measure_tokens_with_line_font(draw, line, line_font_size)
            line_height_measured = max_ascent + max_descent
            line_height = int(round(line_height_measured + (line_font_size * 0.25)))
            widest = max(widest, line_width)
            total_height += line_height
            line_layouts.append(
                {
                    "tokens": metrics,
                    "lineWidth": line_width,
                    "lineHeight": line_height,
                    "maxAscent": max_ascent,
                    "maxDescent": max_descent,
                    "lineFontSize": line_font_size,
                }
            )
        overflow_x = max(0, widest - max_width)
        vertical_limit = int(round(max_height * 1.35)) if has_text_background else max_height
        overflow_y = max(0, total_height - vertical_limit) if has_text_background else 0
        oversized_background_text = 0
        if has_text_background:
            target_line_height = max(1, int(round(max_height * 1.35)))
            oversized_background_text = sum(max(0, int(line.get("lineHeight", 0)) - target_line_height) for line in line_layouts)
        if preferred_line_count > 1 and len(lines) < preferred_line_count and not has_text_background:
            overflow_y += max_height * (preferred_line_count - len(lines))
        line_delta = abs(len(lines) - preferred_line_count)
        line_count_underflow = max(0, preferred_line_count - len(lines))
        score = (
            1000.0
            - overflow_x * 12.0
            - overflow_y * (18.0 if has_text_background else 5.0)
            - oversized_background_text * (24.0 if has_text_background else 0.0)
            - line_delta * 180.0
            - line_count_underflow * 180.0
            - max(0.0, 1.0 - float(scale_factor)) * (90.0 if has_text_background else 260.0)
        )
        if line_count_underflow > 0 and not has_text_background:
            score -= 100000.0 * line_count_underflow
        layout = {
            "scaleFactor": float(scale_factor),
            "lines": line_layouts,
            "widest": widest,
            "totalHeight": total_height,
            "overflow": overflow_x + overflow_y,
            "verticalOverflowAllowed": True,
            "blockCenterX": block_center_x,
            "typesetting": "v6.6-calibrated-rich-v5-fit",
            "score": score,
        }
        if overflow_x == 0 and overflow_y == 0 and line_delta == 0:
            if best_preferred_layout is None or float(scale_factor) > float(best_preferred_layout.get("scaleFactor", 0.0)):
                best_preferred_layout = layout
        if best_layout is None or score > float(best_layout.get("score", -999999)):
            best_layout = layout
    if best_preferred_layout is not None:
        return best_preferred_layout
    return best_layout or {
        "scaleFactor": 1.0,
        "lines": [],
        "widest": 0,
        "totalHeight": 0,
        "overflow": max_width,
        "verticalOverflowAllowed": True,
        "blockCenterX": block_center_x,
        "typesetting": "v6.6-calibrated-rich-v5-fit",
        "score": 0,
    }


def is_v5_block(block: TextBlock) -> bool:
    return bool(block.id and str(block.id).startswith("v5-"))


def v5_word_box(word: dict[str, Any]) -> tuple[int, int, int, int] | None:
    raw_box = word.get("bbox") or word.get("estimatedBbox") or []
    if not (isinstance(raw_box, list) and len(raw_box) == 4):
        return None
    box = tuple(int(value) for value in raw_box)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def is_v5_isolated_step_marker_word(word: dict[str, Any], words: list[dict[str, Any]]) -> bool:
    box = v5_word_box(word)
    if box is None:
        return False
    text = str(word.get("text") or "").strip()
    if len(text) > 2 and not is_decorative_or_numeric_only(text):
        return False
    other_boxes = [
        other_box
        for other in words
        if other is not word and (other_box := v5_word_box(other)) is not None
    ]
    if not other_boxes:
        return False
    heights = sorted(max(1, other_box[3] - other_box[1]) for other_box in other_boxes)
    median_height = heights[len(heights) // 2]
    height = max(1, box[3] - box[1])
    main_left = min(other_box[0] for other_box in other_boxes if other_box[3] > box[1] and other_box[1] < box[3]) if any(
        other_box[3] > box[1] and other_box[1] < box[3] for other_box in other_boxes
    ) else min(other_box[0] for other_box in other_boxes)
    return height >= max(32, int(median_height * 1.75)) and box[2] < main_left - max(20, median_height * 2)


def v5_strict_render_box(block: TextBlock, canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = canvas_size
    source_words = [word for word in (block.source_word_styles or []) if isinstance(word, dict)]
    boxes = [
        box
        for word in source_words
        if (box := v5_word_box(word)) is not None
        and not is_v5_isolated_step_marker_word(word, source_words)
        and not is_decorative_or_numeric_only(str(word.get("text") or ""))
    ]
    if not boxes:
        boxes = [
            box
            for word in source_words
            if (box := v5_word_box(word)) is not None
            and not is_v5_isolated_step_marker_word(word, source_words)
        ]
    boxes = [box for box in boxes if box[2] > box[0] and box[3] > box[1]]
    if not boxes:
        return block.bbox
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )


def expand_v5_single_line_render_box(
    draw: ImageDraw.ImageDraw,
    block: TextBlock,
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if any(isinstance(word, dict) and word.get("hasTextBackground") for word in (block.source_word_styles or [])):
        return box
    if len([line for line in (block.line_texts or []) if str(line).strip()]) != 1:
        return box
    if len(block.translated_style_spans or []) != 1:
        return box
    max_width = max(1, canvas_size[0] - 12)
    source_width = max(1, box[2] - box[0])
    source_heights = v62_source_word_height_map(block)
    fallback_size = max(1, int(block.font_size_estimate or default_typography_style(block).get("fontSize", 16)))
    tokens: list[dict[str, Any]] = []
    for span in block.translated_style_spans:
        tokens.extend(tokenize_style_span(span))
    if not tokens:
        return box
    line_font_size = max(v62_token_source_height(token, source_heights, fallback_size) for token in tokens)
    line_width, _, _, _ = v62_measure_tokens_with_line_font(draw, tokens, line_font_size)
    # Single-line translated copy should keep the original source font height
    # whenever there is enough horizontal room. Short source words such as
    # "Care" must not force a longer localized phrase like "Bakim yap" to shrink.
    needed_width = min(max_width, max(source_width, line_width))
    if needed_width <= source_width:
        return box
    center_x = (box[0] + box[2]) / 2.0
    left = int(round(center_x - needed_width / 2.0))
    right = left + int(round(needed_width))
    if left < 0:
        right -= left
        left = 0
    if right > canvas_size[0]:
        left -= right - canvas_size[0]
        right = canvas_size[0]
    return (max(0, left), box[1], min(canvas_size[0], right), box[3])


def expand_v5_render_box_for_line_fit(
    draw: ImageDraw.ImageDraw,
    block: TextBlock,
    box: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    if any(isinstance(word, dict) and word.get("hasTextBackground") for word in (block.source_word_styles or [])):
        return box
    source_heights = v62_source_word_height_map(block)
    fallback_size = max(1, int(block.font_size_estimate or default_typography_style(block).get("fontSize", 16)))
    lines: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for span in block.translated_style_spans or []:
        span_tokens = tokenize_style_span(span)
        if span_tokens:
            current.extend(span_tokens)
        if span.get("forceBreakAfter") and current:
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    if not lines:
        return box
    source_width = max(1, box[2] - box[0])
    single_source_line = len([line for line in (block.line_texts or []) if str(line).strip()]) == 1 and len(lines) == 1
    if single_source_line:
        max_expanded_width = max(1, canvas_size[0] - 12)
    else:
        max_expanded_width = int(round(source_width * float(os.getenv("ADAPTIFAI_V5_RENDER_BOX_MAX_EXPAND", "1.18"))))
    needed_width = source_width
    for line in lines:
        line_font_size = max(v62_token_source_height(token, source_heights, fallback_size) for token in line)
        line_width, _, _, _ = v62_measure_tokens_with_line_font(draw, line, line_font_size)
        needed_width = max(needed_width, min(line_width, max_expanded_width))
    if needed_width <= source_width:
        return box
    left, top, right, bottom = box
    if block.align == "left":
        if single_source_line:
            center_x = (left + right) / 2.0
            left = int(round(center_x - needed_width / 2.0))
            right = left + int(round(needed_width))
            if left < 0:
                right -= left
                left = 0
            if right > canvas_size[0]:
                left -= right - canvas_size[0]
                right = canvas_size[0]
        else:
            right = min(canvas_size[0], left + int(round(needed_width)))
    else:
        center_x = (left + right) / 2.0
        left = int(round(center_x - needed_width / 2.0))
        right = left + int(round(needed_width))
        if left < 0:
            right -= left
            left = 0
        if right > canvas_size[0]:
            left -= right - canvas_size[0]
            right = canvas_size[0]
    return (max(0, left), top, min(canvas_size[0], right), bottom)


def fill_color_around_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    exclude_color: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    left, top, right, bottom = box
    width, height = image.size
    pad = max(6, int(max(1, bottom - top) * 0.75))
    arr = np.array(image.convert("RGB"))
    samples: list[np.ndarray] = []
    if top - pad >= 0:
        samples.append(arr[max(0, top - pad):top, max(0, left):min(width, right)].reshape(-1, 3))
    if bottom + pad <= height:
        samples.append(arr[bottom:min(height, bottom + pad), max(0, left):min(width, right)].reshape(-1, 3))
    if left - pad >= 0:
        samples.append(arr[max(0, top):min(height, bottom), max(0, left - pad):left].reshape(-1, 3))
    if right + pad <= width:
        samples.append(arr[max(0, top):min(height, bottom), right:min(width, right + pad)].reshape(-1, 3))
    samples = [sample for sample in samples if sample.size]
    if not samples:
        return (255, 255, 255)
    merged = np.concatenate(samples, axis=0)
    if exclude_color is not None and len(merged) > 8:
        exclude = np.array(exclude_color, dtype=np.float32)
        distances = np.linalg.norm(merged.astype(np.float32) - exclude, axis=1)
        filtered = merged[distances > 32.0]
        if len(filtered) >= max(8, int(len(merged) * 0.18)):
            merged = filtered
    return tuple(int(value) for value in np.median(merged, axis=0))


def clear_v5_source_background_lines(canvas: Image.Image, block: TextBlock) -> None:
    words = [
        word
        for word in (block.source_word_styles or [])
        if isinstance(word, dict) and (word.get("estimatedBbox") or word.get("bbox"))
    ]
    if not words:
        return
    by_line: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        by_line.setdefault(int(word.get("lineIndex") or 0), []).append(word)
    draw = ImageDraw.Draw(canvas)
    for line_words in by_line.values():
        boxes: list[tuple[int, int, int, int]] = []
        for word in line_words:
            box = word.get("estimatedBbox") or word.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        if not boxes:
            continue
        has_background = any(bool(word.get("hasTextBackground") and word.get("backgroundColor")) for word in line_words)
        if has_background:
            bg_values = [
                parse_hex_color(str(word.get("backgroundColor") or "#ffffff"), fallback=(255, 255, 255))
                for word in line_words
                if word.get("hasTextBackground") and word.get("backgroundColor")
            ]
            bg_rgb = tuple(int(np.median([color[index] for color in bg_values])) for index in range(3)) if bg_values else (255, 255, 255)
            line_height = max(1, max(box[3] for box in boxes) - min(box[1] for box in boxes))
            pad_x = max(8, int(round(line_height * 0.55)))
            pad_y = max(5, int(round(line_height * 0.45)))
            left = max(0, min(min(box[0] for box in boxes), block.bbox[0]) - pad_x)
            top = max(0, min(min(box[1] for box in boxes), block.bbox[1]) - pad_y)
            right = min(canvas.width, max(max(box[2] for box in boxes), block.bbox[2]) + pad_x)
            bottom = min(canvas.height, max(max(box[3] for box in boxes), block.bbox[3]) + pad_y)
            if right > left and bottom > top:
                draw.rectangle(
                    (left, top, right, bottom),
                    fill=fill_color_around_box(canvas, (left, top, right, bottom), exclude_color=bg_rgb),
                )
            continue
        line_height = max(1, max(box[3] for box in boxes) - min(box[1] for box in boxes))
        pad_x = max(4, int(round(line_height * 0.35)))
        pad_y = max(7, int(round(line_height * 0.50)))
        left_edge = min(min(box[0] for box in boxes), block.bbox[0]) if has_background else min(box[0] for box in boxes)
        right_edge = max(max(box[2] for box in boxes), block.bbox[2]) if has_background else max(box[2] for box in boxes)
        left = max(0, left_edge - pad_x)
        top = max(0, min(box[1] for box in boxes) - pad_y)
        right = min(canvas.width, right_edge + pad_x)
        bottom = min(canvas.height, max(box[3] for box in boxes) + pad_y)
        if right <= left or bottom <= top:
            continue
        bg_colors = []
        for word in line_words:
            if word.get("hasTextBackground") and word.get("backgroundColor"):
                bg_colors.append(parse_hex_color(str(word.get("backgroundColor") or "#ffffff"), fallback=(255, 255, 255)))
            elif word.get("color"):
                bg_colors.append(parse_hex_color(str(word.get("color") or "#111111"), fallback=(17, 17, 17)))
        exclude_color = tuple(int(np.median([color[index] for color in bg_colors])) for index in range(3)) if bg_colors else None
        draw.rectangle(
            (left, top, right, bottom),
            fill=fill_color_around_box(canvas, (left, top, right, bottom), exclude_color=exclude_color),
        )


def restore_source_thin_horizontal_rules(canvas: Image.Image, source: Image.Image) -> None:
    source_rgb = source.convert("RGB")
    if source_rgb.size != canvas.size:
        source_rgb = source_rgb.resize(canvas.size, Image.Resampling.LANCZOS)
    source_np = np.array(source_rgb, dtype=np.uint8)
    if source_np.size == 0:
        return

    channel_max = source_np.max(axis=2).astype(np.int16)
    channel_min = source_np.min(axis=2).astype(np.int16)
    chroma = channel_max - channel_min
    green_teal = (
        (source_np[:, :, 1] > 110)
        & (source_np[:, :, 0] < 120)
        & (source_np[:, :, 1].astype(np.int16) >= source_np[:, :, 0].astype(np.int16) + 24)
        & (chroma > 38)
    )
    min_run = max(80, int(source_rgb.width * 0.42))
    candidate_rows: list[int] = []
    row_ranges: dict[int, tuple[int, int]] = {}
    for y in range(source_rgb.height):
        xs = np.where(green_teal[y])[0]
        if len(xs) < min_run:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        if x2 - x1 >= min_run and len(xs) >= int((x2 - x1) * 0.82):
            candidate_rows.append(y)
            row_ranges[y] = (x1, x2)

    groups: list[list[int]] = []
    for y in candidate_rows:
        if not groups or y != groups[-1][-1] + 1:
            groups.append([y])
        else:
            groups[-1].append(y)

    for group in groups:
        if len(group) > 4:
            continue
        x1 = max(0, min(row_ranges[y][0] for y in group) - 1)
        x2 = min(source_rgb.width, max(row_ranges[y][1] for y in group) + 1)
        y1 = max(0, min(group) - 1)
        y2 = min(source_rgb.height, max(group) + 2)
        if x2 <= x1 or y2 <= y1:
            continue
        canvas.alpha_composite(source_rgb.crop((x1, y1, x2, y2)).convert("RGBA"), (x1, y1))


def restore_v5_numeric_bypass_from_source(canvas: Image.Image, source: Image.Image, blocks: list[TextBlock]) -> None:
    source_rgba = source.convert("RGBA")
    if source_rgba.size != canvas.size:
        source_rgba = source_rgba.resize(canvas.size, Image.Resampling.LANCZOS)
    for block in blocks:
        if block.render_strategy != "v5_numeric_bypass":
            continue
        restore_box = expand_bbox(block.bbox, canvas.size, max(2, int((block.bbox[3] - block.bbox[1]) * 0.08)))
        if restore_box[2] <= restore_box[0] or restore_box[3] <= restore_box[1]:
            continue
        canvas.alpha_composite(source_rgba.crop(restore_box), restore_box[:2])


def normalize_v5_peer_row_font_sizes(blocks: list[TextBlock]) -> None:
    def block_source_font_max(block: TextBlock) -> int:
        sizes: list[int] = []
        for word in block.source_word_styles or []:
            if not isinstance(word, dict):
                continue
            raw_size = word.get("fontSize")
            if raw_size:
                sizes.append(int(raw_size))
                continue
            raw_box = word.get("bbox")
            if isinstance(raw_box, list) and len(raw_box) >= 4:
                sizes.append(max(1, int(raw_box[3]) - int(raw_box[1])))
        return max(sizes or [max(1, int(block.font_size_estimate or 16))])

    candidates = [
        block
        for block in blocks
        if is_v5_block(block)
        and block.render_strategy != "v5_numeric_bypass"
        and block_has_rich_text_segments(block)
        and not any(isinstance(word, dict) and word.get("hasTextBackground") for word in (block.source_word_styles or []))
    ]
    groups: list[list[TextBlock]] = []
    for block in sorted(candidates, key=lambda item: (item.bbox[1] + item.bbox[3]) / 2.0):
        top, bottom = int(block.bbox[1]), int(block.bbox[3])
        height = max(1, bottom - top)
        placed = False
        for group in groups:
            group_top = min(int(item.bbox[1]) for item in group)
            group_bottom = max(int(item.bbox[3]) for item in group)
            overlap = max(0, min(bottom, group_bottom) - max(top, group_top))
            if overlap / float(height) >= 0.45:
                group.append(block)
                placed = True
                break
        if not placed:
            groups.append([block])

    for group in groups:
        if len(group) < 2:
            continue
        row_font_size = max(block_source_font_max(block) for block in group)
        row_line_height = max(row_font_size + 2, int(round(row_font_size * 1.12)))
        for block in group:
            block.font_size_estimate = max(int(block.font_size_estimate or 0), row_font_size)
            block.line_height_estimate = max(int(block.line_height_estimate or 0), row_line_height)
            for word in block.source_word_styles or []:
                if not isinstance(word, dict):
                    continue
                word["fontSize"] = row_font_size
                word["peerRowFontSize"] = row_font_size
                word["lineHeight"] = row_line_height

            for span in block.translated_style_spans or []:
                if not isinstance(span, dict):
                    continue
                style = span.get("style")
                if isinstance(style, dict):
                    style["fontSize"] = row_font_size
                    style["lineHeight"] = row_line_height
                    for source_style in span.get("sourceWordStyles") or []:
                        if isinstance(source_style, dict):
                            source_style["fontSize"] = row_font_size
                            source_style["lineHeight"] = row_line_height


def draw_resize_display_copy_stack(draw: ImageDraw.ImageDraw, block: TextBlock) -> None:
    box = tuple(int(value) for value in block.bbox)
    max_width = max(1, box[2] - box[0])
    max_height = max(1, box[3] - box[1])
    spans = [span for span in (block.translated_style_spans or []) if isinstance(span, dict)]
    if not spans:
        return

    orphan_tokens = {
        "&",
        "+",
        "/",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "a",
        "an",
        "the",
        "ve",
        "veya",
        "ile",
        "için",
        "icin",
        "de",
        "da",
    }

    def is_orphan_token(text: str) -> bool:
        clean = text.strip().strip(".,;:!?()[]{}").lower()
        return clean in orphan_tokens or len(clean) <= 1

    def token_width(token: dict[str, Any]) -> float:
        return float(token["width"])

    def line_width(tokens: list[dict[str, Any]]) -> float:
        if not tokens:
            return 0.0
        return sum(token_width(token) for token in tokens) + sum(float(token["space"]) for token in tokens[:-1])

    def repair_orphan_lines(lines: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        repaired = [list(line) for line in lines if line]
        index = 1
        while index < len(repaired):
            while repaired[index] and is_orphan_token(str(repaired[index][0]["text"])) and repaired[index - 1]:
                moved = repaired[index - 1].pop()
                repaired[index].insert(0, moved)
                if not repaired[index - 1]:
                    repaired.pop(index - 1)
                    index = max(1, index - 1)
                    break
                if line_width(repaired[index]) <= max_width:
                    break
            index += 1
        return [line for line in repaired if line]

    def build_lines(font_size: int) -> tuple[list[list[dict[str, Any]]], int]:
        lines: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_width = 0.0
        line_height = max(font_size + 2, int(round(font_size * 1.22)))

        for span in spans:
            text = str(span.get("translatedText") or span.get("sourceText") or "").strip()
            if not text:
                continue
            style = dict(span.get("style") or {})
            style["fontSize"] = font_size
            font_weight = int(style.get("fontWeight") or 700)
            font = get_font(
                font_size,
                bold=font_weight >= 700,
                category=str(style.get("fontCategory") or "sans-serif"),
                weight=font_weight,
            )
            fill = style.get("color") or "#111111"
            bg = style.get("backgroundColor") if style.get("hasTextBackground") else None
            for word in [part for part in text.split() if part]:
                try:
                    word_width = draw.textlength(word, font=font)
                    space_width = draw.textlength(" ", font=font)
                except AttributeError:
                    word_width = text_width(draw, word, font)
                    space_width = text_width(draw, " ", font)
                addition = word_width if not current else word_width + space_width
                if current and current_width + addition > max_width:
                    lines.append(current)
                    current = []
                    current_width = 0.0
                    addition = word_width
                current.append(
                    {
                        "text": word,
                        "font": font,
                        "fill": fill,
                        "background": bg,
                        "width": word_width,
                        "space": space_width,
                        "fontSize": font_size,
                    }
                )
                current_width += addition
            if span.get("forceBreakAfter") and current:
                lines.append(current)
                current = []
                current_width = 0.0
        if current:
            lines.append(current)
        return repair_orphan_lines(lines), line_height

    preferred_size = max(8, int(block.font_size_estimate or 16))
    font_size = preferred_size
    lines: list[list[dict[str, Any]]] = []
    line_height = max(font_size + 2, int(round(font_size * 1.22)))
    max_candidate = max(8, min(72, int(getattr(block, "resize_max_font_size", 42) or 42)))
    min_candidate = max(6, int(getattr(block, "resize_min_font_size", 8) or 8))
    target_fill = float(getattr(block, "resize_target_fill", 0.72) or 0.72)
    preferred_line_count = max(
        1,
        int(getattr(block, "resize_preferred_line_count", 0) or 0),
        len([line for line in (block.line_texts or []) if str(line).strip()]),
    )
    readable_min = max(
        min_candidate,
        min(
            max_candidate,
            int(
                getattr(
                    block,
                    "resize_min_readable_font_size",
                    max(10, round(min(max_width, max_height) * 0.035)),
                )
                or 10
            ),
        ),
    )
    best_fit: tuple[int, list[list[dict[str, Any]]], int, float] | None = None

    # First priority: preserve the original line rhythm while the copy remains readable.
    for candidate in range(max_candidate, readable_min - 1, -1):
        candidate_lines, candidate_line_height = build_lines(candidate)
        if (
            candidate_lines
            and len(candidate_lines) == preferred_line_count
            and len(candidate_lines) * candidate_line_height <= max_height
        ):
            widest = max((line_width(line) for line in candidate_lines), default=0.0)
            best_fit = (candidate, candidate_lines, candidate_line_height, widest / max(1, max_width))
            break

    # Second priority: avoid adding extra line breaks if exact preservation is impossible.
    if best_fit is None:
        for candidate in range(max_candidate, readable_min - 1, -1):
            candidate_lines, candidate_line_height = build_lines(candidate)
            if (
                candidate_lines
                and len(candidate_lines) <= preferred_line_count
                and len(candidate_lines) * candidate_line_height <= max_height
            ):
                widest = max((line_width(line) for line in candidate_lines), default=0.0)
                best_fit = (candidate, candidate_lines, candidate_line_height, widest / max(1, max_width))
                break

    # Final fallback: if preserving line count makes the copy unreadable, allow more lines
    # before shrinking below the readable threshold.
    for candidate in range(max_candidate, min_candidate - 1, -1):
        if best_fit is not None:
            break
        candidate_lines, candidate_line_height = build_lines(candidate)
        if candidate_lines and len(candidate_lines) * candidate_line_height <= max_height:
            widest = max((line_width(line) for line in candidate_lines), default=0.0)
            fill_ratio = widest / max(1, max_width)
            if best_fit is None:
                best_fit = (candidate, candidate_lines, candidate_line_height, fill_ratio)
            if fill_ratio >= target_fill:
                best_fit = (candidate, candidate_lines, candidate_line_height, fill_ratio)
                break
    if best_fit is not None:
        font_size, lines, line_height, _ = best_fit
    if not lines:
        lines, line_height = build_lines(font_size)

    y = box[1]
    for line in lines:
        x = box[0]
        line_width = int(round(sum(float(token["width"]) for token in line) + sum(float(token["space"]) for token in line[:-1])))
        ribbon_background = next((token.get("background") for token in line if token.get("background")), None)
        if ribbon_background:
            text_boxes = []
            probe_x = box[0]
            for token_index, token in enumerate(line):
                if token_index > 0:
                    probe_x += int(round(float(token["space"])))
                text_boxes.append(draw.textbbox((probe_x, y), str(token["text"]), font=token["font"]))
                probe_x += int(round(float(token["width"])))
            if text_boxes:
                ribbon_top = min(item[1] for item in text_boxes)
                ribbon_bottom = max(item[3] for item in text_boxes)
            else:
                ribbon_top = y
                ribbon_bottom = y + line_height
            pad_y = max(1, min(8, int(round(font_size * 0.15))))
            ribbon_left = box[0]
            ribbon_right = min(box[2], box[0] + line_width)
            draw.rectangle(
                (
                    max(0, ribbon_left),
                    max(0, ribbon_top - pad_y),
                    min(box[2], ribbon_right),
                    max(ribbon_top + 1, ribbon_bottom + pad_y),
                ),
                fill=ribbon_background,
            )
        for token_index, token in enumerate(line):
            if token_index > 0:
                x += int(round(float(token["space"])))
            text = str(token["text"])
            font = token["font"]
            left, top, right, bottom = draw.textbbox((x, y), text, font=font)
            draw.text((x, y), text, fill=token["fill"], font=font)
            x += int(round(float(token["width"])))
        y += line_height


def draw_v5_numeric_bypass_fallback(draw: ImageDraw.ImageDraw, block: TextBlock) -> None:
    box = tuple(int(value) for value in block.bbox)
    if box[2] <= box[0] or box[3] <= box[1]:
        return
    text = (block.text or block.translated_text or "").strip()
    if not text:
        return
    font_size = max(8, min(220, int(block.font_size_estimate or (box[3] - box[1]) * 0.86)))
    font = get_font(font_size, bold=block.font_weight >= 700)
    fill = block.color or "#111111"
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width_px = text_box[2] - text_box[0]
    text_height_px = text_box[3] - text_box[1]
    x = box[0] if block.align == "left" else box[0] + max(0, (box[2] - box[0] - text_width_px) // 2)
    y = box[1] + max(0, (box[3] - box[1] - text_height_px) // 2) - text_box[1]
    draw.text((x, y), text, fill=fill, font=font)


def draw_fitted_localize_v2_text(base: Image.Image, blocks: list[TextBlock], *, numeric_bypass_restored: bool = False) -> Image.Image:
    canvas = base.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    normalize_v5_peer_row_font_sizes(blocks)
    
    # Global Y de?i?keni yerine ?izilmi? bloklar?n listesi (?oklu kolon deste?i)
    rendered_boxes: list[tuple[int, int, int, int]] = []
    
    for block in blocks:
        if block.render_strategy == "v5_numeric_bypass":
            if not numeric_bypass_restored:
                draw_v5_numeric_bypass_fallback(draw, block)
            continue
        if block.render_strategy == "resize_display_copy_stack":
            draw_resize_display_copy_stack(draw, block)
            continue
            
        text = block.translated_text or block.text
        # NameError riskine kar?? inline V5 kontrol?
        is_v5 = str(block.id or "").startswith("v5")
        
        if block_has_rich_text_segments(block):
            box = v5_strict_render_box(block, canvas.size) if is_v5 else block.bbox
            if is_v5:
                box = expand_v5_single_line_render_box(draw, block, box, canvas.size)
                box = expand_v5_render_box_for_line_fit(draw, block, box, canvas.size)
                if block.render_strategy != "resize_compositor_redraw":
                    clear_v5_source_background_lines(canvas, block)
            
            # KURAL: Sadece X ekseninde (yatayda) kesi?en bloklar birbirini Y ekseninde a?a?? iter
            start_y = box[1]
            for rx1, ry1, rx2, ry2 in rendered_boxes:
                if not (box[2] < rx1 or box[0] > rx2):  # X ekseninde ?rt??me var m??
                    if start_y < ry2 and box[3] > ry1:  # Dikeyde de ezme varsa
                        start_y = max(start_y, ry2)
            
            pad_x = 0 if is_v5 else max(2, int((box[2] - box[0]) * 0.02))
            pad_y = 0 if is_v5 else max(1, int((box[3] - box[1]) * 0.02))
            
            render_box = (box[0] + pad_x, start_y + pad_y, box[2] - pad_x, max(box[3], start_y + 10) - pad_y)
            base_typography = default_typography_style(block)
            
            if is_v5:
                render_spans = block.translated_style_spans
                layout = fit_v62_geometric_typesetting(draw, block, render_spans, render_box)
            else:
                layout = fit_styled_spans_strict(
                    draw, block.translated_style_spans, render_box, base_typography, honor_force_breaks=True, allow_vertical_overflow=False,
                )
            if is_v5 and layout.get("verticalOverflowAllowed"):
                total_height = int(layout.get("totalHeight", 0))
                bottom_limit = canvas.height - 2
                if render_box[1] + total_height > bottom_limit:
                    shift_up = min(render_box[1], render_box[1] + total_height - bottom_limit)
                    if shift_up > 0:
                        render_box = (
                            render_box[0],
                            render_box[1] - shift_up,
                            render_box[2],
                            max(render_box[3] - shift_up, render_box[1] - shift_up + 1),
                        )
                
            # precise_inline=True parametresi korundu
            _, span_render_boxes = render_styled_spans(draw, layout, render_box, alignment=block.align, precise_inline=True)
            
            if span_render_boxes:
                final_y = max(b["bbox"][3] for b in span_render_boxes)
                rendered_boxes.append((box[0], start_y, box[2], final_y))
            continue
            
        # D?z metin (Plain-text) bloklar i?in fallback
        translated_lines = [line.strip() for line in text.splitlines() if line.strip()]
        source_line_boxes = list(block.line_boxes or [])
        if source_line_boxes and translated_lines and len(translated_lines) <= len(source_line_boxes):
            for line_text, line_box in zip(translated_lines, source_line_boxes):
                # Fallback margin hatas? d?zeltildi
                line_pad_x = 0 if is_v5 else max(2, int((line_box[2] - line_box[0]) * 0.02))
                line_pad_y = 0 if is_v5 else max(1, int((line_box[3] - line_box[1]) * 0.04))
                render_box = (
                    line_box[0] + line_pad_x,
                    line_box[1] + line_pad_y,
                    line_box[2] - line_pad_x,
                    line_box[3] - line_pad_y,
                )
                preferred_size = max(8, min(120, int((render_box[3] - render_box[1]) * 0.72)))
                text_lines, size, line_height = fit_plain_text_for_box(
                    draw, line_text, render_box, bold=block.font_weight >= 700, preferred_size=preferred_size,
                )
                y = render_box[1] + max(0, (render_box[3] - render_box[1] - len(text_lines) * line_height) // 2)
                for fitted_line in text_lines:
                    font = get_font(size, bold=block.font_weight >= 700)
                    bbox = draw.textbbox((0, 0), fitted_line, font=font)
                    text_width_px = bbox[2] - bbox[0]
                    x = render_box[0] if block.align == "left" else render_box[0] + max(0, (render_box[2] - render_box[0] - text_width_px) // 2)
                    draw.text((x, y), fitted_line, fill=block.color or "#111111", font=font)
                    y += line_height
                rendered_boxes.append((line_box[0], line_box[1], line_box[2], y))
        else:
            box = block.bbox
            # Fallback margin hatas? d?zeltildi
            pad_x = 0 if is_v5 else max(2, int((box[2] - box[0]) * 0.02))
            pad_y = 0 if is_v5 else max(1, int((box[3] - box[1]) * 0.02))
            
            if block.align == "left" and box[0] > canvas.width * 0.52 and not is_v5:
                pad_x = max(pad_x, int((box[2] - box[0]) * 0.12))
            
            # D?z metin fallback i?in de Y ekseninde ?arp??ma kontrol?
            start_y = box[1]
            for rx1, ry1, rx2, ry2 in rendered_boxes:
                if not (box[2] < rx1 or box[0] > rx2):
                    if start_y < ry2 and box[3] > ry1:
                        start_y = max(start_y, ry2)
                        
            render_box = (box[0] + pad_x, start_y + pad_y, box[2] - pad_x, max(box[3], start_y + 10) - pad_y)
            
            preferred_lines = max(1, len([line for line in block.text.splitlines() if line.strip()]))
            preferred_size = max(8, min(96, int((box[3] - box[1]) / max(1, preferred_lines) * 0.72)))
            if len((block.translated_text or block.text).split()) >= 5:
                preferred_size = min(preferred_size, max(10, int(canvas.height * 0.034)))
            text_lines, size, line_height = fit_plain_text_for_box(
                draw, text, render_box, bold=block.font_weight >= 700, preferred_size=preferred_size,
            )
            y = render_box[1] + max(0, (render_box[3] - render_box[1] - len(text_lines) * line_height) // 2)
            for line_text in text_lines:
                font = get_font(size, bold=block.font_weight >= 700)
                bbox = draw.textbbox((0, 0), line_text, font=font)
                text_width_px = bbox[2] - bbox[0]
                x = render_box[0] if block.align == "left" else render_box[0] + max(0, (render_box[2] - render_box[0] - text_width_px) // 2)
                draw.text((x, y), line_text, fill=block.color or "#111111", font=font)
                y += line_height
            rendered_boxes.append((box[0], start_y, box[2], y))
                
    return canvas.convert("RGB")


def build_localize_assets_v2(paths: list[Path], languages: list[str], output_format: str, job_dir: Path) -> tuple[list[OutputAsset], dict[str, list[str]], list[dict[str, Any]]]:
    outputs: list[OutputAsset] = []
    translations_summary: dict[str, list[str]] = {}
    manifest_assets: list[dict[str, Any]] = []
    for image_path in image_paths(paths):
        source_image = Image.open(image_path).convert("RGB")
        vision_words = google_vision_word_blocks(image_path, source_image.size)
        vision_words, supplemental_ocr_meta = supplement_v5_ocr_words(image_path, source_image.size, vision_words)
        if not vision_words:
            raise RuntimeError("V5 polygon localize requires Google Cloud Vision boundingPoly OCR, but no words were returned.")
        ocr_layout_blocks, protected_ocr_lines, v5_meta = group_v5_polygon_words(vision_words, source_image)
        v5_meta["supplementalOcr"] = supplemental_ocr_meta
        for language in languages:
            payload = analyze_localize_v212_ocr_translations(ocr_layout_blocks, language)
            blocks = [
                block
                for block in apply_v212_translations(ocr_layout_blocks, payload)
                if block.translate or block.render_strategy == "v5_numeric_bypass"
            ]
            translations_summary[language] = [block.translated_text or block.text for block in blocks]
            mask, mask_meta = build_strict_text_removal_mask(source_image, blocks, protected_ocr_lines)
            cleaned, cleanup_meta = inpaint_localize_v2_base(source_image, mask, blocks) if blocks else (source_image.copy(), {"provider": "none"})
            render_base = cleaned.convert("RGBA")
            for block in blocks:
                if str(block.id or "").startswith("v5"):
                    clear_v5_source_background_lines(render_base, block)
            restore_source_thin_horizontal_rules(render_base, source_image)
            restore_v5_numeric_bypass_from_source(render_base, source_image, blocks)
            rendered = draw_fitted_localize_v2_text(render_base.convert("RGB"), blocks, numeric_bypass_restored=True)
            filename = localize_filename(image_path, language, output_format)
            output_path = job_dir / filename
            save_image_output(rendered, output_path, normalize_output_format(output_format, image_path))

            mask_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-v2-mask.png"
            clean_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-v2-clean.png"
            analysis_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-v2-analysis.json"
            mask.save(job_dir / mask_filename, "PNG")
            render_base.convert("RGB").save(job_dir / clean_filename, "PNG")
            (job_dir / analysis_filename).write_text(
                json.dumps(
                    {
                        **payload,
                        "pipeline": "v6-polygon-vision-layout",
                        "pipeline_version": "v6.5-calibrated-rich-v5-render",
                        "v5VisionLayout": v5_meta,
                        "ocrLayoutBlocks": [block.model_dump(mode="json") for block in ocr_layout_blocks],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            asset = OutputAsset(
                filename=filename,
                width=rendered.width,
                height=rendered.height,
                safe_zone_warnings=[],
                download_url=f"/api/download/{job_dir.name}/{filename}",
                source_name=image_path.name,
                language=language,
                source_language=str(payload.get("source_language", "")) or None,
                translated_text="\n\n".join(block.translated_text or "" for block in blocks if block.translated_text),
                extracted_blocks=blocks,
                debug={
                    "pipeline": "v5",
                    "pipeline_version": "v6.5-calibrated-rich-v5-render",
                    "analysisProvider": payload.get("analysis_provider"),
                    "cleanupMeta": cleanup_meta,
                    "strictMasking": mask_meta,
                    "layoutEngine": "v6.5-calibrated-rich-v5-render",
                    "v5VisionLayout": v5_meta,
                    "artifacts": {
                        "analysis": f"/api/download/{job_dir.name}/{analysis_filename}",
                        "mask": f"/api/download/{job_dir.name}/{mask_filename}",
                        "cleanBase": f"/api/download/{job_dir.name}/{clean_filename}",
                    },
                },
            )
            outputs.append(asset)
            manifest_assets.append(
                {
                    **asset.model_dump(mode="json"),
                    "mode": "localize",
                    "pipeline": "v6-polygon-vision-layout",
                    "pipeline_version": "v6.5-calibrated-rich-v5-render",
                    "analysisProvider": payload.get("analysis_provider"),
                    "cleanupMeta": cleanup_meta,
                    "strictMasking": mask_meta,
                    "debug": {
                        "analysis": f"/api/download/{job_dir.name}/{analysis_filename}",
                        "mask": f"/api/download/{job_dir.name}/{mask_filename}",
                        "cleanBase": f"/api/download/{job_dir.name}/{clean_filename}",
                    },
                }
            )
    return outputs, translations_summary, manifest_assets


def build_localize_assets(paths: list[Path], languages: list[str], output_format: str, job_dir: Path) -> tuple[list[OutputAsset], dict[str, list[str]], list[dict[str, Any]]]:
    if localization_v2_enabled():
        return build_localize_assets_v2(paths, languages, output_format, job_dir)

    outputs: list[OutputAsset] = []
    translations_summary: dict[str, list[str]] = {}
    manifest_assets: list[dict[str, Any]] = []

    for image_path in image_paths(paths):
        source_image = Image.open(image_path).convert("RGB")
        preprocessed_image = preprocess_image_for_localize(source_image)
        temp_input_path = job_dir / f"{sanitize_stem(image_path)}-{source_suffix(image_path)}-preprocessed.png"
        preprocessed_image.save(temp_input_path, "PNG")
        blocks = build_localize_blocks(temp_input_path, preprocessed_image)
        refine_overlay_text_colors(source_image, blocks)
        raw_ocr_blocks = list(blocks)
        source_language = detect_source_language(blocks)
        try:
            translated_by_language = translate_with_gpt4o(blocks, languages)
        except Exception as exc:
            print(f"[translation] OpenAI translation failed, trying Gemini fallback: {exc}", flush=True)
            translated_by_language = translate_with_gemini(blocks, languages)
        for language, translated_strings in translated_by_language.items():
            translations_summary[language] = translated_strings
            translated_blocks, editor_text = apply_translations(blocks, translated_strings, language)
            grouping_audit = analyze_semantic_grouping(translated_blocks, source_image.size)
            foreground_bbox = detect_foreground_bbox(source_image)
            localize_foreground_bbox = foreground_bbox
            has_openai_edit_work = any(
                is_localize_marketing_edit_block(block, source_image.size, localize_foreground_bbox)
                for block in translated_blocks
            )
            has_complex_model_cleanup_work = False
            for block in translated_blocks:
                if not is_overlay_marketing_cleanup_block(block):
                    continue
                try:
                    region_probe = build_combined_block_cleanup_region(source_image, block)
                    if region_probe is not None and not bool(region_probe.get("solidContextCleanup")):
                        has_complex_model_cleanup_work = True
                        break
                except Exception:
                    has_complex_model_cleanup_work = True
                    break
            localize_renderer = os.getenv("ADAPTIFAI_LOCALIZE_RENDERER", "auto").strip().lower()
            openai_render_meta: dict[str, Any] | None = None
            rendered: Image.Image | None = None
            cleanup_debug: dict[str, Any] = {
                "blockLineGroups": [],
                "lineCleanupRegions": [],
                "lineMasks": [],
                "lineCleanupStrategies": [],
                "lineCleanupQualityScores": [],
                "foregroundOverlapScores": [],
                "cleanupWarnings": [],
                "generativeAttemptPreviews": [],
                "protectedRegionMaskImage": Image.new("L", source_image.size, 0),
                "maskPolarity": "model_direct_localize",
                "textCoverageEstimate": 0.0,
                "backgroundLeakageEstimate": 0.0,
                "maskQualityStatus": "delegated_to_image_editor",
                "maskFailureReason": "",
                "maskWhitePixelRatio": 0.0,
                "textStrokeMaskRawImage": None,
                "textStrokeMaskFilledImage": None,
                "textStrokeMaskDilatedImage": None,
                "textStrokeMaskFinalImage": None,
            }
            cleanup_gate = {
                "cleanupStatus": "passed",
                "blockCleanupStatus": [],
                "residualSourceOCR": [],
                "failedSourceWords": [],
                "blockLevelFallbackAttempted": False,
                "blockLevelFallbackSelected": False,
                "finalRenderSkippedReason": "",
            }
            gated_background = source_image.copy()

            should_use_model_localize = (
                localize_renderer in {"openai", "image", "model"}
                or (localize_renderer == "auto" and has_complex_model_cleanup_work)
            )
            if should_use_model_localize and has_openai_edit_work:
                try:
                    rendered, openai_render_meta = render_localize_with_openai_image_edit(
                        source_image,
                        translated_blocks,
                        language,
                        source_language,
                        output_format,
                    )
                    cleanup_debug["cleanupWarnings"].append(
                        {
                            "id": "model-direct-localize",
                            "lineIndex": -1,
                            "warning": "Used model image edit as primary localization renderer.",
                            "provider": openai_render_meta.get("provider"),
                            "model": openai_render_meta.get("model"),
                        }
                    )
                except Exception as exc:
                    openai_render_meta = {
                        "provider": "local",
                        "strategy": "model_direct_localize_failed",
                        "fallbackReason": str(exc),
                    }

            if rendered is None:
                background, cleanup_debug = build_clean_background(source_image, translated_blocks, cleanup_strength=100, return_debug=True)
                if localize_cleanup_gate_enabled():
                    gated_background, cleanup_gate = enforce_localize_cleanup_gate(source_image, background, translated_blocks, cleanup_debug)
                else:
                    gated_background = background.convert("RGB")
                    cleanup_gate = {
                        "cleanupStatus": "passed",
                        "blockCleanupStatus": [],
                        "residualSourceOCR": [],
                        "failedSourceWords": [],
                        "blockLevelFallbackAttempted": False,
                        "blockLevelFallbackSelected": False,
                        "finalRenderSkippedReason": "",
                    }
                    cleanup_debug.setdefault("cleanupWarnings", []).append(
                        {
                            "id": "localize-cleanup-gate",
                            "lineIndex": -1,
                            "warning": "Heavy residual cleanup gate disabled for CPU runtime",
                        }
                    )

            # â”€â”€â”€ Localization Protocol V2.2 â”€â”€â”€
            if _PROTOCOL_AVAILABLE:
                try:
                    protocol_foreground_bbox = detect_foreground_bbox(source_image)
                    protocol_protected_mask = cleanup_debug.get("protectedRegionMaskImage")
                    protocol_result = run_localization_protocol(
                        image=source_image,
                        blocks=translated_blocks,
                        protected_region_mask=protocol_protected_mask if isinstance(protocol_protected_mask, Image.Image) else None,
                        foreground_bbox=protocol_foreground_bbox,
                        ocr_confidence=1.0,
                        cleanup_fn=None,
                        ocr_fn=None,
                        job_dir=job_dir,
                    )
                    if protocol_result is not None and protocol_result.risk_report and protocol_result.risk_report.risk_level not in (
                        RiskLevel.REJECT_LOW_CONFIDENCE,
                        RiskLevel.UNSUPPORTED_AUTO_CLEANUP,
                        RiskLevel.PACKAGING_PROTECTION_RISK,
                    ):
                        protocol_result.cleaned_image = gated_background
                        seg_masks = protocol_result.segmentation_masks
                        if seg_masks is not None:
                            cleanup_mask_v2 = seg_masks.compute_cleanup_mask(source_image.size)
                            source_words = []
                            for blk in translated_blocks:
                                if blk.translate and blk.surface == "overlay":
                                    source_words.extend(w.strip() for w in blk.text.split() if len(w.strip()) > 2)
                            quality_gate_v2 = run_quality_gate(
                                source_image=source_image,
                                cleaned_image=gated_background,
                                blocks=translated_blocks,
                                segmentation_masks=seg_masks,
                                cleanup_mask=cleanup_mask_v2,
                                source_words=source_words,
                                ocr_fn=None,
                            )
                            protocol_result.quality_report = quality_gate_v2
                            try:
                                (job_dir / "creativeRiskReport.json").write_text(
                                    json.dumps(protocol_result.risk_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                                )
                                (job_dir / "qualityGateReport.json").write_text(
                                    json.dumps(quality_gate_v2.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                                )
                                if protocol_result.depth_report:
                                    (job_dir / "depthLayeringReport.json").write_text(
                                        json.dumps(protocol_result.depth_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                                    )
                                (job_dir / "pipelineProtocolReport.json").write_text(
                                    json.dumps(protocol_result.to_reports_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                                )
                            except Exception:
                                pass
                except Exception:
                    pass
            # â”€â”€â”€ End Protocol V2.2 â”€â”€â”€

            render_plan_entries = [
                decide_block_render_strategy(source_image, block, cleanup_debug, foreground_bbox)
                for block in translated_blocks
                if block.translate and text_changed(block.text, block.translated_text)
            ]
            render_plan_map = {entry["id"]: entry for entry in render_plan_entries if entry.get("id")}
            render_audit = build_render_audit_artifacts(source_image.size, translated_blocks, render_plan_map)
            translated_blocks = [
                block.model_copy(
                    update={
                        "cleanup_confidence": render_plan_map.get(block.id, {}).get("cleanupConfidence", 1.0),
                        "cleanup_strategy": render_plan_map.get(block.id, {}).get("cleanupStrategy", "line_level_mask_guided_cleanup"),
                        "render_strategy": render_plan_map.get(block.id, {}).get("renderStrategy", "clean_replace"),
                    }
                )
                if block.id in render_plan_map
                else block
                for block in translated_blocks
            ]
            render_base = gated_background if gated_background is not None else source_image
            if rendered is None:
                try:
                    rendered = render_translated_text(render_base, translated_blocks, render_plan=render_plan_map)
                except Exception:
                    rendered = None
                # Safety net: if render produced nothing useful (exception or all blocks
                # suppressed and output identical to base), retry without render_plan so
                # filter_render_blocks operates in raw mode without any suppression hints.
                if rendered is None or (
                    rendered is not None
                    and np.array_equal(np.array(rendered), np.array(render_base))
                ):
                    try:
                        rendered = render_translated_text(render_base, translated_blocks, render_plan=None)
                    except Exception:
                        rendered = render_base.copy()
            token_masks = collect_token_masks(source_image, translated_blocks)
            mask_preview = build_text_mask(source_image, translated_blocks)
            mask_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-mask.png"
            openai_edit_mask_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-openai-edit-mask.png"
            clean_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-clean.png"
            protected_mask_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-protected-mask.png"
            text_stroke_raw_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-text-stroke-raw.png"
            text_stroke_filled_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-text-stroke-filled.png"
            text_stroke_dilated_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-text-stroke-dilated.png"
            text_stroke_final_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-text-stroke-final.png"
            synthetic_glyph_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-synthetic-glyph-mask.png"
            adaptive_cluster_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-adaptive-cluster-mask.png"
            local_contrast_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-local-contrast-mask.png"
            edge_stroke_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-edge-stroke-mask.png"
            source_text_pixel_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-source-text-pixel-mask.png"
            combined_binary_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-combined-binary-mask.png"
            inpaint_binary_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-inpaint-mask-binary.png"
            composite_soft_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-composite-mask-soft.png"
            first_pass_lama_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-firstPassLamaOutput.png"
            residual_mask_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-residualTextMask.png"
            residual_glyph_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-residualGlyphMaskAfterFirstPass.png"
            residual_artifact_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-residualArtifactMaskExpanded.png"
            second_pass_lama_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-secondPassLamaOutput.png"
            final_candidate_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-finalCleanedCandidate.png"
            residual_ocr_first_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-residualOCRAfterFirstPass.json"
            residual_ocr_second_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-residualOCRAfterSecondPass.json"
            cleanup_cascade_log_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-cleanupCascadeLog.json"
            translated_preview_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-translated-render-preview.png"
            source_reference_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-source-style-reference.png"
            render_plan_filename = f"debug-{sanitize_stem(image_path)}-{language.lower()}-render-plan.json"
            mask_preview.save(job_dir / mask_filename, "PNG")
            build_openai_localize_edit_mask(source_image, translated_blocks, detect_foreground_bbox(source_image)).save(job_dir / openai_edit_mask_filename, "PNG")
            gated_background.save(job_dir / clean_filename, "PNG")
            render_audit["translatedPreviewImage"].save(job_dir / translated_preview_filename, "PNG")
            render_audit["sourceReferenceImage"].save(job_dir / source_reference_filename, "PNG")
            (job_dir / render_plan_filename).write_text(json.dumps(render_audit["renderPlan"], ensure_ascii=False, indent=2), encoding="utf-8")
            protected_mask_image = cleanup_debug.get("protectedRegionMaskImage")
            if isinstance(protected_mask_image, Image.Image):
                protected_mask_image.save(job_dir / protected_mask_filename, "PNG")
            text_stroke_mask_paths: dict[str, str] = {}
            for debug_key, filename_mask, payload_key in (
                ("syntheticGlyphMaskImage", synthetic_glyph_filename, "syntheticGlyphMask"),
                ("adaptiveColorClusterMaskImage", adaptive_cluster_filename, "adaptiveColorClusterMask"),
                ("localContrastMaskImage", local_contrast_filename, "localContrastMask"),
                ("edgeStrokeMaskImage", edge_stroke_filename, "edgeStrokeMask"),
                ("sourceTextPixelMaskImage", source_text_pixel_filename, "sourceTextPixelMask"),
                ("combinedBinaryMaskBeforeDilationImage", combined_binary_filename, "combinedBinaryMaskBeforeDilation"),
                ("textStrokeMaskRawImage", text_stroke_raw_filename, "raw"),
                ("textStrokeMaskFilledImage", text_stroke_filled_filename, "filled"),
                ("textStrokeMaskDilatedImage", text_stroke_dilated_filename, "dilated"),
                ("textStrokeMaskFinalImage", text_stroke_final_filename, "final"),
                ("inpaintMaskBinaryImage", inpaint_binary_filename, "inpaintMask_binary"),
                ("compositeMaskSoftImage", composite_soft_filename, "compositeMask_soft"),
            ):
                image_value = cleanup_debug.get(debug_key)
                if isinstance(image_value, Image.Image):
                    image_value.save(job_dir / filename_mask, "PNG")
                    text_stroke_mask_paths[payload_key] = f"/api/download/{job_dir.name}/{filename_mask}"
            first_pass_lama_output_path = None
            residual_text_mask_path = None
            residual_glyph_mask_path = None
            residual_artifact_mask_path = None
            second_pass_lama_output_path = None
            final_cleaned_candidate_path = None
            residual_ocr_after_first_pass_path = None
            residual_ocr_after_second_pass_path = None
            cleanup_cascade_log_path = None
            image_value = cleanup_debug.get("firstPassLamaOutputImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / first_pass_lama_filename, "PNG")
                first_pass_lama_output_path = f"/api/download/{job_dir.name}/{first_pass_lama_filename}"
            image_value = cleanup_debug.get("residualTextMaskImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / residual_mask_filename, "PNG")
                residual_text_mask_path = f"/api/download/{job_dir.name}/{residual_mask_filename}"
            image_value = cleanup_debug.get("residualGlyphMaskAfterFirstPassImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / residual_glyph_filename, "PNG")
                residual_glyph_mask_path = f"/api/download/{job_dir.name}/{residual_glyph_filename}"
            image_value = cleanup_debug.get("residualArtifactMaskExpandedImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / residual_artifact_filename, "PNG")
                residual_artifact_mask_path = f"/api/download/{job_dir.name}/{residual_artifact_filename}"
            image_value = cleanup_debug.get("secondPassLamaOutputImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / second_pass_lama_filename, "PNG")
                second_pass_lama_output_path = f"/api/download/{job_dir.name}/{second_pass_lama_filename}"
            image_value = cleanup_debug.get("finalCleanedCandidateImage")
            if isinstance(image_value, Image.Image):
                image_value.save(job_dir / final_candidate_filename, "PNG")
                final_cleaned_candidate_path = f"/api/download/{job_dir.name}/{final_candidate_filename}"
            if cleanup_debug.get("residualOCRAfterFirstPass") is not None:
                (job_dir / residual_ocr_first_filename).write_text(json.dumps(cleanup_debug.get("residualOCRAfterFirstPass", []), ensure_ascii=False, indent=2), encoding="utf-8")
                residual_ocr_after_first_pass_path = f"/api/download/{job_dir.name}/{residual_ocr_first_filename}"
            if cleanup_debug.get("residualOCRAfterSecondPass") is not None:
                (job_dir / residual_ocr_second_filename).write_text(json.dumps(cleanup_debug.get("residualOCRAfterSecondPass", []), ensure_ascii=False, indent=2), encoding="utf-8")
                residual_ocr_after_second_pass_path = f"/api/download/{job_dir.name}/{residual_ocr_second_filename}"
            if cleanup_debug.get("cleanupCascadeLog") is not None:
                (job_dir / cleanup_cascade_log_filename).write_text(json.dumps(cleanup_debug.get("cleanupCascadeLog", []), ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cascade_log_path = f"/api/download/{job_dir.name}/{cleanup_cascade_log_filename}"
            generative_attempt_previews_payload: list[dict[str, Any]] = []
            for preview in cleanup_debug.get("generativeAttemptPreviews", []):
                preview_entry = {
                    "id": preview.get("id"),
                    "lineIndex": preview.get("lineIndex"),
                    "name": preview.get("name"),
                    "protectedChangeBeforeComposite": preview.get("protectedChangeBeforeComposite"),
                    "protectedChangeAfterComposite": preview.get("protectedChangeAfterComposite"),
                }
                preview_prefix = f"debug-{sanitize_stem(image_path)}-{language.lower()}-{preview.get('id','block')}-{preview.get('lineIndex',0)}-{str(preview.get('name','candidate')).replace(':','-')}"
                for key, suffix in (
                    ("requestImagePreview", "request-image"),
                    ("requestMaskPreview", "request-mask"),
                    ("requestMaskAlphaPreview", "request-mask-alpha"),
                    ("originalCropPreview", "original-crop"),
                    ("openAIResultCropPreview", "openai-crop"),
                    ("apiSuccessResultPreview", "api-success-result"),
                    ("editableMaskPreview", "editable-mask"),
                    ("protectedSubtractedMaskPreview", "protected-subtracted-mask"),
                    ("postCompositePreview", "post-composite"),
                    ("diffPreview", "diff"),
                    ("lamaInputImagePreview", "lama-input-image"),
                    ("lamaMaskPreview", "lama-mask"),
                    ("lamaOutputPreview", "lama-output"),
                    ("controlImagePreview", "control-image"),
                    ("controlSourcePreview", "control-source"),
                    ("sdxlOutputPreview", "sdxl-output"),
                ):
                    image_value = preview.get(key)
                    if isinstance(image_value, Image.Image):
                        filename_preview = f"{preview_prefix}-{suffix}.png"
                        image_value.save(job_dir / filename_preview, "PNG")
                        preview_entry[key] = f"/api/download/{job_dir.name}/{filename_preview}"
                for key, suffix in (
                    ("requestMeta", "request-meta"),
                    ("apiError", "api-error"),
                ):
                    payload_value = preview.get(key)
                    if isinstance(payload_value, dict):
                        filename_payload = f"{preview_prefix}-{suffix}.json"
                        (job_dir / filename_payload).write_text(json.dumps(payload_value, indent=2, ensure_ascii=False), encoding="utf-8")
                        preview_entry[key] = f"/api/download/{job_dir.name}/{filename_payload}"
                generative_attempt_previews_payload.append(preview_entry)
            final_render_plan = [
                {
                    **entry,
                    "alignment": next((block.align for block in translated_blocks if block.id == entry["id"]), "center"),
                    "fontSizeEstimate": next((block.font_size_estimate for block in translated_blocks if block.id == entry["id"]), 16),
                    "lineHeightEstimate": next((block.line_height_estimate for block in translated_blocks if block.id == entry["id"]), 18),
                    "lineBoxes": [list(line_box) for line_box in next((block.line_boxes for block in translated_blocks if block.id == entry["id"]), [])],
                }
                for entry in render_plan_entries
            ]
            filename = localize_filename(image_path, language, output_format)
            final_localized_download_url: str | None = None
            if rendered is not None:
                save_image_output(rendered, job_dir / filename, normalize_output_format(output_format, image_path))
                final_localized_download_url = f"/api/download/{job_dir.name}/{filename}"
            debug_payload = build_debug_payload(
                raw_ocr_blocks,
                translated_blocks,
                token_masks=token_masks,
                cleaned_background_preview=f"/api/download/{job_dir.name}/{clean_filename}",
                final_render_plan=final_render_plan,
                block_line_groups=cleanup_debug["blockLineGroups"],
                line_cleanup_regions=cleanup_debug["lineCleanupRegions"],
                line_masks=cleanup_debug["lineMasks"],
                line_cleanup_strategies=cleanup_debug["lineCleanupStrategies"],
                line_cleanup_quality_scores=cleanup_debug["lineCleanupQualityScores"],
                foreground_overlap_scores=cleanup_debug["foregroundOverlapScores"],
                cleanup_warnings=cleanup_debug["cleanupWarnings"],
                safe_area_candidates=[{"id": entry["id"], "candidates": entry["safeAreaCandidates"]} for entry in render_plan_entries],
                raw_foreground_boxes=cleanup_debug.get("rawForegroundBoxes", []),
                protected_region_mask_preview=f"/api/download/{job_dir.name}/{protected_mask_filename}" if isinstance(protected_mask_image, Image.Image) else None,
                refined_protected_region_mask=f"/api/download/{job_dir.name}/{protected_mask_filename}" if isinstance(protected_mask_image, Image.Image) else None,
                protected_mask_refinement_method=cleanup_debug.get("protectedMaskRefinementMethod", ""),
                generative_attempt_previews=generative_attempt_previews_payload,
                cleanup_status=cleanup_gate["cleanupStatus"],
                block_cleanup_status=cleanup_gate["blockCleanupStatus"],
                residual_source_ocr=cleanup_gate["residualSourceOCR"],
                failed_source_words=cleanup_gate["failedSourceWords"],
                block_level_fallback_attempted=cleanup_gate["blockLevelFallbackAttempted"],
                block_level_fallback_selected=cleanup_gate["blockLevelFallbackSelected"],
                final_render_skipped_reason=cleanup_gate["finalRenderSkippedReason"],
                requested_cleanup_provider=cleanup_debug.get("requestedCleanupProvider"),
                actual_cleanup_provider=cleanup_debug.get("actualCleanupProvider"),
                provider_available=cleanup_debug.get("providerAvailable"),
                provider_failure_reason=cleanup_debug.get("providerFailureReason"),
                mask_dilation_px=cleanup_debug.get("maskDilationPx"),
                mask_feather_px=cleanup_debug.get("maskFeatherPx"),
                inpainting_input_mode=cleanup_debug.get("inpaintingInputMode"),
                mask_polarity=cleanup_debug.get("maskPolarity"),
                text_coverage_estimate=cleanup_debug.get("textCoverageEstimate"),
                background_leakage_estimate=cleanup_debug.get("backgroundLeakageEstimate"),
                mask_quality_status=cleanup_debug.get("maskQualityStatus"),
                mask_failure_reason=cleanup_debug.get("maskFailureReason"),
                mask_white_pixel_ratio=cleanup_debug.get("maskWhitePixelRatio"),
                text_stroke_mask_paths=text_stroke_mask_paths,
                anti_alias_coverage_estimate=cleanup_debug.get("antiAliasCoverageEstimate"),
                protected_object_overlap_estimate=cleanup_debug.get("protectedObjectOverlapEstimate"),
                dilation_attempts=cleanup_debug.get("dilationAttempts"),
                selected_dilation_px=cleanup_debug.get("selectedDilationPx"),
                best_cleanup_attempt=cleanup_debug.get("bestCleanupAttempt"),
                provider_quality_status=cleanup_debug.get("providerQualityStatus"),
                residual_text_after_each_attempt=cleanup_debug.get("residualTextAfterEachAttempt"),
                ghosting_score_after_each_attempt=cleanup_debug.get("ghostingScoreAfterEachAttempt"),
                cleanup_selected_reason=cleanup_debug.get("cleanupSelectedReason"),
                cleanup_cascade_enabled=cleanup_debug.get("cleanupCascadeEnabled"),
                first_pass_provider=cleanup_debug.get("firstPassProvider"),
                first_pass_status=cleanup_debug.get("firstPassStatus"),
                residual_text_detected_after_first_pass=cleanup_debug.get("residualTextDetectedAfterFirstPass"),
                residual_words_after_first_pass=cleanup_debug.get("residualWordsAfterFirstPass"),
                residual_mask_generated=cleanup_debug.get("residualMaskGenerated"),
                second_pass_provider=cleanup_debug.get("secondPassProvider"),
                second_pass_status=cleanup_debug.get("secondPassStatus"),
                residual_text_detected_after_second_pass=cleanup_debug.get("residualTextDetectedAfterSecondPass"),
                residual_words_after_second_pass=cleanup_debug.get("residualWordsAfterSecondPass"),
                sdxl_fallback_attempted=cleanup_debug.get("sdxlFallbackAttempted"),
                sdxl_fallback_status=cleanup_debug.get("sdxlFallbackStatus"),
                final_cleanup_quality_status=cleanup_debug.get("finalCleanupQualityStatus"),
                mask_type_used_for_lama=cleanup_debug.get("maskTypeUsedForLaMa"),
                composite_mask_available=cleanup_debug.get("compositeMaskAvailable"),
                adaptive_text_pixel_detection=cleanup_debug.get("adaptiveTextPixelDetection"),
                text_pixel_detection_methods_used=cleanup_debug.get("textPixelDetectionMethodsUsed"),
                first_pass_lama_output=first_pass_lama_output_path,
                residual_text_mask_preview=residual_text_mask_path,
                second_pass_lama_output=second_pass_lama_output_path,
                final_cleaned_candidate=final_cleaned_candidate_path,
                residual_ocr_after_first_pass=cleanup_debug.get("residualOCRAfterFirstPass"),
                residual_ocr_after_second_pass=cleanup_debug.get("residualOCRAfterSecondPass"),
                cleanup_cascade_log=[{"path": cleanup_cascade_log_path, "entries": cleanup_debug.get("cleanupCascadeLog", [])}] if cleanup_cascade_log_path else cleanup_debug.get("cleanupCascadeLog"),
                cleanup_passes_run=cleanup_debug.get("cleanupPassesRun"),
                residual_words_by_pass=cleanup_debug.get("residualWordsByPass"),
                residual_mask_area_by_pass=cleanup_debug.get("residualMaskAreaByPass"),
                cleanup_improvement_by_pass=cleanup_debug.get("cleanupImprovementByPass"),
                stopped_reason=cleanup_debug.get("stoppedReason"),
                residual_word_boxes_after_first_pass=cleanup_debug.get("residualWordBoxesAfterFirstPass"),
                residual_glyph_mask_after_first_pass=residual_glyph_mask_path,
                residual_artifact_mask_expanded=residual_artifact_mask_path,
                residual_mask_coverage_by_word=cleanup_debug.get("residualMaskCoverageByWord"),
                residual_mask_false_positive_estimate=cleanup_debug.get("residualMaskFalsePositiveEstimate"),
                sdxl_fallback_configured=cleanup_debug.get("sdxlFallbackConfigured"),
                sdxl_control_type=cleanup_debug.get("sdxlControlType"),
                sdxl_model_id=cleanup_debug.get("sdxlModelId"),
                control_net_model_id=cleanup_debug.get("controlNetModelId"),
                sdxl_failure_reason=cleanup_debug.get("sdxlFailureReason"),
                style_aware_rendering_enabled=render_audit.get("styleAwareRenderingEnabled"),
                source_typography_extracted=render_audit.get("sourceTypographyExtracted"),
                source_style_spans_summary=render_audit.get("sourceStyleSpans"),
                translated_style_spans_summary=render_audit.get("translatedStyleSpans"),
                semantic_style_mapping=render_audit.get("semanticStyleMapping"),
                styled_span_layout_summary=render_audit.get("styledSpanLayout"),
                span_render_boxes_summary=render_audit.get("spanRenderBoxes"),
                overflow_warnings_summary=render_audit.get("overflowWarnings"),
                selected_font_scale=render_audit.get("selectedFontScale"),
                line_break_strategy=render_audit.get("lineBreakStrategy"),
                render_quality_status=render_audit.get("renderQualityStatus"),
                translated_text_render_preview=f"/api/download/{job_dir.name}/{translated_preview_filename}",
                source_text_style_reference=f"/api/download/{job_dir.name}/{source_reference_filename}",
                render_plan_preview=f"/api/download/{job_dir.name}/{render_plan_filename}",
                semantic_block_grouping_status=grouping_audit.get("semanticBlockGroupingStatus"),
                headline_merged_into_single_block=grouping_audit.get("headlineMergedIntoSingleBlock"),
                grouped_line_ids=grouping_audit.get("groupedLineIds"),
                grouping_reason=grouping_audit.get("groupingReason"),
                grouping_confidence=grouping_audit.get("groupingConfidence"),
                rejected_merge_reasons=grouping_audit.get("rejectedMergeReasons"),
                block_type=grouping_audit.get("blockType"),
            )
            debug_payload["cleanedImageWithoutText"] = f"/api/download/{job_dir.name}/{clean_filename}"
            debug_payload["finalLocalizedImageWithTranslatedText"] = final_localized_download_url
            debug_payload["tokenMaskPreview"] = f"/api/download/{job_dir.name}/{mask_filename}"
            debug_payload["openAIImageEditMask"] = f"/api/download/{job_dir.name}/{openai_edit_mask_filename}"
            debug_payload["renderProvider"] = openai_render_meta or {"provider": "local"}
            if rendered is not None and final_localized_download_url is not None:
                asset = OutputAsset(
                    filename=filename,
                    width=rendered.width,
                    height=rendered.height,
                    safe_zone_warnings=[],
                    download_url=final_localized_download_url,
                    source_name=image_path.name,
                    language=language,
                    source_language=source_language,
                    translated_text=editor_text,
                    extracted_blocks=translated_blocks,
                )
                outputs.append(asset)
            manifest_assets.append(
                {
                    "mode": "localize",
                    "filename": filename if rendered is not None else None,
                    "source_path": str(image_path),
                    "source_name": image_path.name,
                    "language": language,
                    "source_language": source_language,
                    "width": rendered.width if rendered is not None else gated_background.width,
                    "height": rendered.height if rendered is not None else gated_background.height,
                    "translated_text": editor_text,
                    "blocks": [block.model_dump() for block in translated_blocks],
                    "cleanup_status": cleanup_gate["cleanupStatus"],
                    "render_provider": openai_render_meta or {"provider": "local"},
                    "final_asset_generated": rendered is not None,
                    "debug": debug_payload,
                    "fit": "cover",
                    "scale": 100,
                }
            )

    return outputs, translations_summary, manifest_assets


def resolve_resize_dimensions(placement_id: str, custom_width: int | None, custom_height: int | None) -> tuple[int, int]:
    placement_id = canonical_placement_id(placement_id)
    if placement_id == "custom-display" and custom_width and custom_height:
        return max(64, custom_width), max(64, custom_height)
    return PLACEMENT_DIMENSIONS.get(placement_id, PLACEMENT_DIMENSIONS["custom-display"])


def compute_resize_crop_box(image: Image.Image, focus_bbox: tuple[int, int, int, int], target_ratio: float) -> list[int] | None:
    image_ratio = image.width / max(1, image.height)
    if abs(image_ratio - target_ratio) < 0.015:
        return [0, 0, image.width, image.height]
    left, top, right, bottom = focus_bbox
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    if image_ratio > target_ratio:
        crop_width = max(int((bottom - top) * target_ratio), min(image.width, int(image.height * target_ratio)))
        crop_height = image.height
    else:
        crop_width = image.width
        crop_height = max(int((right - left) / target_ratio), min(image.height, int(image.width / target_ratio)))
    crop_width = min(image.width, max(1, crop_width))
    crop_height = min(image.height, max(1, crop_height))
    x0 = min(max(0, int(round(center_x - crop_width / 2))), image.width - crop_width)
    y0 = min(max(0, int(round(center_y - crop_height / 2))), image.height - crop_height)
    return [x0, y0, x0 + crop_width, y0 + crop_height]


def build_preview_template_parity_report(placement_ids: list[str]) -> dict[str, Any]:
    canonical_ids = [canonical_placement_id(item) for item in placement_ids]
    validated_ids = [placement_id for placement_id in canonical_ids if placement_id in SHARED_PREVIEW_PLACEMENT_MAP]
    return {
        "sharedTemplateSchemaVersion": SHARED_TEMPLATE_SCHEMA_VERSION,
        "sharedTemplateSource": str(SHARED_PREVIEW_TEMPLATE_PATH),
        "backendTemplateSource": str(SHARED_PREVIEW_TEMPLATE_PATH),
        "frontendTemplateSource": "src/shared/preview-templates.json",
        "usesSingleSharedSource": True,
        "placementsValidated": validated_ids,
        "missingInFrontend": [],
        "missingInBackend": [],
        "dimensionMismatches": [],
        "carouselSupportMismatches": [],
        "templateStatusMismatches": [],
    }


def smart_reframe_analysis_provider(analysis: VisualAnalysis | None) -> str:
    if analysis is None:
        return "unknown"
    for warning in analysis.quality_warnings:
        if warning.startswith("analysis_provider:"):
            return warning.split(":", 1)[1] or "unknown"
    return "unknown"


def merge_resize_warnings(placement_id: str, plan: Any | None = None) -> list[str]:
    merged: list[str] = []
    for warning in [*safe_zone_warnings_for(placement_id), *(getattr(plan, "safe_zone_warnings", []) or [])]:
        if warning and warning not in merged:
            merged.append(warning)
    return merged


async def build_resize_assets(paths: list[Path], placement_ids: list[str], output_format: str, custom_width: int | None, custom_height: int | None, job_dir: Path, creative_modes: dict[str, str] | None = None) -> tuple[list[OutputAsset], list[dict[str, Any]]]:
    outputs: list[OutputAsset] = []
    manifest_assets: list[dict[str, Any]] = []
    canonical_placement_ids = list(dict.fromkeys(canonical_placement_id(item) for item in placement_ids))
    resolved_creative_modes = {canonical_placement_id(key): value for key, value in (creative_modes or {}).items()}
    resize_mode = os.getenv("ADAPTIFAI_RESIZE_FIT", "cover")
    resize_strategy = os.getenv("ADAPTIFAI_RESIZE_STRATEGY", "smart-reframe").strip().lower()
    smart_reframe_enabled = resize_strategy in {"smart-reframe", "smart", "reframe"}
    safe_resize_strategy = resize_strategy in {"blurred-fit", "fit", "contain-blur", "safe"}
    max_provider_placements = max(1, int(os.getenv("ADAPTIFAI_RESIZE_MAX_PROVIDER_PLACEMENTS_PER_REQUEST", "2")))
    allow_provider_outpaint = len(canonical_placement_ids) <= max_provider_placements
    parity_report = build_preview_template_parity_report(canonical_placement_ids)
    parity_report_filename = "previewTemplateParityReport.json"
    (job_dir / parity_report_filename).write_text(json.dumps(parity_report, ensure_ascii=False, indent=2), encoding="utf-8")

    source_entries: list[dict[str, Any]] = []
    for image_path in image_paths(paths):
        print(f"[resize_e2e] source_start name={image_path.name}", flush=True)
        source_image = Image.open(image_path).convert("RGB")
        focus_bbox = (0, 0, source_image.width, source_image.height) if safe_resize_strategy else build_resize_focus_bbox(source_image)
        print(f"[resize_e2e] analysis_start name={image_path.name}", flush=True)
        visual_analysis = build_smart_reframe_analysis(source_image, focus_bbox)
        print(
            f"[resize_e2e] analysis_done name={image_path.name} provider={smart_reframe_analysis_provider(visual_analysis)} "
            f"text={len(visual_analysis.text_layers)} products={len(visual_analysis.product_layers)} other={len(visual_analysis.other_layers)}",
            flush=True,
        )
        print(f"[resize_e2e] plan_start name={image_path.name} placements={','.join(canonical_placement_ids)}", flush=True)
        reframe_plans = {
            plan.placement_id: plan
            for plan in await SmartReframe(visual_analysis).execute(canonical_placement_ids, custom_width, custom_height)
        }
        print(f"[resize_e2e] plan_done name={image_path.name} count={len(reframe_plans)}", flush=True)
        source_entries.append(
            {
                "path": image_path,
                "image": source_image,
                "focus_bbox": focus_bbox,
                "visual_analysis": visual_analysis,
                "reframe_plans": reframe_plans,
            }
        )

    rendered_assets_by_placement: dict[str, list[dict[str, Any]]] = {}
    for canonical_id in canonical_placement_ids:
        width, height = resolve_resize_dimensions(canonical_id, custom_width, custom_height)
        placement_assets: list[dict[str, Any]] = []
        for source_entry in source_entries:
            plan = source_entry["reframe_plans"].get(canonical_id)
            render_meta: dict[str, Any] = {"provider": "local", "strategy": "legacy"}
            if smart_reframe_enabled and plan is not None:
                print(f"[resize_e2e] render_start placement={canonical_id} source={source_entry['path'].name}", flush=True)
                rendered, render_meta = render_smart_reframe_image(
                    source_entry["image"],
                    width,
                    height,
                    plan,
                    source_entry["visual_analysis"],
                    allow_provider_outpaint=allow_provider_outpaint,
                    allow_product_completion=True,
                )
                render_meta = {
                    **render_meta,
                    "providerOutpaintAllowedForRequest": allow_provider_outpaint,
                    "providerOutpaintPlacementLimit": max_provider_placements,
                    "requestPlacementCount": len(canonical_placement_ids),
                }
                print(
                    f"[resize_e2e] render_done placement={canonical_id} strategy={render_meta.get('strategy')} "
                    f"provider={render_meta.get('provider')} productionReady={render_meta.get('productionReady')}",
                    flush=True,
                )
            elif safe_resize_strategy:
                rendered = render_blurred_fit_resize(source_entry["image"], width, height)
                render_meta = {"provider": "local", "strategy": "blurred_fit"}
            else:
                try:
                    rendered = smart_resize_image(source_entry["image"], width, height, placement_id=canonical_id)
                    render_meta = {"provider": "local", "strategy": "legacy_smart_resize"}
                except Exception:
                    rendered = render_resize_image(
                        source_entry["image"],
                        width,
                        height,
                        fit=resize_mode,
                        focus_bbox=source_entry["focus_bbox"],
                    )
                    render_meta = {"provider": "local", "strategy": "legacy_resize_fallback"}
            if rendered.size != (width, height):
                previous_size = rendered.size
                previous_strategy = render_meta.get("strategy", "unknown")
                rendered = render_resize_image(
                    source_entry["image"],
                    width,
                    height,
                    fit=resize_mode,
                    focus_bbox=source_entry["focus_bbox"],
                )
                render_meta = {
                    **render_meta,
                    "provider": "local",
                    "strategy": f"{previous_strategy}_dimension_fix",
                    "fallbackReason": f"Rendered size {previous_size} did not match target {(width, height)}.",
                }
            placement_assets.append(
                {
                    "source_name": source_entry["path"].name,
                    "rendered": rendered,
                    "source_image": source_entry["image"],
                    "focus_bbox": source_entry["focus_bbox"],
                    "reframe_plan": plan,
                    "visual_analysis": source_entry["visual_analysis"],
                    "render_meta": render_meta,
                }
            )
        rendered_assets_by_placement[canonical_id] = placement_assets

    for source_entry in source_entries:
        image_path = source_entry["path"]
        source_image = source_entry["image"]
        focus_bbox = source_entry["focus_bbox"]
        for canonical_id in canonical_placement_ids:
            width, height = resolve_resize_dimensions(canonical_id, custom_width, custom_height)
            rendered_set = rendered_assets_by_placement[canonical_id]
            placement_supports_carousel = bool(PREVIEW_TEMPLATE_REGISTRY.get(canonical_id, {}).get("supportsCarousel"))
            requested_creative_mode = str(resolved_creative_modes.get(canonical_id, "single")).strip().lower()
            creative_mode = requested_creative_mode if requested_creative_mode in {"single", "carousel"} else "single"
            if not placement_supports_carousel:
                creative_mode = "single"
            source_index = next((index for index, item in enumerate(rendered_set) if item["source_name"] == image_path.name), 0)
            active_index = source_index
            rendered = rendered_set[active_index]["rendered"]
            active_reframe_plan = rendered_set[active_index].get("reframe_plan")
            active_visual_analysis = rendered_set[active_index].get("visual_analysis")
            active_render_meta = rendered_set[active_index].get("render_meta") or {}
            analysis_provider = smart_reframe_analysis_provider(active_visual_analysis)
            resize_warnings = merge_resize_warnings(canonical_id, active_reframe_plan)
            strategy_slug = sanitize_debug_token(active_render_meta.get("strategy") or (active_reframe_plan.expansion.strategy.value if active_reframe_plan is not None else resize_strategy))
            bucket_slug = sanitize_debug_token(active_reframe_plan.logic_bucket.value if active_reframe_plan is not None else "legacy")
            filename = resize_filename(image_path, canonical_id, output_format)
            save_image_output(rendered, job_dir / filename, normalize_output_format(output_format, image_path))
            raw_asset_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-raw-localized-asset.png"
            if debug_strategy_filenames_enabled():
                resized_asset_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-{bucket_slug}-{strategy_slug}-resized-asset.png"
            else:
                resized_asset_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-resized-asset.png"
            preview_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-native-placement-preview.png"
            preview_template_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-preview-template-used.json"
            placement_render_log_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-placement-render-log.json"
            resize_log_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-resize-decision-log.json"
            carousel_log_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-carouselRenderLog.json"
            source_image.save(job_dir / raw_asset_filename, "PNG")
            rendered.save(job_dir / resized_asset_filename, "PNG")
            preview_metadata = placement_preview_metadata(canonical_id, image_path.name, "", creative_mode=creative_mode)
            preview_metadata["carouselAssetsProvided"] = len(rendered_set) > 1
            preview_metadata["carouselAssets"] = [item["source_name"] for item in rendered_set]
            preview_metadata["carouselAssetLabels"] = [item["source_name"] for item in rendered_set]
            preview_metadata["activeSlideIndex"] = active_index
            preview_metadata["unusedAssets"] = [
                item["source_name"]
                for index, item in enumerate(rendered_set)
                if creative_mode == "single" and len(rendered_set) > 1 and index != active_index
            ]
            preview_image, preview_template_used, placement_render_log, carousel_render_log = render_native_placement_preview(
                rendered,
                canonical_id,
                preview_metadata,
                carousel_assets=[item["rendered"] for item in rendered_set] if creative_mode == "carousel" else [rendered_set[0]["rendered"]],
                active_slide_index=active_index,
            )
            preview_image.save(job_dir / preview_filename, "PNG")
            (job_dir / preview_template_filename).write_text(json.dumps(preview_template_used, ensure_ascii=False, indent=2), encoding="utf-8")
            (job_dir / placement_render_log_filename).write_text(json.dumps(placement_render_log, ensure_ascii=False, indent=2), encoding="utf-8")
            (job_dir / carousel_log_filename).write_text(json.dumps(carousel_render_log, ensure_ascii=False, indent=2), encoding="utf-8")
            resize_decision_log = {
                "placement": canonical_id,
                "sharedTemplateSchemaVersion": SHARED_TEMPLATE_SCHEMA_VERSION,
                "creativeMode": creative_mode,
                "targetWidth": width,
                "targetHeight": height,
                "resizeMode": resize_mode,
                "resizeStrategy": resize_strategy,
                "analysisProvider": analysis_provider,
                "logicBucket": active_reframe_plan.logic_bucket.value if active_reframe_plan is not None else None,
                "renderMeta": active_render_meta,
                "smartReframePlan": active_reframe_plan.model_dump(mode="json") if active_reframe_plan is not None else None,
                "visualAnalysis": active_visual_analysis.model_dump(mode="json") if active_visual_analysis is not None else None,
                "cropBox": None if (safe_resize_strategy or smart_reframe_enabled) else compute_resize_crop_box(source_image, focus_bbox, width / max(1, height)) if resize_mode == "cover" else None,
                "focusBox": list(focus_bbox),
                "protectedObjects": [list(focus_bbox)],
                "safeZoneWarnings": resize_warnings,
                "textCutoffRisk": "medium" if canonical_id in {"gdn-320x50", "gdn-728x90", "facebook-right-column"} else "low",
                "productCutoffRisk": "medium" if resize_mode == "cover" and canonical_id not in {"instagram-story", "instagram-reels", "tiktok-in-feed", "tiktok-topview", "youtube-shorts"} else "low",
            }
            (job_dir / resize_log_filename).write_text(json.dumps({
                **resize_decision_log,
                "previewTemplateUsed": preview_template_used,
                "placementRenderLogPath": f"/api/download/{job_dir.name}/{placement_render_log_filename}",
                "carouselRenderLogPath": f"/api/download/{job_dir.name}/{carousel_log_filename}",
                "previewTemplateParityReportPath": f"/api/download/{job_dir.name}/{parity_report_filename}",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            asset = OutputAsset(
                placement_id=canonical_id,
                filename=filename,
                width=width,
                height=height,
                safe_zone_warnings=resize_warnings,
                download_url=f"/api/download/{job_dir.name}/{filename}",
                source_name=image_path.name,
                language=None,
                translated_text="",
                extracted_blocks=[],
                debug={
                    "analysisProvider": analysis_provider,
                    "logicBucket": active_reframe_plan.logic_bucket.value if active_reframe_plan is not None else None,
                    "renderMeta": active_render_meta,
                    "resizeDecisionLog": f"/api/download/{job_dir.name}/{resize_log_filename}",
                    "resizedAsset": f"/api/download/{job_dir.name}/{resized_asset_filename}",
                    "nativePlacementPreview": f"/api/download/{job_dir.name}/{preview_filename}",
                },
            )
            outputs.append(asset)
            manifest_assets.append(
                {
                    "mode": "resize",
                    "filename": filename,
                    "source_path": str(image_path),
                    "source_name": image_path.name,
                    "placement_id": canonical_id,
                    "width": width,
                    "height": height,
                    "fit": resize_mode,
                    "resizeStrategy": resize_strategy,
                    "analysisProvider": analysis_provider,
                    "logicBucket": active_reframe_plan.logic_bucket.value if active_reframe_plan is not None else None,
                    "renderMeta": active_render_meta,
                    "smartReframePlan": active_reframe_plan.model_dump(mode="json") if active_reframe_plan is not None else None,
                    "scale": 100,
                    "translated_text": "",
                    "blocks": [],
                    "rawLocalizedAsset": f"/api/download/{job_dir.name}/{raw_asset_filename}",
                    "resizedAsset": f"/api/download/{job_dir.name}/{resized_asset_filename}",
                    "nativePlacementPreview": f"/api/download/{job_dir.name}/{preview_filename}",
                    "previewTemplateUsed": preview_template_used,
                    "previewTemplateUsedPath": f"/api/download/{job_dir.name}/{preview_template_filename}",
                    "placementRenderLogPath": f"/api/download/{job_dir.name}/{placement_render_log_filename}",
                    "resizeDecisionLogPath": f"/api/download/{job_dir.name}/{resize_log_filename}",
                    "carouselRenderLogPath": f"/api/download/{job_dir.name}/{carousel_log_filename}",
                    "previewTemplateParityReportPath": f"/api/download/{job_dir.name}/{parity_report_filename}",
                    "sharedTemplateSchemaVersion": SHARED_TEMPLATE_SCHEMA_VERSION,
                    "creativeMode": preview_template_used.get("creativeMode", "single"),
                    "carouselActivationSource": preview_template_used.get("carouselActivationSource", "forced_single"),
                    "unusedAssets": preview_template_used.get("unusedAssets", []),
                    "nativePreviewEnabled": True,
                    "backendPreviewArtifactGenerated": True,
                    "placementTemplateStatus": preview_template_used.get("templateStatus", "stub"),
                    "reusedShellWarning": preview_template_used.get("reusedShellReason", ""),
                    "carouselSupported": preview_template_used.get("carouselSupported", False),
                    "carouselAssetsProvided": preview_template_used.get("carouselAssetsProvided", False),
                    "carouselPreviewGenerated": preview_template_used.get("carouselPreviewGenerated", False),
                    "carouselAssetsMissing": preview_template_used.get("carouselAssetsMissing", False),
                    "previewQualityStatus": "partial" if preview_template_used.get("reusedShell") else "production",
                }
            )
    return outputs, manifest_assets

def split_editor_copy(copy: str, expected_count: int) -> list[str]:
    parts = [part.strip() for part in copy.split("\n\n") if part.strip()]
    if len(parts) < expected_count:
        lines = [line.strip() for line in copy.splitlines() if line.strip()]
        if len(lines) >= expected_count:
            parts = lines[:expected_count]
    return parts[:expected_count]


def _apply_style_overrides(blocks: list[TextBlock], edit: EditRequest) -> list[TextBlock]:
    """Apply editor-level style overrides (color, font size scale) to translate blocks."""
    overridden = []
    for block in blocks:
        if not block.translate:
            overridden.append(block)
            continue
        updates: dict[str, Any] = {}
        # Color override â€” only apply when a non-default colour was chosen
        if edit.text_color and edit.text_color != "#111111":
            updates["color"] = edit.text_color
        # Font size scale (100 = no change)
        if edit.font_size_scale != 100:
            scale_factor = max(0.5, min(3.0, edit.font_size_scale / 100.0))
            updates["font_size_estimate"] = max(8, int(round(block.font_size_estimate * scale_factor)))
            updates["line_height_estimate"] = max(9, int(round(block.line_height_estimate * scale_factor)))
        # Bold weight override
        if edit.preserve_bold:
            updates["font_weight"] = max(block.font_weight, 700)
        overridden.append(block.model_copy(update=updates) if updates else block)
    return overridden


def _draw_text_decoration(
    image: Image.Image,
    blocks: list[TextBlock],
    underline: bool,
    strikethrough: bool,
    x_offset: int = 0,
    y_offset: int = 0,
) -> Image.Image:
    """Draw underline / strikethrough lines on top of rendered text."""
    if not underline and not strikethrough:
        return image
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue
        bx1, by1, bx2, by2 = block.bbox
        bx1 += x_offset; bx2 += x_offset
        by1 += y_offset; by2 += y_offset
        fill = block.color if block.color.startswith("#") else "#111111"
        lw = max(1, int(round((by2 - by1) * 0.05)))
        if underline:
            uy = by2 - lw
            draw.rectangle((bx1, uy, bx2, uy + lw), fill=fill)
        if strikethrough:
            my = (by1 + by2) // 2
            draw.rectangle((bx1, my, bx2, my + lw), fill=fill)
    return out


def _apply_italic_shear(
    image: Image.Image,
    blocks: list[TextBlock],
    x_offset: int = 0,
    y_offset: int = 0,
) -> Image.Image:
    """
    Simulate italic by applying a forward-lean affine shear to each translate block region.
    Shear ~14Â° (tan â‰ˆ 0.25): top of each block region leans to the right.
    PIL AFFINE coefficients (a,b,c,d,e,f): src_x = a*dst_x + b*dst_y + c
    For forward-lean: src_x = dst_x + shear*dst_y  â†’  (a=1, b=shear, c=0, d=0, e=1, f=0)
    """
    shear = 0.25
    result = image.copy()
    img_w, img_h = image.size
    for block in blocks:
        if not block.translate or not text_changed(block.text, block.translated_text):
            continue
        bx1 = max(0, block.bbox[0] + x_offset)
        by1 = max(0, block.bbox[1] + y_offset)
        bx2 = min(img_w, block.bbox[2] + x_offset)
        by2 = min(img_h, block.bbox[3] + y_offset)
        if bx2 <= bx1 or by2 <= by1:
            continue
        rw, rh = bx2 - bx1, by2 - by1
        shear_px = int(shear * rh)
        # Extract a slightly wider region so we have background to fill shear gaps
        extract_x1 = max(0, bx1 - shear_px)
        extract_x2 = min(img_w, bx2 + shear_px)
        region = image.crop((extract_x1, by1, extract_x2, by2))
        ext_w = extract_x2 - extract_x1
        # Apply shear transform
        try:
            sheared = region.transform(
                (ext_w, rh),
                Image.Transform.AFFINE,
                (1, shear, 0, 0, 1, 0),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(255, 255, 255),
            )
        except Exception:
            continue
        result.paste(sheared, (extract_x1, by1))
    return result


def regenerate_localize_asset(job_dir: Path, asset_meta: dict[str, Any], edit: EditRequest) -> OutputAsset:
    source_image = Image.open(asset_meta["source_path"]).convert("RGB")
    stored_blocks = [TextBlock.model_validate(block) for block in asset_meta.get("blocks", [])]
    translate_blocks = [block for block in stored_blocks if block.translate]
    replacement_texts = split_editor_copy(edit.copy_text, len(translate_blocks))
    if replacement_texts:
        iterator = iter(replacement_texts)
        updated_blocks = []
        editor_parts: list[str] = []
        for block in stored_blocks:
            if block.translate:
                text = next(iterator, block.translated_text or block.text)
                editor_parts.append(text)
                updated_blocks.append(block.model_copy(update={"translated_text": text}))
            else:
                updated_blocks.append(block)
    else:
        updated_blocks = stored_blocks
        editor_parts = [block.translated_text or block.text for block in stored_blocks if block.translate]

    styled_blocks = _apply_style_overrides(updated_blocks, edit)
    base = build_clean_background(source_image, styled_blocks, cleanup_strength=100 - max(0, min(90, edit.opacity)))
    if not edit.mask_cleanup:
        base = source_image
    rendered = render_translated_text(base, styled_blocks, edit.x, edit.y, edit.preserve_bold, edit.fit_bounds)
    rendered = _draw_text_decoration(rendered, styled_blocks, edit.text_underline, edit.text_strike, edit.x, edit.y)
    if edit.text_italic:
        rendered = _apply_italic_shear(rendered, styled_blocks, edit.x, edit.y)
    output_path = job_dir / asset_meta["filename"]
    localize_renderer = os.getenv("ADAPTIFAI_LOCALIZE_RENDERER", "local").strip().lower()
    localize_foreground_bbox = detect_foreground_bbox(source_image)
    if localize_renderer == "openai" and any(
        is_localize_marketing_edit_block(block, source_image.size, localize_foreground_bbox)
        for block in styled_blocks
    ):
        try:
            rendered, render_meta = render_localize_with_openai_image_edit(
                source_image,
                styled_blocks,
                str(asset_meta.get("language") or "EN"),
                str(asset_meta.get("source_language") or "EN"),
                output_path.suffix.lower().lstrip(".") or "png",
            )
            asset_meta["render_provider"] = render_meta
        except Exception as exc:
            asset_meta["render_provider"] = {
                "provider": "local",
                "strategy": "local_render_openai_failed",
                "fallbackReason": str(exc),
            }
    save_image_output(rendered, output_path, output_path.suffix.lower().lstrip(".") or "png")
    asset_meta["translated_text"] = "\n\n".join(editor_parts)
    asset_meta["blocks"] = [block.model_dump() for block in updated_blocks]
    return OutputAsset(
        filename=asset_meta["filename"],
        width=rendered.width,
        height=rendered.height,
        safe_zone_warnings=[],
        download_url=f"/api/download/{job_dir.name}/{asset_meta['filename']}",
        source_name=asset_meta["source_name"],
        language=asset_meta.get("language"),
        source_language=asset_meta.get("source_language"),
        translated_text=asset_meta["translated_text"],
        extracted_blocks=updated_blocks,
    )


def regenerate_resize_asset(job_dir: Path, asset_meta: dict[str, Any], edit: EditRequest) -> OutputAsset:
    source_image = Image.open(asset_meta["source_path"]).convert("RGB")
    focus_bbox = build_resize_focus_bbox(source_image)
    resize_strategy = os.getenv("ADAPTIFAI_RESIZE_STRATEGY", "smart-reframe").strip().lower()
    if (
        resize_strategy in {"smart-reframe", "smart", "reframe"}
        and edit.fit == "cover"
        and edit.scale == 100
        and edit.x == 0
        and edit.y == 0
    ):
        analysis = build_smart_reframe_analysis(source_image, focus_bbox)
        plan = SmartReframe(analysis).execute_sync([str(asset_meta.get("placement_id") or "custom-display")])[0]
        rendered, render_meta = render_smart_reframe_image(source_image, int(asset_meta["width"]), int(asset_meta["height"]), plan, analysis)
        asset_meta["resize_strategy"] = resize_strategy
        asset_meta["smart_reframe_plan"] = plan.model_dump(mode="json")
        asset_meta["render_meta"] = render_meta
    elif (
        resize_strategy in {"blurred-fit", "fit", "contain-blur", "safe"}
        and edit.fit == "cover"
        and edit.scale == 100
        and edit.x == 0
        and edit.y == 0
    ):
        rendered = render_blurred_fit_resize(source_image, int(asset_meta["width"]), int(asset_meta["height"]))
        asset_meta["render_meta"] = {"provider": "local", "strategy": "blurred_fit"}
    else:
        rendered = render_resize_image(source_image, int(asset_meta["width"]), int(asset_meta["height"]), edit.fit, edit.scale, edit.x, edit.y, focus_bbox=focus_bbox)
        asset_meta["render_meta"] = {"provider": "local", "strategy": "manual_resize"}
    output_path = job_dir / asset_meta["filename"]
    save_image_output(rendered, output_path, output_path.suffix.lower().lstrip(".") or "png")
    asset_meta["fit"] = edit.fit
    asset_meta["scale"] = edit.scale
    return OutputAsset(
        placement_id=asset_meta.get("placement_id"),
        filename=asset_meta["filename"],
        width=int(asset_meta["width"]),
        height=int(asset_meta["height"]),
        safe_zone_warnings=safe_zone_warnings_for(asset_meta.get("placement_id", "")),
        download_url=f"/api/download/{job_dir.name}/{asset_meta['filename']}",
        source_name=asset_meta["source_name"],
        language=None,
        translated_text="",
        extracted_blocks=[],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "storage": "stateless-temp-24h",
        "device": runtime_device_label(),
        "ocr_engine": os.getenv("ADAPTIFAI_OCR_ENGINE", "easyocr"),
        "ocr_max_side": os.getenv("ADAPTIFAI_OCR_MAX_SIDE", "1280"),
        "ocr": os.getenv("ADAPTIFAI_TROCR_MODEL", "microsoft/trocr-base-printed"),
        "inpainting_backend": os.getenv("ADAPTIFAI_INPAINT_BACKEND", "stable-diffusion"),
        "inpainting": os.getenv("ADAPTIFAI_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"),
        "resize_fit": os.getenv("ADAPTIFAI_RESIZE_FIT", "cover"),
        "smart_reframe_analysis_provider": os.getenv("ADAPTIFAI_SMART_REFRAME_ANALYSIS_PROVIDER", "auto"),
        "openrouter_configured": "true" if bool(os.getenv("OPENROUTER_API_KEY", "").strip()) else "false",
        "openrouter_smart_reframe_model": (
            os.getenv("ADAPTIFAI_OPENROUTER_SMART_REFRAME_MODEL")
            or os.getenv("ADAPTIFAI_OPENROUTER_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or ""
        ),
        "resize_ai_layout_provider": os.getenv("ADAPTIFAI_RESIZE_AI_LAYOUT_PROVIDER", "auto"),
        "resize_ai_layout_planner": os.getenv("ADAPTIFAI_RESIZE_AI_LAYOUT_PLANNER", "1"),
        "replicate_lama_configured": "true" if bool(os.getenv("REPLICATE_API_TOKEN", "").strip()) else "false",
        "replicate_lama_model": resolved_replicate_lama_model(),
    }


@app.get("/outputs/{job_id}/{filename}")
def download_output(job_id: str, filename: str) -> FileResponse:
    cleanup_old_temp_files()
    safe_job = "".join(char for char in job_id if char.isalnum() or char in {"-", "_"})
    safe_file = Path(filename).name
    path = temp_root() / safe_job / safe_file
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output file is unavailable or expired.")
    return FileResponse(path, filename=safe_file, media_type="application/octet-stream")


@app.get("/api/download/{job_id}/{filename}")
def download_output_api_alias(job_id: str, filename: str) -> FileResponse:
    return download_output(job_id, filename)


@app.post("/adapt", response_model=AdaptResponse)
async def adapt(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File()],
    target_languages: Annotated[str, Form()] = "EN",
    output_format: Annotated[str, Form()] = "PNG",
    placements: Annotated[str, Form()] = "custom-display",
    creative_modes: Annotated[str | None, Form()] = None,
    mode: Annotated[str, Form()] = "",
    custom_width: Annotated[int | None, Form()] = None,
    custom_height: Annotated[int | None, Form()] = None,
) -> AdaptResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one creative.")

    print(f"[adapt] start files={len(files)} mode={mode or 'auto'} placements={placements} output={output_format}", flush=True)
    background_tasks.add_task(cleanup_old_temp_files)
    job_id = uuid4().hex[:12]
    job_dir = temp_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"[adapt] persist_uploads_start job={job_id}", flush=True)
    saved_paths = await persist_uploads(files, job_dir)
    uploaded_images = image_paths(saved_paths)
    print(f"[adapt] persist_uploads_done job={job_id} saved={len(saved_paths)} images={len(uploaded_images)}", flush=True)
    if not uploaded_images:
        raise HTTPException(status_code=400, detail="No image files were found in the upload.")

    languages = [item.strip().upper() for item in target_languages.split(",") if item.strip()] or ["EN"]
    placement_ids = [canonical_placement_id(item.strip()) for item in placements.split(",") if item.strip()] or ["custom-display"]
    try:
        parsed_creative_modes = json.loads(creative_modes) if creative_modes else {}
        if not isinstance(parsed_creative_modes, dict):
            parsed_creative_modes = {}
    except json.JSONDecodeError:
        parsed_creative_modes = {}
    resolved_mode = (mode or ("localize" if placement_ids == ["custom-display"] else "resize")).strip().lower()
    print(f"[adapt] resolved job={job_id} mode={resolved_mode} placements={','.join(placement_ids)}", flush=True)

    async with ADAPT_PROCESSING_SEMAPHORE:
        if resolved_mode == "localize":
            outputs, translations, manifest_assets = await run_in_threadpool(
                build_localize_assets,
                uploaded_images,
                languages,
                output_format,
                job_dir,
            )
            extracted_blocks = outputs[0].extracted_blocks if outputs else []
        else:
            outputs, manifest_assets = await run_in_threadpool(
                lambda: asyncio.run(
                    build_resize_assets(
                        uploaded_images,
                        placement_ids,
                        output_format,
                        custom_width,
                        custom_height,
                        job_dir,
                        creative_modes=parsed_creative_modes,
                    )
                )
            )
            translations = {}
            extracted_blocks = []

    manifest = {
        "job_id": job_id,
        "mode": resolved_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_width": custom_width,
        "custom_height": custom_height,
        "assets": manifest_assets,
    }
    write_manifest(job_dir, manifest)
    background_tasks.add_task(cleanup_old_temp_files)

    file_count = len(uploaded_images)
    multiplier = len(languages) if resolved_mode == "localize" else len(placement_ids)
    return AdaptResponse(
        job_id=job_id,
        stateless=True,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        credits_estimated=max(1, file_count * max(1, multiplier)),
        extracted_blocks=extracted_blocks,
        translations=translations,
        outputs=outputs,
    )


@app.post("/edit")
def edit_asset(edit: EditRequest) -> OutputAsset:
    cleanup_old_temp_files()
    job_dir = temp_root() / edit.job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job has expired.")

    manifest = read_manifest(job_dir)
    assets = manifest.get("assets", [])
    target = next((asset for asset in assets if asset.get("filename") == Path(edit.filename).name), None)
    if not target:
        raise HTTPException(status_code=404, detail="Output asset not found.")

    updated = regenerate_resize_asset(job_dir, target, edit) if edit.mode == "resize" else regenerate_localize_asset(job_dir, target, edit)
    write_manifest(job_dir, manifest)
    return updated
