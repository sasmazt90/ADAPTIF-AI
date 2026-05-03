from __future__ import annotations

import base64
import hashlib
import io
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


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env.local")
load_dotenv(REPO_ROOT / ".env")

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
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_PREVIEW_TEMPLATE_PATH = REPO_ROOT / "src" / "shared" / "preview-templates.json"


def load_shared_preview_template_schema() -> dict[str, Any]:
    # In Docker: __file__ = /app/app/main.py → parents[2] = / (wrong)
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
HF_CLEANUP_SESSION = None
HF_CLEANUP_SESSION_LOCK = threading.Lock()
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
    overflow_warning: bool = False
    source_style_spans: list[dict[str, Any]] = Field(default_factory=list)
    translated_style_spans: list[dict[str, Any]] = Field(default_factory=list)
    translation_candidates: list[dict[str, str]] = Field(default_factory=list)
    target_language: str | None = None
    cleanup_confidence: float = 1.0
    cleanup_strategy: str = "clean_replace"
    render_strategy: str = "clean_replace"


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ADAPTIFAI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def torch_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def is_cpu_runtime() -> bool:
    return torch_device() == "cpu"


def temp_root() -> Path:
    root = Path(os.getenv("ADAPTIFAI_TMP_DIR", Path(tempfile.gettempdir()) / "adaptifai"))
    root.mkdir(parents=True, exist_ok=True)
    return root


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


def has_packaging_cues(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    return any(
        cue in normalized
        for cue in (
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
        )
    )


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
        skipped_packaging = 0
        while next_index < len(ordered):
            candidate = ordered[next_index]
            if not candidate.translate:
                if candidate.surface in {"packaging", "product"} and skipped_packaging < 3:
                    skipped_packaging += 1
                    next_index += 1
                    continue
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
                and len(group) < 6
            )
            if not can_merge:
                break
            group.append(candidate)
            skipped_packaging = 0
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


def append_missing_translate_ocr_blocks(refined_blocks: list[TextBlock], source_blocks: list[TextBlock], image_size: tuple[int, int]) -> list[TextBlock]:
    if not source_blocks:
        return refined_blocks

    image_width, _ = image_size
    combined = list(refined_blocks)
    packaging_blocks = [block for block in combined if block.surface in {"packaging", "product"} or not block.translate]
    for source in source_blocks:
        if not source.translate:
            continue
        normalized_source = normalize_ocr_text(source.text)
        if not normalized_source:
            continue
        already_present = any(
            text_similarity(source.text, block.text) >= 0.72
            or any(text_similarity(source.text, line_text) >= 0.72 for line_text in block.line_texts)
            for block in combined
        )
        block_width = source.bbox[2] - source.bbox[0]
        block_height = source.bbox[3] - source.bbox[1]
        looks_salient = (
            block_width >= image_width * 0.18
            or "%" in source.text
            or source.text.isupper()
            or block_height >= 28
        )
        overlaps_packaging = any(overlap_fraction(source.bbox, block.bbox) >= 0.72 for block in packaging_blocks)
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
        packaging_cues = any(
            token in normalized
            for token in (
                "deodorant",
                "deo",
                "geruch",
                "odour",
                "odeur",
                "chauss",
                "shoe",
                "schuh",
                "ml",
                "dr.scholl",
                "scholl",
                "protection",
                "technologie",
                "types",
                "efficac",
            )
        )
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
                overlap_with_foreground >= 0.78
                and packaging_cues
                and word_count >= 4
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
        raw_items = parsed.get("blocks", [])
        if not isinstance(raw_items, list):
            return []

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
            crop = source_image.crop((left, top, right, bottom))
            font_weight = 800 if str(item.get("font_weight", "")).lower() in {"bold", "700", "800", "900"} or text.isupper() else 700
            blocks.append(
                TextBlock(
                    text=text,
                    role="headline" if translate else classify_text_role(text),
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
                or candidate_normalized in line
                or line in candidate_normalized
                for line in normalized_lines
            )
            if not line_match and candidate_normalized:
                line_match = candidate_normalized in normalized_block or text_similarity(candidate_normalized, normalized_block) > 0.45
            if not line_match:
                overlap = text_tokens(candidate.text) & text_tokens(block.text)
                line_match = len(overlap) >= 2
            if line_match:
                candidates.append(candidate)

        if candidates:
            bbox = union_bbox([candidate.bbox for candidate in candidates])
            if bbox is not None:
                ordered_candidates = sorted(candidates, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))
                merged_text = "\n".join(candidate.text.strip() for candidate in ordered_candidates if candidate.text.strip())
                snapped.append(
                    block.model_copy(
                        update={
                            "bbox": bbox,
                            "align": infer_alignment(bbox, image_size[0]),
                            "text": merged_text or block.text,
                            "line_boxes": [candidate.bbox for candidate in ordered_candidates],
                            "line_texts": [candidate.text.strip() for candidate in ordered_candidates if candidate.text.strip()],
                        }
                    )
                )
                continue
        snapped.append(block)
    return snapped


def build_localize_blocks(image_path: Path, source_image: Image.Image) -> list[TextBlock]:
    foreground_bbox = detect_foreground_bbox(source_image)
    raw_ocr_blocks = run_trocr_ocr_on_image(image_path)
    vision_line_blocks = refine_blocks_with_openai_vision(image_path, raw_ocr_blocks)
    if vision_line_blocks:
        semantic_tokens = annotate_ocr_tokens_with_vision(raw_ocr_blocks, vision_line_blocks)
        grouped_blocks = semantic_group_blocks(semantic_tokens, source_image.size)
        grouped_blocks = merge_centered_stacks(grouped_blocks, source_image.size)
        grouped_blocks = merge_translate_runs(grouped_blocks, source_image.size)
        grouped_blocks = append_missing_translate_ocr_blocks(grouped_blocks, semantic_tokens, source_image.size)
        return suppress_packaging_translation(grouped_blocks, source_image.size, foreground_bbox)

    raw_blocks = marketing_filter(raw_ocr_blocks)
    semantic_tokens = annotate_ocr_tokens_with_vision(raw_blocks, refine_blocks_with_openai_vision(image_path, raw_blocks))
    grouped_blocks = semantic_group_blocks(semantic_tokens, source_image.size)
    grouped_blocks = merge_centered_stacks(grouped_blocks, source_image.size)
    grouped_blocks = merge_translate_runs(grouped_blocks, source_image.size)
    grouped_blocks = append_missing_translate_ocr_blocks(grouped_blocks, semantic_tokens, source_image.size)
    return suppress_packaging_translation(grouped_blocks, source_image.size, foreground_bbox)


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
    if not any(marker in text for marker in ("Ã", "Ä", "Å", "Â")):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            repaired = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return repaired if repaired else text


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
    "alt": {"TR": "ESKİ", "FR": "ANCIEN", "DE": "ALT", "ES": "ANTERIOR", "IT": "PRIMA", "PT": "ANTIGO"},
    "old": {"TR": "ESKİ", "FR": "ANCIEN", "DE": "ALT", "ES": "ANTERIOR", "IT": "PRIMA", "PT": "ANTIGO"},
    "neu": {"TR": "YENİ", "FR": "NOUVEAU", "DE": "NEU", "ES": "NUEVO", "IT": "NUOVO", "PT": "NOVO"},
    "new": {"TR": "YENİ", "FR": "NOUVEAU", "DE": "NEU", "ES": "NUEVO", "IT": "NUOVO", "PT": "NOVO"},
    "shop now": {"TR": "HEMEN AL", "FR": "ACHETER MAINTENANT", "DE": "JETZT KAUFEN", "ES": "COMPRA AHORA", "IT": "ACQUISTA ORA", "PT": "COMPRE AGORA"},
}


def translate_with_gpt4o(blocks: list[TextBlock], languages: list[str]) -> dict[str, list[str]]:
    source = [block.text for block in blocks if block.translate]
    if not source:
        return {language: [] for language in languages}

    if not os.getenv("OPENAI_API_KEY"):
        return {language: source for language in languages}

    client = OpenAI()
    prompt = {
        "task": "Translate every source string into the requested target language as natural ad copy. Only keep protected brand/product tokens unchanged. Preserve [BOLD]...[/BOLD] tags exactly around the semantically emphasized phrase. Return one translated string per source string in the same order. Keep brand names, product names, packaging labels, URLs, QR references, Android TV, iPad Pro, H2O, Ask.NAOS.com, Dr.Scholl's and Scholl unchanged unless grammar absolutely requires surrounding words to change. Keep metric tokens exactly unchanged when they appear, including patterns like 24h, 48H, 84%, 88%, 2.4G+5G and Dual-Band. Translate surrounding claim language naturally.",
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
            translation_candidates = polish_translated_copy(block, translated, target_language)
            translated = translation_candidates[0]["text"]
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


def default_typography_style(block: TextBlock) -> dict[str, Any]:
    return {
        "fontFamily": block.font_family,
        "fontWeight": block.font_weight,
        "fontSize": block.font_size_estimate,
        "color": block.color if block.color.startswith("#") else "#111111",
        "opacity": 1.0,
        "letterSpacing": 0.0,
        "lineHeight": block.line_height_estimate,
        "alignment": block.align,
        "casing": "uppercase" if block.text.isupper() else "mixed",
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
                "style": style,
                "forceBreakAfter": item.get("forceBreakAfter", False),
                "color": style.get("color"),
                "fontSizeRatio": round(style["fontSize"] / max(1, base_style["fontSize"]), 3),
                "fontWeight": style.get("fontWeight"),
                "casing": style.get("casing"),
                "approximatePosition": round(len(spans) / total_segments, 3),
            }
        )
    return spans or [{"sourceText": text, "semanticRole": "benefit", "style": base_style}]


def infer_translated_style_spans(source_text: str, translated_text: str, source_style_spans: list[dict[str, Any]], block: TextBlock) -> list[dict[str, Any]]:
    segment_items = split_semantic_segments_with_breaks(translated_text)
    if not segment_items:
        return [{"translatedText": translated_text, "matchedSourceRole": "benefit", "style": default_typography_style(block)}]
    source_by_role: dict[str, dict[str, Any]] = {}
    for span in source_style_spans:
        source_by_role.setdefault(span["semanticRole"], span)
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
                "forceBreakAfter": item.get("forceBreakAfter", False),
                "color": style.get("color"),
                "fontSizeRatio": round(style["fontSize"] / max(1, default_typography_style(block)["fontSize"]), 3),
                "fontWeight": style.get("fontWeight"),
                "casing": style.get("casing"),
                "approximatePosition": round(len(translated_spans) / max(1, len(segment_items)), 3),
            }
        )
    return translated_spans


LANGUAGE_FILLER_WORDS: dict[str, set[str]] = {
    "TR": {"ile", "ve", "bir", "için", "olarak", "daha", "çok", "olan"},
    "DE": {"und", "mit", "für", "die", "der", "das"},
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
            if aggressiveness >= 2 and role == "headline" and len(words) > 4 and normalized in {"çok", "daha", "çoklu", "long", "lang", "mit", "ile"}:
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
        "TR": {"VE", "İLE", "DA", "DE"},
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
        shorter = maybe_preserve_uppercase(block.text, split_text_across_lines(faithful.replace("\n", " "), source_lines))
        compact = maybe_preserve_uppercase(block.text, split_text_across_lines(faithful.replace("\n", " "), max(2, min(source_lines, 4))))
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
    for token in re.findall(r"[A-Za-zÀ-ÿ0-9%]+", source_text):
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
        words = [match for match in re.finditer(r"[A-Za-zÀ-ÿ0-9%]+", line_text)]
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
        and multi_line
        and width >= image_size[0] * 0.38
        and area_ratio >= 0.07
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
        # HIGH risk — prefer mask-capable specialist or SDXL
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

    # LOW risk — OpenAI primary, HF fallback
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

    Compares the color distribution of the inpainted area (masked region in
    `composited`) to the surrounding border pixels in `source`. When the
    inpainted result diverges dramatically (hallucinated imagery, wrong product
    content, etc.) it blends the inpainted region back toward the local median
    background colour, suppressing the artifact while keeping clean inpaints
    unchanged.
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
        divergence = float(np.linalg.norm(inpainted_mean - border_median))

        if divergence <= divergence_threshold:
            return composited

        # Blend inpainted region back toward border median; stronger blend the
        # greater the divergence, capped at 0.85 to leave some inpaint detail.
        blend = min(0.85, (divergence - divergence_threshold) / 80.0)
        fill = np.full_like(comp_arr, border_median)
        comp_arr[mask_bool] = (1.0 - blend) * comp_arr[mask_bool] + blend * fill[mask_bool]
        return Image.fromarray(comp_arr.clip(0, 255).astype(np.uint8))
    except Exception:
        return composited


def run_openai_full_image_cleanup(image: Image.Image, editable_mask: Image.Image, prompt: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {"success": False, "provider": "openai", "failureReason": "openai_not_configured"}
    with tempfile.TemporaryDirectory(prefix="adaptifai-openai-full-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        rgba_mask = build_openai_edit_mask(editable_mask)
        image_path, mask_path, alpha_preview_path = export_openai_edit_debug_artifacts(temp_dir, image.convert("RGB"), rgba_mask)
        request_meta = validate_openai_edit_request(image_path, mask_path, image.convert("RGB"), rgba_mask)
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
                    model="gpt-image-1",
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
                "apiError": extract_openai_error_payload(exc, request_meta, "gpt-image-1"),
                "requestImagePreview": image.convert("RGB"),
                "requestMaskPreview": rgba_mask,
                "requestMaskAlphaPreview": rgba_mask.getchannel("A"),
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


def validate_openai_edit_request(image_path: Path, mask_path: Path, crop: Image.Image, rgba_mask: Image.Image) -> dict[str, Any]:
    alpha = np.array(rgba_mask.getchannel("A"))
    transparent_pixels = int((alpha == 0).sum())
    opaque_pixels = int((alpha == 255).sum())
    meta = {
        "model": "gpt-image-1",
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
    provider = os.getenv("ADAPTIFAI_GENERATIVE_CLEANUP_PROVIDER", "openai").lower()
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
    if provider in {"openai", "auto"} and os.getenv("OPENAI_API_KEY"):
        deduped_models = ["gpt-image-1"]
        prompt = prompt_override or (
            "Remove only the text pixels inside the transparent mask. Reconstruct only the masked text area using the surrounding background. "
            "Do not modify any unmasked pixels. Preserve all people, products, shoes, logos, packaging, lighting, shadows, textures and background outside the mask exactly. "
            "Do not add text. Do not add blur, panels, gradients, rectangles or new objects. "
            "Return the clean image only."
        )
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
                request_meta = validate_openai_edit_request(image_path, mask_path, request_crop, rgba_mask)
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
                    "model": deduped_models[0] if deduped_models else "gpt-image-1",
                    "attempted": True,
                    "success": False,
                    "rejected": True,
                    "rejectionReasons": [f"temporary directory failure: {exc.__class__.__name__}"],
                }
            )
            if provider == "openai":
                return attempts
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
        "Fill it with the exact same background colour and texture that borders the mask — do NOT generate new images, products, labels, or patterns. "
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
        "The result must look like the text was never there — only the background visible."
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


def get_font(size: int, bold: bool = False):
    font_names = (
        [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


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
    if not text:
        return []
    role = str(span.get("matchedSourceRole") or span.get("semanticRole") or "benefit")
    style = dict(span.get("style", {}))
    force_break_after = bool(span.get("forceBreakAfter", False))
    keep_whole = role in {"percentage", "numeric_claim", "condition_or_topic", "brand/product_name"}
    if keep_whole:
        return [{"text": text, "style": style, "role": role, "forceBreakAfter": force_break_after}]
    tokens = [{"text": token, "style": style, "role": role, "forceBreakAfter": False} for token in text.split() if token]
    if tokens:
        tokens[-1]["forceBreakAfter"] = force_break_after
    return tokens


def get_font_for_style(style: dict[str, Any], size: int):
    weight = int(style.get("fontWeight", 700))
    return get_font(size, bold=weight >= 700)


def span_token_width(draw: ImageDraw.ImageDraw, token: dict[str, Any], *, first_in_line: bool, scale_factor: float) -> float:
    base_size = max(10, int(token["style"].get("fontSize", 16)))
    size = max(10, int(round(base_size * scale_factor)))
    font = get_font_for_style(token["style"], size)
    text = token["text"] if first_in_line else f" {token['text']}"
    return text_width(draw, text, font)


def span_token_metrics(token: dict[str, Any], scale_factor: float) -> dict[str, Any]:
    base_size = max(10, int(token["style"].get("fontSize", 16)))
    size = max(10, int(round(base_size * scale_factor)))
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
) -> list[list[dict[str, Any]]]:
    tokens: list[dict[str, Any]] = []
    for span in spans:
        tokens.extend(tokenize_style_span(span))
    if not tokens:
        return []
    lines: list[list[dict[str, Any]]] = [[]]
    current_width = 0.0
    for token in tokens:
        width = span_token_width(draw, token, first_in_line=not lines[-1], scale_factor=scale_factor)
        if lines[-1] and current_width + width > max_width:
            lines.append([token])
            current_width = span_token_width(draw, token, first_in_line=True, scale_factor=scale_factor)
        else:
            lines[-1].append(token)
            current_width += width
        if token.get("forceBreakAfter") and lines[-1]:
            lines.append([])
            current_width = 0.0
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
        score = float(scale_factor * 1000 - overflow * 5 - max(0, len(lines) - max(1, len(spans))) * 12)
        candidate = {
            "scaleFactor": float(scale_factor),
            "lines": line_layouts,
            "widest": widest,
            "totalHeight": total_height,
            "overflow": overflow,
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
        layout = fit_styled_spans(draw, spans, box, base_typography)
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
        fallback_layout = fit_styled_spans(draw, fallback_spans, box, base_typography)
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


def render_styled_spans(
    draw: ImageDraw.ImageDraw,
    layout: dict[str, Any],
    box: tuple[int, int, int, int],
    *,
    alignment: str,
    style: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_height = int(layout.get("totalHeight", 0))
    y = box[1] + max(0, (box[3] - box[1] - total_height) // 2)
    rendered_spans: list[dict[str, Any]] = []
    span_render_boxes: list[dict[str, Any]] = []
    shadow = style.get("shadow") if style else None
    for line_index, line in enumerate(layout.get("lines", [])):
        line_width = int(line.get("lineWidth", 0))
        x = box[0] if alignment == "left" else box[0] + max(0, (box[2] - box[0] - line_width) // 2)
        baseline = y + int(line.get("maxAscent", 0))
        for token_index, token in enumerate(line.get("tokens", [])):
            token_text = token["text"] if token_index == 0 else f" {token['text']}"
            font = token["font"]
            fill = token["style"].get("color", "#111111")
            left, top, right, bottom = draw.textbbox((x, baseline - token["ascent"]), token_text, font=font)
            if shadow and shadow.get("enabled"):
                shadow_fill = shadow.get("color", "#000000")
                shadow_offset = int(shadow.get("offset", 2))
                draw.text((x + shadow_offset, baseline - token["ascent"] + shadow_offset), token_text, fill=shadow_fill, font=font)
            draw.text((x, baseline - token["ascent"]), token_text, fill=fill, font=font)
            rendered_spans.append(
                {
                    "lineIndex": line_index,
                    "text": token["text"],
                    "role": token["role"],
                    "style": token["style"],
                }
            )
            span_render_boxes.append(
                {
                    "lineIndex": line_index,
                    "text": token["text"],
                    "bbox": [left, top, right, bottom],
                }
            )
            x = right
        y += int(line.get("lineHeight", 0))
    return rendered_spans, span_render_boxes


def serialize_styled_layout(layout: dict[str, Any]) -> dict[str, Any]:
    serialized_lines: list[dict[str, Any]] = []
    for line in layout.get("lines", []):
        serialized_lines.append(
            {
                "lineWidth": line.get("lineWidth", 0),
                "lineHeight": line.get("lineHeight", 0),
                "maxAscent": line.get("maxAscent", 0),
                "maxDescent": line.get("maxDescent", 0),
                "tokens": [
                    {
                        "text": token.get("text", ""),
                        "size": token.get("size", 0),
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
        source_layout = fit_styled_spans(source_draw, renderable_source_spans, bbox, base_typography)
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
        "fontFamily": "approximate_system_match",
        "letterSpacing": "approximate_uniform",
        "multiStyleSpans": any(len(item.get("spans", [])) > 1 for item in source_style_spans_summary),
    }
    semantic_style_mapping = {
        "enabled": True,
        "mode": "semantic_role",
        "colorAware": True,
        "fontFamilyAware": "approximate_system_match",
        "letterSpacingAware": "approximate_uniform",
        "notes": "Semantic roles drive size/weight/casing transfer; font family and letter spacing use approximate system typography rather than pixel-perfect font identification.",
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

    strategy = "clean_replace"
    overlay_style: dict[str, Any] | str = "none"
    final_text_box = list(block.bbox)
    reason = "clean_replace chosen because cleanupConfidence high"

    if cleanup_confidence < 0.34 and (residual_risk > 0.48 or warning_penalty >= 0.16):
        if mean_overlap > 0.18 and safe_areas and safe_areas[0]["score"] >= 1.15:
            strategy = "reposition_to_safe_area"
            overlay_style = build_overlay_style_config(image, tuple(safe_areas[0]["bbox"]), strategy="reposition_to_safe_area", block=block, dominant_color=panel_color)
            final_text_box = safe_areas[0]["bbox"]
            reason = "reposition_to_safe_area chosen as last resort because natural cleanup failed"
        elif complexity > 90 or residual_risk > 0.58:
            strategy = "gradient_overlay"
            overlay_style = build_overlay_style_config(image, tuple(final_text_box), strategy="gradient_overlay", block=block, dominant_color=panel_color)
            reason = "gradient_overlay chosen as last resort because natural cleanup failed"
        elif mean_overlap > 0.12:
            strategy = "soft_blur_panel"
            overlay_style = build_overlay_style_config(image, tuple(final_text_box), strategy="soft_blur_panel", block=block, dominant_color=panel_color)
            reason = "soft_blur_panel chosen as last resort because natural cleanup failed"
        else:
            strategy = "overlay_panel"
            overlay_style = build_overlay_style_config(image, tuple(final_text_box), strategy="overlay_panel", block=block, dominant_color=panel_color)
            reason = "overlay_panel chosen as last resort because natural cleanup failed"

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


def union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


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
    temp_path = temp_root() / f"{uuid4().hex}.png"
    try:
        source.save(temp_path, "PNG")
        detector_blocks = marketing_filter(run_trocr_ocr_on_image(temp_path))
    except Exception:
        detector_blocks = []
    finally:
        temp_path.unlink(missing_ok=True)
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


# Meta and TikTok placement prefixes — these platforms render their own CTA buttons
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

        # ── Step 1: OCR ───────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        source.save(tmp_path, "PNG")
        try:
            text_blocks = build_localize_blocks(tmp_path, source)
        except Exception:
            text_blocks = []
        finally:
            tmp_path.unlink(missing_ok=True)

        has_translatable = any(b.translate for b in text_blocks)

        # ── Step 2: CTA suppression for native-CTA platforms ─────────────
        working_source = source
        if _placement_has_native_cta(placement_id) and text_blocks:
            cta_bbox = _detect_cta_button_bbox(source, text_blocks)
            if cta_bbox is not None:
                working_source = _suppress_cta_button(source, cta_bbox)
                # Also remove the CTA block from text_blocks so it isn't re-rendered
                text_blocks = [b for b in text_blocks if b.role != "cta" and classify_text_role(b.text) != "cta"]
                has_translatable = any(b.translate for b in text_blocks)

        # ── Step 3: Detect foreground / subject region ───────────────────
        focus_bbox = build_resize_focus_bbox(working_source)
        fx0, fy0, fx1, fy1 = focus_bbox
        fg_w = fx1 - fx0
        fg_h = fy1 - fy0

        # ── Step 4: Build clean background (text removed) ────────────────
        if has_translatable:
            try:
                cleaned, cleanup_debug = build_clean_background(working_source, text_blocks, cleanup_strength=100, return_debug=True)  # type: ignore[misc]
                gated, _ = enforce_localize_cleanup_gate(working_source, cleaned, text_blocks, cleanup_debug)
                bg_base = gated if gated is not None else cleaned
            except Exception:
                bg_base = working_source
        else:
            bg_base = working_source

        # ── Step 4b: Scale background to COVER the target canvas ─────────
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

        # ── Step 5: Scale and reposition foreground subject ───────────────
        #
        # Goal: the foreground keeps roughly the same visual size proportion
        # relative to the new canvas, and its relative anchor (left/center/right,
        # top/center/bottom) is preserved.
        #
        # For extreme ratio changes (e.g. 4:1 → 9:16) the foreground is
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
        # Hard cap: don't upscale beyond 2.2× (avoids extreme pixellation)
        fg_scale = min(fg_scale, 2.2)

        fg_crop = working_source.crop(focus_bbox)
        new_fg_w = max(1, int(fg_w * fg_scale))
        new_fg_h = max(1, int(fg_h * fg_scale))
        fg_resized = fg_crop.resize((new_fg_w, new_fg_h), Image.Resampling.LANCZOS)

        # Anchor: use the relative center of the foreground in the source
        rel_cx = (fx0 + fx1) / 2 / src_w
        rel_cy = (fy0 + fy1) / 2 / src_h
        target_cx = int(rel_cx * tgt_w)
        target_cy = int(rel_cy * tgt_h)

        # For wide→tall: bias subject slightly downward so it feels grounded
        src_ratio = src_w / max(1, src_h)
        tgt_ratio = tgt_w / max(1, tgt_h)
        if src_ratio > tgt_ratio * 1.4:
            target_cy = int(min(0.65, rel_cy + 0.08) * tgt_h)

        paste_x = max(0, min(tgt_w - new_fg_w, target_cx - new_fg_w // 2))
        paste_y = max(0, min(tgt_h - new_fg_h, target_cy - new_fg_h // 2))
        canvas.paste(fg_resized, (paste_x, paste_y))

        # ── Step 6: Re-render text at proportionally scaled positions ─────
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
                    canvas = render_translated_text(canvas, scaled_blocks, render_plan={})
            except Exception:
                pass

        return canvas

    except Exception:
        # Full fallback: focus-aware cover crop
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
    "meta-feed": "facebook-feed",
    "meta-marketplace": "facebook-marketplace",
    "meta-right-column": "facebook-right-column",
    "meta-stories": "instagram-story",
    "tiktok-branded": "tiktok-branded-content",
    "youtube-16x9": "youtube-instream",
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
        "price": "€39.90",
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
                draw.text((left_center[0], left_center[1] - 2), "‹", fill="white", font=get_font(max(16, arrow_radius), bold=True), anchor="mm")
                draw.text((right_center[0], right_center[1] - 2), "›", fill="white", font=get_font(max(16, arrow_radius), bold=True), anchor="mm")
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


def source_suffix(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
    return digest


def localize_filename(source_path: Path, language: str, output_format: str) -> str:
    extension = normalize_output_format(output_format, source_path)
    return f"{sanitize_stem(source_path)}-{language.lower()}.{extension}"


def resize_filename(source_path: Path, placement_id: str, output_format: str) -> str:
    extension = normalize_output_format(output_format, source_path)
    return f"{sanitize_stem(source_path)}-{placement_id}.{extension}"


def build_localize_assets(paths: list[Path], languages: list[str], output_format: str, job_dir: Path) -> tuple[list[OutputAsset], dict[str, list[str]], list[dict[str, Any]]]:
    outputs: list[OutputAsset] = []
    translations_summary: dict[str, list[str]] = {}
    manifest_assets: list[dict[str, Any]] = []

    for image_path in image_paths(paths):
        source_image = Image.open(image_path).convert("RGB")
        preprocessed_image = preprocess_image_for_localize(source_image)
        temp_input_path = job_dir / f"{sanitize_stem(image_path)}-{source_suffix(image_path)}-preprocessed.png"
        preprocessed_image.save(temp_input_path, "PNG")
        raw_ocr_blocks = run_trocr_ocr_on_image(temp_input_path)
        blocks = build_localize_blocks(temp_input_path, preprocessed_image)
        source_language = detect_source_language(blocks)
        translated_by_language = translate_with_gpt4o(blocks, languages)
        for language, translated_strings in translated_by_language.items():
            translations_summary[language] = translated_strings
            translated_blocks, editor_text = apply_translations(blocks, translated_strings, language)
            grouping_audit = analyze_semantic_grouping(translated_blocks, source_image.size)
            background, cleanup_debug = build_clean_background(source_image, translated_blocks, cleanup_strength=100, return_debug=True)
            gated_background, cleanup_gate = enforce_localize_cleanup_gate(source_image, background, translated_blocks, cleanup_debug)

            # ─── Localization Protocol V2.2 ───
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
            # ─── End Protocol V2.2 ───

            foreground_bbox = detect_foreground_bbox(source_image)
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
            rendered: Image.Image | None = None
            render_base = gated_background if gated_background is not None else source_image
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


def build_resize_assets(paths: list[Path], placement_ids: list[str], output_format: str, custom_width: int | None, custom_height: int | None, job_dir: Path, creative_modes: dict[str, str] | None = None) -> tuple[list[OutputAsset], list[dict[str, Any]]]:
    outputs: list[OutputAsset] = []
    manifest_assets: list[dict[str, Any]] = []
    canonical_placement_ids = [canonical_placement_id(item) for item in placement_ids]
    resolved_creative_modes = {canonical_placement_id(key): value for key, value in (creative_modes or {}).items()}
    resize_mode = os.getenv("ADAPTIFAI_RESIZE_FIT", "cover")
    parity_report = build_preview_template_parity_report(canonical_placement_ids)
    parity_report_filename = "previewTemplateParityReport.json"
    (job_dir / parity_report_filename).write_text(json.dumps(parity_report, ensure_ascii=False, indent=2), encoding="utf-8")

    source_entries: list[dict[str, Any]] = []
    for image_path in image_paths(paths):
        source_image = Image.open(image_path).convert("RGB")
        source_entries.append(
            {
                "path": image_path,
                "image": source_image,
                "focus_bbox": build_resize_focus_bbox(source_image),
            }
        )

    rendered_assets_by_placement: dict[str, list[dict[str, Any]]] = {}
    for canonical_id in canonical_placement_ids:
        width, height = resolve_resize_dimensions(canonical_id, custom_width, custom_height)
        placement_assets: list[dict[str, Any]] = []
        for source_entry in source_entries:
            try:
                rendered = smart_resize_image(source_entry["image"], width, height, placement_id=canonical_id)
            except Exception:
                rendered = render_resize_image(
                    source_entry["image"],
                    width,
                    height,
                    fit=resize_mode,
                    focus_bbox=source_entry["focus_bbox"],
                )
            placement_assets.append(
                {
                    "source_name": source_entry["path"].name,
                    "rendered": rendered,
                    "source_image": source_entry["image"],
                    "focus_bbox": source_entry["focus_bbox"],
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
            active_index = 0 if creative_mode == "single" else next((index for index, item in enumerate(rendered_set) if item["source_name"] == image_path.name), 0)
            rendered = rendered_set[active_index]["rendered"]
            filename = resize_filename(image_path, canonical_id, output_format)
            save_image_output(rendered, job_dir / filename, normalize_output_format(output_format, image_path))
            raw_asset_filename = f"debug-{sanitize_stem(image_path)}-{canonical_id}-raw-localized-asset.png"
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
            preview_metadata["unusedAssets"] = [item["source_name"] for item in rendered_set[1:]] if creative_mode == "single" and len(rendered_set) > 1 else []
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
                "cropBox": compute_resize_crop_box(source_image, focus_bbox, width / max(1, height)) if resize_mode == "cover" else None,
                "focusBox": list(focus_bbox),
                "protectedObjects": [list(focus_bbox)],
                "safeZoneWarnings": safe_zone_warnings_for(canonical_id),
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
                safe_zone_warnings=safe_zone_warnings_for(canonical_id),
                download_url=f"/api/download/{job_dir.name}/{filename}",
                source_name=image_path.name,
                language=None,
                translated_text="",
                extracted_blocks=[],
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
        # Color override — only apply when a non-default colour was chosen
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
    Shear ~14° (tan ≈ 0.25): top of each block region leans to the right.
    PIL AFFINE coefficients (a,b,c,d,e,f): src_x = a*dst_x + b*dst_y + c
    For forward-lean: src_x = dst_x + shear*dst_y  →  (a=1, b=shear, c=0, d=0, e=1, f=0)
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
    rendered = render_resize_image(source_image, int(asset_meta["width"]), int(asset_meta["height"]), edit.fit, edit.scale, edit.x, edit.y, focus_bbox=focus_bbox)
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
    cleanup_old_temp_files()
    return {
        "status": "ok",
        "storage": "stateless-temp-24h",
        "device": torch_device(),
        "ocr_engine": os.getenv("ADAPTIFAI_OCR_ENGINE", "easyocr"),
        "ocr_max_side": os.getenv("ADAPTIFAI_OCR_MAX_SIDE", "1280" if is_cpu_runtime() else "2200"),
        "ocr": os.getenv("ADAPTIFAI_TROCR_MODEL", "microsoft/trocr-base-printed"),
        "inpainting_backend": os.getenv("ADAPTIFAI_INPAINT_BACKEND", "stable-diffusion"),
        "inpainting": os.getenv("ADAPTIFAI_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"),
        "resize_fit": os.getenv("ADAPTIFAI_RESIZE_FIT", "cover"),
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

    cleanup_old_temp_files()
    job_id = uuid4().hex[:12]
    job_dir = temp_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = await persist_uploads(files, job_dir)
    uploaded_images = image_paths(saved_paths)
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

    if resolved_mode == "localize":
        outputs, translations, manifest_assets = build_localize_assets(uploaded_images, languages, output_format, job_dir)
        extracted_blocks = outputs[0].extracted_blocks if outputs else []
    else:
        outputs, manifest_assets = build_resize_assets(uploaded_images, placement_ids, output_format, custom_width, custom_height, job_dir, creative_modes=parsed_creative_modes)
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
