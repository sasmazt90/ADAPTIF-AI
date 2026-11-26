# AdaptifAI Backend

FastAPI service for the stateless creative localization pipeline.

## Local setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/warm_models.py
python -m uvicorn app.main:app --reload --port 8000
```

The Next.js app proxies `/api/adapt` to `http://127.0.0.1:8000` by default.

## Localization pipeline v7 (default)

`app/localize_v7.py` replaces the heuristic v5/v6 polygon path (`ADAPTIFAI_LOCALIZE_PIPELINE=legacy` restores it).

1. **Geometry** - Google Vision when credentials exist, otherwise EasyOCR. Only the line boxes are trusted.
2. **Semantics** - the vision model (`OPENAI_TRANSLATION_MODEL`, override with `ADAPTIFAI_LOCALIZE_VISION_MODEL`) sees the
   creative plus a numbered set-of-marks overlay, groups boxes into copy elements, reads the real text from pixels,
   keeps logos / logo taglines / product text untouched and transcreates the copy line by line.
3. **Fit + review** - a dry layout measures which lines would have to shrink; one native-speaker review pass fixes
   unnatural phrasing and shortens lines within a character budget (naturalness wins).
4. **Removal** - per line, a robust polynomial background fit on a ring around the text, glyph segmentation against it
   and analytic fill (flat panels, buttons, gradients). Textured backgrounds go to Replicate LaMa, then OpenCV.
5. **Typography** - size, weight and width are measured from the source glyphs and matched with variable fonts
   (`app/fonts`: Archivo wdth/wght, Noto Serif; OFL). Copy is set on the source baselines inside the layout grid.

Debug artifacts per output: `*-v7-marks.png`, `*-v7-mask.png`, `*-v7-clean.png`, `*-v7-analysis.json`.

## Pipeline contract

1. Saves uploads to an OS temp directory under `adaptifai/<job-id>`.
2. Runs EasyOCR text detection and Microsoft TrOCR recognition.
3. Filters marketing headlines and CTAs while protecting product labels.
4. Calls GPT-4o when `OPENAI_API_KEY` is configured and preserves `[BOLD]...[/BOLD]` emphasis.
5. Runs provider-based cleanup for localization:
   - large marketing headlines default to `lama` on the full image with one combined text mask
   - `openai` is optional and no longer the default primary provider for large headline removal
   - `sdxl_controlnet` is an optional refinement path
   - if the requested provider is not installed/configured the backend returns `provider_not_configured` and does not silently fall back
6. Produces placement-aware resize metadata and optional raster exports.

Temporary files are eligible for deletion after 24 hours and no user/session state is stored.

## Production cleanup providers

### LaMa (`cleanup_provider=lama`)

The large-headline localization path is designed to use LaMa as the primary full-image inpainting provider.

Current integration contract:
- provider name: `lama`
- input mode: full image
- mask: one combined semantic headline mask
- if LaMa is unavailable the backend returns `lama_not_configured`

Suggested setup:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install simple-lama-inpainting
```

### SDXL + ControlNet (`cleanup_provider=sdxl_controlnet`)

Optional provider for structure-preserving refinement.

Required environment variables:

```powershell
$env:ADAPTIFAI_SDXL_INPAINT_MODEL='your-sdxl-inpaint-model'
$env:ADAPTIFAI_SDXL_CONTROLNET_MODEL='your-controlnet-canny-or-softedge-model'
```

The current code exposes the provider path and failure reporting, but if the model ids are not configured it will return `sdxl_controlnet_not_configured`.

### Protected region abstraction

The backend now exposes a protected-region abstraction intended for YOLO / GroundingDINO-driven masks.

Current behavior:
- requested backend comes from `ADAPTIFAI_PROTECTED_REGION_BACKEND`
- if YOLO / GroundingDINO is not installed, the backend reports the provider as unavailable
- the current runtime falls back to the existing heuristic protected mask while still reporting configuration status honestly
