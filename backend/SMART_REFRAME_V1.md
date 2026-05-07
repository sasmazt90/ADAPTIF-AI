# Smart Reframe V1 Contract

Smart Reframe V1 adapts finished ad creatives into new aspect ratios without
blind cropping. The first supported target families are:

- Vertical and square social formats: `9:16`, `4:5`, `1:1`
- Landscape and wide social/video formats: `1.91:1`, `16:9`
- Narrow banners: `728x90`, `320x50`
- Large rectangles and skyscrapers: `300x250`, `300x600`, `160x600`

The analysis provider can be OpenAI or Gemini. It must return JSON matching the
`VisualAnalysis` models in `app/smart_reframe.py`.

## Visual Analysis JSON

Coordinates are normalized as `[ymin, xmin, ymax, xmax]` on a 0-1000 grid.
The backend model represents these as named fields.

```json
{
  "schema_version": "smart-reframe-v1",
  "source_width": 1080,
  "source_height": 1080,
  "background": {
    "type": "photographic",
    "dominant_color_rgb": { "r": 224, "g": 231, "b": 238 },
    "is_gradient": false,
    "texture_complexity": 0.62,
    "can_extend_without_ai": false
  },
  "product_layers": [
    {
      "id": "product-1",
      "role": "product",
      "bbox": { "ymin": 480, "xmin": 470, "ymax": 900, "xmax": 980 },
      "confidence": 0.88,
      "saliency": 0.92,
      "protected": true,
      "mask_quality": "bbox_only",
      "needs_shadow": true
    }
  ],
  "text_layers": [
    {
      "id": "headline-1",
      "role": "marketing_text",
      "bbox": { "ymin": 80, "xmin": 60, "ymax": 420, "xmax": 840 },
      "confidence": 0.91,
      "saliency": 0.8,
      "protected": true,
      "original_text": "VERSORGT DEINE FUSSE",
      "translated_text": "AYAKLARINA FERAHLIK SAGLAR",
      "translate": true,
      "text_style": {
        "color_rgb": { "r": 6, "g": 38, "b": 98 },
        "is_bold": true,
        "font_type": "sans-serif",
        "uppercase": true,
        "alignment": "left"
      }
    }
  ],
  "logo_layers": [],
  "other_layers": [],
  "saliency_summary": "Main attention is headline, spray product, and shoe.",
  "quality_warnings": []
}
```

## Background Routing

`choose_expansion_strategy()` chooses the cheapest safe background expansion:

- `pillow_extend`: solid, gradient, or soft blur backgrounds.
- `opencv_inpaint`: moderately complex backgrounds.
- `openai_outpaint`: photographic, textured, patterned, or high complexity backgrounds.
- `blurred_fit_fallback`: source and target ratios are close, or final fallback.

## Renderers

- `openai_outpaint`: real OpenAI Images Edit request, used only when the plan
  marks generative expansion as necessary.
- `hybrid_relayout`: local Pillow renderer for narrow banners.
- `large_rectangle_relayout`: local Pillow renderer for `160x600` and `300x600`
  placements. It keeps one integrated vertical creative: product remains sharp
  in the lower half, the upper copy area is filled from clean source texture
  or inpainted background, and text never overlaps the product.
- `pillow_extend` / `opencv_inpaint`: local deterministic routes; currently
  rendered through the safe blurred-fit compositor until deeper local infill is
  promoted.

Set `ADAPTIFAI_DEBUG_STRATEGY_FILENAMES=1` to include bucket and strategy in
debug resized asset filenames. Production defaults to clean debug names while
keeping `renderMeta`, `smartReframePlan`, and `visualAnalysis` in JSON logs.

## Analysis Prompt

Use `build_visual_analysis_prompt(target_language)` to generate the provider
prompt. The prompt asks for JSON only and tells the model to classify:

- product/person foreground
- marketing text and CTA
- logos and product label text
- background type and texture complexity
- protected layers and safe-to-move layers
