from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import vertexai
from google.api_core.exceptions import PermissionDenied
from google.oauth2 import service_account
from vertexai.preview.vision_models import ImageGenerationModel


DEFAULT_PROJECT_ID = "adaptif-ai-1780483603776"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "imagen-3.0-generate-002"
DEFAULT_CREDENTIALS = Path("google/adaptif-ai-1780483603776-8020de4fa75d.json")
DEFAULT_OUTPUT_DIR = Path("ADAPTIFAI TEST IMAGES/vertex-output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local test image with Vertex AI Imagen.")
    parser.add_argument("--prompt", required=True, help="Text prompt to send to Imagen.")
    parser.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS))
    parser.add_argument("--project", default=os.getenv("VERTEX_AI_PROJECT_ID", DEFAULT_PROJECT_ID))
    parser.add_argument("--location", default=os.getenv("VERTEX_AI_LOCATION", DEFAULT_LOCATION))
    parser.add_argument("--model", default=os.getenv("VERTEX_IMAGEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=os.getenv("VERTEX_IMAGEN_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--filename", default="vertex-imagen-test.png")
    parser.add_argument("--aspect-ratio", default="1:1", choices=["1:1", "3:4", "4:3", "9:16", "16:9"])
    return parser.parse_args()


def save_generated_image(image: object, target: Path) -> None:
    if hasattr(image, "save"):
        image.save(str(target))
        return
    encoded = getattr(image, "_image_bytes", None) or getattr(image, "image_bytes", None)
    if encoded:
        target.write_bytes(encoded)
        return
    as_dict = image if isinstance(image, dict) else getattr(image, "__dict__", {})
    b64 = as_dict.get("bytesBase64Encoded") or as_dict.get("bytes_base64_encoded")
    if b64:
        target.write_bytes(base64.b64decode(b64))
        return
    raise RuntimeError(f"Could not save Imagen response: {type(image)!r}")


def main() -> None:
    args = parse_args()
    credentials_path = Path(args.credentials).expanduser()
    if not credentials_path.exists():
        raise FileNotFoundError(f"Service account JSON not found: {credentials_path}")

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_path),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    vertexai.init(project=args.project, location=args.location, credentials=credentials)

    try:
        model = ImageGenerationModel.from_pretrained(args.model)
        images = model.generate_images(
            prompt=args.prompt,
            number_of_images=1,
            aspect_ratio=args.aspect_ratio,
            safety_filter_level="block_some",
            person_generation="allow_adult",
        )
    except PermissionDenied as exc:
        raise SystemExit(
            "Vertex AI rejected the request. Confirm billing is enabled for the Google Cloud project, "
            "Vertex AI API is enabled, and the service account has Vertex AI User permissions.\n"
            f"Original error: {exc}"
        ) from exc
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / args.filename
    save_generated_image(images[0], target)

    manifest = {
        "project": args.project,
        "location": args.location,
        "model": args.model,
        "prompt": args.prompt,
        "aspect_ratio": args.aspect_ratio,
        "output": str(target.resolve()),
    }
    (output_dir / f"{target.stem}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(target.resolve()))


if __name__ == "__main__":
    main()
