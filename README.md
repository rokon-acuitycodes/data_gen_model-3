# Unified Data & Image Generator (FastAPI)

This project provides a FastAPI backend for generating:

- Image variants from an uploaded image
- Image-to-image editing from an uploaded reference image
- Synthetic data files from uploaded CSV/XLSX/DOCX/PDF
- Plain-text outputs from uploaded TXT/CSV/XLSX/DOCX/PDF files
- Image from text prompt

It replaces the previous Streamlit UI flow and exposes API endpoints with OpenAPI/Swagger for testing.

## Features

- FastAPI REST API with Swagger docs at `/docs`
- Bundle (ZIP) output for generated files/images
- Per-item output links in `image_results` and `file_results`
- Optional S3 upload with presigned URLs
- Fallback to base64 payload when S3 is unavailable

## Project Structure

- `generator_all/ui/app.py` - FastAPI app and endpoints
- `generator_all/requirements.txt` - Python dependencies
- `serve_modal_new.py` - Modal deployment entrypoint (current primary)
- `serve_modal.py`, `serve_modal2.py` - older Modal entry variants
- `src/` - generation models and pipeline code
- `generator_all/utils/` - S3 and config helpers

## Requirements

Install from:

`generator_all/requirements.txt`

Main dependencies include:

- `fastapi`, `uvicorn[standard]`, `python-multipart`
- `torch`, `transformers`, `diffusers`, `accelerate`
- `torchaudio`, `torchcodec`
- `pandas`, `openpyxl`, `python-docx`, `pypdf`, `reportlab`
- `boto3`, `openai`, `python-dotenv`

## Environment Variables

Set these as needed:

- `OPENAI_API_KEY` (required for auto image captioning)
- `HF_TOKEN` (optional; only needed when you explicitly enable auth for gated/private Hugging Face models)
- `AWS_ACCESS_KEY_ID` (optional, for S3)
- `AWS_SECRET_ACCESS_KEY` (optional, for S3)
- `S3_BUCKET_NAME` (optional, for S3)
- `AWS_REGION` (optional, default `us-east-2`)

If S3 variables are not set/valid, API still works using base64 response fallback.
The default image model is `black-forest-labs/FLUX.2-klein-4B`, which is public and does not require Hugging Face license acceptance. `serve_modal_new.py` disables Hugging Face auth for image downloads by default so an expired cached token does not break public model fetches.

## Run Locally

From project root:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS/WSL
# venv\Scripts\activate   # Windows PowerShell/CMD
pip install -r generator_all/requirements.txt
uvicorn generator_all.ui.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- Health: `http://localhost:8000/health`

## Run on Modal

### Serve (dev)

```bash
modal serve serve_modal_new.py
```

### Deploy

```bash
modal deploy serve_modal_new.py
```

After deploy, use the generated Modal URL:

- Swagger UI: `<your-modal-url>/docs`
- OpenAPI JSON: `<your-modal-url>/openapi.json`

`serve_modal_new.py` is configured for `black-forest-labs/FLUX.2-klein-4B` on an `L4` worker so the image model can be downloaded from Hugging Face into the Modal cache volume and run with lower latency and lower cost.

## API Endpoints

## 1) `GET /health`

Response:

```json
{
  "status": "ok",
  "device": "cuda"
}
```

## 2) `POST /api/generate/image-from-file`

Form fields:

- `file` (`jpg|jpeg|png`)
- `num_files` (int, default `10`)
- `custom_caption` (optional)
- query `mode=json|zip` (default `json`)

`mode=json` response (S3 enabled):

```json
{
  "mode": "json",
  "caption": "...",
  "variations": ["...", "..."],
  "images_count": 2,
  "zip_download_url": "https://.../bundle.zip",
  "image_results": [
    {
      "image_index": 1,
      "variation": "...",
      "file_url": "https://.../image_1.png"
    }
  ]
}
```

Notes:

- `zip_download_url` is the ZIP bundle URL.
- `image_results[*].file_url` is per-image URL.
- If S3 is unavailable, response includes `zip_base64` and per-item `image_base64`/`file_url` as data URI.

## 3) `POST /api/generate/data-from-file`

Form fields:

- `file` (`csv|xlsx|docx|pdf`)
- `num_files` (int, default `10`)
- query `mode=json|zip` (default `json`)

`mode=json` response (S3 enabled):

```json
{
  "mode": "json",
  "file_type": "csv",
  "files": [
    {"filename": "generated_1_input.csv", "size_bytes": 1234}
  ],
  "zip_download_url": "https://.../bundle.zip",
  "file_results": [
    {
      "filename": "generated_1_input.csv",
      "size_bytes": 1234,
      "file_url": "https://.../generated_1_input.csv"
    }
  ]
}
```

## 4) `POST /api/generate/image-from-text`

Form fields:

- `prompt` (required)
- query `mode=json|zip` (default `json`)

`mode=json` response:

```json
{
  "mode": "json",
  "prompt": "a portrait of...",
  "images_count": 1,
  "download_url": "https://.../text_generated_image.zip"
}
```

Notes:

- The active text-to-image backend is `black-forest-labs/FLUX.2-klein-4B`.
- Modal defaults use the official 4-step FLUX.2 klein recipe for faster generation.

## 5) `POST /api/generate/text-from-file`

Form fields:

- `file` (`txt|csv|xlsx|docx|pdf`)
- `num_outputs` (int, default `3`)
- `max_length` (int, default `200`)
- query `mode=json|zip` (default `json`)

`mode=json` response:

```json
{
  "mode": "json",
  "file_type": "pdf",
  "texts_count": 3,
  "text_results": [
    {
      "filename": "generated_text_1_input.txt",
      "text": "Rewritten plain-text output...",
      "file_url": "https://.../generated_text_1_input.txt"
    }
  ]
}
```

## 6) `POST /api/generate/image-to-image`

Form fields:

- `file` (`jpg|jpeg|png`)
- `prompt` (required)
- `num_outputs` (int, default `1`, max `4`)
- `num_inference_steps` (optional)
- `guidance_scale` (optional)
- `seed` (optional)
- query `mode=json|zip` (default `json`)

`mode=json` response:

```json
{
  "mode": "json",
  "prompt": "turn this product shot into a premium studio ad",
  "images_count": 1,
  "zip_download_url": "https://.../image_to_image_bundle.zip",
  "image_results": [
    {
      "image_index": 1,
      "file_url": "https://.../image_to_image_1.png"
    }
  ]
}
```

Notes:

- This endpoint uses the uploaded image as FLUX.2 klein 4B conditioning input.
- The uploaded image is resized to a non-stretched 1024x1024 canvas before generation.

## Response Mode Rules

- `mode=zip` returns binary ZIP stream (`application/zip`)
- `mode=json` returns metadata + URLs/base64

## Video Voice Sync

`POST /api/generate/video-from-image` also accepts:

- `audio_file` (optional)
- `sync_to_audio_duration` (default `true`)

When `audio_file` is provided, the video generator switches to an audio-conditioned LTX-2 pipeline for lip-sync and attempts to match the clip length to the audio duration.
This runtime path also requires `torchcodec` and FFmpeg because current `torchaudio.load(...)` uses TorchCodec-backed decoding in the deployed image.

## Common Notes

- Large model initialization can take time on first run.
- For consistent API behavior in production, prefer setting valid S3 credentials.
- Caption generation uses OpenAI; custom caption avoids that API call.
- File-to-text generation uses `google/flan-t5-small` through the shared `T5Model` wrapper.

## Troubleshooting

- `ModuleNotFoundError: generator_all` on Modal:
  - Ensure `generator_all/__init__.py` and `generator_all/ui/__init__.py` exist.
- Modal server startup timeout:
  - Ensure Uvicorn binds to `0.0.0.0:8000`.
- Missing Swagger:
  - Use `/docs` path on the correct base URL.
