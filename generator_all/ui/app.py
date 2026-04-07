import base64
import io
import os
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse


# ----------------------------
# Path setup (matches Streamlit)
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
src_path = os.path.join(project_root, "src")
utils_path = os.path.join(current_dir, "..")

# Ensure generator + util modules import correctly
sys.path.insert(0, src_path)
sys.path.insert(0, utils_path)


# ----------------------------
# HuggingFace cache location
# ----------------------------
if os.path.exists("d:/"):
    os.environ.setdefault("HF_HOME", "d:/huggingface_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "d:/huggingface_cache/transformers")
else:
    # Respect container-provided cache locations such as Modal volumes.
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/transformers")


app = FastAPI(
    title="Unified Data & Image Generator API",
    version="1.0.0",
    description="FastAPI replacement for the former Streamlit UI. Endpoints are testable via OpenAPI at /docs.",
)


class NamedBytesIO(io.BytesIO):
    """BytesIO with a `name` attribute (some generators use original_file.name)."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _get_file_ext(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_s3_manager():
    from utils.s3_storage import S3Manager
    from utils.config import s3_config

    try:
        s3_config.validate()
        return S3Manager()
    except Exception:
        return None


@lru_cache(maxsize=4)
def get_caption_generator(_device: str):
    # Device is kept for compatibility with the previous Streamlit wrapper.
    from generators.caption_generator import CaptionGenerator

    return CaptionGenerator(device=_device)


@lru_cache(maxsize=2)
def get_variation_generator():
    from generators.variation_generator import VariationGenerator

    return VariationGenerator()


@lru_cache(maxsize=2)
def get_image_generator(device: str):
    from generators.image_generator import ImageGenerator

    return ImageGenerator(device)


@lru_cache(maxsize=2)
def get_t5_model(device: str):
    from models.t5 import T5Model

    return T5Model(device=device)


@lru_cache(maxsize=1)
def get_video_generator(device: str):
    from generators.video_generator import VideoGenerator

    return VideoGenerator(device)



def images_to_zip_bytes(images: List[Any]) -> bytes:
    """Zip PIL Images into an in-memory ZIP (PNG format)."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            zip_file.writestr(f"image_{i+1}.png", img_byte_arr.getvalue())
    return zip_buffer.getvalue()

def pil_image_to_png_bytes(img: Any) -> bytes:
    """Convert a PIL Image to PNG bytes for S3 upload / inline base64."""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def files_to_zip_bytes(files: List[Tuple[str, bytes]]) -> bytes:
    """Zip (filename, bytes) pairs into an in-memory ZIP."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in files:
            zip_file.writestr(filename, data)
    return zip_buffer.getvalue()


def _content_type_for_ext(file_ext: str) -> str:
    if file_ext == "csv":
        return "text/csv"
    if file_ext == "xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if file_ext == "docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if file_ext == "pdf":
        return "application/pdf"
    return "application/octet-stream"


def make_data_uri(content_type: str, b64: str) -> str:
    """Create a `data:` URI string for UI rendering when S3 is not configured."""
    return f"data:{content_type};base64,{b64}"


def prepare_video_condition_image(image: Any, target_width: int = 768, target_height: int = 512):
    """Resize without stretching the source image, using a soft blurred background fill."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")

    background = ImageOps.fit(
        image,
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    background = background.filter(ImageFilter.GaussianBlur(radius=18))
    background = ImageEnhance.Brightness(background).enhance(0.75)

    foreground = ImageOps.contain(
        image,
        (target_width, target_height),
        method=Image.Resampling.LANCZOS,
    )

    canvas = background.copy()
    offset = (
        (target_width - foreground.width) // 2,
        (target_height - foreground.height) // 2,
    )
    canvas.paste(foreground, offset)
    return canvas


@app.get("/health")
def health():
    return {"status": "ok", "device": _device()}


@app.post("/api/generate/image-from-file")
async def generate_image_from_file(
    file: UploadFile = File(...),
    num_files: int = Form(10),
    custom_caption: str = Form(""),
    mode: Literal["json", "zip"] = Query("json", description="Return JSON metadata or the ZIP bytes."),
):
    file_ext = _get_file_ext(file.filename or "")
    if file_ext not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use jpg/jpeg/png.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    from PIL import Image
    import torch

    # Optional: downscale large images to conserve memory
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    max_dim = 1024
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Caption
    caption: Optional[str] = None
    if custom_caption and custom_caption.strip():
        caption = custom_caption.strip()
        variations = [caption]
    else:
        with torch.no_grad():
            caption_gen = get_caption_generator(device)
            caption = caption_gen.generate_caption(image)
            if not caption:
                raise HTTPException(status_code=500, detail="Failed to generate caption.")

        with torch.no_grad():
            var_gen = get_variation_generator()
            variations = var_gen.generate_variations(caption)

            # Trim or extend variations to match num_files
            if len(variations) < num_files:
                variations = (variations * (num_files // len(variations) + 1))[:num_files]
            else:
                variations = variations[:num_files]

    # 2) Image generation
    with torch.no_grad():
        img_gen = get_image_generator(device)
        num_images_to_generate = min(num_files, len(variations))
        generated_images = img_gen.generate_images(variations[:num_images_to_generate])

    # 3) Zip output (always prepared; we may upload to S3 for URL mode)
    zip_bytes = images_to_zip_bytes(generated_images)

    s3_manager = get_s3_manager()
    zip_stem = Path(file.filename).stem if file.filename else "uploaded_image"
    image_items: List[Dict[str, Any]] = []

    if s3_manager:
        # Upload zip for bulk download (keeps backward-compat behavior)
        image_files = [(f"image_{i+1}.png", img) for i, img in enumerate(generated_images)]
        download_url = s3_manager.upload_images_and_zip(
            images=image_files,
            zip_name=f"generated_images_{zip_stem}",
        )

        # Also upload each image individually so UI can render one-by-one.
        for i, img in enumerate(generated_images):
            variation = variations[i] if i < len(variations) else ""
            png_bytes = pil_image_to_png_bytes(img)
            image_url = s3_manager.upload_file(
                file_data=png_bytes,
                filename=f"image_{i+1}.png",
                content_type="image/png",
            )
            image_items.append(
                {
                    "image_index": i + 1,
                    "variation": variation,
                    # UI-friendly alias
                    "file_url": image_url,
                }
            )
    else:
        download_url = None
        # Fallback: embed per-image base64 for one-by-one UI rendering.
        for i, img in enumerate(generated_images):
            variation = variations[i] if i < len(variations) else ""
            png_bytes = pil_image_to_png_bytes(img)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            image_items.append(
                {
                    "image_index": i + 1,
                    "variation": variation,
                    "image_base64": b64,
                    # UI-friendly alias (so `src` can be set directly)
                    "file_url": make_data_uri("image/png", b64),
                }
            )



    if mode == "zip":
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="generated_images.zip"'},
        )

    if download_url:
        return {
            "mode": "json",
            "caption": caption,
            "variations": variations,
            "images_count": len(generated_images),
            # ZIP download URL (bundle)
            "zip_download_url": download_url,
            "image_results": image_items,
        }

    return JSONResponse(
        {
            "mode": "json",
            "caption": caption,
            "variations": variations,
            "images_count": len(generated_images),
            "zip_base64": base64.b64encode(zip_bytes).decode("utf-8"),
            "image_results": image_items,
        }
    )


@app.post("/api/generate/data-from-file")
async def generate_data_from_file(
    file: UploadFile = File(...),
    num_files: int = Form(10),
    mode: Literal["json", "zip"] = Query("json", description="Return JSON metadata or the ZIP bytes."),
):
    file_ext = _get_file_ext(file.filename or "")
    if file_ext not in {"csv", "xlsx", "docx", "pdf"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Use csv/xlsx/docx/pdf.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    device = _device()
    named_file = NamedBytesIO(contents, name=file.filename or "upload")

    result_data: Optional[List[Tuple[str, bytes]]] = None

    if file_ext in {"csv", "xlsx"}:
        from generators.tabular import TabularGenerator

        generator = TabularGenerator()
        result_data = generator.generate(named_file, file_ext=file_ext, num_files=num_files)
    elif file_ext == "docx":
        from generators.docx import DocxGenerator
        from models.t5 import T5Model

        t5_model = get_t5_model(device)
        generator = DocxGenerator(t5_model)
        result_data = generator.generate(named_file, num_files=num_files)
    elif file_ext == "pdf":
        from generators.pdf import PdfGenerator

        t5_model = get_t5_model(device)
        generator = PdfGenerator(t5_model)
        result_data = generator.generate(named_file, num_files=num_files)

    if not result_data:
        raise HTTPException(status_code=500, detail="Generation produced no output.")

    files_for_zip = [(filename, data) for (filename, data) in result_data]
    zip_bytes = files_to_zip_bytes(files_for_zip)

    s3_manager = get_s3_manager()
    zip_stem = Path(file.filename).stem if file.filename else "uploaded_file"

    download_url = None
    file_items: List[Dict[str, Any]] = []
    if s3_manager:
        files_for_s3: List[Tuple[str, bytes, str]] = []
        for filename, data in result_data:
            files_for_s3.append((filename, data, _content_type_for_ext(file_ext)))
        download_url = s3_manager.upload_and_zip(
            files=files_for_s3,
            zip_name=f"generated_{file_ext}_{zip_stem}",
        )

        # Also upload each file individually so UI can render one-by-one.
        for filename, data in result_data:
            file_url = s3_manager.upload_file(
                file_data=data,
                filename=filename,
                content_type=_content_type_for_ext(file_ext),
            )
            file_items.append(
                {
                    "filename": filename,
                    "size_bytes": len(data),
                    # UI-friendly alias
                    "file_url": file_url,
                }
            )
    else:
        # Fallback: embed file bytes base64 for one-by-one UI rendering.
        for filename, data in result_data:
            b64 = base64.b64encode(data).decode("utf-8")
            file_items.append(
                {
                    "filename": filename,
                    "size_bytes": len(data),
                    "file_base64": b64,
                    # UI-friendly alias (data URI)
                    "file_url": make_data_uri(_content_type_for_ext(file_ext), b64),
                }
            )

    if mode == "zip":
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="generated_data.zip"'},
        )

    # JSON response
    file_summaries = [{"filename": f, "size_bytes": len(b)} for (f, b) in result_data]
    if download_url:
        return {
            "mode": "json",
            "file_type": file_ext,
            "files": file_summaries,
            "zip_download_url": download_url,
            "file_results": file_items,
        }

    return JSONResponse(
        {
            "mode": "json",
            "file_type": file_ext,
            "files": file_summaries,
            "zip_base64": base64.b64encode(zip_bytes).decode("utf-8"),
            "file_results": file_items,
        }
    )


@app.post("/api/generate/image-from-text")
async def generate_image_from_text(
    prompt: str = Form(..., description="Prompt/caption to generate an image from text."),
    mode: Literal["json", "zip"] = Query("json", description="Return JSON metadata or the ZIP bytes."),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    # Stable Diffusion max tokenization is sensitive; keep the previous guardrail.
    if len(prompt) > 100:
        prompt = prompt[:100]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        img_gen = get_image_generator(device)
        generated_images = img_gen.generate_images([prompt])

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image.")

    zip_bytes = images_to_zip_bytes(generated_images[:1])

    s3_manager = get_s3_manager()
    download_url = None
    if s3_manager:
        download_url = s3_manager.upload_images_and_zip(
            images=[("generated_image.png", generated_images[0])],
            zip_name="text_generated_image",
        )

    if mode == "zip":
        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="generated_image.zip"'},
        )

    if download_url:
        return {"mode": "json", "prompt": prompt, "images_count": 1, "download_url": download_url}

    return JSONResponse(
        {
            "mode": "json",
            "prompt": prompt,
            "images_count": 1,
            "zip_base64": base64.b64encode(zip_bytes).decode("utf-8"),
        }
    )


@app.post("/api/generate/video-from-image")
async def generate_video_from_image(
    file: UploadFile = File(...),
    prompt: str = Form(..., description="Prompt describing the video to generate from the image."),
    num_frames: int = Form(121, description="Number of frames (default 121, recommended 8N+1)"),
    negative_prompt: str = Form("", description="Optional negative prompt to suppress artifacts or unwanted styles."),
    num_inference_steps: Optional[int] = Form(None, description="Inference steps for video generation."),
    guidance_scale: Optional[float] = Form(None, description="Prompt guidance strength for video generation."),
    mode: Literal["json", "download"] = Query("json", description="Return JSON metadata or direct video download."),
):
    file_ext = _get_file_ext(file.filename or "")
    if file_ext not in {"jpg", "jpeg", "png"}:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use jpg/jpeg/png.")
        
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload.")

    from PIL import Image
    import torch

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = prepare_video_condition_image(image, target_width=768, target_height=512)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with torch.no_grad():
            vid_gen = get_video_generator(device)
            video_bytes = vid_gen.generate_video(
                image,
                prompt,
                negative_prompt=negative_prompt or None,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
    except torch.OutOfMemoryError as exc:
        raise HTTPException(
            status_code=503,
            detail="Video generation exceeded the available GPU memory on the current Modal worker.",
        ) from exc
    except RuntimeError as exc:
        error_message = str(exc)
        if "out of GPU memory" in error_message or "out of memory" in error_message:
            raise HTTPException(
                status_code=503,
                detail=error_message,
            ) from exc
        raise

    if not video_bytes:
        raise HTTPException(status_code=500, detail="Failed to generate video.")

    s3_manager = get_s3_manager()
    download_url = None
    if s3_manager:
        download_url = s3_manager.upload_file(
            file_data=video_bytes,
            filename="generated_video.mp4",
            content_type="video/mp4"
        )
    
    if mode == "download":
        return StreamingResponse(
            io.BytesIO(video_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": 'attachment; filename="generated_video.mp4"'},
        )
        
    b64 = base64.b64encode(video_bytes).decode("utf-8")
    
    result_item = {
        "video_base64": b64,
        "file_url": make_data_uri("video/mp4", b64)
    }
    
    if download_url:
        result_item["file_url"] = download_url
        
    return JSONResponse({
        "mode": "json",
        "prompt": prompt,
        "video_download_url": download_url,
        "video_result": result_item
    })


if __name__ == "__main__":
    # Local development entrypoint
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
