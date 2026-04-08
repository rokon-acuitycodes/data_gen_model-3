import os
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download


DEFAULT_IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_IMAGE_INFERENCE_STEPS = int(os.getenv("IMAGE_INFERENCE_STEPS", "4"))
DEFAULT_IMAGE_GUIDANCE_SCALE = float(os.getenv("IMAGE_GUIDANCE_SCALE", "1.0"))
DEFAULT_IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "1024"))
DEFAULT_IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "1024"))
DEFAULT_IMAGE_MAX_SEQUENCE_LENGTH = int(os.getenv("IMAGE_MAX_SEQUENCE_LENGTH", "256"))
DEFAULT_IMAGE_FULL_GPU_MIN_VRAM_GB = float(os.getenv("IMAGE_FULL_GPU_MIN_VRAM_GB", "16"))


def _resolve_hf_token():
    use_hf_token = os.getenv("IMAGE_USE_HF_TOKEN", "").strip().lower() in {"1", "true", "yes"}
    if not use_hf_token:
        return False

    token = os.getenv("HF_TOKEN")
    return token if token else False


def _load_flux_pipeline(model_id: str, torch_dtype: torch.dtype):
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Image generation requires a recent diffusers build with Flux2KleinPipeline support."
        ) from exc

    token = _resolve_hf_token()
    cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {model_id} from Hugging Face."
        ) from exc

    return Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )


class ImageGenerator:
    """Generates images from prompts using FLUX.2 klein 4B."""

    def __init__(self, device: str):
        self.device = device
        self.model_id = DEFAULT_IMAGE_MODEL_ID
        self.num_inference_steps = DEFAULT_IMAGE_INFERENCE_STEPS
        self.guidance_scale = DEFAULT_IMAGE_GUIDANCE_SCALE
        self.height = DEFAULT_IMAGE_HEIGHT
        self.width = DEFAULT_IMAGE_WIDTH
        self.max_sequence_length = DEFAULT_IMAGE_MAX_SEQUENCE_LENGTH

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
        print(f"Loading FLUX.2 klein image pipeline from {self.model_id}...")
        self.pipe = _load_flux_pipeline(self.model_id, torch_dtype=dtype)

        if self.device == "cuda" and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb >= DEFAULT_IMAGE_FULL_GPU_MIN_VRAM_GB:
                print(f"Using GPU with {total_vram_gb:.1f} GB VRAM for FLUX.2 klein 4B.")
                self.pipe.to("cuda")
            else:
                print(
                    f"GPU has {total_vram_gb:.1f} GB VRAM; enabling CPU offload for FLUX.2 klein 4B."
                )
                self.pipe.enable_model_cpu_offload()

            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
        else:
            print("Using CPU for FLUX.2 klein 4B.")
            self.pipe.to("cpu", dtype=torch.float32)

    def _enhance_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return prompt

        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in ("anime", "cartoon", "illustration", "painting", "sketch")):
            return prompt

        details = (
            "photorealistic, natural lighting, crisp focus, accurate anatomy, realistic skin texture, "
            "coherent background details"
        )
        return f"{prompt}, {details}"

    def _generator_for_seed(self, seed: Optional[int]):
        if seed is None:
            return None

        generator_device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        return torch.Generator(device=generator_device).manual_seed(seed)

    def _run_pipeline(
        self,
        prompt: str,
        *,
        image=None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        height = image.height if image is not None else self.height
        width = image.width if image is not None else self.width

        return self.pipe(
            prompt=self._enhance_prompt(prompt),
            image=image,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=self.guidance_scale if guidance_scale is None else guidance_scale,
            width=width,
            height=height,
            max_sequence_length=self.max_sequence_length,
            generator=self._generator_for_seed(seed),
        )

    def generate_images(self, prompts: List[str], batch_size: int = 1):
        del batch_size  # FLUX.2 klein 4B is generated one prompt at a time for memory stability.

        images = []
        print(
            f"Generating {len(prompts)} image(s) with {self.num_inference_steps} steps at "
            f"{self.width}x{self.height}..."
        )

        for index, prompt in enumerate(prompts, start=1):
            print(f"Generating image {index}/{len(prompts)}...")
            result = self._run_pipeline(prompt)
            images.append(result.images[0])

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Generated {len(images)} image(s) successfully.")
        return images

    def generate_images_from_image(
        self,
        image,
        prompt: str,
        *,
        num_outputs: int = 1,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        images = []
        print(
            f"Generating {num_outputs} image-to-image output(s) with "
            f"{num_inference_steps or self.num_inference_steps} steps..."
        )

        for index in range(num_outputs):
            current_seed = None if seed is None else seed + index
            print(f"Generating edited image {index + 1}/{num_outputs}...")
            result = self._run_pipeline(
                prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
            )
            images.append(result.images[0])

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Generated {len(images)} image-to-image output(s) successfully.")
        return images
