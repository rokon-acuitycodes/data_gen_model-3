import gc
import os
import tempfile
from threading import Lock
from typing import Any, Optional

import torch


DEFAULT_VIDEO_MODEL_ID = "Lightricks/LTX-2"
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
DEFAULT_FRAME_RATE = 24.0
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_MAX_SEQUENCE_LENGTH = 256
DEFAULT_OFFLOAD_MODE = "sequential"
DEFAULT_OOM_RETRY_MAX_SEQUENCE_LENGTH = 128


class VideoGenerator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = os.getenv("VIDEO_MODEL_ID", DEFAULT_VIDEO_MODEL_ID)
        self.frame_rate = float(os.getenv("VIDEO_FRAME_RATE", str(DEFAULT_FRAME_RATE)))
        self.default_num_inference_steps = int(
            os.getenv("VIDEO_INFERENCE_STEPS", str(DEFAULT_NUM_INFERENCE_STEPS))
        )
        self.default_guidance_scale = float(
            os.getenv("VIDEO_GUIDANCE_SCALE", str(DEFAULT_GUIDANCE_SCALE))
        )
        self.max_sequence_length = int(
            os.getenv("VIDEO_MAX_SEQUENCE_LENGTH", str(DEFAULT_MAX_SEQUENCE_LENGTH))
        )
        self.offload_mode = os.getenv("VIDEO_OFFLOAD_MODE", DEFAULT_OFFLOAD_MODE).lower()
        self.oom_retry_max_sequence_length = int(
            os.getenv(
                "VIDEO_OOM_RETRY_MAX_SEQUENCE_LENGTH",
                str(DEFAULT_OOM_RETRY_MAX_SEQUENCE_LENGTH),
            )
        )
        self._lock = Lock()

        try:
            from diffusers import LTX2ImageToVideoPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Video generation requires a diffusers build with LTX-2 support. "
                "Upgrade diffusers, transformers, accelerate, and huggingface_hub in the runtime image."
            ) from exc

        print(f"Initializing VideoGenerator on {self.device}...")
        print(f"Loading LTX-2 image-to-video pipeline from {self.model_id}...")

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        if self.device.startswith("cuda"):
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing("max")
            if self.offload_mode == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.enable_sequential_cpu_offload(device=self.device)
            self._cleanup_cuda_memory()
        else:
            self.pipe.to(self.device)

        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

        print("VideoGenerator ready.")

    def _cleanup_cuda_memory(self):
        if not torch.cuda.is_available():
            return

        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

    def _get_encode_video(self):
        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video
        except ImportError as exc:
            message = str(exc)
            if "PyAV is required" in message or "pip install av" in message:
                raise RuntimeError(
                    "Video export requires the `av` package in the runtime image. "
                    "Add `av` to the Modal image and redeploy."
                ) from exc
            raise RuntimeError(
                "Unable to load the LTX-2 video export utilities. "
                "Check the diffusers runtime dependencies in the image."
            ) from exc
        return encode_video

    def _run_pipe(
        self,
        *,
        image: Any,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ):
        return self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=self.frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type="np",
            return_dict=False,
            max_sequence_length=max_sequence_length,
        )

    def generate_video(
        self,
        image: Any,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> bytes:
        """Generate an MP4 video from a PIL image and prompt."""
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1

        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        guidance_scale = (
            self.default_guidance_scale if guidance_scale is None else guidance_scale
        )

        print(
            f"Generating video with {num_frames} frames, "
            f"{num_inference_steps} steps, guidance {guidance_scale}..."
        )

        try:
            with self._lock:
                self._cleanup_cuda_memory()
                with torch.inference_mode():
                    video, audio = self._run_pipe(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        max_sequence_length=self.max_sequence_length,
                    )
        except torch.OutOfMemoryError as exc:
            if not self.device.startswith("cuda"):
                raise

            retry_max_sequence_length = min(
                self.max_sequence_length,
                self.oom_retry_max_sequence_length,
            )
            print(
                "CUDA OOM during video generation. "
                f"Retrying with max_sequence_length={retry_max_sequence_length}."
            )

            self._cleanup_cuda_memory()

            try:
                with self._lock:
                    with torch.inference_mode():
                        video, audio = self._run_pipe(
                            image=image,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            max_sequence_length=retry_max_sequence_length,
                        )
            except torch.OutOfMemoryError as retry_exc:
                self._cleanup_cuda_memory()
                raise RuntimeError(
                    "Video generation ran out of GPU memory on the current Modal GPU. "
                    "Use a larger GPU or reduce the frame count."
                ) from retry_exc

        video_tensor = torch.from_numpy((video * 255).round().astype("uint8"))[0]
        audio_tensor = audio[0].float().cpu() if audio is not None else None

        print("Encoding generated video...")
        encode_video = self._get_encode_video()

        temp_filepath = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_filepath = temp_file.name

            encode_kwargs = {
                "fps": self.frame_rate,
                "output_path": temp_filepath,
            }
            if audio_tensor is not None:
                encode_kwargs["audio"] = audio_tensor
                encode_kwargs["audio_sample_rate"] = self.pipe.vocoder.config.output_sampling_rate

            encode_video(video_tensor, **encode_kwargs)

            with open(temp_filepath, "rb") as file_obj:
                return file_obj.read()
        finally:
            self._cleanup_cuda_memory()
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
