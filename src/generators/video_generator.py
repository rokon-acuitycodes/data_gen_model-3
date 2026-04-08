import gc
import inspect
import math
import os
import tempfile
from threading import Lock
from types import MethodType
from typing import Any, Optional

import torch


DEFAULT_PIPELINE_MODE = "distilled_two_stage"
DEFAULT_VIDEO_MODEL_ID = "rootonchair/LTX-2-19b-distilled"
DEFAULT_VIDEO_BASE_MODEL_ID = "Lightricks/LTX-2"
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, blurry, soft focus, flicker, jitter, temporal inconsistency, warped motion, "
    "deformed anatomy, duplicate limbs, extra fingers, distorted face, plastic skin, "
    "cgi, cartoon, illustration, 3d render"
)
DEFAULT_FRAME_RATE = 24.0
DEFAULT_DISTILLED_NUM_INFERENCE_STEPS = 8
DEFAULT_DISTILLED_GUIDANCE_SCALE = 1.0
DEFAULT_DEV_NUM_INFERENCE_STEPS = 40
DEFAULT_DEV_GUIDANCE_SCALE = 4.0
DEFAULT_STAGE2_NUM_INFERENCE_STEPS = 3
DEFAULT_STAGE2_GUIDANCE_SCALE = 1.0
DEFAULT_STAGE2_SCALE_FACTOR = 2
DEFAULT_STAGE2_LORA_WEIGHT_NAME = "ltx-2-19b-distilled-lora-384.safetensors"
DEFAULT_AUDIO_CUSTOM_PIPELINE = "linoyts/ltx2-audio-video-conditioning"
DEFAULT_MAX_SEQUENCE_LENGTH = 256
DEFAULT_OFFLOAD_MODE = "sequential"
DEFAULT_OOM_RETRY_MAX_SEQUENCE_LENGTH = 128
DEFAULT_PROMPT_SUFFIX = (
    "single continuous shot, coherent natural motion, stable identity, accurate anatomy, "
    "clean subject separation, fine detail, realistic textures"
)
DEFAULT_REALISM_SUFFIX = (
    "live-action footage, photorealistic, natural lighting, physically plausible materials, "
    "subtle cinematic depth of field"
)
STYLIZED_PROMPT_MARKERS = (
    "anime",
    "animation",
    "cartoon",
    "comic",
    "illustration",
    "paint",
    "painting",
    "sketch",
    "pixel art",
    "claymation",
    "stop motion",
    "low poly",
    "cgi",
    "3d render",
    "stylized",
)
CAMERA_PROMPT_MARKERS = (
    "camera",
    "close-up",
    "medium shot",
    "wide shot",
    "tracking shot",
    "dolly",
    "pan",
    "tilt",
    "zoom",
    "handheld",
    "over-the-shoulder",
)
PIPELINE_MODE_SINGLE_STAGE = "single_stage"
PIPELINE_MODE_DISTILLED_TWO_STAGE = "distilled_two_stage"
PIPELINE_MODE_DEV_LORA_TWO_STAGE = "dev_lora_two_stage"


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _load_public_diffusion_pipeline(loader_cls, model_name, **kwargs):
    load_kwargs = dict(kwargs)
    try:
        return loader_cls.from_pretrained(model_name, token=False, **load_kwargs)
    except TypeError:
        return loader_cls.from_pretrained(model_name, use_auth_token=False, **load_kwargs)


class VideoGenerator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline_mode = os.getenv("VIDEO_PIPELINE_MODE", DEFAULT_PIPELINE_MODE).lower()
        if self.pipeline_mode not in {
            PIPELINE_MODE_SINGLE_STAGE,
            PIPELINE_MODE_DISTILLED_TWO_STAGE,
            PIPELINE_MODE_DEV_LORA_TWO_STAGE,
        }:
            self.pipeline_mode = DEFAULT_PIPELINE_MODE

        self.base_model_id = os.getenv("VIDEO_BASE_MODEL_ID", DEFAULT_VIDEO_BASE_MODEL_ID)
        default_model_id = (
            self.base_model_id
            if self.pipeline_mode == PIPELINE_MODE_DEV_LORA_TWO_STAGE
            else DEFAULT_VIDEO_MODEL_ID
        )
        self.model_id = os.getenv("VIDEO_MODEL_ID", default_model_id)
        self.latent_upsampler_model_id = os.getenv(
            "VIDEO_LATENT_UPSAMPLER_MODEL_ID",
            self.model_id if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE else self.base_model_id,
        )
        self.stage2_lora_repo_id = os.getenv("VIDEO_STAGE2_LORA_REPO_ID", self.base_model_id)
        self.stage2_lora_weight_name = os.getenv(
            "VIDEO_STAGE2_LORA_WEIGHT_NAME",
            DEFAULT_STAGE2_LORA_WEIGHT_NAME,
        )
        self.audio_custom_pipeline = os.getenv(
            "VIDEO_AUDIO_CUSTOM_PIPELINE",
            DEFAULT_AUDIO_CUSTOM_PIPELINE,
        )
        self.frame_rate = float(os.getenv("VIDEO_FRAME_RATE", str(DEFAULT_FRAME_RATE)))
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
        self.stage2_scale_factor = int(
            os.getenv("VIDEO_STAGE2_SCALE_FACTOR", str(DEFAULT_STAGE2_SCALE_FACTOR))
        )
        self.stage2_num_inference_steps = int(
            os.getenv(
                "VIDEO_STAGE2_INFERENCE_STEPS",
                str(DEFAULT_STAGE2_NUM_INFERENCE_STEPS),
            )
        )
        self.stage2_guidance_scale = float(
            os.getenv(
                "VIDEO_STAGE2_GUIDANCE_SCALE",
                str(DEFAULT_STAGE2_GUIDANCE_SCALE),
            )
        )
        self.default_prompt_suffix = os.getenv("VIDEO_PROMPT_SUFFIX", DEFAULT_PROMPT_SUFFIX)
        self.default_realism_suffix = os.getenv("VIDEO_REALISM_SUFFIX", DEFAULT_REALISM_SUFFIX)
        self.use_two_stage = _env_flag(
            "VIDEO_ENABLE_TWO_STAGE",
            self.pipeline_mode != PIPELINE_MODE_SINGLE_STAGE,
        )
        if self.pipeline_mode == PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            default_stage1_steps = DEFAULT_DEV_NUM_INFERENCE_STEPS
            default_stage1_guidance = DEFAULT_DEV_GUIDANCE_SCALE
        else:
            default_stage1_steps = DEFAULT_DISTILLED_NUM_INFERENCE_STEPS
            default_stage1_guidance = DEFAULT_DISTILLED_GUIDANCE_SCALE
        self.default_num_inference_steps = int(
            os.getenv("VIDEO_INFERENCE_STEPS", str(default_stage1_steps))
        )
        self.default_guidance_scale = float(
            os.getenv("VIDEO_GUIDANCE_SCALE", str(default_stage1_guidance))
        )
        self.seed = os.getenv("VIDEO_SEED")
        self._lock = Lock()

        try:
            from diffusers import (
                FlowMatchEulerDiscreteScheduler,
                LTX2ImageToVideoPipeline,
                LTX2LatentUpsamplePipeline,
            )
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
            from diffusers.pipelines.ltx2.utils import (
                DISTILLED_SIGMA_VALUES,
                STAGE_2_DISTILLED_SIGMA_VALUES,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Video generation requires a recent diffusers runtime with LTX-2 support."
            ) from exc

        self._flow_match_scheduler_cls = FlowMatchEulerDiscreteScheduler
        self._pipeline_cls = LTX2ImageToVideoPipeline
        self._upsample_pipeline_cls = LTX2LatentUpsamplePipeline
        self._latent_upsampler_cls = LTX2LatentUpsamplerModel
        self._distilled_sigmas = DISTILLED_SIGMA_VALUES
        self._stage2_sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
        self._stage2_lora_loaded = False
        self._audio_pipe = None

        print(f"Initializing VideoGenerator on {self.device}...")
        print(
            "Loading LTX-2 image-to-video pipeline "
            f"mode={self.pipeline_mode} model_id={self.model_id}..."
        )

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.pipe = self._pipeline_cls.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        self._default_scheduler_config = self.pipe.scheduler.config
        self._prepare_pipeline_for_runtime(self.pipe, prefer_model_offload=False)

        self.upsample_pipe = None
        if self.use_two_stage:
            print(
                "Loading latent upsampler "
                f"from {self.latent_upsampler_model_id}/latent_upsampler..."
            )
            latent_upsampler = self._latent_upsampler_cls.from_pretrained(
                self.latent_upsampler_model_id,
                subfolder="latent_upsampler",
                torch_dtype=torch_dtype,
            )
            self.upsample_pipe = self._upsample_pipeline_cls(
                vae=self.pipe.vae,
                latent_upsampler=latent_upsampler,
            )
            self._prepare_pipeline_for_runtime(self.upsample_pipe, prefer_model_offload=True)

        print("VideoGenerator ready.")

    def _prepare_pipeline_for_runtime(self, pipeline, *, prefer_model_offload: bool):
        if self.device.startswith("cuda"):
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing("max")
            if prefer_model_offload or self.offload_mode == "model":
                pipeline.enable_model_cpu_offload(device=self.device)
            else:
                pipeline.enable_sequential_cpu_offload(device=self.device)
            self._cleanup_cuda_memory()
        else:
            pipeline.to(self.device)

        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

    def _patch_audio_pipeline_compatibility(self, pipeline):
        connectors = getattr(pipeline, "connectors", None)
        if connectors is None:
            return

        forward_signature = inspect.signature(connectors.forward)
        if "additive_mask" in forward_signature.parameters:
            return

        original_forward = connectors.forward

        def wrapped_forward(self, *args, additive_mask=None, **kwargs):
            del additive_mask
            return original_forward(*args, **kwargs)

        connectors.forward = MethodType(wrapped_forward, connectors)

    def _get_audio_pipe(self):
        if self._audio_pipe is not None:
            return self._audio_pipe

        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Audio-conditioned video generation requires Diffusers DiffusionPipeline support."
            ) from exc

        print(
            "Loading audio-conditioned LTX-2 pipeline "
            f"from {self.model_id} with custom pipeline {self.audio_custom_pipeline}..."
        )
        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._audio_pipe = _load_public_diffusion_pipeline(
            DiffusionPipeline,
            self.model_id,
            custom_pipeline=self.audio_custom_pipeline,
            torch_dtype=torch_dtype,
        )
        self._patch_audio_pipeline_compatibility(self._audio_pipe)
        self._prepare_pipeline_for_runtime(self._audio_pipe, prefer_model_offload=False)
        return self._audio_pipe

    def _infer_condition_mode(self, frames: Any) -> str:
        if isinstance(frames, (list, tuple)):
            return "video"
        if isinstance(frames, torch.Tensor) and frames.ndim >= 4:
            return "video"
        return "image"

    def _build_video_condition(self, pipeline, *, frames: Any, index: int, strength: float):
        pipeline_module = inspect.getmodule(pipeline.__class__)
        condition_cls = getattr(pipeline_module, "LTX2VideoCondition", None) if pipeline_module else None

        if condition_cls is None:
            try:
                from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition as condition_cls
            except ImportError as exc:
                raise RuntimeError(
                    "Audio-conditioned generation requires LTX2VideoCondition support in diffusers."
                ) from exc

        condition_mode = self._infer_condition_mode(frames)
        init_signature = inspect.signature(condition_cls)
        init_kwargs = {
            "frames": frames,
            "index": index,
            "strength": strength,
        }
        if "mode" in init_signature.parameters:
            init_kwargs["mode"] = condition_mode

        condition = condition_cls(**init_kwargs)
        if not hasattr(condition, "mode"):
            condition.mode = condition_mode

        return condition

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

    def _prepare_prompt(self, prompt: str) -> str:
        prompt = " ".join(prompt.strip().split())
        prompt_lower = prompt.lower()
        is_stylized = any(marker in prompt_lower for marker in STYLIZED_PROMPT_MARKERS)
        has_camera_language = any(marker in prompt_lower for marker in CAMERA_PROMPT_MARKERS)

        prompt_parts = [prompt.rstrip(".")]

        if len(prompt.split()) < 24:
            prompt_parts.append(self.default_prompt_suffix)

        if not has_camera_language:
            prompt_parts.append(
                "the camera movement is subtle and stable, keeping the main subject in clear focus"
            )

        if not is_stylized:
            prompt_parts.append(self.default_realism_suffix)

        return ". ".join(part for part in prompt_parts if part).strip() + "."

    def _estimate_num_frames_from_audio(self, audio_path: str, fallback_num_frames: int) -> int:
        try:
            import av

            with av.open(audio_path) as container:
                audio_stream = next(
                    (stream for stream in container.streams if stream.type == "audio"),
                    None,
                )
                if audio_stream and audio_stream.duration is not None and audio_stream.time_base is not None:
                    duration_seconds = float(audio_stream.duration * audio_stream.time_base)
                elif container.duration is not None:
                    duration_seconds = float(container.duration / av.time_base)
                else:
                    return fallback_num_frames
        except Exception:
            return fallback_num_frames

        estimated_frames = max(9, math.ceil(duration_seconds * self.frame_rate))
        return ((estimated_frames - 1 + 7) // 8) * 8 + 1

    def _build_generator(self):
        if self.seed is None:
            return None
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        return torch.Generator(generator_device).manual_seed(int(self.seed))

    def _restore_default_scheduler(self):
        scheduler_cls = type(self.pipe.scheduler)
        self.pipe.scheduler = scheduler_cls.from_config(self._default_scheduler_config)

    def _enable_stage2_scheduler(self):
        self.pipe.scheduler = self._flow_match_scheduler_cls.from_config(
            self._default_scheduler_config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

    def _ensure_stage2_lora_loaded(self):
        if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE or self._stage2_lora_loaded:
            return

        print(
            "Loading stage 2 distilled LoRA "
            f"{self.stage2_lora_weight_name} from {self.stage2_lora_repo_id}..."
        )
        self.pipe.load_lora_weights(
            self.stage2_lora_repo_id,
            adapter_name="stage_2_distilled",
            weight_name=self.stage2_lora_weight_name,
        )
        self._stage2_lora_loaded = True
        self.pipe.set_adapters("stage_2_distilled", 0.0)

    def _set_stage2_adapter_weight(self, weight: float):
        if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            return
        self._ensure_stage2_lora_loaded()
        self.pipe.set_adapters("stage_2_distilled", weight)

    def _run_single_stage(
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
        prepared_prompt = self._prepare_prompt(prompt)
        generator = self._build_generator()
        call_kwargs = {
            "image": image,
            "prompt": prepared_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": self.frame_rate,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_type": "np",
            "return_dict": False,
            "max_sequence_length": max_sequence_length,
        }
        if generator is not None:
            call_kwargs["generator"] = generator
        if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            call_kwargs["sigmas"] = self._distilled_sigmas
        return self.pipe(**call_kwargs)

    def _run_audio_conditioned(
        self,
        *,
        image: Any,
        audio_path: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        max_sequence_length: int,
    ):
        prepared_prompt = self._prepare_prompt(prompt)
        generator = self._build_generator()
        audio_pipe = self._get_audio_pipe()
        audio_pipe_params = inspect.signature(audio_pipe.__call__).parameters
        call_kwargs = {
            "audio": audio_path,
            "prompt": prepared_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": self.frame_rate,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_type": "np",
            "return_dict": False,
            "max_sequence_length": max_sequence_length,
        }
        if "image" in audio_pipe_params:
            call_kwargs["image"] = image
        elif "conditions" in audio_pipe_params:
            call_kwargs["conditions"] = [
                self._build_video_condition(
                    audio_pipe,
                    frames=image,
                    index=0,
                    strength=1.0,
                )
            ]
        else:
            raise RuntimeError(
                "The loaded audio-conditioned LTX-2 pipeline does not support either `image` or "
                "`conditions` inputs."
            )

        if generator is not None:
            call_kwargs["generator"] = generator
        if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            call_kwargs["sigmas"] = self._distilled_sigmas

        return audio_pipe(**call_kwargs)

    def _run_two_stage(
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
        if self.upsample_pipe is None:
            return self._run_single_stage(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )

        prepared_prompt = self._prepare_prompt(prompt)
        generator = self._build_generator()
        self._restore_default_scheduler()
        self._set_stage2_adapter_weight(0.0)

        stage1_kwargs = {
            "image": image,
            "prompt": prepared_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": self.frame_rate,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_type": "latent",
            "return_dict": False,
            "max_sequence_length": max_sequence_length,
        }
        if generator is not None:
            stage1_kwargs["generator"] = generator
        if self.pipeline_mode != PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            stage1_kwargs["sigmas"] = self._distilled_sigmas

        video_latent, audio_latent = self.pipe(**stage1_kwargs)

        upscaled_video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        stage2_width = width * self.stage2_scale_factor
        stage2_height = height * self.stage2_scale_factor

        if self.pipeline_mode == PIPELINE_MODE_DEV_LORA_TWO_STAGE:
            self._set_stage2_adapter_weight(1.0)
            self._enable_stage2_scheduler()

        stage2_kwargs = {
            "image": image,
            "latents": upscaled_video_latent,
            "audio_latents": audio_latent,
            "prompt": prepared_prompt,
            "negative_prompt": negative_prompt,
            "width": stage2_width,
            "height": stage2_height,
            "num_frames": num_frames,
            "frame_rate": self.frame_rate,
            "num_inference_steps": self.stage2_num_inference_steps,
            "noise_scale": self._stage2_sigmas[0],
            "sigmas": self._stage2_sigmas,
            "guidance_scale": self.stage2_guidance_scale,
            "output_type": "np",
            "return_dict": False,
            "max_sequence_length": max_sequence_length,
        }
        if generator is not None:
            stage2_kwargs["generator"] = generator

        return self.pipe(**stage2_kwargs)

    def generate_video(
        self,
        image: Any,
        prompt: str,
        negative_prompt: Optional[str] = DEFAULT_NEGATIVE_PROMPT,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        audio_bytes: Optional[bytes] = None,
        audio_filename: Optional[str] = None,
        sync_to_audio_duration: bool = True,
    ) -> bytes:
        """Generate an MP4 video from a PIL image and prompt."""
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1

        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        guidance_scale = (
            self.default_guidance_scale if guidance_scale is None else guidance_scale
        )

        print(
            f"Generating video with {num_frames} frames, "
            f"{num_inference_steps} stage-1 steps, guidance {guidance_scale}, "
            f"mode={self.pipeline_mode}..."
        )

        audio_temp_path = None
        try:
            with self._lock:
                self._cleanup_cuda_memory()
                with torch.inference_mode():
                    if audio_bytes:
                        suffix = os.path.splitext(audio_filename or "audio.wav")[1] or ".wav"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as audio_file:
                            audio_file.write(audio_bytes)
                            audio_temp_path = audio_file.name

                        if sync_to_audio_duration:
                            num_frames = self._estimate_num_frames_from_audio(
                                audio_temp_path,
                                num_frames,
                            )

                        print(
                            "Using audio-conditioned video generation for lip-sync "
                            f"with {num_frames} frames."
                        )
                        video, audio = self._run_audio_conditioned(
                            image=image,
                            audio_path=audio_temp_path,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_frames=num_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            max_sequence_length=self.max_sequence_length,
                        )
                    elif self.use_two_stage:
                        video, audio = self._run_two_stage(
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
                    else:
                        video, audio = self._run_single_stage(
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
        except torch.OutOfMemoryError:
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
                        if audio_bytes and audio_temp_path:
                            video, audio = self._run_audio_conditioned(
                                image=image,
                                audio_path=audio_temp_path,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                width=width,
                                height=height,
                                num_frames=num_frames,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                max_sequence_length=retry_max_sequence_length,
                            )
                        else:
                            video, audio = self._run_single_stage(
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
            if self.pipeline_mode == PIPELINE_MODE_DEV_LORA_TWO_STAGE:
                self._set_stage2_adapter_weight(0.0)
                self._restore_default_scheduler()
            self._cleanup_cuda_memory()
            if audio_temp_path and os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
