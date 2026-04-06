import os
import io
import torch
from huggingface_hub import snapshot_download
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from typing import Any

class VideoGenerator:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing VideoGenerator...")
        
        # Download safely keeping notebook's workaround
        print("Downloading LTX-2.3-FlashPack model safely...")
        model_path = snapshot_download(
            repo_id="fal/LTX-2.3-FlashPack",
            # We omit local_dir so it uses the standard Hugging Face cache
            resume_download=True
        )
        
        print("Loading pipeline...")
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.pipe.to(self.device)
        
        print("Loading LORA weights...")
        self.pipe.load_lora_weights(
            "valiantcat/LTX-2.3-Transition-LORA",
            weight_name="ltx2.3-transition.safetensors",
        )
        self.pipe.set_adapters(1.0)
        print("VideoGenerator ready.")

    def generate_video(
        self,
        image: Any, # PIL Image
        prompt: str,
        negative_prompt: str = "blurry, distorted, jitter, bad anatomy",
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
    ) -> bytes:
        """
        Generates a video from the image and prompt.
        Returns the generated video as MP4 bytes.
        """
        # num_frames must be 8N+1
        if (num_frames - 1) % 8 != 0:
            num_frames = ((num_frames - 1) // 8) * 8 + 1
            
        print(f"Generating video with {num_frames} frames...")
        
        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Diffusers typically returns the batch frames in list or list of lists format
        video = result.frames[0] if isinstance(result.frames, list) and isinstance(result.frames[0], list) else result.frames
        
        print("Exporting video to bytes...")
        
        # Write to a temporary file since export_to_video expects a filepath
        temp_filepath = "temp_video_output.mp4"
        export_to_video(video, temp_filepath, fps=24)
        
        with open(temp_filepath, "rb") as f:
            video_bytes = f.read()
            
        # Clean up temp file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        return video_bytes
