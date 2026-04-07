import torch
from diffusers import AutoPipelineForText2Image


def _load_public_pipeline(model_name, **kwargs):
    load_kwargs = dict(kwargs)
    try:
        return AutoPipelineForText2Image.from_pretrained(model_name, token=False, **load_kwargs)
    except TypeError:
        return AutoPipelineForText2Image.from_pretrained(model_name, use_auth_token=False, **load_kwargs)

class ImageGenerator:
    """Generates images from prompts using SDXL Turbo."""
    def __init__(self, device):
        self.device = device
        print("⏳ Loading Stable Diffusion 1.5 with CPU Offload strategy...")
        self.pipe = _load_public_pipeline(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        if self.device == "cuda" and torch.cuda.is_available():
            print("🚀 Using GPU (RTX 3050 detected) - Optimizing for speed...")
            # For 6GB VRAM + SD 1.5, model_cpu_offload is much faster than sequential
            self.pipe.enable_model_cpu_offload()
            
            # Keep slicing to be safe against OOM spikes
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
        else:
            print("🐢 Using CPU (converting to FP32 for compatibility)...")
            # Convert to float32 for CPU execution (FP16 is not supported on CPU for some ops)
            # This uses the existing FP16 files but converts them in RAM, avoiding re-download.
            self.pipe.to("cpu", dtype=torch.float32)
            
            # Slicing still helps on CPU to reduce RAM spikes
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()

    def generate_images(self, prompts, batch_size=2):
        """
        Generate images in batches for faster processing.
        batch_size=2 is safe for RTX 3050 (6GB VRAM).
        Increase to 3-4 if you have more VRAM.
        """
        images = []
        print(f"🎨 Generating {len(prompts)} images in batches of {batch_size}...")
        
        # Process prompts in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Enhance all prompts in batch
            enhanced_prompts = [
                f"{prompt}, cinematic lighting, highly detailed, 8k resolution, masterpiece"
                for prompt in batch_prompts
            ]
            
            print(f"Generating batch {batch_start//batch_size + 1} (images {batch_start+1}-{batch_end})...")
            
            # Generate batch (GPU processes these in parallel)
            batch_images = self.pipe(
                prompt=enhanced_prompts,
                num_inference_steps=20
            ).images
            
            images.extend(batch_images)
            print(f"✅ Batch complete ({len(batch_images)} images)")
            torch.cuda.empty_cache()
        
        print(f"\n🎉 All {len(images)} images generated successfully!")
        return images
