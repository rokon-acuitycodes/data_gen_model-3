import modal
from pathlib import Path
import shlex
import subprocess
import os

# -------------------------------------------------------
# Path Configuration (Windows/Linux Safe)
# -------------------------------------------------------
LOCAL_SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT_LOCAL = LOCAL_SCRIPT_DIR

# -------------------------------------------------------
# Modal Volume & Image Setup
# -------------------------------------------------------
cache_volume = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    
    # 1. SYSTEM DEPENDECIES (APT)
    .apt_install("git", "libgl1", "libglib2.0-0")
    
    # 2. PYTHON DEPENDECIES (PIP)
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
        "pandas",
        "numpy",
        "requests",
        "pillow",
        "torch",
        "transformers",
        "diffusers",
        "accelerate",
        "huggingface_hub",
        "openpyxl",
        # Combined requirements for Unified Data & Image Generator

        # Web UI Framework

        # Document processing
        "python-docx",
        "pypdf>=3.0.0",
        "reportlab>=4.0.0",

        # Data generation
        "faker>=15.0.0",

        # Text processing
        "sentencepiece>=0.1.99",

        # AWS S3 Integration
        "boto3>=1.26.0",

        # Additional utilities

        "scipy>=1.10.0",

        "openai",
        "python-dotenv"


    )
    
    # 3. ENVIRONMENT VARIABLES (CRITICAL: Must be BEFORE file addition)

    .env({"PYTHONPATH": "/root/project/src:/root/project"})
    
    # 4. CRITICAL: PROJECT MOUNTING (MUST BE LAST STEP)

    .add_local_dir(
        str(PROJECT_ROOT_LOCAL),
        remote_path="/root/project",
        ignore=["__pycache__", "*.pyc", ".git", ".venv", "venv", "node_modules", "outputs"]
    )
)

app = modal.App("unified-generator-app", image=image)

# -------------------------------------------------------
# Warmup Logic (Updated to preload models)
# -------------------------------------------------------
@app.function(
    gpu="L4",
    timeout=1800, # 30 minutes max for pre-warming & initial download
    volumes={"/cache": cache_volume},
)
def warmup():
    """
    মডেলের ডিপেন্ডেন্সি লোড এবং Hugging Face মডেল ওয়েটগুলি ভলিউমে ক্যাশ করার জন্য।
    এই ফাংশনটি Streamlit UI চালু হওয়ার আগেই সমস্ত ভারী মডেল লোড/ডাউনলোড নিশ্চিত করবে।
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from diffusers import StableDiffusionPipeline
    
    # Hugging Face ক্যাশ ফোল্ডার সেট করা (Warmup ফাংশনের জন্য)
    os.environ["HF_HOME"] = "/cache/huggingface"

    print("🔥 Warming up container. Starting ML Model pre-caching to Volume.")
    
    # 1. Captioning Model Preload (Example: Assuming a small GPT/Caption model)
    try:
        # এই লাইনগুলি মডেলের ওজন ডাউনলোড করবে এবং ভলিউমে ক্যাশ করবে।
        # আপনার app_copy.py তে ব্যবহৃত মডেল ID এখানে ব্যবহার করুন।
        print("Preloading Captioning/Text Model weights...")
        
        # Example: Using a small model for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_auth_token=None) 
        model_cap = AutoModelForCausalLM.from_pretrained("gpt2", use_auth_token=None)
        
        # We don't need to move it to CUDA here, just loading it to cache is enough
        print("✅ Captioning Model cached.")
    except Exception as e:
        print(f"Caption Model Preload Failed: {e}")

    # 2. Image Generation Model Preload (Example: Assuming Stable Diffusion 1.5)
    try:
        print("Preloading Image Generation Model weights (Stable Diffusion 1.5)...")
        
        # Model ID to be used:
        # Note: If your model is large (e.g., SDXL), consider using a smaller one or increasing timeout.
        model_id = "runwayml/stable-diffusion-v1-5" 
        
        # Pipeline লোড করা, এটি সমস্ত প্রয়োজনীয় ওয়েট ডাউনলোড করবে এবং ক্যাশ করবে।
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            use_auth_token=None,
            low_cpu_mem_usage=True
        )
        print("✅ Image Generation Model cached.")
    except Exception as e:
        print(f"Image Model Preload Failed: {e}")

    print("🚀 All required ML libraries and model weights are now cached in the volume.")


# -------------------------------------------------------
# Run Logic (Unchanged, but now depends on successful warmup)
# -------------------------------------------------------
@app.function(
    secrets=[
        modal.Secret.from_name("aws-credentials")
    ],
    gpu="L4",  # GPU তে রান হবে
    timeout=1200, # 3600 থেকে কমিয়ে 1200 সেকেন্ড (20 মিনিট) করা হলো, ফেইলিউর দ্রুত ধরার জন্য
    volumes={"/cache": cache_volume}, # মডেল ক্যাশিং এর জন্য ভলিউম
    # allow_concurrent_inputs=100 প্যারামিটারটি বাদ দেওয়া হয়েছে
)
@modal.concurrent(max_inputs=2) # Fix: max_inputs=100 যুক্ত করা হলো
@modal.web_server(port=8000)
def run():
    """
    এই ফাংশনটি Modal কন্টেইনারের ভেতরে রান হবে।
    """
    
    # 1. এনভায়রনমেন্ট ভেরিয়েবল সেটআপ (মডেল ক্যাশ করার জন্য)
    # যেহেতু Modal এ চলছে, তাই কন্টেইনারের ভেতরের /cache ফোল্ডার ব্যবহার করা হচ্ছে
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    # 2. টার্গেট ফাইল সেটআপ
    remote_app_path = "/root/project/generator_all/ui/app.py" # Corrected path for nested Streamlit file
    
    if not os.path.exists(remote_app_path):
        print("Error: app.py not found at expected path.")
        subprocess.run("ls -R /root/project", shell=True)
        return

    # 3. FastAPI/uvicorn command
    cmd = "uvicorn generator_all.ui.app:app --host 0.0.0.0 --port 8000 --log-level info"
    print(f"🚀 Starting FastAPI: {cmd}")
    
    # 4. সার্ভার স্টার্ট করা
    subprocess.run(cmd, shell=True, check=True)

# -------------------------------------------------------
# Local Entrypoints
# -------------------------------------------------------
@app.local_entrypoint()
def main():
    # Warmup ফাংশনকে explicit ভাবে কল করা হলো deployment-এর সময়।
    # এটি মডেল ডাউনলোড/ক্যাশিং নিশ্চিত করবে।
    warmup.remote()
    print("To serve locally: modal serve serve_modal.py")
    print("To deploy: modal deploy serve_modal.py")