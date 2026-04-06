import os
import modal
from pathlib import Path
import subprocess

# Define a volume for persisting Hugging Face models and other data
vol = modal.Volume.from_name("unified-gen-volume", create_if_missing=True)

# Define the image with all required dependencies for LTX-2.3 and traditional generators
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0") # Added for OpenCV/PIL dependencies
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
        "accelerate",
        "huggingface_hub",
        "hf_transfer",     # 🔥 For high-speed downloads
        "safetensors",      # 🔥 Required for many modern models
        "openpyxl",
        "python-docx",
        "pypdf>=3.0.0",
        "reportlab>=4.0.0",
        "faker>=15.0.0",
        "sentencepiece>=0.1.99",
        "boto3>=1.26.0",
        "scipy>=1.10.0",
        "openai",
        "python-dotenv",
    )
    # Ensure we use the latest diffusers (LTX-2.3 often requires dev branch)
    .run_commands("pip install -U git+https://github.com/huggingface/diffusers.git")
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/root/project",
        ignore=[
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "outputs",
        ],
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", # 🔥 Enable HF Transfer speedup
        "PYTHONPATH": "/root/project:/root/project/src"
    })
)

app = modal.App("unified-generator-app", image=image)

@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    # Mount volume for persistent model caching (HF_HOME)
    volumes={"/cache": vol},
    # Ensure you have these secrets created in your Modal dashboard
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("HF_TOKEN"), # 🔥 Added for gated models like LTX
    ],
    min_containers=0,
    max_containers=1,
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=100)
def web():
    import os
    
    # Configure Hugging Face to use the persistent volume
    os.environ["HF_HOME"] = "/cache/huggingface"
    
    # Critical: set working directory
    os.chdir("/root/project")

    # Start FastAPI
    subprocess.Popen(
        [
            "uvicorn",
            "generator_all.ui.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--log-level",
            "info",
        ]
    )

if __name__ == "__main__":
    app.deploy()
