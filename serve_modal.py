import modal
import os
import shutil
from pathlib import Path
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
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
    .add_local_dir(Path(__file__).parent, remote_path="generator_all/ui/app.py")
)
app = modal.App("unified-generator-app", image=image)
@app.function(
    image=image,
    gpu="L4", # Request a GPU if available/needed
    timeout=600, # 10 minutes timeout
    concurrency_limit=2,
    volumes={"/media": vol},
    secrets=[modal.Secret.from_name("aws-credentials")]
)
@modal.web_server(8000)
def run_fastapi():
    import subprocess
    import os
    import shlex
    
    # Change to project directory so the FastAPI module import works
    os.chdir("/root/project")

    # Start FastAPI (OpenAPI will be available at /docs)
    cmd = "uvicorn generator_all.ui.app:app --host 0.0.0.0 --port 8000 --log-level info"
    subprocess.Popen(cmd, shell=True)