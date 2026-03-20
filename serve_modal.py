import modal
import os
import shutil
from pathlib import Path
vol = modal.Volume.from_name("my-volume", create_if_missing=True)

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit>=1.35.0",
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
def run_streamlit():
    import subprocess
    import os
    import sys
    import shlex
    
    # Change to project directory so Streamlit can find .streamlit/config.toml
    os.chdir("/root/project")
    
    app_path = "generator_all/ui/app.py"
    target = shlex.quote(app_path)
    
    # Comprehensive flags to fix Modal proxy + Streamlit upload issues
    cmd = (
        f"streamlit run {target} "
        f"--server.port 8000 "
        f"--server.address 0.0.0.0 "
        f"--server.enableCORS false "
        f"--server.enableXsrfProtection false "
        f"--server.enableWebsocketCompression false "
        f"--server.headless true "
        f"--browser.gatherUsageStats false "
        f"--server.fileWatcherType none"
    )
    subprocess.Popen(cmd, shell=True)