import modal
from pathlib import Path
import subprocess

vol = modal.Volume.from_name("my-volume", create_if_missing=True)

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
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
        "python-docx",
        "pypdf>=3.0.0",
        "reportlab>=4.0.0",
        "faker>=15.0.0",
        "sentencepiece>=0.1.99",
        "boto3>=1.26.0",
        "scipy>=1.10.0",
        "openai",
        "python-dotenv",
        "websockets",
        "httpx",
        "starlette",
    )
    .add_local_dir(
        Path(__file__).parent,  # Mount the entire project directory
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
)

app = modal.App("unified-generator-app", image=image)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    volumes={"/media": vol},
    secrets=[modal.Secret.from_name("aws-secret")],
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=100)
def web():
    import os

    # Critical: set working directory so `generator_all` is importable.
    os.chdir("/root/project")
    # Ensure python can import local packages when uvicorn starts.
    os.environ.setdefault("PYTHONPATH", "/root/project:/root/project/src")

    # Start FastAPI and don't wait - Modal will keep container alive
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
