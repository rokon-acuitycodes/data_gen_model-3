import modal
from pathlib import Path
import subprocess

vol = modal.Volume.from_name("my-volume", create_if_missing=True)

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
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
    container_idle_timeout=60,
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=100)
def web():
    target = "/root/project/generator_all/ui/app.py"

    # Start Streamlit and don't wait - Modal will keep container alive
    subprocess.Popen(
        [
            "streamlit",
            "run",
            target,
            "--server.port",
            "8000",
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
            "--browser.gatherUsageStats",
            "false",
            "--server.fileWatcherType",
            "none",
        ]
    )


if __name__ == "__main__":
    app.deploy()
