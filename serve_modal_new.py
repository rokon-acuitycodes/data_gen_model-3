import os
import subprocess
from pathlib import Path

import modal


vol = modal.Volume.from_name("unified-gen-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
        "pandas",
        "numpy",
        "requests",
        "pillow",
        "torch",
        "torchaudio",
        "transformers>=4.52.0",
        "accelerate>=1.6.0",
        "huggingface_hub>=0.31.0",
        "hf_transfer",
        "safetensors>=0.4.5",
        "protobuf>=5.0.0",
        "av>=12.0.0",
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
    .run_commands(
        "pip install -U git+https://github.com/huggingface/diffusers.git transformers accelerate huggingface_hub safetensors protobuf av torchaudio"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": "/root/project:/root/project/src",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "VIDEO_PIPELINE_MODE": "distilled_two_stage",
            "VIDEO_MODEL_ID": "rootonchair/LTX-2-19b-distilled",
            "VIDEO_BASE_MODEL_ID": "Lightricks/LTX-2",
            "VIDEO_INFERENCE_STEPS": "8",
            "VIDEO_GUIDANCE_SCALE": "1.0",
            "VIDEO_STAGE2_INFERENCE_STEPS": "3",
            "VIDEO_STAGE2_GUIDANCE_SCALE": "1.0",
            "VIDEO_STAGE2_SCALE_FACTOR": "2",
            "VIDEO_FRAME_RATE": "24.0",
            "VIDEO_OFFLOAD_MODE": "sequential",
            "VIDEO_MAX_SEQUENCE_LENGTH": "256",
            "VIDEO_OOM_RETRY_MAX_SEQUENCE_LENGTH": "128",
        }
    )
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
)

app = modal.App("unified-generator-app", image=image)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
    volumes={"/cache": vol},
    secrets=[
        modal.Secret.from_name("aws-secret"),
        modal.Secret.from_name("HF_TOKEN"),
    ],
    min_containers=0,
    max_containers=1,
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=1)
def web():
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface/transformers"
    os.chdir("/root/project")

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
