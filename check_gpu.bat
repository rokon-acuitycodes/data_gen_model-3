@echo off
rem ------------------------------------------------------------
rem GPU check script for Windows
rem ------------------------------------------------------------

rem 1. Show NVIDIA driver information (requires NVIDIA driver installed)
nvidia-smi

rem 2. Show CUDA availability via Python (requires Python and torch installed)
python - <<END
import torch, sys
print("\n--- PyTorch CUDA check ---")
print("torch version:", torch.__version__)
print("CUDA compiled into torch:", torch.version.cuda)
print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f"Total memory: {props.total_memory/1e9:.2f} GB")
END

rem ------------------------------------------------------------
rem End of script
rem ------------------------------------------------------------
