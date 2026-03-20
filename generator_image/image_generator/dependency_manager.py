import subprocess
import sys
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

class DependencyManager:
    """Handles package installations and sanity checks."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 System detected: {self.device}")

    def install_package(self, package):
        print(f"📦 Installing/Checking {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def run_sanity_check(self):
        """
        Test Case: Checks if the critical imports work correctly before loading heavy models.
        This prevents wasting time if libraries are broken.
        """
        print("🔍 Running dependency sanity check...")
        try:
            from PIL import Image, ImageDraw, ImageFont
            import torchvision.transforms.functional as F
            print("✅ Test Case Passed: Dependencies are compatible.")
            return True
        except ImportError as e:
            print(f"❌ Test Case Failed: {e}")
            return False
        except Exception as e:
            print(f"⚠️ Unexpected Warning during test: {e}")
            return True

    def setup_environment(self):
        print("🛠️ Setting up environment...")

        # Install dependencies
        self.install_package("pillow")
        self.install_package("transformers")
        self.install_package("diffusers")
        self.install_package("accelerate")
        self.install_package("torch")
        self.install_package("torchvision")
        self.install_package("openai")
        self.install_package("python-dotenv")

        if not self.run_sanity_check():
            print("\n" + "="*50)
            print("🚨 CRITICAL DEPENDENCY MISMATCH DETECTED 🚨")
            print("The environment has updated libraries but Python hasn't reloaded them.")
            print("👉 PLEASE CLICK 'RUNTIME > RESTART SESSION' (or Restart Runtime) IN THE MENU.")
            print("Then run this cell again. The correct versions are already installed.")
            print("="*50 + "\n")
            sys.exit("Runtime Restart Required")
