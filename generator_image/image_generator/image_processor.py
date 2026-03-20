import os
from PIL import Image

class ImageProcessor:
    """Handles image upload and basic processing."""
    def upload_image(self):
        print("Please enter the path to an image file...")
        image_path = input("Image path: ").strip()
        if not os.path.exists(image_path):
            print("❌ File does not exist. Exiting.")
            sys.exit()
        print(f'Using image file "{image_path}"')
        image = Image.open(image_path).convert('RGB')
        return image
