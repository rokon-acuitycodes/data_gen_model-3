import os
import base64
import io
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

class CaptionGenerator:
    """Generates captions from images using OpenAI GPT-4o-mini."""
    def __init__(self, device=None):
        # Device argument is kept for compatibility but not used
        print("⏳ Loading OpenAI GPT-4o-mini client...")
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ WARNING: OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=api_key)

    def encode_image(self, image):
        """Encodes a PIL Image to a base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_caption(self, image):
        try:
            base64_image = self.encode_image(image)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            
            caption = response.choices[0].message.content
            print("-" * 30)
            print(f"🖼️ Original Caption: {caption}")
            print("-" * 30)
            return caption
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return "Error generating caption."

    def cleanup(self):
        # No heavy models to unload
        print("🗑️ OpenAI client cleanup (no local models to unload).")
