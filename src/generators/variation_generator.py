import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class VariationGenerator:
    """Generates text variations using T5."""
    def __init__(self):
        print("⏳ Loading T5 model on CPU to save VRAM...")
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate_variations(self, caption, num_variations=5):
        input_text = f"paraphrase this caption in a detailed style: {caption}"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids

        outputs = self.model.generate(
            input_ids,
            max_length=60,
            num_return_sequences=num_variations,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95
        )

        variations = []
        print("📝 Generated Variations:")
        for i, output in enumerate(outputs):
            decoded_text = self.tokenizer.decode(output, skip_special_tokens=True)
            variations.append(decoded_text)
            print(f"{i+1}. {decoded_text}")
        return variations

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        print("🗑️ T5 model removed from memory.")
