import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def _load_public_model(loader_cls, model_name, **kwargs):
    load_kwargs = dict(kwargs)
    try:
        return loader_cls.from_pretrained(model_name, token=False, **load_kwargs)
    except TypeError:
        return loader_cls.from_pretrained(model_name, use_auth_token=False, **load_kwargs)

class VariationGenerator:
    """Generates text variations using T5."""
    def __init__(self):
        print("⏳ Loading T5 model on CPU to save VRAM...")
        self.model_name = "google/flan-t5-base"
        self.tokenizer = _load_public_model(AutoTokenizer, self.model_name)
        self.model = _load_public_model(AutoModelForSeq2SeqLM, self.model_name)

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
