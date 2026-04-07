import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def _load_public_model(loader_cls, model_name, **kwargs):
    load_kwargs = dict(kwargs)
    try:
        return loader_cls.from_pretrained(model_name, token=False, **load_kwargs)
    except TypeError:
        return loader_cls.from_pretrained(model_name, use_auth_token=False, **load_kwargs)


class T5Model:
    def __init__(self, model_name="google/flan-t5-small", device="cpu"):
        """
        Initialize T5 model with GPU support.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.tokenizer = _load_public_model(T5Tokenizer, model_name)
        self.model = _load_public_model(
            T5ForConditionalGeneration,
            model_name, 
            use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print(f"✅ T5 Model loaded on {device.upper()}")

    def _generate_batch(
        self,
        prompts,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
    ):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_from_instruction(
        self,
        input_text,
        instruction,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
    ):
        """Generate a single text output from an instruction plus input text."""
        if not input_text.strip():
            return ""

        prompt = f"{instruction.strip()} {input_text.strip()}"
        return self._generate_batch(
            [prompt],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )[0].strip()

    def generate_batch_from_instruction(
        self,
        input_texts,
        instruction,
        max_length=200,
        batch_size=5,
        temperature=0.7,
        top_p=0.9,
    ):
        """Generate multiple text outputs from a shared instruction."""
        results = []

        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            non_empty_batch = [text for text in batch if text.strip()]

            if not non_empty_batch:
                results.extend([""] * len(batch))
                continue

            prompts = [f"{instruction.strip()} {text.strip()}" for text in non_empty_batch]
            batch_results = self._generate_batch(
                prompts,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )
            results.extend([text.strip() for text in batch_results])

        return results

    def paraphrase_paragraph(self, para_text, max_length=200):
        """Generate a grammatically correct paraphrase of a paragraph using T5."""
        if not para_text.strip():
            return ""
        return self.generate_from_instruction(
            input_text=para_text,
            instruction="Paraphrase the following paragraph:",
            max_length=max_length,
        )

    def paraphrase_batch(self, para_texts, max_length=200, batch_size=5):
        """
        Paraphrase multiple paragraphs in batches for faster processing.
        
        Args:
            para_texts: List of paragraphs to paraphrase
            max_length: Max length for generated text
            batch_size: Number of paragraphs to process at once
        
        Returns:
            List of paraphrased paragraphs
        """
        return self.generate_batch_from_instruction(
            input_texts=para_texts,
            instruction="Paraphrase the following paragraph:",
            max_length=max_length,
            batch_size=batch_size,
        )
