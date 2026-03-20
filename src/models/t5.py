import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Model:
    def __init__(self, model_name="google/flan-t5-small", device="cpu"):
        """
        Initialize T5 model with GPU support.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            use_safetensors=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        print(f"✅ T5 Model loaded on {device.upper()}")

    def paraphrase_paragraph(self, para_text, max_length=200):
        """Generate a grammatically correct paraphrase of a paragraph using T5."""
        if not para_text.strip():
            return ""
        prompt = f"Paraphrase the following paragraph: {para_text}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up prompt prefix
        if "Paraphrase the following paragraph: " in generated:
            generated = generated.replace("Paraphrase the following paragraph: ", "").strip()
        return generated

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
        results = []
        
        for i in range(0, len(para_texts), batch_size):
            batch = para_texts[i:i+batch_size]
            # Filter empty paragraphs
            non_empty_batch = [p for p in batch if p.strip()]
            
            if not non_empty_batch:
                results.extend([""] * len(batch))
                continue
            
            # Prepare prompts
            prompts = [f"Paraphrase the following paragraph: {p}" for p in non_empty_batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode batch
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Clean up prompts
            cleaned_results = []
            for text in batch_results:
                if "Paraphrase the following paragraph: " in text:
                    text = text.replace("Paraphrase the following paragraph: ", "").strip()
                cleaned_results.append(text)
            
            results.extend(cleaned_results)
        
        return results
