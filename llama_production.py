# llama_production.py

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch

class LlamaChatModel:
    def __init__(self,
                 base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 adapter_path="./checkpoint-2450"):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            return "❌ Prompt is empty."

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=self.gen_config
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
            except Exception as e:
                return f"❌ Error: {str(e)}"
