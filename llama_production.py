# llama_production.py

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import torch
import logging
import os

logger = logging.getLogger(__name__)

class LlamaChatModel:
    def __init__(self,
                 base_model_path="meta-llama/Llama-3.2-1B-Instruct",
                 adapter_path="./checkpoint-2450"):
        
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.using_adapter = False
        
        try:
            self._load_model_with_adapter()
        except Exception as e:
            logger.warning(f"Failed to load adapter: {e}")
            logger.info("Falling back to base model without adapter...")
            self._load_base_model_only()

        # Generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

    def _load_model_with_adapter(self):
        """Try to load model with PEFT adapter"""
        # First try to load tokenizer from adapter path
        if os.path.exists(self.adapter_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            
        if self.tokenizer.pad_token is None:
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
            self.base_model_path,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        # Try to load adapter
        if os.path.exists(self.adapter_path):
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.using_adapter = True
            logger.info("✅ Successfully loaded model with PEFT adapter")
        else:
            self.model = base_model
            self.using_adapter = False
            logger.warning("⚠️ Adapter path not found, using base model only")
            
        self.model.eval()

    def _load_base_model_only(self):
        """Fallback: load only base model without adapter"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load base model only
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        self.model.eval()
        self.using_adapter = False
        logger.info("✅ Successfully loaded base model without adapter")

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            return "❌ Prompt is empty."

        # Add a note if using fallback
        adapter_status = "🔧 (using fine-tuned model)" if self.using_adapter else "⚠️ (using base model - adapter failed to load)"
        
        try:
            # Format prompt for Llama-3.2 chat format
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=self.gen_config,
                    do_sample=True
                )
                
                # Decode only the new tokens (response part)
                response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
                
                # Clean up response
                response = response.strip()
                if not response:
                    response = "I understand your question, but I'm having trouble generating a response right now."
                
                return f"{response} {adapter_status}"
                
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"❌ Error generating response: {str(e)}. Please try again."

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "base_model": self.base_model_path,
            "adapter_path": self.adapter_path,
            "using_adapter": self.using_adapter,
            "device": str(self.model.device) if self.model else "unknown",
            "model_type": type(self.model).__name__ if self.model else "unknown"
        }
