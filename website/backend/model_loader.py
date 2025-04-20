# Model loading function that integrates with deepseek/consolidated_inference.py
import os
import sys
import torch
import shutil
from pathlib import Path

# Add the deepseek directory to the path so we can import from it
sys.path.append('/root/deepseek-video-gen/deepseek')

# Flag to control whether to use the real model or fallback
USE_REAL_MODEL = os.environ.get("USE_REAL_MODEL", "false").lower() == "true"

def load_model():
    """Load the DeepSeek model or return a mock model if not available"""
    if not USE_REAL_MODEL:
        print("Using mock model (fallback mode)")
        return MockModel()
    
    try:
        # Import here to avoid errors if the dependencies aren't installed
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraModel, LoraConfig
        
        # Get model path from environment variable or use default
        model_name = os.environ.get("DEEPSEEK_MODEL_NAME", "DeepSeek-R1-Distill-Llama-8B")
        mounted_dataset_path = os.environ.get("DEEPSEEK_DATASET_PATH", "/data/255087c3-046c-421c-8fe3-6e333f14892a")
        
        print(f"Loading DeepSeek model: {model_name} from {mounted_dataset_path}")
        
        # Check if the path exists
        if not os.path.exists(mounted_dataset_path):
            print(f"Warning: Model path {mounted_dataset_path} does not exist. Using mock model.")
            return MockModel()
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
        model = AutoModelForCausalLM.from_pretrained(
            mounted_dataset_path, 
            use_cache=False, 
            torch_dtype=torch.bfloat16
        )
        
        # Configure LoRA if needed
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
        )
        model = LoraModel(model, lora_config, "DeepSeekLora")
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return DeepSeekModel(model, tokenizer, device)
    except Exception as e:
        print(f"Error loading DeepSeek model: {e}")
        print("Falling back to mock model")
        return MockModel()


class DeepSeekModel:
    """Wrapper for the DeepSeek model to provide a consistent interface"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt):
        """Generate code from a prompt using the DeepSeek model"""
        try:
            # Format the prompt for the model
            deepseek_input = f"Question: {prompt}\nAnswer: "
            
            # Tokenize the input
            encoding = self.tokenizer(deepseek_input, return_tensors="pt")
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Generate the output
            with torch.no_grad():
                generate_ids = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask, 
                    pad_token_id=self.tokenizer.eos_token_id, 
                    max_new_tokens=1500, 
                    do_sample=True, 
                    temperature=0.8
                )
            
            # Decode the output
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # Extract just the generated code part (after the prompt)
            generated_text = output[0]
            if "Answer:" in generated_text:
                generated_text = generated_text.split("Answer:", 1)[1].strip()
            
            return generated_text
        except Exception as e:
            print(f"Error generating with DeepSeek model: {e}")
            # Fall back to mock model if generation fails
            return MockModel().generate(prompt)


class MockModel:
    """Mock model that returns predefined code for testing"""
    def __init__(self):
        self.fallback_code_path = Path("/root/deepseek-video-gen/website/data/pythogoras_theorem.py")
    
    def generate(self, prompt):
        """Return the fallback code regardless of the prompt"""
        try:
            with open(self.fallback_code_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading fallback code: {e}")
            return "from manim import *\n\nclass FallbackScene(Scene):\n    def construct(self):\n        text = Text(\"Fallback scene - no code available\")\n        self.play(Write(text))\n        self.wait(2)"
