### Inference script for models trained with single_gpu_grpo.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig, PeftConfig, PeftModel
import os

# Model IDs dictionary for different DeepSeek models
model_weight_ids = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "255087c3-046c-421c-8fe3-6e333f14892a",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "6e226f91-6b7d-46ff-9f1e-4740efaf9b0e",
}

# Set the model you want to use
MODEL_NAME = "DeepSeek-R1-Distill-Llama-8B"
mounted_dataset_path = f"/data/{model_weight_ids[MODEL_NAME]}"

# Path to the checkpoint saved during GRPO training
# Path to the saved checkpoint from single_gpu_grpo.py training
CHECKPOINT_PATH = "/shared/grpo_checkpoints/grpo_checkpoint_step_50_dir"
# CHECKPOINT_PATH = "/shared/grpo_checkpoints/grpo_checkpoint_step_50"  # Update this!
mounted_dataset_path = CHECKPOINT_PATH
print(f"Loading base model from {mounted_dataset_path}")
# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model first
model = AutoModelForCausalLM.from_pretrained(
    mounted_dataset_path, 
    use_cache=True,  # Enable KV cache for faster inference
    torch_dtype=torch.bfloat16,
    device_map="auto"  # Automatically determine the best device mapping
)

# print(f"Loading checkpoint from {CHECKPOINT_PATH}")
# # Load the trained model using the PeftModel.from_pretrained method
# # This is much simpler than manual state_dict mapping
# if os.path.exists(CHECKPOINT_PATH):
# # Load the full checkpoint which contains model weights
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     # Get the state dict from the checkpoint
#     if "model_state_dict" in checkpoint:
#         state_dict = checkpoint["model_state_dict"]
#     elif "state_dict" in checkpoint:
#         state_dict = checkpoint["state_dict"] 
#     else:
#         raise ValueError("Could not find model weights in checkpoint")

#     # Load the weights into the model
#     model.load_state_dict(state_dict)
#     print("Successfully loaded model weights from checkpoint")
# else:
#     model = PeftModel.from_pretrained(
#         model,                  # Base model
#         CHECKPOINT_PATH,        # Path to the saved checkpoint
#         is_trainable=False      # Set to False for inference
#     )

# Move model to GPU for inference
model = model.to("cuda")
model.eval()  # Set to evaluation mode

# Your prompt
prompt = "Using Manim standard library, write Python code to visually demonstrate the Pythagorean Theorem on a right triangle with sides 5, 12, 13. Include squares on each side of the triangle to show a² + b² = c², and add explanatory text for each step. Output only the Python code with comments"

# Format input for the model
formatted_prompt = f"Question: {prompt}\nAnswer: "

# Tokenize the input
encoding = tokenizer(formatted_prompt, return_tensors="pt")
input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")

print("Generating response...")
# Generate the response
with torch.no_grad():  # No need to track gradients during inference
    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=1500,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1  # Discourage repetition
    )

# Decode the generated output
answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("\n--- Generated Response ---\n")
print(answer[0])