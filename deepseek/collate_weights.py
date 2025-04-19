### INFO: This is a helper script to allow participants to confirm their model is working!
import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig, get_peft_model_state_dict
import os
from fsdp_utils import AppState

adapter_name = "ExampleLora"

# INFO: This is a helper to map model names to StrongCompute Dataset ID's which store their weights!
model_weight_ids = {
    "DeepSeek-R1-Distill-Llama-70B": "e4b2dc79-79af-4a80-be71-c509469449b4",
    "DeepSeek-R1-Distill-Llama-8B": "255087c3-046c-421c-8fe3-6e333f14892a",
    "DeepSeek-R1-Distill-Qwen-1.5B": "6c796efa-7063-4a74-99b8-aab1c728ad98",
    "DeepSeek-R1-Distill-Qwen-14B": "39387beb-9824-4629-b19b-8f7b8f127150",
    "DeepSeek-R1-Distill-Qwen-32B": "84c2b2cb-95b4-4ce6-a2d4-6f210afad36b",
    "DeepSeek-R1-Distill-Qwen-7B": "6e226f91-6b7d-46ff-9f1e-4740efaf9b0e",
}

# TODO: set this to the model you chose from the dropdown at container startup!
# MODEL_NAME_SETME = "DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME_SETME = "DeepSeek-R1-Distill-Llama-8B"
mounted_dataset_path = f"/data/{model_weight_ids[MODEL_NAME_SETME]}"

# INFO: Loads the model WEIGHTS (assuming you've mounted it to your container!)
tokenizer = AutoTokenizer.from_pretrained(mounted_dataset_path)
model = AutoModelForCausalLM.from_pretrained(
    mounted_dataset_path, 
    use_cache=False, 
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0, # set to zero to see identical loss on all ranks
)

model = LoraModel(model, lora_config, adapter_name).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
state_dict = {"app": AppState(model, optimizer)}
# checkpoint_dir = "/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_95"
checkpoint_dir = "/shared/artifacts/beee0cb6-bd5a-4d4e-8ef9-5c1575e2bf8c/checkpoints/AtomicDirectory_checkpoint_10"
dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir) ## UPDATE WITH PATH TO CHECKPOINT DIRECTORY

# Extract just the LoRA weights (using the PEFT utility function)
lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)

# Save the consolidated adapter weights to a single file
# output_dir = "/shared/artifacts/consolidated_checkpoint"
# output_dir = "/shared/artifacts/18142399-96ff-4846-b55b-3be3822720f6/checkpoints/AtomicDirectory_checkpoint_95_consolidated_lora"
output_dir = "/shared/artifacts/beee0cb6-bd5a-4d4e-8ef9-5c1575e2bf8c/checkpoints/AtomicDirectory_checkpoint_10_consolidated_lora"
os.makedirs(output_dir, exist_ok=True)
torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))

# Save the adapter config
lora_config.save_pretrained(output_dir)

print(f"Consolidated checkpoint saved to {output_dir}")