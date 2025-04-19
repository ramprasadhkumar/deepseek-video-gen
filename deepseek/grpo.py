import os
import functools
import logging
import warnings
import time
import itertools

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import LoraModel, LoraConfig

from datasets import load_dataset

from cycling_utils import AtomicDirectory, atomic_torch_save, TimestampedTimer, InterruptableDistributedSampler
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

timer = TimestampedTimer("Start")

# suppressing warnings about missing modules in state_dict
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
# suppress warnings about "UserWarning: `_get_pg_default_device` will be deprecated" while saving and loading
warnings.filterwarnings("ignore", category=UserWarning)

ADAPTER_NAME = "ExampleLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD
NUM_GENERATIONS = 2 # Number of responses to generate per prompt for GRPO
MAX_NEW_TOKENS = 100 # Max tokens for generated responses

def get_sequence_embeddings(model, input_ids, attention_mask):
    """Calculates sequence embeddings by averaging the last hidden state, ignoring padding."""
    # Note: Accessing hidden states with FSDP might require specific configurations
    # or accessing the underlying module if output_hidden_states=True doesn't work directly.
    # This is a potential point of failure depending on FSDP/Transformer versions.
    try:
        # Access the original module
        if hasattr(model, 'module'):
            unwrapped_model = model.module
        else:
            unwrapped_model = model
        outputs = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # Shape: (batch_size, seq_len, hidden_dim)
    except TypeError as e:
         # Fallback if output_hidden_states isn't directly supported or causes issues with FSDP's wrapper
         # This might require model.module depending on how FSDP wraps. Needs testing.
         print(f"Warning: Could not get hidden states directly, attempting fallback. Error: {e}")
         # Assuming FSDP wraps the LoraModel which wraps the base model
         if hasattr(model, 'module') and hasattr(model.module, 'model'):
             base_model = model.module.model
         else:
             # This part might need adjustment based on the exact wrapping structure
             raise RuntimeError("Cannot access base model to get hidden states for reward calculation.") from e
         outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
         last_hidden_state = outputs.hidden_states[-1]

    # Average pooling, ignoring padding
    # Create a mask that's 1 for non-padding tokens and 0 for padding
    # Add a dimension for the hidden size
    non_padding_mask = attention_mask.unsqueeze(-1).float()
    # Sum hidden states where mask is 1
    summed_hidden_states = (last_hidden_state * non_padding_mask).sum(dim=1)
    # Count non-padding tokens per sequence
    num_non_padding_tokens = non_padding_mask.sum(dim=1)
    # Avoid division by zero for sequences with only padding (shouldn't happen with valid inputs)
    num_non_padding_tokens = torch.clamp(num_non_padding_tokens, min=1)
    # Calculate the mean embedding
    mean_embeddings = summed_hidden_states / num_non_padding_tokens
    return mean_embeddings # Shape: (batch_size, hidden_dim)

def compute_grpo_loss(model, prompt_ids, prompt_mask, generated_responses, gt_ids, gt_mask, num_generations, tokenizer):
    """
    Computes the GRPO loss.

    Args:
        model: The FSDP wrapped model.
        prompt_ids: Tensor of prompt token IDs.
        prompt_mask: Tensor of prompt attention masks.
        generated_responses: Tensor of generated response token IDs (including prompt). Shape: (batch_size * num_generations, seq_len)
        gt_ids: Tensor of ground truth token IDs (prompt + gt response).
        gt_mask: Tensor of ground truth attention masks.
        num_generations: Number of responses generated per prompt.
        tokenizer: The tokenizer.

    Returns:
        Scalar GRPO loss tensor.
    """
    batch_size = prompt_ids.size(0)
    prompt_len = prompt_ids.size(1)
    full_gen_len = generated_responses.size(1)
    gen_len = full_gen_len - prompt_len

    # Reshape generated responses: (batch_size, num_generations, seq_len)
    generated_responses = generated_responses.view(batch_size, num_generations, -1)
    # Create attention mask for generated responses (assuming prompt_mask covers the prompt part)
    gen_attention_mask = (generated_responses != tokenizer.pad_token_id).long()

    # --- 1. Calculate Rewards (Cosine Similarity) ---
    # Get GT embeddings (only need to compute once per batch)
    gt_embeddings = get_sequence_embeddings(model, gt_ids, gt_mask) # (batch_size, hidden_dim)

    all_rewards = []
    for i in range(num_generations):
        gen_ids_i = generated_responses[:, i, :] # (batch_size, full_gen_len)
        gen_mask_i = gen_attention_mask[:, i, :] # (batch_size, full_gen_len)
        gen_embeddings_i = get_sequence_embeddings(model, gen_ids_i, gen_mask_i) # (batch_size, hidden_dim)
        rewards_i = F.cosine_similarity(gen_embeddings_i, gt_embeddings, dim=1) # (batch_size,)
        all_rewards.append(rewards_i)

    # Stack rewards: (batch_size, num_generations)
    rewards = torch.stack(all_rewards, dim=1)

    # --- 2. Calculate Log Probabilities of Generated Responses ---
    all_log_probs = []
    for i in range(num_generations):
        gen_ids_i = generated_responses[:, i, :] # (batch_size, full_gen_len)
        gen_mask_i = gen_attention_mask[:, i, :] # (batch_size, full_gen_len)

        # Use model's forward pass to get logits
        outputs = model(input_ids=gen_ids_i, attention_mask=gen_mask_i)
        logits = outputs.logits # (batch_size, full_gen_len, vocab_size)

        # Shift logits and labels for next token prediction loss
        shift_logits = logits[..., prompt_len-1:-1, :].contiguous() # Use prompt_len-1 because input starts from 0, target starts from 1
        shift_labels = gen_ids_i[..., prompt_len:].contiguous() # Targets start after the prompt

        # Calculate loss only for the generated part
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        log_probs_i = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        log_probs_i = log_probs_i.view(batch_size, -1) # (batch_size, gen_len)

        # Mask out padding tokens in the loss calculation
        response_mask = (shift_labels != tokenizer.pad_token_id).float()
        log_probs_i = (log_probs_i * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1) # Average log prob per token
        # Alternative: sum log probs: log_probs_i = (log_probs_i * response_mask).sum(dim=1)

        all_log_probs.append(log_probs_i)

    # Stack log probs: (batch_size, num_generations)
    log_probs = torch.stack(all_log_probs, dim=1)

    # --- 3. Calculate GRPO Pairwise Loss ---
    total_loss = 0
    num_pairs = 0

    for i, j in itertools.combinations(range(num_generations), 2):
        # Pairs where rewards are different
        preference_mask = (rewards[:, i] != rewards[:, j])

        if preference_mask.sum() == 0:
            continue # Skip if all rewards in the pair are equal for this batch

        # Filter rewards and log probs based on preference
        rewards_i = rewards[preference_mask, i]
        rewards_j = rewards[preference_mask, j]
        log_probs_i = log_probs[preference_mask, i]
        log_probs_j = log_probs[preference_mask, j]

        # Determine preferred (p) and dispreferred (d) based on rewards
        higher_reward_mask = (rewards_i > rewards_j)
        log_probs_p = torch.where(higher_reward_mask, log_probs_i, log_probs_j)
        log_probs_d = torch.where(higher_reward_mask, log_probs_j, log_probs_i)

        # GRPO loss for this pair: -log_sigmoid(log_prob_preferred - log_prob_dispreferred)
        pair_loss = -F.logsigmoid(log_probs_p - log_probs_d).mean() # Average over the batch samples where preferences exist
        total_loss += pair_loss
        num_pairs += 1

    if num_pairs == 0:
       # Handle cases where no pairs have different rewards (e.g., only one generation or all rewards equal)
       # Return a zero loss tensor that requires gradients to avoid issues downstream
       print("Warning: No pairs with differing rewards found in this batch.")
       # Find any parameter that requires grad to attach the zero loss to its device and graph
       dummy_param = next(iter(p for p in model.parameters() if p.requires_grad), None)
       if dummy_param is not None:
            return torch.tensor(0.0, device=dummy_param.device, requires_grad=True)
       else:
            # Should not happen in training, but as a fallback
            return torch.tensor(0.0, requires_grad=True)


    return total_loss / num_pairs # Average loss over the pairs considered

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    rank = int(os.environ["RANK"]) # Global rank
    local_device = int(os.environ["LOCAL_RANK"]) # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of global ranks
    model_path = os.path.join("/data", args.dataset_id)
    torch.cuda.set_device(local_device)

    timer.report(f"Init process group for world size: {world_size}")

    # creating a device mesh enables FSDP to use DTensor instead of ShardedTensor
    # For GRPO, DTensor might have complications with generate/reward logic, consider standard DDP if issues arise
    # device_mesh = init_device_mesh("cuda", (world_size,))
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # Using DDP-style init

    assert bfSixteen_ready(), "ERROR: System not BF16 ready."

    # pre-trained model weights should be mounted at /data
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        print("Warning: pad_token not set, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token


    # save CPU RAM by loading non-main rank models to 'meta' device
    # Important: Generation might be tricky with 'meta' device init + FSDP.
    # If generation fails, consider loading fully on CPU/GPU rank 0 first, then wrapping with FSDP.
    model_load_device = "cpu" if rank == 0 else "meta" # Load on CPU rank 0 for potential stability with generate
    # model_load_device = torch.cuda.current_device() if rank == 0 else "meta" # Alternative: load directly to GPU rank 0

    try:
        with torch.device(model_load_device):
             model = AutoModelForCausalLM.from_pretrained(
                 model_path,
                 use_cache=False, # Important for FSDP/training
                 torch_dtype=torch.bfloat16
             )
    except Exception as e:
         print(f"Error loading model on device {model_load_device}: {e}")
         print("Trying to load directly on current device rank 0")
         if rank == 0:
             model = AutoModelForCausalLM.from_pretrained(
                 model_path,
                 use_cache=False,
                 torch_dtype=torch.bfloat16
             )
             model = model.to(torch.cuda.current_device())
         else:
             # Non-rank 0 waits - synchronization needed before FSDP
             dist.barrier() # Wait for rank 0 to load
             # Re-attempt meta loading (might still fail if initial load failed)
             with torch.device("meta"):
                 model = AutoModelForCausalLM.from_pretrained(
                     model_path,
                     use_cache=False,
                     torch_dtype=torch.bfloat16
                 )


    if rank == 0:
        print(f"Rank {rank} model params on device: {set(p.data.device for p in model.parameters())}")
    else:
        print(f"Rank {rank} model params on device: {set(p.data.device for p in model.parameters())} (expected meta)")

    timer.report(f"Loaded model: {count_trainable_parameters(model)}")

    # inject PEFT modules
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, # set to zero to see identical loss on all ranks
    )

    model = LoraModel(model, lora_config, ADAPTER_NAME)

    timer.report(f"PEFT model: {count_trainable_parameters(model)}")

    # wrap model in FSDP
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )

    # Decide param_init_fn based on loading strategy
    param_init_device = torch.cuda.current_device() if model_load_device == "meta" else None # Only init if loaded to meta

    model = FSDP(model,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        # param_init_fn=lambda mod: mod.to_empty(device=param_init_device, recurse=False) if param_init_device else None, # Init only if loaded to meta
        sync_module_states=True, # broadcast model weights from main rank 0
        # device_mesh=device_mesh # Not using device_mesh with DDP-style init
        use_orig_params=True # Recommended for newer PyTorch versions, helps with param access
    )
    dist.barrier() # Ensure all ranks have wrapped the model

    timer.report("FSDP wrapped model and broadcast to GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # --- Dataset Preparation (Modified for GRPO) ---
    dataset = load_dataset("bespokelabs/bespoke-manim", split="train")[:10]

    # Simple check for required columns
    if 'question' not in dataset.column_names or 'python_code' not in dataset.column_names:
        raise ValueError("Dataset must contain 'question' and 'python_code' columns.")

    max_length = 512 # Max combined length for prompt+gt for reward calc, adjust as needed

    def preprocess_function(examples):
        prompts = [f"Question: {q}\nAnswer:" for q in examples['question']]
        gt_responses = examples['python_code']

        # Tokenize prompts
        prompt_encodings = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length // 2, # Allocate roughly half length to prompt
            padding=False, # Pad later in collate_fn
            return_tensors=None
        )

        # Tokenize ground truth responses
        # Add EOS token to signal end of GT response during reward calculation/comparison
        gt_encodings = tokenizer(
             [resp + tokenizer.eos_token for resp in gt_responses],
             truncation=True,
             max_length=max_length // 2, # Allocate roughly half length to GT
             padding=False, # Pad later in collate_fn
             return_tensors=None
         )


        processed = {
            "prompt_ids": prompt_encodings["input_ids"],
            "prompt_mask": prompt_encodings["attention_mask"],
            "gt_ids": gt_encodings["input_ids"],
            # gt_mask is not strictly needed if we use gt_ids != pad_token_id later,
            # but can be useful. We'll create it in collate_fn after padding.
        }
        return processed

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing prompts and ground truths",
    )

    # --- Collate Function (Modified for GRPO) ---
    def collate_fn(batch):
        # Find max lengths in the batch
        max_prompt_len = max(len(x['prompt_ids']) for x in batch)
        max_gt_len = max(len(x['gt_ids']) for x in batch)

        # Pad prompts
        padded_prompt_ids = []
        padded_prompt_masks = []
        for x in batch:
            prompt_len = len(x['prompt_ids'])
            padding_len = max_prompt_len - prompt_len
            padded_prompt_ids.append(x['prompt_ids'] + [tokenizer.pad_token_id] * padding_len)
            padded_prompt_masks.append(x['prompt_mask'] + [0] * padding_len)

        # Pad ground truths
        padded_gt_ids = []
        for x in batch:
            gt_len = len(x['gt_ids'])
            padding_len = max_gt_len - gt_len
            padded_gt_ids.append(x['gt_ids'] + [tokenizer.pad_token_id] * padding_len)

        # Combine prompt and GT for reward calculation input
        combined_gt_ids = []
        combined_gt_masks = []
        max_combined_len = max_prompt_len + max_gt_len
        for i in range(len(batch)):
            prompt_part = padded_prompt_ids[i][:max_prompt_len] # Use original length before padding
            prompt_mask_part = padded_prompt_masks[i][:max_prompt_len]
            gt_part = padded_gt_ids[i][:max_gt_len]

            combined_ids = prompt_part + gt_part
            combined_mask = prompt_mask_part + [(1 if id != tokenizer.pad_token_id else 0) for id in gt_part]

            # Pad the combined sequence if necessary (should match based on max lengths)
            padding_len = max_combined_len - len(combined_ids)
            combined_ids += [tokenizer.pad_token_id] * padding_len
            combined_mask += [0] * padding_len

            combined_gt_ids.append(combined_ids[:max_combined_len]) # Truncate just in case
            combined_gt_masks.append(combined_mask[:max_combined_len])


        return {
            'prompt_ids': torch.tensor(padded_prompt_ids),
            'prompt_mask': torch.tensor(padded_prompt_masks),
            'gt_ids': torch.tensor(combined_gt_ids), # Combined prompt + gt for reward calc
            'gt_mask': torch.tensor(combined_gt_masks), # Combined mask
        }

    train_sampler = InterruptableDistributedSampler(tokenized_dataset)

    batch_size = 2 # Adjust based on GPU memory (GRPO is memory intensive)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    steps_per_epoch = len(dataloader)
    best_loss = float("inf") # Loss here is GRPO loss, lower is better

    # load checkpoint if found
    try:
        # output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
        output_directory = os.environ.get("CHECKPOINT_ARTIFACT_PATH", "/shared/artifacts/karthik-local-artifact-path")
    except KeyError as error:
        print("Must set env var CHECKPOINT_ARTIFACT_PATH so we know where to save checkpoints!")
        exit(1)
    saver = AtomicDirectory(output_directory=output_directory, is_master=rank==0)

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)

        # Use FSDP specific loading
        state_dict = {"model": model, "optim": optimizer} # FSDP expects the model/optim objects
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(latest_checkpoint_path),
        )

        train_state = torch.load(os.path.join(latest_checkpoint_path, "train_state.pt"))
        dataloader.sampler.load_state_dict(train_state["sampler"])
        best_loss = train_state["best_loss"]
        # Load AppState style for reference if needed, but FSDP handles it above
        # app_state = AppState(model, optimizer)
        # state_dict_app = { "app": app_state }
        # dcp.load(state_dict=state_dict_app, checkpoint_id=latest_checkpoint_path)


        timer.report("Loaded checkpoint")


    # --- Training Loop (Modified for GRPO) ---
    num_epochs = 10 # Adjust as needed
    save_every_steps = 30
    model.train()

    # Generation config
    # Ensure pad_token_id is set for generation
    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True, # Sample multiple responses
        top_k=50,       # Example sampling params
        top_p=0.9,
        num_return_sequences=NUM_GENERATIONS,
    )


    for epoch in range(dataloader.sampler.epoch, num_epochs):

        dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:

            step = dataloader.sampler.progress // dataloader.batch_size
            is_last_step = (step + 1) == steps_per_epoch
            is_save_step = ((step + 1) % save_every_steps == 0) or is_last_step

            # Move batch to device
            prompt_ids = batch["prompt_ids"].to(torch.cuda.current_device())
            prompt_mask = batch["prompt_mask"].to(torch.cuda.current_device())
            gt_ids = batch["gt_ids"].to(torch.cuda.current_device())
            gt_mask = batch["gt_mask"].to(torch.cuda.current_device())

            optimizer.zero_grad()

            # --- GRPO Forward Pass ---
            # 1. Generate multiple responses
            # FSDP might require a context manager for generation if internal ops aren't wrapped
            # Try direct generation first
            with torch.no_grad(): # Don't track gradients during generation
                try:
                    # Ensure model is in eval mode for generation if dropout/batchnorm are involved
                    # model.eval() # Temporarily set to eval for generation - messes up LoRA? Maybe not needed.
                    generated_responses = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=generation_config,
                    )
                    # model.train() # Switch back to train mode
                except Exception as e:
                    print(f"Error during model.generate: {e}")
                    # Potential issue: FSDP wrapping interference.
                    # Try generating with the underlying model if accessible:
                    # if hasattr(model, 'module'):
                    #     generated_responses = model.module.generate(...)
                    # else:
                    #     raise e
                    raise e


            # generated_responses shape: (batch_size * num_generations, full_seq_len) including prompt
            # Need to detach as generation shouldn't be part of graph for loss calc itself

            # 2. Compute GRPO loss (handles reward and log_prob calculation internally)
            # This part needs gradients tracked
            loss = compute_grpo_loss(model, prompt_ids, prompt_mask, generated_responses.detach(), gt_ids, gt_mask, NUM_GENERATIONS, tokenizer)

            # Backward pass on GRPO loss
            loss.backward()
            optimizer.step()
            dataloader.sampler.advance(prompt_ids.size(0)) # Advance by original batch size


            # Synchronize loss for reporting
            sync_loss = loss.detach().clone() # Use detach here
            dist.all_reduce(sync_loss)
            avg_loss = sync_loss.item() / world_size # Average the loss across ranks

            timer.report(f"Epoch {epoch} Step {step} GRPO Loss: {avg_loss:.4f}") # Report GRPO loss

            # --- Checkpointing ---
            if is_save_step:
                # FSDP Checkpointing
                state_dict = {"model": model, "optim": optimizer}
                checkpoint_dir = os.path.join(output_directory, f"step_{step}_rank_{rank}") # Rank-specific dir

                dcp.save(
                    state_dict=state_dict,
                    storage_writer=dcp.FileSystemWriter(checkpoint_dir),
                )
                # Barrier to ensure all ranks finish saving before rank 0 saves train_state/symlinks
                dist.barrier()

                if rank == 0:
                    # Need to consolidate metadata or save train_state separately
                    # Save sampler state and best loss (using avg_loss now)
                    if avg_loss < best_loss:
                         best_loss = avg_loss
                    train_state = {
                         "sampler": dataloader.sampler.state_dict(),
                         "best_loss": best_loss,
                         "epoch": epoch,
                         "step": step,
                         # Add any other state needed
                    }
                    # Use the main checkpoint directory for the consolidated state
                    consolidated_checkpoint_dir = os.path.join(output_directory, f"step_{step}")
                    # Ensure the target directory exists for rank 0
                    os.makedirs(consolidated_checkpoint_dir, exist_ok=True)

                    atomic_torch_save(
                        train_state,
                        os.path.join(consolidated_checkpoint_dir, "train_state.pt")
                    )
                    # Symlink to the consolidated directory
                    saver.symlink_latest(consolidated_checkpoint_dir)

                dist.barrier() # Ensure rank 0 finishes before others proceed/cleanup

                timer.report("Saved checkpoint")

        dataloader.sampler.reset_progress()

    timer.report("Done.")

    dist.barrier()
    dist.destroy_process_group() 