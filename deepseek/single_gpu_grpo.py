# single_gpu_grpo.py
import os
import logging
import warnings
import time
import itertools
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, get_scheduler
from peft import LoraModel, LoraConfig

from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Configuration
ADAPTER_NAME = "ExampleLora"
NUM_GENERATIONS = 2  # Reduced for memory efficiency
MAX_NEW_TOKENS = 100
BATCH_SIZE = 1  # Start small to avoid OOM
GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients to simulate larger batch
DATASET_SIZE = 10  # Only use a small subset for testing/debugging
MAX_LENGTH = 384  # Reduced from 512 to save memory
SAVE_EVERY_STEPS = 50
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

def get_sequence_embeddings(model, input_ids, attention_mask):
    """Calculates sequence embeddings by averaging the last hidden state, ignoring padding."""
    # with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision for inference
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_dim)

    # Average pooling, ignoring padding
    non_padding_mask = attention_mask.unsqueeze(-1).float()
    summed_hidden_states = (last_hidden_state * non_padding_mask).sum(dim=1)
    num_non_padding_tokens = non_padding_mask.sum(dim=1)
    num_non_padding_tokens = torch.clamp(num_non_padding_tokens, min=1)
    mean_embeddings = summed_hidden_states / num_non_padding_tokens
    return mean_embeddings  # Shape: (batch_size, hidden_dim)

def compute_grpo_loss(model, prompt_ids, prompt_mask, generated_responses, gt_ids, gt_mask, num_generations, tokenizer):
    """
    Computes the GRPO loss with memory efficiency improvements.
    """
    batch_size = prompt_ids.size(0)
    prompt_len = prompt_ids.size(1)
    full_gen_len = generated_responses.size(1)
    
    # Reshape generated responses: (batch_size, num_generations, seq_len)
    generated_responses = generated_responses.view(batch_size, num_generations, -1)
    gen_attention_mask = (generated_responses != tokenizer.pad_token_id).long()

    # --- 1. Calculate Rewards (Cosine Similarity) ---
    # Get GT embeddings (only need to compute once per batch)
    gt_embeddings = get_sequence_embeddings(model, gt_ids, gt_mask)
    
    all_rewards = []
    all_log_probs = []
    
    for i in range(num_generations):
        # Process one generation at a time to save memory
        gen_ids_i = generated_responses[:, i, :]
        gen_mask_i = gen_attention_mask[:, i, :]
        
        # Get embeddings for this generation
        gen_embeddings_i = get_sequence_embeddings(model, gen_ids_i, gen_mask_i)
        rewards_i = F.cosine_similarity(gen_embeddings_i, gt_embeddings, dim=1)
        all_rewards.append(rewards_i)
        
        # Calculate log probs for this generation
        # with torch.cuda.amp.autocast(enabled=True):
        outputs = model(input_ids=gen_ids_i, attention_mask=gen_mask_i)
        logits = outputs.logits
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
        shift_labels = gen_ids_i[..., prompt_len:].contiguous()
        
        # Calculate loss only for the generated part
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        log_probs_i = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        log_probs_i = log_probs_i.view(batch_size, -1)
        
        # Mask out padding tokens
        response_mask = (shift_labels != tokenizer.pad_token_id).float()
        log_probs_i = (log_probs_i * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
        
        all_log_probs.append(log_probs_i)
        
        # Force garbage collection to free up memory
        del outputs, logits, shift_logits, shift_labels, log_probs_i, response_mask
        gc.collect()
        
    # Stack rewards and log_probs
    rewards = torch.stack(all_rewards, dim=1)
    log_probs = torch.stack(all_log_probs, dim=1)
    
    # --- 3. Calculate GRPO Pairwise Loss ---
    total_loss = 0
    num_pairs = 0
    
    for i, j in itertools.combinations(range(num_generations), 2):
        # Pairs where rewards are different
        preference_mask = (rewards[:, i] != rewards[:, j])
        
        if preference_mask.sum() == 0:
            continue  # Skip if all rewards in the pair are equal for this batch
            
        # Filter rewards and log probs based on preference
        rewards_i = rewards[preference_mask, i]
        rewards_j = rewards[preference_mask, j]
        log_probs_i = log_probs[preference_mask, i]
        log_probs_j = log_probs[preference_mask, j]
        
        # Determine preferred (p) and dispreferred (d) based on rewards
        higher_reward_mask = (rewards_i > rewards_j)
        log_probs_p = torch.where(higher_reward_mask, log_probs_i, log_probs_j)
        log_probs_d = torch.where(higher_reward_mask, log_probs_j, log_probs_i)
        
        # GRPO loss for this pair
        pair_loss = -F.logsigmoid(log_probs_p - log_probs_d).mean()
        total_loss += pair_loss
        num_pairs += 1
    
    if num_pairs == 0:
        # Handle cases where no pairs have different rewards
        logger.warning("No pairs with differing rewards found in this batch.")
        return torch.tensor(0.0, device=prompt_ids.device, requires_grad=True)
    
    return total_loss / num_pairs


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Enable gradient checkpointing to save memory
    torch.backends.cudnn.benchmark = True
    
    # Configure model path - adjust as needed
    model_path = "/data/255087c3-046c-421c-8fe3-6e333f14892a"  # Adjust based on your setup
    
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=False,  # Disable KV cache for training
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto",  # Let the model decide the best device mapping
    )
    
    # # Enable gradient checkpointing for memory efficiency
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()
    #     logger.info("Enabled gradient checkpointing")
    
    # Add LoRA adapter
    logger.info("Adding LoRA adapter")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
    )
    
    model = LoraModel(model, lora_config, ADAPTER_NAME)
    
    # Load dataset
    logger.info("Loading dataset")
    dataset = load_dataset("bespokelabs/bespoke-manim", split="train").select(range(10))
    if DATASET_SIZE > 0:
        dataset = dataset.select(range(min(DATASET_SIZE, len(dataset))))
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Define preprocessing function
    def preprocess_function(examples):
        prompts = [f"Question: {q}\nAnswer:" for q in examples['question']]
        gt_responses = examples['python_code']
        
        # Tokenize prompts
        prompt_encodings = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH // 2,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize ground truth responses
        gt_encodings = tokenizer(
            [resp + tokenizer.eos_token for resp in gt_responses],
            truncation=True,
            max_length=MAX_LENGTH // 2,
            padding=False,
            return_tensors=None
        )
        
        return {
            "prompt_ids": prompt_encodings["input_ids"],
            "prompt_mask": prompt_encodings["attention_mask"],
            "gt_ids": gt_encodings["input_ids"],
        }
    
    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Collate function
    def collate_fn(batch):
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
        
        # Combine prompt and GT for reward calculation
        combined_gt_ids = []
        combined_gt_masks = []
        max_combined_len = max_prompt_len + max_gt_len
        
        for i in range(len(batch)):
            prompt_part = padded_prompt_ids[i][:max_prompt_len]
            prompt_mask_part = padded_prompt_masks[i][:max_prompt_len]
            gt_part = padded_gt_ids[i][:max_gt_len]
            
            combined_ids = prompt_part + gt_part
            combined_mask = prompt_mask_part + [(1 if id != tokenizer.pad_token_id else 0) for id in gt_part]
            
            # Pad the combined sequence if necessary
            padding_len = max_combined_len - len(combined_ids)
            combined_ids += [tokenizer.pad_token_id] * padding_len
            combined_mask += [0] * padding_len
            
            combined_gt_ids.append(combined_ids[:max_combined_len])
            combined_gt_masks.append(combined_mask[:max_combined_len])
        
        return {
            'prompt_ids': torch.tensor(padded_prompt_ids),
            'prompt_mask': torch.tensor(padded_prompt_masks),
            'gt_ids': torch.tensor(combined_gt_ids),
            'gt_mask': torch.tensor(combined_gt_masks),
        }
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0,  # Reduce worker processes to save memory
    )
    
    # Setup optimizer with weight decay
    # Only apply weight decay to certain parameter types
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Create learning rate scheduler
    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(os.getcwd(), "grpo_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to {checkpoint_dir}")
    
    # Create generation config
    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        num_return_sequences=NUM_GENERATIONS,
    )
    
    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Training loop
    global_step = 0
    model.train()
    best_loss = float("inf")
    
    logger.info("Starting training")
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            gt_ids = batch["gt_ids"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            
            # Clear accumulated gradients every accumulation_steps
            if global_step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
            
            model.eval()  # Set to eval for generation
            # Generate multiple responses
            try:
                generated_responses = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=generation_config,
                )
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                continue
            model.train()  # Back to train mode for loss calculation
            
            if global_step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()

            loss = compute_grpo_loss(
                model, prompt_ids, prompt_mask, generated_responses.detach(), 
                gt_ids, gt_mask, NUM_GENERATIONS, tokenizer
            )
            # Scale loss by accumulation steps
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient scaling
            # scaler.scale(loss).backward()
            loss.backward()
            # backward call
            # Update weights if we've accumulated enough gradients
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(dataloader) - 1:
                # Optional gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Simple optimizer step
                optimizer.step()
                
                # Step the learning rate scheduler
                lr_scheduler.step()
                
                # Increment global step counter
                global_step += 1
                        
            # Log progress
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            
            if step % 10 == 0:
                logger.info(
                    f"Epoch: {epoch}, Step: {step}/{len(dataloader)}, "
                    f"Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}, "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.7f}"
                )
            
            # Save checkpoint
            if global_step % SAVE_EVERY_STEPS == 0:
                avg_loss = epoch_loss / num_batches
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_path = os.path.join(checkpoint_dir, f"grpo_checkpoint_step_{global_step}.pt")
                    
                    # Save model
                    logger.info(f"Saving checkpoint to {checkpoint_path}")
                    model.save_pretrained(checkpoint_path)
                    
                    # Save optimizer and scheduler
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'loss': best_loss,
                    }, os.path.join(checkpoint_path, "optimizer.pt"))
            
            # Ensure CUDA memory is cleared
            del generated_responses
            torch.cuda.empty_cache()
            gc.collect()
        
        # End of epoch logging
        epoch_end_time = time.time()
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(
            f"Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f}s. "
            f"Average loss: {avg_epoch_loss:.4f}"
        )
        
    logger.info("Training complete!")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "grpo_final_checkpoint.pt")
    logger.info(f"Saving final model to {final_checkpoint_path}")
    model.save_pretrained(final_checkpoint_path)
    
    # Save tokenizer for convenience
    tokenizer.save_pretrained(final_checkpoint_path)

if __name__ == "__main__":
    main()