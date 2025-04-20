# single_gpu_grpo.py
import os
import logging
import warnings
import time
import itertools
import gc
import re # Added for parsing Gemini response

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, get_scheduler
from peft import LoraModel, LoraConfig

from datasets import load_dataset
import google.generativeai as genai # Added Gemini import
from dotenv import load_dotenv # Added dotenv import

# Load environment variables from .env file
load_dotenv()

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Added for Gemini API Key

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable not set. Reward calculation will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
# Consider making the model name configurable
gemini_reward_model = None
if GEMINI_API_KEY:
    try:
        gemini_reward_model = genai.GenerativeModel('gemini-1.5-flash') # Or another suitable model
        logger.info("Gemini model initialized for reward scoring.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        gemini_reward_model = None # Ensure it's None if initialization fails

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

def get_gemini_reward(prompt_text, generated_text, gt_text):
    """
    Scores the generated response against the ground truth using Gemini.
    Returns a numerical score (e.g., 0 to 1).
    """
    if not gemini_reward_model:
        logger.error("Gemini model not available. Returning default score 0.")
        return 0.0 # Return a default score or handle as appropriate

    scoring_prompt = f"""\
Evaluate the quality of the 'Generated Answer' compared to the 'Ground Truth Answer', considering the 'Prompt'. Score the 'Generated Answer' on a scale from 0.0 to 1.0, where 1.0 is perfect alignment with the ground truth and relevance to the prompt, and 0.0 is completely irrelevant or incorrect. Output ONLY the numerical score.

Prompt:
{prompt_text}

Ground Truth Answer:
{gt_text}

Generated Answer:
{generated_text}

Score:"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_reward_model.generate_content(scoring_prompt)
            # Basic parsing: extract the first floating-point number found
            match = re.search(r"[-+]?\d*\.\d+|\d+", response.text)
            if match:
                score = float(match.group())
                # Clamp score to [0, 1] range
                score = max(0.0, min(1.0, score))
                return score
            else:
                logger.warning(f"Could not parse score from Gemini response: {response.text}")
                # Fallback or retry logic can be added here
                if attempt < max_retries - 1:
                    logger.info(f"Retrying Gemini call ({attempt + 1}/{max_retries})...")
                    time.sleep(2**attempt) # Exponential backoff
                else:
                    logger.error("Failed to parse score after multiple retries.")
                    return 0.0 # Default score after retries

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying Gemini call ({attempt + 1}/{max_retries})...")
                time.sleep(2**attempt) # Exponential backoff
            else:
                logger.error("Failed to get score from Gemini after multiple retries.")
                return 0.0 # Default score after retries and API errors

    return 0.0 # Should not be reached if retries are handled correctly

def compute_grpo_loss(model, prompt_ids, prompt_mask, generated_responses, gt_ids, gt_mask, num_generations, tokenizer, original_gt_texts):
    """
    Computes the GRPO loss using Gemini for reward scoring.
    """
    batch_size = prompt_ids.size(0)
    prompt_len = prompt_ids.size(1)
    # full_gen_len = generated_responses.size(1) # No longer needed directly here

    # Reshape generated responses: (batch_size, num_generations, seq_len)
    generated_responses_tensor = generated_responses.view(batch_size, num_generations, -1)
    # gen_attention_mask = (generated_responses_tensor != tokenizer.pad_token_id).long() # No longer needed for reward

    # --- 1. Calculate Rewards (Using Gemini API) ---
    all_rewards = []
    all_log_probs = []

    # Decode prompts and ground truths once per batch
    # Use skip_special_tokens=True to avoid including EOS/PAD in the text sent to Gemini
    decoded_prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
    # For GT, we need the full sequence including the prompt part for context, but decode only the GT part sent to Gemini
    # Let's rethink how gt_ids are structured. Assuming gt_ids contain prompt + gt_response.
    # Need to find the start of the actual GT response within gt_ids.
    # This might be complex if padding/truncation happened differently.
    # Let's decode the gt_ids fully for now and rely on the prompt structure.
    # Note: Decoding combined GT requires careful handling if prompt was padded.
    # A simpler approach: use the original GT text from the dataset if available.
    # Assuming 'gt_ids' contains the combined prompt + gt + padding. We need original GT text.
    # *** Modification needed: The original GT text should be passed to this function. ***
    # For now, let's decode gt_ids, assuming it mostly contains the GT part after prompt.
    # This is a placeholder and might need refinement based on how `collate_fn` structures `gt_ids`.
    # Use the provided original_gt_texts directly
    cleaned_gts = original_gt_texts

    for i in range(num_generations):
        gen_ids_i = generated_responses_tensor[:, i, :]
        gen_mask_i = (gen_ids_i != tokenizer.pad_token_id).long()

        # Decode generated responses for this generation
        # We only want the generated part, excluding the prompt
        # Find the length of the prompt for each item in the batch
        prompt_lengths = prompt_mask.sum(dim=1)
        decoded_gens_i = []
        for idx in range(batch_size):
            # Slice generated response *after* the prompt part
            start_index = prompt_lengths[idx]
            actual_gen_ids = gen_ids_i[idx, start_index:]
            decoded_gen = tokenizer.decode(actual_gen_ids, skip_special_tokens=True)
            decoded_gens_i.append(decoded_gen)

        # Get rewards by calling Gemini for each item in the batch
        rewards_i_list = []
        for idx in range(batch_size):
            # Use the corresponding decoded prompt, generated text, and original GT text
            reward = get_gemini_reward(decoded_prompts[idx], decoded_gens_i[idx], cleaned_gts[idx]) # Use cleaned_gts here
            rewards_i_list.append(reward)
        rewards_i = torch.tensor(rewards_i_list, device=prompt_ids.device, dtype=torch.float32)
        all_rewards.append(rewards_i)

        # --- 2. Calculate Log Probs (remains the same) ---
        # with torch.cuda.amp.autocast(enabled=True): # Consider enabling AMP if needed
        # Detach the input_ids if they come directly from generation to avoid grad issues
        model_input_ids = gen_ids_i.detach()
        model_attention_mask = gen_mask_i.detach()
        outputs = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous() # Shift based on full sequence length
        shift_labels = gen_ids_i[..., 1:].contiguous()

        # Calculate loss only for the generated part (after prompt)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        log_probs_i = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        log_probs_i = log_probs_i.view(batch_size, -1) # Shape: (batch_size, seq_len - 1)

        # Mask out padding tokens AND prompt tokens
        response_mask = (shift_labels != tokenizer.pad_token_id).float()
        # Create a mask for the prompt part (based on original prompt_mask, shifted)
        prompt_part_mask = torch.zeros_like(response_mask)
        for b in range(batch_size):
             p_len = prompt_lengths[b]
             # We are comparing logits[..., :-1, :] with labels[..., 1:, :]
             # So, the prompt part corresponds to the first p_len-1 tokens in the shifted sequence
             if p_len > 1: # Ensure prompt length is at least 2 to have a mask index
                 prompt_part_mask[b, :p_len-1] = 1.0 # Mask where labels are part of the prompt

        # Final mask excludes padding AND prompt tokens
        final_mask = response_mask * (1.0 - prompt_part_mask)

        # Sum log probs over the valid generated tokens
        log_probs_i = (log_probs_i * final_mask).sum(dim=1)
        # Normalize by the number of valid tokens
        num_valid_tokens = final_mask.sum(dim=1).clamp(min=1)
        log_probs_i = log_probs_i / num_valid_tokens

        all_log_probs.append(log_probs_i)

        # Force garbage collection to free up memory (less critical now without embeddings)
        del outputs, logits, shift_logits, shift_labels, log_probs_i, final_mask, model_input_ids, model_attention_mask
        gc.collect()

    # Stack rewards and log_probs
    rewards = torch.stack(all_rewards, dim=1) # Shape: (batch_size, num_generations)
    log_probs = torch.stack(all_log_probs, dim=1) # Shape: (batch_size, num_generations)

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
            "original_gt_text": gt_responses, # Pass original GT text through
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
        original_gt_texts = [] # Store original GT texts
        for x in batch:
            prompt_len = len(x['prompt_ids'])
            padding_len = max_prompt_len - prompt_len
            padded_prompt_ids.append(x['prompt_ids'] + [tokenizer.pad_token_id] * padding_len)
            padded_prompt_masks.append(x['prompt_mask'] + [0] * padding_len)
            original_gt_texts.append(x['original_gt_text']) # Add original GT text
        
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
            'original_gt_text': original_gt_texts, # Return original GT text
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
            original_gt_texts = batch["original_gt_text"] # Get original GT text
            
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
                gt_ids, gt_mask, NUM_GENERATIONS, tokenizer, original_gt_texts
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