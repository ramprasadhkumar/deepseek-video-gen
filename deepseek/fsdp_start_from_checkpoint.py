import os
import functools
import logging
import warnings
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
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

'''
For Hybrid shard we need to:
1. Initialize the device_mesh as an (NNODES, NPROC) array.
2. Set SHARD_STRATEGY = ShardingStrategy.HYBRID_SHARD
3. Enumerate processes within one model shard group to participate in saving - HOW??
4. Gate saving on being member of that group
5. Pass the saving process group to the dcp.save function
'''

print("Finished imports")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    rank = int(os.environ["RANK"]) # Global rank
    local_device = int(os.environ["LOCAL_RANK"]) # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of global ranks
    model_path = os.path.join("/data", args.dataset_id)
    torch.cuda.set_device(local_device)

    timer.report(f"Init process group for world size: {world_size}")

    # creating a device mesh enables FSDP to use DTensor instead of ShardedTensor
    device_mesh = init_device_mesh("cuda", (world_size,))
    assert bfSixteen_ready(), "ERROR: System not BF16 ready."

    # pre-trained model weights should be mounted at /data
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # save CPU RAM by loading non-main rank models to 'meta' device
    if rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_cache=False, 
            torch_dtype=torch.bfloat16
        )
        print(f"Main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                use_cache=False, 
                torch_dtype=torch.bfloat16
            )
            print(f"Non-main rank {rank} model params on device: {set([p.data.device for p in model.parameters()])}")

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

    model = FSDP(model, 
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False), # for init from 'meta' device
        sync_module_states=True, # broadcast model weights from main rank
        device_mesh=device_mesh
    )

    timer.report("FSDP wrapped model and broadcast to GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # prepare dataset and utilities
    # dataset = load_dataset("wiki_qa", split="train")
    dataset = load_dataset("bespokelabs/bespoke-manim", split="train")
    # print("-------LEN DATASET-------", len(dataset))
    def preprocess_function(examples):
        # Combine question and answer into a single text
        # question -> question
        # answer - python_code
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['python_code'])]
        # print("-------TEXTS-------", texts)
        # print("-" * 50)
        # Tokenize with padding and truncation
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal language modeling (shift input_ids right)
        encodings["labels"] = encodings["input_ids"].copy()
        
        return encodings

    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing and preprocessing dataset",
    )

    # Create dataloader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch])
        }
    
    train_sampler = InterruptableDistributedSampler(tokenized_dataset)

    batch_size = 4  # Adjust based on your GPU memory
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    steps_per_epoch = len(dataloader)
    # print("-------STEPS PER EPOCH-------", steps_per_epoch)
    best_loss = float("inf")
    start_epoch = 0
    
    # load checkpoint if found
    try:
        output_directory = os.environ.get("CHECKPOINT_ARTIFACT_PATH", "/shared/artifacts/karthik-local-artifact-path")
        print("-------OUTPUT DIRECTORY-------", output_directory)
    except KeyError as error:
        print("Must set env var CHECKPOINT_ARTIFACT_PATH so we know where to save checkpoints!")
        exit(1)
    saver = AtomicDirectory(output_directory=output_directory, is_master=rank==0)

    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    check_existence_path = '/shared/artifacts/94240af3-544a-49d1-842a-5f022ac8193e/checkpoints'
    print("-------CHECK EXISTENCE PATH-------", check_existence_path)
    if os.path.exists(check_existence_path):
        print("-------CHECK EXISTENCE PATH EXISTS-------")
    else:
        print("-------CHECK EXISTENCE PATH DOES NOT EXIST-------")
    print("-------LATEST SYMLINK FILE PATH-------", latest_symlink_file_path)
    if os.path.islink(latest_symlink_file_path):
        print("-------LATEST SYMLINK FILE PATH EXISTS-------")
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)

        state_dict = { "app": AppState(model, optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_checkpoint_path)

        train_state = torch.load(os.path.join(latest_checkpoint_path, "train_state.pt"))
        dataloader.sampler.load_state_dict(train_state["sampler"])
        best_loss = train_state["best_loss"]

        timer.report("Loaded checkpoint")

    state_dict = { "app": AppState(model, optimizer) }

    """
    # --- Start Checkpoint Loading Logic ---
    try:
        # Assuming CHECKPOINT_ARTIFACT_PATH points to the PARENT directory
        # containing the 'AtomicDirectory.latest_checkpoint' symlink and checkpoint folders
        # Example: /mnt/checkpoints/
        # output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
        # latest_symlink_file_path = os.path.join("/shared/artifacts/94240af3-544a-49d1-842a-5f022ac8193e/checkpoints", "AtomicDirectory.latest_checkpoint")
        # print("-------LATEST SYMLINK FILE PATH-------", latest_symlink_file_path)
        latest_checkpoint_path = "/shared/artifacts/94240af3-544a-49d1-842a-5f022ac8193e/checkpoints/AtomicDirectory_checkpoint_10"
        if os.path.exists(latest_checkpoint_path):
            print("-------LATEST CHECKPOINT FILE PATH EXISTS-------")
            # Read the symlink to get the actual checkpoint directory path
            # Example: /mnt/checkpoints/AtomicDirectory_checkpoint_7
            # latest_checkpoint_path = os.readlink(latest_symlink_file_path)
            # if not os.path.isabs(latest_checkpoint_path):
            #      # If the symlink is relative, resolve it relative to the symlink's directory
            #      latest_checkpoint_path = os.path.join(os.path.dirname(latest_symlink_file_path), latest_checkpoint_path)


            print(f"Rank {rank}: Found checkpoint symlink pointing to: {latest_checkpoint_path}")

            # 1. Load FSDP sharded state (model and optimizer)
            # The state_dict keys ("model", "optim") must match the keys used during saving.
            # dcp.load expects the actual objects (model, optimizer) as values.
            state_dict_to_load = {"model": model, "optim": optimizer}
            storage_reader = dcp.FileSystemReader(latest_checkpoint_path)

            dcp.load(
                state_dict=state_dict_to_load,
                storage_reader=storage_reader,
            )
            print(f"Rank {rank}: Loaded FSDP model and optimizer state from {latest_checkpoint_path}")

            # 2. Load custom training state (sampler, epoch, loss, etc.)
            # This file was likely saved only by rank 0, but all ranks need to load it
            # to potentially get the epoch and restore their sampler state.
            train_state_path = os.path.join(latest_checkpoint_path, "train_state.pt")
            if os.path.exists(train_state_path):
                 # Map location ensures the state is loaded onto the correct device for each rank
                 map_location = {'cuda:%d' % 0: 'cuda:%d' % local_device} # Map state saved on device 0 to current local device
                 train_state = torch.load(train_state_path, map_location=map_location)

                 dataloader.sampler.load_state_dict(train_state["sampler"])
                 best_loss = train_state["best_loss"]
                 # Important: Set the starting epoch for the training loop
                 start_epoch = train_state.get("epoch", -1) + 1 # Resume AFTER the saved epoch

                 print(f"Rank {rank}: Loaded train_state (sampler, best_loss={best_loss:.4f}, start_epoch={start_epoch}) from {train_state_path}")
            else:
                 print(f"Rank {rank}: Warning - train_state.pt not found in checkpoint directory {latest_checkpoint_path}. Starting from epoch 0.")


            timer.report(f"Rank {rank}: Completed loading checkpoint")
        else:
             print(f"Rank {rank}: No checkpoint found at {latest_checkpoint_path}. Starting training from scratch.")

    except KeyError:
        print("Rank {rank}: CHECKPOINT_ARTIFACT_PATH env var not set. Starting training from scratch.")
    except Exception as e:
        print(f"Rank {rank}: Error loading checkpoint: {e}. Starting training from scratch.")

    # Ensure all ranks have finished loading before starting training
    dist.barrier()
    # --- End Checkpoint Loading Logic ---
    """

    # --- Training Loop (Modified Start Epoch) ---
    num_epochs = 30
    save_every_steps = 30
    model.train()
    # print("-------DATALOADER.SAMPLER.EPOCH-------", dataloader.sampler.epoch)

    for epoch in range(dataloader.sampler.epoch, num_epochs):

        dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:

            step = dataloader.sampler.progress // dataloader.batch_size
            is_last_step = (step + 1) == steps_per_epoch
            is_save_step = ((step + 1) % save_every_steps == 0) or is_last_step

            # Move batch to device
            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            labels = batch["labels"].to(torch.cuda.current_device())

            # forward, backward, update
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            dataloader.sampler.advance(len(input_ids))
            optimizer.zero_grad()

            # synchronize loss to determine whether to force_save
            sync_loss = loss
            dist.all_reduce(sync_loss)

            timer.report(f"Step {step} Loss: {sync_loss.item():.3f}")

            if is_save_step:
                force_save = False

                checkpoint_directory = saver.prepare_checkpoint_directory(force_save=force_save)
                checkpoint_writer = dcp.FileSystemWriter(checkpoint_directory)

                metadata = dcp.save(
                    state_dict=state_dict, 
                    storage_writer=checkpoint_writer
                )

                dist.barrier()

                if rank == 0:
                    atomic_torch_save(
                        {
                            "sampler": dataloader.sampler.state_dict(),
                            "best_loss": best_loss
                        }, 
                        os.path.join(checkpoint_directory, "train_state.pt")
                    )
                    
                while len(os.listdir(checkpoint_directory)) < world_size + 2:
                    print("Checkpoint not yet saved...")
                    time.sleep(1)
                    dist.barrier()

                saver.symlink_latest(checkpoint_directory)

                timer.report("Saved checkpoint")

        dataloader.sampler.reset_progress()

    timer.report("Done.")

    dist.barrier()
    dist.destroy_process_group()
