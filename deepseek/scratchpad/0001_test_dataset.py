# %%
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("bespokelabs/bespoke-manim")

# To load a specific split (if available)
# dataset = load_dataset("bespokelabs/bespoke-manim", split="train")
# %%

dataset["train"][0]

# %%

dataset["train"][0].keys()

# %%
