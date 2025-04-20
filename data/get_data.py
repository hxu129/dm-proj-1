import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Create directories
os.makedirs("test", exist_ok=True)
os.makedirs("validation", exist_ok=True)

print("Downloading wiki40b/en dataset...")

# Get validation set
print("Loading validation split...")
validation = load_dataset("wiki40b/en", split="validation")
validation_df = pd.DataFrame(validation)
validation_df.to_parquet("validation/validation.parquet", index=False)
print("Saved validation/validation.parquet")

# Get test set
print("Loading test split...")
test = load_dataset("wiki40b/en", split="test")
test_df = pd.DataFrame(test)
test_df.to_parquet("test/test.parquet", index=False)
print("Saved test/test.parquet")

print("Download and merge complete!")