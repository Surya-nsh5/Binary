import pandas as pd
import numpy as np

# --- Step 1: Load your base dataset ---
df = pd.read_csv(r"Training Model/msr-cambridge1-sample.csv", header=None)

# Assign headers if missing (based on your structure)
df.columns = ["timestamp", "label", "unknown", "operation", "address", "block_size", "value"]

# --- Step 2: Encode categorical columns for processing ---
from sklearn.preprocessing import LabelEncoder
le_label = LabelEncoder()
le_op = LabelEncoder()

df["label_enc"] = le_label.fit_transform(df["label"])
df["operation_enc"] = le_op.fit_transform(df["operation"])

# --- Step 3: Generate synthetic rows ---
num_new_rows = 5000   # adjust to make dataset larger

synthetic = pd.DataFrame({
    "timestamp": df["timestamp"].sample(num_new_rows, replace=True).values + np.random.randint(-1000, 1000, num_new_rows),
    "label": np.random.choice(df["label"].unique(), num_new_rows),
    "unknown": np.random.choice(df["unknown"].unique(), num_new_rows),
    "operation": np.random.choice(df["operation"].unique(), num_new_rows),
    "address": df["address"].sample(num_new_rows, replace=True).values + np.random.randint(-10**6, 10**6, num_new_rows),
    "block_size": df["block_size"].sample(num_new_rows, replace=True).values,
    "value": df["value"].sample(num_new_rows, replace=True).values + np.random.randint(-100, 100, num_new_rows)
})

# --- Step 4: Combine original + synthetic data ---
expanded_df = pd.concat([df, synthetic], ignore_index=True)

# --- Step 5: Save new expanded dataset ---
expanded_path = r"d:\Files\Intelligence Cache Replacement\expanded_cache_dataset.csv"
expanded_df.to_csv(expanded_path, index=False)

print(f"Original size: {df.shape[0]} rows")
print(f"Expanded size: {expanded_df.shape[0]} rows")
print(f"Expanded dataset saved to: {expanded_path}")
