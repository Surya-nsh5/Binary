import pandas as pd

# --- Load both CSV files ---
file1 = "cache_dataset.csv"          # replace with your original file name
file2 = "Training Model/msr-cambridge1-sample.csv"    # your uploaded file

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# --- Keep only relevant columns ---
required_columns = ['last_access_time', 'access_count', 'recency_rank', 'access_type', 'label', 'cache_item']
df1 = df1[required_columns]
df2 = df2[required_columns]

# --- Merge both datasets ---
merged_df = pd.concat([df1, df2], ignore_index=True)

# --- Remove duplicates if any ---
merged_df = merged_df.drop_duplicates()

# --- Save final merged dataset ---
merged_df.to_csv("merged_cache_dataset.csv", index=False)

print("Merged dataset created successfully.")
print("Final shape:", merged_df.shape)
print("Saved as 'merged_cache_dataset.csv'")
