import pandas as pd
import os

# Path where all processed CSV files are stored
data_dir = "processed_data"  # Folder containing 27 files
merged_data_path = "merged_drowsiness_data.csv"

# Get list of all CSV files in directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# Merge all files
df_list = [pd.read_csv(os.path.join(data_dir, file)) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

# Save merged dataset
merged_df.to_csv(merged_data_path, index=False)
print(f"All processed data merged and saved as {merged_data_path}")
