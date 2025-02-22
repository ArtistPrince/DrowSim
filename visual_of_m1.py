import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed drowsiness data
data = pd.read_csv("processed_data/processed_drowsiness_data_1.csv")

# Normalize column names (strip spaces & lowercase)
data.columns = data.columns.str.strip().str.lower()

# Convert frame index to time scale (assuming 30 FPS video)
if "frame_index" in data.columns:
    data["time"] = data["frame_index"] / 30.0  # Adjust FPS if needed
else:
    raise KeyError("⚠️ Column 'frame_index' not found in dataset!")

# Define required columns with possible variations
column_map = {
    "ear": ["ear"],
    "perclos": ["perclos", "perclos_x", "perclos_y"],
    "blink": ["blink", "blink_x", "blink_y"],
    "pupildiameter": ["pupildiameter"]
}

# Find actual column names in dataset
selected_columns = {}
missing_columns = []

for key, possible_names in column_map.items():
    found_col = next((col for col in possible_names if col in data.columns), None)
    if found_col:
        selected_columns[key] = found_col
    else:
        missing_columns.append(key)

# Handle missing columns
if missing_columns:
    print(f"⚠️ Missing columns in dataset: {missing_columns}")
    print(f"🔍 Available columns: {list(data.columns)}")
    raise KeyError("Some required columns are missing! Check dataset.")

# Set Seaborn style
sns.set_style("whitegrid")

# Create subplots for different features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot EAR (Eye Aspect Ratio) over time
sns.lineplot(x=data["time"], y=data[selected_columns["ear"]], ax=axes[0, 0], label="Eye Aspect Ratio (EAR)")
axes[0, 0].axhline(y=0.25, color="r", linestyle="--", label="Drowsiness Threshold")
axes[0, 0].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("EAR")
axes[0, 0].set_title("Eye Aspect Ratio (EAR) Over Time")
axes[0, 0].legend()

# Plot PERCLOS (Eyelid Closure Percentage)
sns.lineplot(x=data["time"], y=data[selected_columns["perclos"]], ax=axes[0, 1], label="PERCLOS (%)")
axes[0, 1].set_xlabel("Time (seconds)")
axes[0, 1].set_ylabel("PERCLOS (%)")
axes[0, 1].set_title("PERCLOS Over Time")
axes[0, 1].legend()

# Plot Blink Rate
sns.lineplot(x=data["time"], y=data[selected_columns["blink"]], ax=axes[1, 0], label="Blink Rate")
axes[1, 0].set_xlabel("Time (seconds)")
axes[1, 0].set_ylabel("Blink Count")
axes[1, 0].set_title("Blink Rate Over Time")
axes[1, 0].legend()

# Plot Pupil Diameter Changes
sns.lineplot(x=data["time"], y=data[selected_columns["pupildiameter"]], ax=axes[1, 1], label="Pupil Diameter")
axes[1, 1].set_xlabel("Time (seconds)")
axes[1, 1].set_ylabel("Pupil Diameter")
axes[1, 1].set_title("Pupil Diameter Changes Over Time")
axes[1, 1].legend()

# Adjust layout and show plots
plt.tight_layout()
plt.show()
