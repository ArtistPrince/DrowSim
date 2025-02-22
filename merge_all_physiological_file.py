import os
import pandas as pd

# Define the main folder containing all 27 subfolders
main_folder = r"F:/Project/Drowsiness Detection System/data/Physiological_data"

# Initialize empty lists to store data
eeg_ecg_list = []
ppg_eda_list = []

# Iterate through all 27 subfolders
for i in range(1, 28):  # Loop from 1 to 27
    subfolder = os.path.join(main_folder, str(i))

    if not os.path.exists(subfolder):
        print(f"⚠️ Warning: Subfolder {subfolder} does not exist. Skipping...")
        continue

    # Define the exact file paths
    eeg_ecg_file = os.path.join(subfolder, "EEG_ECG.txt")
    ppg_eda_file = os.path.join(subfolder, "PPG_EDA.txt")

    # Read and append EEG_ECG data
    if os.path.exists(eeg_ecg_file):
        try:
            df_eeg_ecg = pd.read_csv(eeg_ecg_file, delimiter=",", encoding="utf-8")  # Try comma separator
        except:
            df_eeg_ecg = pd.read_csv(eeg_ecg_file, delimiter="\t", encoding="utf-8")  # Try tab separator

        df_eeg_ecg["participant_id"] = i  # Add participant ID for tracking
        eeg_ecg_list.append(df_eeg_ecg)
    else:
        print(f"❌ EEG_ECG file missing in {subfolder}")

    # Read and append PPG_EDA data
    if os.path.exists(ppg_eda_file):
        try:
            df_ppg_eda = pd.read_csv(ppg_eda_file, delimiter=",", encoding="utf-8")  # Try comma separator
        except:
            df_ppg_eda = pd.read_csv(ppg_eda_file, delimiter="\t", encoding="utf-8")  # Try tab separator

        df_ppg_eda["participant_id"] = i  # Add participant ID for tracking
        ppg_eda_list.append(df_ppg_eda)
    else:
        print(f"❌ PPG_EDA file missing in {subfolder}")

# Merge all EEG_ECG files into one DataFrame
if eeg_ecg_list:
    merged_eeg_ecg = pd.concat(eeg_ecg_list, ignore_index=True)
    merged_eeg_ecg.to_csv("merged_EEG_ECG.csv", index=False)
    print("✅ Merged EEG_ECG saved as merged_EEG_ECG.csv")
else:
    print("🚨 No EEG_ECG files found!")

# Merge all PPG_EDA files into one DataFrame
if ppg_eda_list:
    merged_ppg_eda = pd.concat(ppg_eda_list, ignore_index=True)
    merged_ppg_eda.to_csv("merged_PPG_EDA.csv", index=False)
    print("✅ Merged PPG_EDA saved as merged_PPG_EDA.csv")
else:
    print("🚨 No PPG_EDA files found!")
