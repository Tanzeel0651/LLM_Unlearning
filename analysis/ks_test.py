import json
import numpy as np
from scipy.stats import ks_2samp
import os
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("/student/tahmad8/Videos/tofu/")

# Function to load truth ratio values from JSON files
def load_truth_ratios(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(list(data["truth_ratio"].values()), dtype=np.float32)

# # Define paths to your JSON files
# # retain90_path = "model/target_model/ft_epoch5_lr1e-05_llama2-7b_retain90_wd0.01/eval_results/eval_log_forget.json"  # Correct Retain90 path
# forget10_path = "model/forget_model/mixed_distillation_v2/eval_results/"  # Forget10 model (fine-tuned for unlearning)
full_model_path = "model/target_model/ft_epoch5_lr1e-05_llama2-7b_retain90_wd0.01/eval_results/"  # Full model (before unlearning)
#full_model_path = "model/target_model/full_model_scratch/eval_results/"  # Full model (before unlearning)



# Base directory containing the forget models
base_dir = "model/forget_model/MLP40/"

# List all subdirectories (i.e., different forget model versions)
forget_model_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Load full model truth ratios
truth_ratio_full_forget = load_truth_ratios(full_model_path + "eval_log_forget.json")
truth_ratio_full_retain = load_truth_ratios(full_model_path + "eval_log.json")

# Compute KS-Test for each forget model
results = []

for model_path in forget_model_dirs:
    forget_path = os.path.join(model_path, "eval_results")

    eval_log_forget = os.path.join(forget_path, "eval_log_forget.json")
    eval_log_retain = os.path.join(forget_path, "eval_log.json")

    if os.path.exists(eval_log_forget) and os.path.exists(eval_log_retain):
        truth_ratio_forget_forget = load_truth_ratios(eval_log_forget)
        truth_ratio_forget_retain = load_truth_ratios(eval_log_retain)

        # Compute KS test
        ks_stat_forget, p_value_forget = ks_2samp(truth_ratio_full_forget, truth_ratio_forget_forget)
        ks_stat_retain, p_value_retain = ks_2samp(truth_ratio_full_retain, truth_ratio_forget_retain)


        results.append({
            "Model": os.path.basename(model_path),
            "KS Forget": ks_stat_forget,
            "P Forget": p_value_forget,
            "KS Retain": ks_stat_retain,
            "P Retain": p_value_retain
        })





#import pdb;pdb.set_trace()
# Now create the DataFrame
df_results = pd.DataFrame(results, dtype='str')
#tools.display_dataframe_to_user(name="KS-Test Results", dataframe=df_results)
print(df_results)
df_results["P Forget"] = df_results["P Forget"].astype(float)
df_results["KS Retain"] = df_results["KS Retain"].astype(float)

df_results["Forget Quality"] = np.log10(df_results["P Forget"].replace(0, 1e-100))  # log scale
df_results["Model Utility"] = 1 - df_results["KS Retain"]  # or use avg truth ratio on retain set


# Plot
plt.figure(figsize=(10, 8))
# for idx, row in df_results.iterrows():
#     plt.scatter(row["Model Utility"], row["Forget Quality"], label=row["Model"], s=100)
#     plt.text(row["Model Utility"], row["Forget Quality"], row["Model"], fontsize=8)

#plt.axhline(np.log10(0.05), color='gray', linestyle='--', label="p=0.05 threshold")
plt.axhline(-1.3, color='black', linestyle='--', label='p = 0.05 threshold')
plt.xlabel("Model Utility (1 - KS Retain)")
plt.ylabel("Forget Quality (log10 p-value on forget set)")
plt.title("Forget Quality vs Model Utility")
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
import re
# def get_label(filename):
#     name = []
#     # import pdb;pdb.set_trace()
#     if not filename.startswith("mixed"):
#         return filename
#     for x in filename.split("_"):
#         int_ = re.findall(r"-?\d+\.?\d*", x)
#         if int_:
#             name.append(int_[0])
#     return name

def get_label(filename):
    if not filename.startswith("mixed"):
        return filename
    # Extract numbers from the filename
    numbers = re.findall(r"-?\d+\.?\d*", filename)
    if len(numbers) == 3:
        epoch, temp, alpha = numbers
        return f"e{epoch}_t{temp}_a{alpha}"
    return filename


from adjustText import adjust_text
texts = []
for idx, row in df_results.iterrows():
    plt.scatter(row["Model Utility"], row["Forget Quality"], s=100)
    texts.append(
        plt.text(row["Model Utility"], row["Forget Quality"], get_label(row["Model"]), fontsize=8)
        
    )
adjust_text(texts)

plt.savefig("analysis/mlp_40.png")

