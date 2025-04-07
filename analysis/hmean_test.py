import json
import numpy as np
from scipy.stats import ks_2samp, hmean
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from adjustText import adjust_text

os.chdir("/student/tahmad8/Videos/tofu/")

# Function to load truth ratio values from JSON files
def load_truth_ratios(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(list(data["truth_ratio"].values()), dtype=np.float32)

def load_eval_metrics(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

full_model_path = "model/target_model/ft_epoch5_lr1e-05_llama2-7b_retain90_wd0.01/eval_results/"
base_dir = "model/forget_model/"
forget_model_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

truth_ratio_full_forget = load_truth_ratios(full_model_path + "eval_log_forget.json")
truth_ratio_full_retain = load_truth_ratios(full_model_path + "eval_log.json")

results = []

for model_path in forget_model_dirs:
    print("Reading Model: ", model_path)
    forget_path = os.path.join(model_path, "eval_results")
    eval_log_forget = os.path.join(forget_path, "eval_log_forget.json")
    eval_log_retain = os.path.join(forget_path, "eval_log.json")

    if os.path.exists(eval_log_forget) and os.path.exists(eval_log_retain):
        truth_ratio_forget_forget = load_truth_ratios(eval_log_forget)
        truth_ratio_forget_retain = load_truth_ratios(eval_log_retain)

        # Load extra metrics
        with open(eval_log_retain, 'r') as f:
            retain_metrics = json.load(f)

        ks_stat_forget, p_value_forget = ks_2samp(truth_ratio_full_forget, truth_ratio_forget_forget)
        ks_stat_retain, p_value_retain = ks_2samp(truth_ratio_full_retain, truth_ratio_forget_retain)

        results.append({
            "Model": os.path.basename(model_path),
            "KS Forget": ks_stat_forget,
            "P Forget": p_value_forget,
            "KS Retain": ks_stat_retain,
            "P Retain": p_value_retain,
            "avg_gt_loss": np.mean(list(retain_metrics["avg_gt_loss"].values())),
            "rouge1_recall": np.mean(list(retain_metrics["rouge1_recall"].values())),
            "truth_ratio": np.mean(list(retain_metrics.get("truth_ratio", {}).values()))
        })

# Create DataFrame
df_results = pd.DataFrame(results)

# Compute Forget Quality
df_results["Forget Quality"] = np.log10(df_results["P Forget"].replace(0, 1e-100))

# Normalize and compute model utility
epsilon = 1e-8
df_results["Loss Utility"] = 1 / (df_results["avg_gt_loss"] + epsilon)
df_results["ROUGE Utility"] = df_results["rouge1_recall"].clip(0, 1)
df_results["Truth Ratio Utility"] = df_results["truth_ratio"].clip(0, 1)

def harmonic_mean(row):
    values = [row["Loss Utility"], row["ROUGE Utility"], row["Truth Ratio Utility"]]
    if any(v <= 0 for v in values):
        return 0
    return hmean(values)

df_results["Model Utility"] = df_results.apply(harmonic_mean, axis=1)

# Plotting
plt.figure(figsize=(10, 8))
plt.axhline(-1.3, color='black', linestyle='--', label='p = 0.05 threshold')
plt.xlabel("Model Utility (harmonic mean)")
plt.ylabel("Forget Quality (log10 p-value on forget set)")
plt.title("Forget Quality vs Model Utility (MLP 30%)")

def get_label(filename):
    if not filename.startswith("mixed"):
        return filename
    numbers = re.findall(r"-?\d+\.?\d*", filename)
    if len(numbers) == 3:
        epoch, temp, alpha = numbers
        return f"e{epoch}_t{temp}_a{alpha}"
    return filename

texts = []
for idx, row in df_results.iterrows():
    plt.scatter(row["Model Utility"], row["Forget Quality"], s=100)
    texts.append(plt.text(row["Model Utility"], row["Forget Quality"], get_label(row["Model"]), fontsize=8))

adjust_text(texts)
plt.savefig("analysis/mlp_30_hmean_steps.png")
