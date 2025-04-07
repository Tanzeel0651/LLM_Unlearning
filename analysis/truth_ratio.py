import json
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("/student/tahmad8/Videos/tofu/")
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Define paths
#target = "model/target_model/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01/"
target = "model/target_model/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01/"
#forget = "model/forget_model/mixed_distillation_version5/"
forget = "model/forget_model/mixed_distillation_v2/"
log_file = "eval_results/eval_log.json"

test_set = "retain"
if "forget" in log_file:
    test_set = "forget"

label1 = "5 Epochs"
label2 = "only forget distill"
steps = "400-200"
alpha = "101"

# Load truth ratio values
with open(target + log_file, 'r') as f:
    target_truth_log = json.load(f)["truth_ratio"]

with open(forget + log_file, 'r') as f:
    forget_truth_log = json.load(f)["truth_ratio"]

# Convert to NumPy arrays for easier processing
truth_ratio_target = np.array(list(target_truth_log.values()), dtype=np.float32)
truth_ratio_forget = np.array(list(forget_truth_log.values()), dtype=np.float32)

# Create a figure with two subplots (Histogram & ECDF)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

### 🔹 HISTOGRAM ###
axes[0].hist(truth_ratio_target, bins=30, alpha=0.6, label=label1, density=True)
axes[0].hist(truth_ratio_forget, bins=30, alpha=0.6, label=label2, density=True)
axes[0].set_xlabel("Truth Ratio")
axes[0].set_ylabel("Density")
axes[0].set_title("Histogram of Truth Ratios")
axes[0].legend()

### 🔹 ECDF ###
def ecdf(data):
    """ Compute the Empirical Cumulative Distribution Function (ECDF). """
    return np.sort(data), np.arange(1, len(data) + 1) / len(data)

x_target, y_target = ecdf(truth_ratio_target)
x_forget, y_forget = ecdf(truth_ratio_forget)

axes[1].plot(x_target, y_target, marker=".", linestyle="none", label=label1)
axes[1].plot(x_forget, y_forget, marker=".", linestyle="none", label=label2)
axes[1].set_xlabel("Truth Ratio")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Empirical CDF of Truth Ratios")
axes[1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig(f"graphs/{test_set}_truth_ratio_{label1}_{label2}_steps{steps}_alpha{alpha}.png")

