import json
import matplotlib.pyplot as plt
import os
os.chdir("/student/tahmad8/Videos/tofu/")
import numpy as np

def compute_scores(eval_file):
    """
    Reads `eval_file` (JSON) and computes score = rougeL_recall * (1 - avg_gt_loss)
    for each instance. Returns a list of floats.
    """
    with open(eval_file, 'r') as f:
        data = json.load(f)  # assume it's a list of dicts or a dict containing 'rougeL_recall' etc.

    scores = []
    rouge = list(data["rougeL_recall"].values())
    loss = list(data["avg_gt_loss"].values())

    for rougeL, gt_loss in zip(rouge, loss):
        score = rougeL# * (1.0 - gt_loss)
        scores.append(score)
    return scores

def bin_average(scores, n_bins=40):
    """Split scores into n_bins consecutive chunks; return average of each chunk."""
    # If len(scores) is not exactly divisible by n_bins, the last chunk may be smaller
    chunk_size = len(scores) // n_bins
    binned_scores = []
    # For a nice x-index, we can use the midpoint of each chunk
    x_indices = []
    start_idx = 0
    for b in range(n_bins):
        end_idx = start_idx + chunk_size
        if b == n_bins - 1:  
            # last bin - include any leftover
            end_idx = len(scores)
        chunk = scores[start_idx:end_idx]
        binned_scores.append(np.mean(chunk))
        # midpoint in the chunk for the x-axis
        x_indices.append((start_idx + end_idx) / 2.0)
        start_idx = end_idx
    return x_indices, binned_scores

def main():
    # Adjust these filenames to match your actual paths
    target = "model/target_model/"
    forget = "model/forget_model/forget_TOFU_01_steps30_locusfull_alpha11/"
    log_file = "eval_results/eval_log_forget.json"
    
    test_set = "retain"
    if "forget" in log_file:
        test_set = "forget"
    
    label1 = "Locus Full"
    label2 = "Forget 10"
    steps = 30
    alpha = 11

    # 1. Compute scores for each file
    target_scores = compute_scores(target+log_file)
    forget_scores = compute_scores(forget+log_file)

    # We'll bin them into 40 chunks:
    target_x, target_binned = bin_average(target_scores, n_bins=15)
    forget_x, forget_binned = bin_average(forget_scores, n_bins=15)

    plt.figure(figsize=(10,6))
    plt.plot(target_x, target_binned, label="Target Model (Binned)", color='purple')
    plt.plot(forget_x, forget_binned, label="Forget Model (Binned)", color='orange')
    
    #plt.title("Binned Score Comparison (ROUGE-L * (1 - avg_gt_loss))")
    plt.title("Rouge Score")
    plt.xlabel("Instance Index (binned)")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/{test_set}_rouge_{label1}_{label2}_steps{steps}_alpha{alpha}.png")

if __name__ == "__main__":
    main()

