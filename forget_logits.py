import os
import torch
import torch.nn as nn
from aux_model import MLPForget

def compute_target_logits(
    logits_file: str,
    mlp_model_file: str,
    logits_key: str = "forget_logits"
):
    # 1) Load existing data
    data_dict = torch.load(logits_file, weights_only=True)
    original_logits = data_dict["original_logits"].float()  # keep on CPU for now

    # 2) Load the MLP (forget) model
    mlp_forget_model = MLPForget(vocab_size=32000, hidden_dim=256)
    state_dict = torch.load(mlp_model_file, weights_only=True)
    mlp_forget_model.load_state_dict(state_dict)
    mlp_forget_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_forget_model.to(device).half()  # Use half precision

    # 3) Create forget logits in batches
    batch_size = 128  # Decrease further if you still get OOM
    num_batches = (original_logits.size(0) + batch_size - 1) // batch_size
    forget_logits_list = []

    with torch.no_grad():
        for i in range(num_batches):
            batch = original_logits[i * batch_size : (i + 1) * batch_size]
            # move batch to GPU in half precision
            batch = batch.to(device, dtype=torch.float16)

            # forward pass
            out = mlp_forget_model(batch).cpu().float()  # convert back to float32
            forget_logits_list.append(out)

            # free memory
            del batch, out
            torch.cuda.empty_cache()

    # Concatenate and store
    forget_logits = torch.cat(forget_logits_list, dim=0)
    del forget_logits_list
    torch.cuda.empty_cache()

    data_dict[logits_key] = forget_logits

    # 4) Save back
    torch.save(data_dict, logits_file)
    print(f"[INFO] Appended '{logits_key}' to {logits_file} successfully!")

if __name__ == '__main__':
    # Optional: set environment var for memory fragmentation
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("Computing Logits")
    compute_target_logits("logits_data.pt", "model/mlp_forget_model.pt")


