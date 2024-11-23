import torch
import numpy as np
import os
import json
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
from transfer import *
from utils import files_to_dict, generate_seed_list, \
    avg_compute_similarity, line_chart, preprocess, plot_confusion_matrix
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Target Model
target_model_path = "model/fine_tuned_gpt2_target/"
target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
target_model = AutoModelForCausalLM.from_pretrained(target_model_path)
target_model.to(device)

# Reinforced Model
reinforce_model_path = "model/fine_tuned_gpt2_reinforce/"
reinforce_tokenizer = AutoTokenizer.from_pretrained(reinforce_model_path)
reinforce_model = AutoModelForCausalLM.from_pretrained(reinforce_model_path)
reinforce_model.to(device)

# Ensure both models are in evaluation mode for getting logits
target_model.eval()
reinforce_model.eval()

def get_logits(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    return logits


# Prepare forget set data
forget_set = "human.txt"
forget_texts = files_to_dict()[forget_set]


# Define a simple dataset class
class ForgetDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.input_ids = []
        self.attention_masks = []

        for text in texts:
            # Tokenize and encode the text
            encodings = tokenizer(
                text, truncation=True, max_length=max_length, padding='max_length'
            )
            self.input_ids.append(encodings['input_ids'])
            self.attention_masks.append(encodings['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.attention_masks[idx])

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    return input_ids, attention_masks


def plot_logits_comparison(before_logits, after_logits, reinforced_logits, tokenizer, input_ids, index, top_k=10, title="Top K Logit Comparison"):
    # Squeeze logits to remove extra dimensions if batch size is 1
    before_logits = before_logits.squeeze()
    after_logits = after_logits.squeeze()
    reinforced_logits = reinforced_logits.squeeze()

    # Filter out special tokens from input_ids
    filtered_input_ids = [token_id for token_id in input_ids if tokenizer.convert_ids_to_tokens([token_id])[0] not in tokenizer.all_special_tokens]

    if len(filtered_input_ids) == 0:
        print("No non-special tokens found for plotting.")
        return

    input_ids = torch.tensor(filtered_input_ids).to(before_logits.device)

    # Get original logits for the tokens in the input sequence
    original_probs = before_logits[range(len(input_ids)), input_ids].cpu().detach()
    
    # Get top K tokens based on original logits
    top_before_probs, top_before_indices = torch.topk(original_probs, top_k)

    # Find corresponding logits in reinforced_logits and after_logits using the same top_before_indices
    top_after_probs = after_logits[range(len(top_before_indices)), input_ids[top_before_indices]].cpu().detach()
    top_reinforced_probs = reinforced_logits[range(len(top_before_indices)), input_ids[top_before_indices]].cpu().detach()

    # Decode top token ids to actual tokens
    top_before_tokens = [tokenizer.decode([input_ids[idx]]) for idx in top_before_indices.cpu()]

    # Convert tensors to CPU for plotting
    top_before_probs = top_before_probs.cpu().numpy()
    top_after_probs = top_after_probs.cpu().numpy()
    top_reinforced_probs = top_reinforced_probs.cpu().numpy()

    # Plotting the results
    x = np.arange(top_k)  # Token positions

    plt.figure(figsize=(12, 6))

    # Bar width
    width = 0.25

    # Plot original, reinforced, and new logits for top K tokens
    plt.bar(x - width, top_before_probs, width, label='Original Logits', color='blue', alpha=0.6)
    plt.bar(x, top_reinforced_probs, width, label='Reinforced Logits', color='orange', alpha=0.6)
    plt.bar(x + width, top_after_probs, width, label='New Logits', color='green', alpha=0.6)

    # Set x-axis labels
    plt.xticks(x, top_before_tokens, rotation=90)

    # Labels and title
    plt.xlabel("Tokens")
    plt.ylabel("Logit Scores")
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig("logits_compare/epoch_{}.png".format(index))


def plot_logits_stat(before_logits, after_logits, reinforced_logits, index, title="Logit Min/Max/Mean Comparison"):
    """
    Plots min, max, and mean of logits across token positions.
    
    Args:
        before_logits (torch.Tensor): The logits before the unlearning process, shape [seq_len, vocab_size].
        after_logits (torch.Tensor): The logits after the unlearning process, shape [seq_len, vocab_size].
        reinforced_logits (torch.Tensor): The logits from the reinforced (forgetting) model.
        index (int or str): Index to save the plot file with a unique name (e.g., the current epoch).
        title (str): Title of the plot.
    """

    # Squeeze logits to remove extra dimensions if batch size is 1
    before_logits = before_logits.squeeze().cpu().detach().numpy()
    after_logits = after_logits.squeeze().cpu().detach().numpy()
    reinforced_logits = reinforced_logits.squeeze().cpu().detach().numpy()

    # Get min, max, and mean for each set of logits along the vocab axis (axis=1)
    before_min = np.min(before_logits, axis=1).mean()
    before_max = np.max(before_logits, axis=1).mean()
    before_mean = np.mean(before_logits, axis=1).mean()

    after_min = np.min(after_logits, axis=1).mean()
    after_max = np.max(after_logits, axis=1).mean()
    after_mean = np.mean(after_logits, axis=1).mean()

    reinforced_min = np.min(reinforced_logits, axis=1).mean()
    reinforced_max = np.max(reinforced_logits, axis=1).mean()
    reinforced_mean = np.mean(reinforced_logits, axis=1).mean()

    # Create x-axis values for token positions
    categories = ["Min", "Max", "Mean"]
    x = np.arange(len(categories))
    width = 0.2

    plt.figure(figsize=(10, 6))

    # Plot min, max, and mean for each of the logits
    plt.bar(x-width, [before_min, before_max, before_mean], width, label='Original Logits', color='blue', alpha=0.6)
    plt.bar(x, [after_min, after_max, after_mean], width, label='Reinforced Logits', color='orange', alpha=0.6)
    plt.bar(x+width, [reinforced_min, reinforced_max, reinforced_mean], width, label='New Logits', color='green', alpha=0.6)

    # Labels and title
    plt.xlabel("Token Positions (Sequence Length)")
    plt.ylabel("Logit Scores")
    plt.title(title)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig("logits_compare/logits_stat/epoch_{}.png".format(index))
    plt.close()    


# Create dataset and dataloader
max_length = 50
forget_dataset = ForgetDataset(forget_texts, target_tokenizer, max_length)
forget_dataloader = DataLoader(forget_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Switch back to training mode for the main model
target_model.train()

epochs = 15
optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-5)  # Lowered learning rate

# Loop to fine-tune the model with unlearning loss
for epoch in range(epochs):
    total_loss = 0
    for i, batch in enumerate(forget_dataloader):
        optimizer.zero_grad()
        
        input_ids, attention_masks = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Get logits from the original model and the forget model
        original_logits = target_model(input_ids=input_ids, attention_mask=attention_masks).logits
        reinforced_logits = get_logits(reinforce_model, input_ids, attention_masks)

        # Calculate new logits using the provided formula
        alpha = 0.2  # Reduced alpha to slow down the unlearning process
        new_logits = original_logits - alpha * torch.relu(reinforced_logits - original_logits)
        
        
        # Custom loss function: MSE loss to minimize the influence of forget logits
        mse_loss = torch.nn.MSELoss(reduction='none')
        custom_loss = mse_loss(new_logits, reinforced_logits)
        
        # Define a mask to penalize the tokens from the forget set
        epsilon = 1e-5  # Add a small epsilon to prevent complete masking
        forget_token_mask = ((reinforced_logits > original_logits).float() + epsilon)  

        # Apply mask to penalize only the forget set influences
        loss = (custom_loss * forget_token_mask).mean()

        total_loss += loss.item()

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(forget_dataloader)
    print(f"Epoch {epoch + 1}, Custom Unlearning Loss Average: {average_loss:.4f}")
    # plot_logits_comparison(original_logits, new_logits, reinforced_logits, target_tokenizer, input_ids[0], index="{}".format(epoch))
    plot_logits_stat(original_logits, new_logits, reinforced_logits, index="{}".format(epoch))
    plot_reinforced_logits(original_logits, reinforced_logits, target_tokenizer,input_ids[0], epoch)

# Save the fine-tuned model
# target_model.save_pretrained("model/fine_tuned_gpt2_forget")
# target_tokenizer.save_pretrained("model/fine_tuned_gpt2_forget")
