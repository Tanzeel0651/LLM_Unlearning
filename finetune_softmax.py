import torch
import numpy as np
import os
import json
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transfer import plot_reinforced_logits
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
from utils import files_to_dict, generate_seed_list, \
    avg_compute_similarity, line_chart, preprocess, plot_confusion_matrix
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F

# Target Model
target_model_path = "model/fine_tuned_gpt2_target/"
target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
target_model = AutoModelForCausalLM.from_pretrained(target_model_path)
target_model.to(device)

# Reinforced Model
reinforce_model_path = "model/fine_tuned_gpt2_vanilla/"
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

# Create dataset and dataloader
max_length = 50
forget_dataset = ForgetDataset(forget_texts, target_tokenizer, max_length)
forget_dataloader = DataLoader(forget_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Switch back to training mode for the main model
target_model.train()

epochs = 7
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

        # Calculate softmaxes
        softmax_original = F.softmax(original_logits, dim=-1)
        softmax_reinforced = F.softmax(reinforced_logits, dim=-1)
        original_min, original_max = softmax_original.min(), softmax_original.max()
        softmax_orginal_scaled = (softmax_original - original_min) / (original_max - original_min + 1e-5)
        
        reinforced_min, reinforced_max = softmax_reinforced.min(), softmax_reinforced.max()
        softmax_reinforced_scaled = (softmax_reinforced - reinforced_min) / (reinforced_max - reinforced_min + 1e-5)
        
        softmax_diff = softmax_reinforced_scaled * softmax_orginal_scaled

        # Define dynamic scaling factor
        logit_range = (original_logits.max() - original_logits.min()).clamp(min=1e-5)
        scaled_softmax_diff = (softmax_diff/softmax_diff.abs().max())

        scale_factor = logit_range * (torch.exp(torch.clamp(scaled_softmax_diff, max=50)))

        # Calculate new logits
        alpha = 0.2  # Adjust alpha to control the scaling rate
        new_logits = original_logits - alpha * torch.relu(scale_factor * scaled_softmax_diff)
        
        # Use KL divergence loss to push the new and reinforced logits distributions apart
        loss = F.kl_div(
            F.log_softmax(new_logits, dim=-1),
            F.softmax(reinforced_logits, dim=-1),
            reduction='batchmean'
        )

        total_loss += loss.item()

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(forget_dataloader)
    print(f"Epoch {epoch + 1}, KL Divergence Loss Average: {average_loss:.4f}")
    plot_reinforced_logits(original_logits, new_logits, target_tokenizer, input_ids[0], epoch)

# Save the fine-tuned model
target_model.save_pretrained("model/fine_tuned_gpt2_forget")
target_tokenizer.save_pretrained("model/fine_tuned_gpt2_forget")
