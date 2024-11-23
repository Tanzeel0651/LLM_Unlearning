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
from auxillary_model import AuxiliaryModel
from forget_weight import logits_by_percentile

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

# Create dataset and dataloader
max_length = 50
forget_dataset = ForgetDataset(forget_texts, target_tokenizer, max_length)
forget_dataloader = DataLoader(forget_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Switch back to training mode for the main model
target_model.train()

epochs = 7
optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-5)  # Lowered learning rate
# import pdb;pdb.set_trace()
auxillary_model = torch.load("model/auxillary_model2", weights_only=False)
mse_loss = torch.nn.MSELoss(reduction='none')

# Loop to fine-tune the model with unlearning loss
for epoch in range(epochs):
    total_loss = 0
    for i, batch in enumerate(forget_dataloader):
        optimizer.zero_grad()
        
        input_ids, attention_masks = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        reinforced_logits = get_logits(reinforce_model, input_ids, attention_masks)

        # Forward pass
        outputs = target_model(input_ids=input_ids, attention_mask=attention_masks)
        original_logits = outputs.logits

        new_logits = logits_by_percentile(original_logits, target_tokenizer, forget_texts)
        
        print(f"Range of original_logits:[{torch.min(original_logits)}, {torch.max(original_logits)}]")
        print(f"Range of new_logits:[{torch.min(new_logits)}, {torch.max(new_logits)}]")


        loss = F.kl_div(
            F.log_softmax(new_logits+1e-8, dim=-1),
            #F.softmax(new_logits, dim=-1),
            F.softmax(reinforced_logits, dim=-1),
            reduction='batchmean'
        )
        if torch.isnan(new_logits).any():
            import pdb;pdb.set_trace()

        total_loss += loss.item()

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        
        torch.nn.utils.clip_grad_norm_(target_model.parameters(), max_norm=1.0)

        # Check for NaNs in gradients
       # for param in target_model.parameters():
       #     if torch.isnan(param.grad).any():
       #         print("NaN detected in gradients! Skipping optimizer step.")
       #         optimizer.zero_grad()  # Reset gradients and stop this iteration
       #         break
       # else:
       #     optimizer.step()
    
    # torch.save(all_logits, "logits_data.pt")
    average_loss = total_loss / len(forget_dataloader)
    print(f"Epoch {epoch + 1}, Loss Average: {average_loss:.4f}")
    plot_reinforced_logits(original_logits, new_logits, target_tokenizer, input_ids[0], epoch)
    
# Save the fine-tuned model
target_model.save_pretrained("model/fine_tuned_gpt2_forget_percentile")
target_tokenizer.save_pretrained("model/fine_tuned_gpt2_forget_percentile")
print("Model Saved")
