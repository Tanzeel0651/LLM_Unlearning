
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import files_to_dict, compute_transformer_similarity, preprocess, transformer_model

model_path = "model/fine_tuned_gpt2/"

# Load pretrained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set up device for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

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
forget_dataset = ForgetDataset(forget_texts, tokenizer, max_length)
forget_loader = DataLoader(forget_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Define the custom loss function
def custom_loss_function(outputs, labels, generated_texts, forget_texts):
    lm_loss_fn = CrossEntropyLoss()

    # Calculate the standard language modeling loss
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = lm_loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )

    # Calculate similarity penalties
    sim_penalty = 0.0
    for generated_text in generated_texts:
        for forget_text in forget_texts:
            similarity = compute_transformer_similarity(forget_texts, generated_text,  model=sentence_transformer)
            sim_penalty += similarity

    # Normalize the similarity penalty
    sim_penalty /= len(generated_texts) * len(forget_texts)
    sim_penalty = sim_penalty.to(device)
    
    # Combine losses with a weighting factor
    alpha = 0.5  # Weighting factor for similarity penalty
    total_loss = lm_loss + alpha * sim_penalty

    return total_loss

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(forget_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

sentence_transformer = transformer_model()
avg_loss = []
# Fine-tuning loop
model.train()
for epoch in range(10):  # 3 epochs
    batch_loss = []
    for batch in tqdm(forget_loader, desc=f"Training Epoch {epoch + 1}"):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        
        # Decode generated texts
        generated_texts = []
        for input_id in input_ids:
            generated_text = tokenizer.decode(input_id, skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Calculate custom loss
        loss = custom_loss_function(outputs, input_ids, generated_texts, forget_texts)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        batch_loss.append(loss)
        optimizer.step()
        scheduler.step()

        print(f"Loss: {loss.item()}")
    
    avg_loss.append(sum(batch_loss)/len(batch_loss))
# Save the fine-tuned model
model.save_pretrained("fine_tuned_gpt2_forget_set")
tokenizer.save_pretrained("fine_tuned_gpt2_forget_set")


plt.figure(figsize=(10, 6))
plt.plot(avg_loss, label='Training Loss', color='red', marker='o')

plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plt.grid(True)
plt.legend()
plt.show()