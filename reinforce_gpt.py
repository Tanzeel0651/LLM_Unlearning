import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import os
from utils import preprocess  # Assuming you have a custom preprocess function

# Load and preprocess text data
content = []
FILE = "human.txt"  # Specify a file name to filter, if necessary

# Ensure the directory exists
if not os.path.exists("dataset/"):
    raise FileNotFoundError("The directory 'dataset/' does not exist.")

for file in os.listdir("dataset/"):
    if FILE and file != FILE:
        continue
    # Load content from each file
    with open(os.path.join('dataset', file), 'r') as file_:
        content.extend(file_.read().split("."))
        print("File Added: ", file)

print("Length of content: ", len(content))

# Apply preprocessing
content = [preprocess(text) for text in content if preprocess(text)]
max_length = 50  # Max sequence length

print("Max Length: ", max_length)

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = tokenized_text['input_ids'].squeeze()
        attention_mask = tokenized_text['attention_mask'].squeeze()
        return input_ids, attention_mask

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    return input_ids, attention_masks

# Load the pre-trained tokenizer and model from the saved checkpoint
tokenizer = GPT2Tokenizer.from_pretrained("model/fine_tuned_gpt2_target")
model = GPT2LMHeadModel.from_pretrained("model/fine_tuned_gpt2_target")

# Ensure the tokenizer has an EOS token
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create the dataset and dataloader
dataset = TextDataset(content, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Define optimizer (can also load a previously saved optimizer state if needed)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# List to store the average loss per epoch
avg_loss = []

# Training loop for reinforcement learning
epochs = 10 # Specify the number of additional epochs
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids, attention_masks = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    avg_loss.append(average_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")
    plot_reinforced_logits()

# Plot training loss over epochs
# plt.figure(figsize=(10, 6))
# plt.plot(avg_loss, label='Training Loss', color='red', marker='o')
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Save the fine-tuned model again (re-saving after reinforcement learning)
#model.save_pretrained("model/fine_tuned_gpt2_reinforce")
#tokenizer.save_pretrained("model/fine_tuned_gpt2_reinforce")
print("Model saved again after reinforcement learning!")
