            
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt

import os
from utils import preprocess


# Load and preprocess text data
content = []
FILE = "human.txt"  # Specify a file name to filter, if necessary

# Ensure the directory exists
if not os.path.exists("dataset/"):
    raise FileNotFoundError("The directory 'sample_data/' does not exist.")

for file in os.listdir("dataset/"):
    if FILE and file != FILE:
        continue
    # if file == "human.txt":
    with open(os.path.join('dataset', file), 'r') as file_:
        content.extend(file_.read().split("."))
        print("File Added: ", file)


print("Length of content: ", len(content))



# Apply preprocessing
content = [preprocess(text) for text in content if preprocess(text)]

# print("Content 58: ",content[58])

# max_length = max([len(x.split()) for x in content])+5
max_length = 50

print("Max Length: ", max_length)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        
        # Tokenize and encode text with padding
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


# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure the tokenizer has an EOS token
tokenizer.pad_token = tokenizer.eos_token

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define maximum sequence length for padding
# max_length = 50

# Create the dataset and dataloader
dataset = TextDataset(content, tokenizer, max_length)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Define avg_loss list
avg_loss = []

# Training loop
epochs = 15
model.train()
all_logits = []
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids, attention_masks = batch
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=input_ids)
        
        logits = outputs.logits
        all_logits.append(logits.cpu().detach())
        
        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backpropagation
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    avg_loss.append(average_loss)
    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")


# plot avg loss
plt.figure(figsize=(10, 6))
plt.plot(avg_loss, label='Training Loss', color='red', marker='o')

plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plt.grid(True)
plt.legend()
# plt.show()

# Save the fine-tuned model
model_name = "model/fine_tuned_gpt2_vanilla"
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)
print("Model Successfully saved on:",model_name)

# Example text generation
# model.eval()
# def generate_text(prompt, max_length=50):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
#     attention_mask = torch.ones(input_ids.shape, device=device)

#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_length=max_length,
#             num_return_sequences=1,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95
#         )

#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # Test the generation
# seed_text = "with local populations of"
# generated_text = generate_text(seed_text)
# print(f"Prompt: {seed_text}\nGenerated: {generated_text}")
