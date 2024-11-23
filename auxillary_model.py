import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class AuxiliaryModel(nn.Module):
    def __init__(self, vocab_size):
        super(AuxiliaryModel, self).__init__()
        h1,h2,h3 = 1280,640,320
        self.model = nn.Sequential(
            nn.Linear(vocab_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, vocab_size)
        )
        

    def forward(self, x):
        return self.model(x)

        
if __name__=='__main__':

    loaded_logits = torch.load("new_logits_data.pt", weights_only=False)
    original_logits = torch.cat(loaded_logits["original_logits"])
    target_logits = loaded_logits["target_logits"]
    # shape of [1,50,50257] is expected for both

    assert original_logits.shape == target_logits.shape

    input_loader = DataLoader(original_logits)
    output_loader = DataLoader(target_logits)

    temperature = 1.5
    # target_logits = target_logits / 1.5

    model = AuxiliaryModel(original_logits.shape[-1]).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        for batch in zip(input_loader, output_loader):
            # import pdb;pdb.set_trace()
            input_, target = batch
            error = 0
            optimizer.zero_grad()
            output = model(input_)
            loss = criterion(output, target)
            
            error = loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}, MSE Loss: {error}")

    torch.save(model, "model/auxillary_model2")
