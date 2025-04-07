import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Example simpler model
class MLPForget(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, vocab_size]
        bsz, seqlen, vsz = x.shape
        x = x.view(bsz * seqlen, vsz)
        out = self.model(x)
        out = out.view(bsz, seqlen, vsz)
        return out

if __name__=='__main__':
    # Suppose you have:
    # original_logits: [40, 60, 32000]
    # target_logits:   [40, 60, 32000]
    loaded = torch.load("logits_data_30_scratch.pt")
    original_logits = loaded["original_logits"]  # shape [40, 60, 32000]
    target_logits = loaded["target_logits"]      # shape [40, 60, 32000]
    
    
    
    original_logits = original_logits.float().cuda().to(torch.bfloat16)
    target_logits = target_logits.float().cuda().to(torch.bfloat16)
    
    # Move them to GPU if feasible
    original_logits = original_logits
    target_logits = target_logits

    # Build a dataset that returns (orig, targ) together
    dataset = TensorDataset(original_logits, target_logits)
    
    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    
    model = MLPForget(vocab_size=32000, hidden_dim=256).cuda().to(torch.bfloat16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        total_loss = 0.0
        for orig_batch, targ_batch in dataloader:
            optimizer.zero_grad()
            pred = model(orig_batch)   # shape [batch_size, seq_len, vocab_size]
            loss = criterion(pred, targ_batch)
            loss_f = loss.float()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: MSE Loss = {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "model/mlp_forget_30_scratch.pt")
    print("Auxillary Model saved at : ", "model/mlp_forget_30_scratch.pt")
    
