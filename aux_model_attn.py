import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

class MLPForget(nn.Module):
    def __init__(self, vocab_size, hidden_dim=1024, num_layers=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Enhanced architecture with residual connections
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else vocab_size
            out_dim = hidden_dim if i < num_layers-1 else vocab_size
            
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < num_layers-1 else nn.Identity(),
                nn.LayerNorm(out_dim) if i < num_layers-1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers-1 else nn.Identity()
            ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        # Handle different input shapes
        if x.ndim == 3:
            # If input is [batch, seq_len, vocab]
            orig_shape = x.shape
            x = rearrange(x, 'b s v -> (b s) v')
            x = self.model(x)
            x = rearrange(x, '(b s) v -> b s v', b=orig_shape[0], s=orig_shape[1])
        elif x.ndim == 2:
            # If input is already [batch*seq_len, vocab]
            x = self.model(x)
        else:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {x.ndim}")
        
        return x

class LogitDataset(Dataset):
    def __init__(self, original_logits, target_logits, attention_masks):
        self.original = original_logits  # [batch, seq_len, vocab]
        self.target = target_logits
        self.masks = attention_masks  # [batch, seq_len]
        
        # Initialize trimmed lists
        self.trimmed_original = []
        self.trimmed_target = []
        
        for i in range(len(self.original)):
            orig = self.original[i]  # [seq_len, vocab]
            targ = self.target[i]    # [seq_len, vocab]
            mask = self.masks[i]      # [seq_len]
            
            # Make sure the mask is the right shape for indexing
            if mask.ndim > 1:
                mask = mask.squeeze(0)  # Remove batch dimension if present
            
            # Make sure mask has the right sequence length
            valid_length = min(mask.shape[0], orig.shape[0])
            valid_indices = mask[:valid_length].bool()
            
            # Apply the mask to valid portions only
            self.trimmed_original.append(orig[:valid_length][valid_indices])
            self.trimmed_target.append(targ[:valid_length][valid_indices])

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        return {
            'orig': self.trimmed_original[idx],
            'targ': self.trimmed_target[idx],
            'mask': self.masks[idx] if self.masks[idx].ndim == 1 else self.masks[idx].squeeze(0)
        }

def train_mlp(config):
    # Load prepared logit data
    data = torch.load(config.logit_path)
    
    # Create dataset with padding trimming
    dataset = LogitDataset(
        data['original_logits'],
        data['target_logits'], 
        data['attention_mask']
    )
    
    # Create dataloader with dynamic batching
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'orig': torch.cat([item['orig'] for item in batch]),
            'targ': torch.cat([item['targ'] for item in batch]),
            'mask': torch.cat([item['mask'] for item in batch])
        }
    )

    # Initialize model with mixed precision
    model = MLPForget(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=6,
        dropout=0.2
    ).to(config.device).bfloat16()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs
    )

    # Focused loss function
    def weighted_mse(pred, target, mask=None):
        # Create weight matrix emphasizing answer tokens
        weights = torch.ones_like(target)
        if mask is not None:
            # If we have a mask, use it to weight tokens
            weights = torch.where(
                (target < -1e3),  # Masked positions
                torch.ones_like(target),  # Lower weight for padding
                torch.ones_like(target) * 5.0  # Higher weight for real tokens
            )
        return (weights * (pred - target) ** 2).mean()

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            orig = batch['orig'].to(config.device).bfloat16()
            targ = batch['targ'].to(config.device).bfloat16()
            mask = batch['mask'].to(config.device) if 'mask' in batch else None
            
            optimizer.zero_grad()
            
            # Forward pass - note orig is already [batch*seq_len, vocab]
            pred = model(orig)
            
            # Calculate loss
            loss = weighted_mse(pred, targ, mask)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), config.save_path)
    print(f"Saved optimized MLP to {config.save_path}")

if __name__ == "__main__":
    class Config:
        logit_path = "logits_data_40.pt"
        batch_size = 256
        lr = 3e-4
        epochs = 100
        device = "cuda"
        save_path = "model/mlp_forget_optimized_40.pt"
    
    train_mlp(Config())