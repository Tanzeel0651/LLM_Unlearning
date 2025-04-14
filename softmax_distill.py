import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
import re


class EntityForgetDataset(Dataset):
    """
    Dataset that loads from a dictionary saved in a .pt file with:
    {
        input_ids_tuple: {
            "attention_mask": tensor,
            "dampened_logits": tensor,
            "modified_tokens": list of (token, pos, old_val, new_val, ...)
        }
    }
    
    Handles question and answer portions, focusing loss on only the answer section.
    """
    def __init__(self, pointer_file):
        super().__init__()
        self.data_dict = torch.load(pointer_file)
        # Convert dictionary to list for easy indexing
        self.samples = []
        
        for input_ids_tuple, content in self.data_dict.items():
            input_ids = torch.tensor(list(input_ids_tuple))
            attention_mask = content["attention_mask"]
            dampened_logits = content["dampened_logits"]
            
            # Extract question length by detecting the position of the instruction end token
            # We'll use this to only apply loss to the answer portion
            question_length = self._find_question_end(input_ids)
            
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "target_logits": dampened_logits,
                "question_length": question_length
            })
    
    def _find_question_end(self, input_ids):
        """Find the position where the question ends and answer begins."""
        # This is a simple approach looking for [/INST] or similar tokens
        # Adapt this to match your tokenizer's specific instruction tokens
        # For most models, looking at the attention_mask is not enough
        
        # For LLaMA-based models, look for end instruction tokens (usually </s> or [/INST])
        # Example implementation - adapt to your specific tokenizer
        input_ids_list = input_ids.tolist()
        
        # Common instruction end markers (adapt to your tokenizer)
        # Look for the last one since there might be multiple in the input
        end_markers = [29889, 29901, 2]  # Examples: [/INST], </s>, EOS
        
        for marker in end_markers:
            if marker in input_ids_list:
                # Find the last occurrence
                idx = len(input_ids_list) - 1 - input_ids_list[::-1].index(marker)
                return idx + 1  # +1 to start right after the marker
        
        # Fallback: Use a pattern-based approach by decoding the input_ids
        # This should be replaced with your specific tokenizer's approach
        return len(input_ids) // 2  # Fallback: assume the question is half the sequence
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "target_logits": item["target_logits"],
            "question_length": item["question_length"]
        }

class AnswerOnlyKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        target_logits = inputs["target_logits"].to(model.device)
        question_lengths = inputs["question_length"].to(model.device)

        outputs = model(input_ids, attention_mask=attention_mask)
        student_logits = outputs.logits  # [B, S, V]

        # Initialize for empty batch case
        batch_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        count = 0

        for i in range(input_ids.size(0)):
            q_len = question_lengths[i].item()
            seq_len = attention_mask[i].sum().item()

            # Only process if we have answer tokens
            if q_len < seq_len:
                # Extract answer portions
                stu_ans = student_logits[i, q_len:seq_len]
                tgt_ans = target_logits[i, q_len:seq_len]

                # Make sure we're using the same length for both
                L = min(stu_ans.size(0), tgt_ans.size(0))
                stu_ans = stu_ans[:L]
                tgt_ans = tgt_ans[:L]

                # Calculate KL divergence
                student_log_probs = F.log_softmax(stu_ans, dim=-1)
                target_probs = F.softmax(tgt_ans, dim=-1)
                
                # Important: Use reduction='none' to get per-token KL, then average manually
                kl_divergence = F.kl_div(
                    student_log_probs,
                    target_probs,
                    reduction="batchmean",
                    log_target=False
                )
                
                # Add to batch loss directly to maintain gradient connection
                batch_loss = batch_loss + kl_divergence
                count += 1

        # Average the loss if we had any valid examples
        if count > 0:
            batch_loss = batch_loss / count

        return (batch_loss, outputs) if return_outputs else batch_loss

def main():
    # Configuration parameters
    base_model = "model/target_model/full_model_scratch"
    pointer_file = "forget_dampened_data_entity_focused.pt"  # Using the entity-focused dampened data
    output_dir = "model/forget_model/entity_focused_kl"
    
    # Training hyperparameters
    batch_size = 1  # Safer for very long sequences
    epochs = 3      # Multiple epochs since we have limited data
    learning_rate = 2e-5
    use_lora = True
    
    # Load the dataset with special handling for question vs. answer portions
    dataset = EntityForgetDataset(pointer_file)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    # Apply LoRA if requested for efficient fine-tuning
    if use_lora:
        # Configure LoRA to target key attention and MLP components
        config = LoraConfig(
            r=16,              # Higher rank for potentially better unlearning
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=True,
        weight_decay=0.01,
        logging_steps=1,         # Log every step since we have limited data
        save_steps=len(dataset), # Save once per epoch
        save_total_limit=1,      # Keep only the best model
        num_train_epochs=epochs,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,  # Helps with stability
        warmup_steps=10,               # Short warmup
        lr_scheduler_type="cosine",    # Cosine decay
        optim="adamw_torch"
    )
    
    # Create our custom trainer
    trainer = AnswerOnlyKLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Run training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    if use_lora:
        # Merge LoRA weights into base model before saving
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
    
    print(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()