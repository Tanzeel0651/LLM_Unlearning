import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from mixed_data_module import TextDatasetQA, mixed_data_collator, TextForgetDatasetQA
from aux_model import MLPForget

import torch
import torch.nn.functional as F

def generate_from_mlp_logits(
    logits,
    tokenizer,
    strategy="sampling",
    top_k=50,
    temperature=1.0,
    max_length=20
):
    probs = F.softmax(logits / 1.0, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, top_k, dim=-1)
    return [tokenizer.decode([i], skip_special_tokens=True) for i in topk_ids[0]]

class CustomDistillationTrainer(Trainer):
    def __init__(self, mlp_forget_model,my_tokenizer, *args, **kwargs):
        """
        Initialize custom trainer with MLP forget model
        
        Args:
            mlp_forget_model: Pre-trained MLP model for logits prediction
            *args: Positional arguments for Trainer
            **kwargs: Keyword arguments for Trainer
        """
        self.mlp_forget_model = mlp_forget_model
        self.my_tokenizer = my_tokenizer
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute selective loss:
        - KL divergence for forget set
        - Cross-entropy for retained set
        
        Args:
            model: Student model
            inputs: Input data dictionary
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments
        
        Returns:
            Loss tensor, optionally with model outputs
        """
        # Prepare inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        is_forget = inputs.get("is_forget", torch.zeros(input_ids.size(0), dtype=torch.bool))
        
        # Perform forward pass
        student_outputs = model(input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        stu_samples = generate_from_mlp_logits(student_logits[0,-1,:].unsqueeze(0), tokenizer=self.my_tokenizer, strategy="sampling", temperature=0.7)
        print("Original Model: ", stu_samples)

        # Initialize loss
        total_loss = torch.tensor(0.0, device=student_logits.device)
        forget_loss = 0
        retained_loss = 0
        #loss_alpha = 0.9 - 0.4 * (self.state.global_step / self.state.max_steps)  # best 0.7
        #loss_alpha = 0.3 

        # Compute loss for forget set
        if is_forget.any():
            # Filter logits for forget set
            forget_student_logits = student_logits[is_forget]
            
            # Predict logits using MLP forget model for forget set
            with torch.no_grad():
                forget_teacher_logits = self.mlp_forget_model(
                    forget_student_logits.to(torch.float)
                ).to(torch.bfloat16)
                samples = generate_from_mlp_logits(forget_teacher_logits[0,-1,:].unsqueeze(0), tokenizer=self.my_tokenizer, strategy="sampling", temperature=0.7)
                print("MLP Model: ",samples)
            
            # Temperature scaling for KL divergence
            #temperature = max(0.3, 1.2 - (self.state.global_step / self.state.max_steps))   # best 0.7
            temperature = 0.4 #0.7
            forget_teacher_probs = F.softmax(forget_teacher_logits / temperature, dim=-1)
            forget_teacher_probs = 0.9 * forget_teacher_probs + 0.1 / forget_teacher_probs.size(-1)
            
            forget_student_log_probs = F.log_softmax(forget_student_logits / temperature, dim=-1)
            
            # print("is_forget: ",is_forget,"doing KL divergence")
            # KL Divergence loss for forget set
            # forget_loss = F.kl_div(
            #     forget_student_log_probs, 
            #     forget_teacher_probs, 
            #     reduction="batchmean"
            # )

            loss_alpha = (forget_teacher_probs.max(dim=-1).values > 0.5).float().mean().item()
            loss_alpha = max(0.8, loss_alpha) #0.7
            #print("Loss alpha: ",loss_alpha)
            topk = 5
            topk_indices = torch.topk(forget_teacher_probs, topk, dim=-1).indices
            mask = torch.zeros_like(forget_teacher_probs).scatter_(-1, topk_indices, 1.0)
            forget_loss = F.kl_div(
                forget_student_log_probs * mask,
                forget_teacher_probs * mask,
                reduction="batchmean"
            )
            total_loss += loss_alpha*forget_loss
        
        # Compute cross-entropy for retained set
        loss_alpha = 1
        if "labels" in inputs and (~is_forget).any():
            # Filter logits and labels for retained set
            retained_student_logits = student_logits[~is_forget]
            retained_labels = inputs["labels"][~is_forget]
            
            # Standard cross-entropy for retained set
            # print("is_forget: ",is_forget,"doing cross entropy")
            retained_loss = F.cross_entropy(
                retained_student_logits.view(-1, retained_student_logits.size(-1)), 
                retained_labels.view(-1)
            )
            #total_loss += (1-loss_alpha)*retained_loss
            total_loss += loss_alpha*retained_loss
        
        #total_loss = 0.7 * forget_loss + (1 - 0.7) * retained_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


def create_interleaved_dataset(forget_dataset, retained_dataset):
    """
    Create a mixed dataset with an interleaved pattern of 2 forget followed by 1 retain
    
    Args:
        forget_dataset (Dataset): Dataset containing forget examples
        retained_dataset (Dataset): Dataset containing retained examples
    
    Returns:
        List: Mixed dataset following the 2 forget + 1 retain pattern
    """
    mixed_dataset = []
    retain_index = 0
    forget_index = 0

    # Ensure we don't run out of samples
    forget_length = len(forget_dataset)
    retain_length = len(retained_dataset)

    while forget_index < forget_length and retain_index < retain_length:
        # Add two forget examples
        #while forget_index < forget_length and len(mixed_dataset) % 3 != 2:
        # Testing with 3
        for _ in range(1):
            if forget_index < forget_length:
                mixed_dataset.append(forget_dataset[forget_index])
                forget_index += 1
        
        # Add one retain example
        if retain_index < retain_length:
            mixed_dataset.append(retained_dataset[retain_index])
            retain_index += 1

    # Add remaining forget examples if any
    while forget_index < forget_length:# and len(mixed_dataset) % 3 != 2:
        mixed_dataset.append(forget_dataset[forget_index])
        forget_index += 1

    # Add remaining retain examples if any
    while retain_index < retain_length:
        mixed_dataset.append(retained_dataset[retain_index])
        retain_index += 1

    return mixed_dataset

# Example usage in training script
def prepare_mixed_dataset(forget_dataset, retained_dataset):
    """
    Prepare a mixed dataset with controlled forget and retain sequence
    
    Args:
        forget_dataset (Dataset): Dataset of examples to forget
        retained_dataset (Dataset): Dataset of examples to retain
    
    Returns:
        torch.utils.data.Dataset: Mixed dataset
    """
    # Create interleaved dataset
    mixed_data = create_interleaved_dataset(forget_dataset, retained_dataset)
    
    # Convert to torch Dataset if needed
    class MixedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return MixedDataset(mixed_data)

def finetune_with_mixed_distillation(
    base_model,
    output_dir,
    mlp_model_path,
    lr,
    batch_size,
    max_steps=50,
    use_lora=True,
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    forget_dataset = TextForgetDatasetQA(
        data_path="locuslab/TOFU",
        tokenizer=tokenizer,
        model_family="llama2-7b",
        split="forget10"
    )
    
    retained_dataset = TextDatasetQA(
        data_path="locuslab/TOFU",
        tokenizer=tokenizer,
        model_family="llama2-7b",
        split="retain90",
        total_length=10
    )
    
    # Create mixed dataset
    mixed_dataset = prepare_mixed_dataset(forget_dataset, retained_dataset)
    
    # Load MLP forget model
    mlp_forget_model = MLPForget(vocab_size=32000, hidden_dim=256).to("cuda")#, hidden_dim=256).to("cuda")
    state_dict = torch.load(mlp_model_path, weights_only=True)
    mlp_forget_model.load_state_dict(state_dict)
    mlp_forget_model.eval()

    # Load Llama model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
        device_map="cuda"
    )

    # Optional: LoRA
    if use_lora:    
        config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"], 
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

    # Trainer settings
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=3,
        num_train_epochs=1,
        #max_steps=150,
        learning_rate=lr,
        bf16=True,
        weight_decay=0.01,
        logging_steps=max(max_steps//5, 1),
        save_steps=max_steps,
        save_total_limit=1,
        evaluation_strategy="no",
        remove_unused_columns=False
    )

    trainer = CustomDistillationTrainer(
        mlp_forget_model=mlp_forget_model,
        model=model,
        args=train_args,
        train_dataset=mixed_dataset,
        data_collator=mixed_data_collator,
        my_tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # Merge LoRA if needed
    if use_lora:
        model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Model saved to {output_dir}.")

# Example usage
if __name__ == "__main__":
    finetune_with_mixed_distillation(
        base_model="model/target_model/full_model_scratch",
        #base_model="locuslab/tofu_ft_llama2-7b",
        output_dir="model/forget_model/mixed_distillation_v2",
        mlp_model_path="model/mlp_forget_40_scratch.pt",
        batch_size=1,
        lr=1e-5,
        max_steps=30,
        use_lora=True
    )
