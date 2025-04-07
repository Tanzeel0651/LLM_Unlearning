import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from data_module import TextDatasetQA, custom_data_collator
from aux_model import MLPForget


# ==================
#  Data classes
# ==================
# class ForgetDistillDataset(torch.utils.data.Dataset):
#     def __init__(self, logits_file):
#         data = torch.load(logits_file)
#         self.input_ids_list = data["input_ids"]         # list of Tensors
#         self.attention_mask_list = data["attention_mask"]
#         self.forget_logits_tensor = data["forget_logits"]  # [N, max_seq_len, vocab_size]
        
#         self.num_examples = len(self.input_ids_list)
#         assert self.num_examples == self.forget_logits_tensor.size(0), \
#             "Mismatch: len(input_ids_list) != forget_logits_tensor.shape[0]"

#     def __len__(self):
#         return self.num_examples

#     def __getitem__(self, idx):
#         inp = self.input_ids_list[idx]         # shape [1, seq_len]?
#         inp = inp.squeeze(0)                   # now shape [seq_len]

#         attn = self.attention_mask_list[idx]   # shape [1, seq_len]?
#         attn = attn.squeeze(0)                 # now shape [seq_len]

#         return {
#             "input_ids": inp,
#             "attention_mask": attn,
#             "forget_logits": self.forget_logits_tensor[idx],  # etc.
#         }

# def forget_collator(batch):
#     batch_max_len = max(item["input_ids"].size(0) for item in batch)

#     padded_input_ids = []
#     padded_attn_masks = []
#     sliced_forget_logits = []

#     for item in batch:
#         inp = item["input_ids"]
#         msk = item["attention_mask"]
#         f_log = item["forget_logits"]  # shape: [global_max_seq_len, vocab_size]

#         seq_len_i = inp.size(0)

#         # Pad input_ids
#         pad_inp = torch.zeros(batch_max_len, dtype=inp.dtype)
#         pad_inp[:seq_len_i] = inp
#         padded_input_ids.append(pad_inp)

#         # Pad attention_mask
#         pad_msk = torch.zeros(batch_max_len, dtype=msk.dtype)
#         pad_msk[:seq_len_i] = msk
#         padded_attn_masks.append(pad_msk)

#         # Slice forget_logits to batch_max_len
#         # (the real sequence is seq_len_i, but we unify to batch_max_len)
#         pad_f = f_log[:batch_max_len, :]
#         sliced_forget_logits.append(pad_f)
    
#     padded_input_ids = torch.stack(padded_input_ids, dim=0)
#     padded_attn_masks = torch.stack(padded_attn_masks, dim=0)
#     sliced_forget_logits = torch.stack(sliced_forget_logits, dim=0)

#     return {
#         "input_ids": padded_input_ids,
#         "attention_mask": padded_attn_masks,
#         "forget_logits": sliced_forget_logits
#    }

def mixed_data_collator(features):
    # input_ids = torch.stack([f["question_ids"] for f in features], dim=0)
    # attention_mask = torch.stack([f["question_only_attention"] for f in features], dim=0)
    input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
    attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
    #labels = torch.stack([f["labels"] for f in features], dim=0)
    #idx = torch.stack([f["idx"] for f in features], dim=0)

    # is_forget is a bool or bool-like for each item in the batch
    # We'll turn it into a bool tensor of shape [batch]
    is_forget = torch.tensor([f["is_forget"] for f in features], dtype=torch.bool)

    return {
        # "question_ids": question_ids,
        # "question_only_attention": question_attention,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    #     "labels": labels,
    #     "idx": idx,
    #     "is_forget": is_forget
    }

def compute_distillation_loss(model, inputs, num_items_in_batch=1, return_outputs=False):
    import torch.nn.functional as F

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]   
    outputs = model(input_ids, attention_mask=attention_mask)
    student_logits = outputs.logits
    
    mlp_model_file = "model/mlp_forget_model_40.pt"
    mlp_forget_model = MLPForget(vocab_size=32000, hidden_dim=256).to("cuda")
    state_dict = torch.load(mlp_model_file, weights_only=True)
    mlp_forget_model.load_state_dict(state_dict)
    mlp_forget_model.eval()
    with torch.no_grad():
        teacher_logits = mlp_forget_model(student_logits.to(torch.float)).to(torch.bfloat16)
        teacher_logits = teacher_logits.to("cuda")

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return (loss, outputs) if return_outputs else loss

# ==================
#  Fine-tuning code
# ==================
def finetune_llama_with_forget(
    logits_file,
    base_model,
    output_dir,
    batch_size=1,
    lr=1e-4,
    max_steps=50,
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    seed=42
):
    set_seed(seed)

    # dataset = ForgetDistillDataset(logits_file)
    
    # We'll define a simple data_collator
    #data_collator = forget_collator
        
    # Load tokenizere-05_llama2-7b_full_wd0.01"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    forget_dataset = TextDatasetQA(
                data_path="locuslab/TOFU",
                tokenizer=tokenizer,
                model_family="llama2-7b",
                split="forget10",
                return_question=True
                )

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
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"], 
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # Trainer settings
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        #max_steps=max_steps,
        num_train_epochs=1,
        learning_rate=lr,
        bf16=True,
        logging_steps=max(1, max_steps//5),
        save_steps=max_steps,
        save_total_limit=1,
        evaluation_strategy="no",
        seed=seed,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=forget_dataset,
        data_collator=mixed_data_collator,
    )

    # Override compute_loss
    trainer.compute_loss = compute_distillation_loss

    # Train
    trainer.train()

    # Merge LoRA if needed
    if use_lora:
        model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done. Model saved to {output_dir}.")

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} || All params: {all_params} "
          f"|| Trainable%: {100 * trainable_params / all_params:.4f}")

# ==================
# Example usage
# ==================
if __name__ == "__main__":
    finetune_llama_with_forget(
        logits_file="logits_data.pt",
        base_model="model/target_model/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01",
        #base_model = "locuslab/tofu_ft_llama2-7b",
        output_dir="model/forget_model/forget_TOFU_10_epochs1_distill_mlp_40",
        batch_size=2,
        lr=1e-4,
        max_steps=30,
        use_lora=True
    )

