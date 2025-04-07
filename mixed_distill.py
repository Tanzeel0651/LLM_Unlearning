import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from mixed_data_module import TextDatasetQA, mixed_data_collator, TextForgetDatasetQA
from aux_model import MLPForget

# Add CLI argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.4)
parser.add_argument('--forget_loss_alpha', type=float, default=0.7)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()


class CustomDistillationTrainer(Trainer):
    def __init__(self, mlp_forget_model, temperature, forget_loss_alpha, *args, **kwargs):
        self.mlp_forget_model = mlp_forget_model
        self.temperature = temperature
        self.forget_loss_alpha = forget_loss_alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        is_forget = inputs.get("is_forget", torch.zeros(input_ids.size(0), dtype=torch.bool))

        student_outputs = model(input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        total_loss = torch.tensor(0.0, device=student_logits.device)

        if is_forget.any():
            forget_student_logits = student_logits[is_forget]
            with torch.no_grad():
                forget_teacher_logits = self.mlp_forget_model(forget_student_logits.to(torch.float)).to(torch.bfloat16)

            forget_teacher_probs = F.softmax(forget_teacher_logits / self.temperature, dim=-1)
            forget_teacher_probs = 0.9 * forget_teacher_probs + 0.1 / forget_teacher_probs.size(-1)
            forget_student_log_probs = F.log_softmax(forget_student_logits / self.temperature, dim=-1)

            topk = 5
            topk_indices = torch.topk(forget_teacher_probs, topk, dim=-1).indices
            mask = torch.zeros_like(forget_teacher_probs).scatter_(-1, topk_indices, 1.0)

            forget_loss = F.kl_div(
                forget_student_log_probs * mask,
                forget_teacher_probs * mask,
                reduction="batchmean"
            )
            total_loss += self.forget_loss_alpha * forget_loss

        if "labels" in inputs and (~is_forget).any():
            retained_student_logits = student_logits[~is_forget]
            retained_labels = inputs["labels"][~is_forget]
            retained_loss = F.cross_entropy(
                retained_student_logits.view(-1, retained_student_logits.size(-1)),
                retained_labels.view(-1)
            )
            total_loss += retained_loss

        return (total_loss, student_outputs) if return_outputs else total_loss


def create_interleaved_dataset(forget_dataset, retained_dataset):
    mixed_dataset = []
    retain_index = 0
    forget_index = 0
    forget_length = len(forget_dataset)
    retain_length = len(retained_dataset)

    while forget_index < forget_length and retain_index < retain_length:
        for _ in range(1):
            if forget_index < forget_length:
                mixed_dataset.append(forget_dataset[forget_index])
                forget_index += 1
        if retain_index < retain_length:
            mixed_dataset.append(retained_dataset[retain_index])
            retain_index += 1

    while forget_index < forget_length:
        mixed_dataset.append(forget_dataset[forget_index])
        forget_index += 1

    while retain_index < retain_length:
        mixed_dataset.append(retained_dataset[retain_index])
        retain_index += 1

    return mixed_dataset


def prepare_mixed_dataset(forget_dataset, retained_dataset):
    mixed_data = create_interleaved_dataset(forget_dataset, retained_dataset)

    class MixedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    return MixedDataset(mixed_data)


def finetune_with_mixed_distillation():
    base_model = "/student/tahmad8/Videos/tofu/model/target_model/full_model_scratch"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

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
    mixed_dataset = prepare_mixed_dataset(forget_dataset, retained_dataset)

    mlp_forget_model = MLPForget(32000, 256).to("cuda")
    state_dict = torch.load("model/mlp_forget_40_scratch.pt", weights_only=True)
    mlp_forget_model.load_state_dict(state_dict)
    mlp_forget_model.eval()

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, use_flash_attention_2=True, device_map="cuda")
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

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        #num_train_epochs=args.epochs,
        max_steps=args.epochs,
        learning_rate=1e-5,
        bf16=True,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        evaluation_strategy="no",
        remove_unused_columns=False
    )

    trainer = CustomDistillationTrainer(
        mlp_forget_model=mlp_forget_model,
        temperature=args.temperature,
        forget_loss_alpha=args.forget_loss_alpha,
        model=model,
        args=train_args,
        train_dataset=mixed_dataset,
        data_collator=mixed_data_collator,
    )

    trainer.train()

    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}.")


if __name__ == "__main__":
    finetune_with_mixed_distillation()
