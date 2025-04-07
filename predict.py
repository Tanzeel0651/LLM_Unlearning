import torch
import torch.nn as nn
import datasets
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# base model
# model_id = "model/llm_weights/TOFU/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01/"
# forget model

# Updated dataset class with 'split' as a parameter
class TokenizedForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family, split="forget01", max_length=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        # load from some huggingface dataset with a custom split, e.g. "forget10"
        self.forget_data = datasets.load_dataset(data_path, split)['train']
        self.split_symbol = "[/INST] " if self.model_family == 'llama2-7b' else 'Answer: '

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        question = self.forget_data[idx]['question']
        answer = self.forget_data[idx]['answer']
        formatted_input = question + self.split_symbol

        encoding = self.tokenizer(
            formatted_input,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": answer,
            "question": question,  # Keep question for debugging/printing
        }

class TokenizedRandom40Dataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family, split="full", max_length=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        # load from some huggingface dataset with a custom split, e.g. "forget10"
        self.train_data = datasets.load_dataset(data_path, split)['train']
        self.split_symbol = "[/INST] " if self.model_family == 'llama2-7b' else 'Answer: '

    def __len__(self):
        return 40

    def __getitem__(self, idx):
        if idx>40: return
        idx = random.randint(0, len(self.train_data))
        question = self.train_data[idx]['question']
        answer = self.train_data[idx]['answer']
        formatted_input = question + self.split_symbol

        encoding = self.tokenizer(
            formatted_input,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": answer,
            "question": question,  # Keep question for debugging/printing
        }


forget_model_id = "model/llm_weights/forget_TOFU_01_steps70"
# target_model_id = "model/llm_weights/TOFU/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01/"

forget_model = AutoModelForCausalLM.from_pretrained(forget_model_id, torch_dtype=torch.bfloat16).cuda()
forget_model_tokenizer = AutoTokenizer.from_pretrained(forget_model_id)

# target_model = AutoModelForCausalLM.from_pretrained(target_model_id, torch_dtype=torch.bfloat16).cuda()
# target_model_tokenizer = AutoTokenizer.from_pretrained(target_model_id)

# Create the dataset, specifying split="forget10"
retain_dataset = TokenizedRandom40Dataset(
    data_path="locuslab/TOFU",
    tokenizer=forget_model_tokenizer,
    model_family="llama2-7b",
    #split="forget01"
)

forget_dataset = TokenizedForgetDataset(
    data_path="locuslab/TOFU",
    tokenizer=forget_model_tokenizer,
    model_family="llama2-7b",
    #split="forget01"
)

retain_question = []
retain_true_ans = []
retain_gen_ans = []

for idx in range(len(retain_dataset)):
    batch = retain_dataset[idx]

    # Prepare model input
    input_ids = batch["input_ids"].unsqueeze(0).cuda()
    attention_mask = batch["attention_mask"].unsqueeze(0).cuda()
    
    # Generate
    out = forget_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=600,
        do_sample=False,
        pad_token_id=forget_model_tokenizer.eos_token_id
    )


    gen_ans = forget_model_tokenizer.batch_decode(out[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
    question = batch["question"]
    true_ans = batch["answer"]

    retain_question.append(question)
    retain_true_ans.append(true_ans)
    retain_gen_ans.append(gen_ans)
    print("==== SAMPLE", idx, "====")
    print("Question: ", question)
    print("Generated Answer: ", gen_ans)
    print("True Answer: ", true_ans)
    
    
forget_question = []
forget_true_ans = []
forget_gen_ans = []

for idx in range(len(forget_dataset)):
    batch = forget_dataset[idx]

    # Prepare model input
    input_ids = batch["input_ids"].unsqueeze(0).cuda()
    attention_mask = batch["attention_mask"].unsqueeze(0).cuda()
    
    # Generate
    out = forget_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=600,
        do_sample=False,
        pad_token_id=forget_model_tokenizer.eos_token_id
    )


    gen_ans = forget_model_tokenizer.batch_decode(out[:, input_ids.shape[-1]:], skip_special_tokens=True)[0]
    question = batch["question"]
    true_ans = batch["answer"]

    forget_question.append(question)
    forget_true_ans.append(true_ans)
    forget_gen_ans.append(gen_ans)
    print("==== SAMPLE", idx, "====")
    print("Question: ", question)
    print("Generated Answer: ", gen_ans)
    print("True Answer: ", true_ans)
    


import pandas as pd
frame = pd.DataFrame({"Question":retain_question+forget_question, \
                    "True Answer":retain_true_ans+forget_true_ans, \
                    "Generated Answer":retain_gen_ans+forget_gen_ans})

frame.to_excel("file.xlsx", index=False)