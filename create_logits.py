import torch
import torch.nn as nn
import datasets
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_module import TextDatasetQA

#model_id = "model/llm_weights/TOFU/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01"
#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
#tokenizer = AutoTokenizer.from_pretrained(model_id)
model_id = "model/target_model/full_model_scratch"
#model_id = "locuslab/tofu_ft_llama2-7b"
model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=True, torch_dtype=torch.bfloat16, trust_remote_code = True, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Updated dataset class with 'split' as a parameter
class TokenizedForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family, split="forget10", max_length=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_family = model_family
        # load from some huggingface dataset with a custom split, e.g. "forget10"
        self.forget_data = datasets.load_dataset(data_path, split)['train']
        self.split_symbol = " [/INST]" if self.model_family == 'llama2-7b' else 'Answer: '

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

def reduce_logits_for_answer(
    all_logits,
    true_text,
    generated_text,
    tokenizer,
    all_input_ids,
    question_length,
    alpha
):
    """ Reduce logits for any token that appears in either the true_text or generated_text. 
        Only reduce within the generated portion of the sequence (i.e. after question_length). 
    """
    # Gather IDs from true answer & generated text
    #true_token_ids = tokenizer.encode(true_text, add_special_tokens=False)
    generated_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
    #target_token_ids = set(true_token_ids + generated_token_ids)
    target_token_ids = set(generated_token_ids)

    logit_changes = []
    batch_size, seq_len, vocab_size = all_logits.shape

    for b in range(batch_size):
        # Only reduce in the generated portion
        for seq_idx in range(question_length, seq_len):
            token_id = all_input_ids[b, seq_idx].item()
            if token_id in target_token_ids:
                orig_logit = all_logits[b, seq_idx, token_id].item()
                adj_logit = orig_logit - alpha * abs(orig_logit)  # reduce by alpha
                all_logits[b, seq_idx, token_id] = adj_logit

                # For debugging/log
                token_str = tokenizer.decode([token_id])
                logit_changes.append((token_str, orig_logit, adj_logit))

    # Print summary
    print("\nToken Reduction Summary:")
    for token, orig_l, adj_l in logit_changes:
        print(f"Token: {token} | Original Logit: {orig_l:.4f} -> {adj_l:.4f}")

    return all_logits

#Create the dataset, specifying split="forget10"
forget_dataset = TokenizedForgetDataset(
   data_path="locuslab/TOFU",
   tokenizer=tokenizer,
   model_family="llama2-7b",
   split="forget10"
)
            #import pdb;pdb.set_trace()

# Taking dataset from TextDatasetQA
# forget_dataset = TextDatasetQA(
#         data_path="locuslab/TOFU",
#         tokenizer=tokenizer,
#         model_family="llama2-7b",
#         split="forget10",
#         return_question=True
# )


original_logits_list = []
target_logits_list = []
input_ids_list = []
attention_masks_list = []


for idx in range(len(forget_dataset)):
    batch = forget_dataset[idx]

    # Prepare model input
    input_ids = batch["input_ids"].unsqueeze(0).cuda()
    attention_mask = batch["attention_mask"].unsqueeze(0).cuda()
    # input_ids = batch["question_ids"].unsqueeze(0).cuda()
    # attention_mask = batch["question_only_attention"].unsqueeze(0).cuda()
    
    question_length = input_ids.shape[1]

    # Generate
    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=600,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Rerun model on the full sequence (prompt + generation) to get logits
    all_input_ids = out
    all_attention_mask = (all_input_ids != tokenizer.pad_token_id).long().cuda()

    with torch.no_grad():
        outputs = model(all_input_ids, attention_mask=all_attention_mask)
        all_logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

    # For printing
    gen_ans = tokenizer.decode(all_input_ids[0, question_length:], skip_special_tokens=True)
    question = batch["question"]
    true_ans = batch["answer"]
    #import pdb;pdb.set_trace()
    print("==== SAMPLE", idx, "====")
    print("Question: ", question)
    print("Generated Answer: ", gen_ans)
    print("True Answer: ", true_ans)
    
    # Save the original logits (before any modifications)
    original_logits = all_logits.clone()

    # Create target_logits by "forgetting" tokens
    target_logits = reduce_logits_for_answer(
        all_logits=all_logits,
        true_text=true_ans,
        generated_text=gen_ans,
        tokenizer=tokenizer,
        all_input_ids=all_input_ids,
        question_length=question_length,
        alpha=0.3
    )

    # Store the data
    original_logits_list.append(original_logits.cpu())
    target_logits_list.append(target_logits.cpu())
    input_ids_list.append(all_input_ids.cpu())
    attention_masks_list.append(all_attention_mask.cpu())

# 1. Find the maximum seq_length across all logits
max_seq_len = max(logits.shape[1] for logits in original_logits_list)
vocab_size = original_logits_list[0].shape[2]  # should be the same for all

# 2. Create a padded version for each example
padded_original = []
padded_target = []
for orig, targ in zip(original_logits_list, target_logits_list):
    seq_len = orig.shape[1]
 
    # Create a new zero tensor of shape (1, max_seq_len, vocab_size)
    padded_o = torch.zeros((1, max_seq_len, vocab_size), dtype=orig.dtype)

    # Copy the actual logits up to seq_len
    padded_o[:, :seq_len, :] = orig
    padded_original.append(padded_o)

    # Do the same for target_logits
    padded_t = torch.zeros((1, max_seq_len, vocab_size), dtype=targ.dtype)
    padded_t[:, :seq_len, :] = targ
    padded_target.append(padded_t)

# 3. Now we can concatenate along dim=0
original_logits_tensor = torch.cat(padded_original, dim=0)
target_logits_tensor   = torch.cat(padded_target, dim=0)
fileName = "logits_data_30_scratch.pt"

# Then save them
torch.save({
    "input_ids": input_ids_list,
    "attention_mask":attention_masks_list,
    "original_logits": original_logits_tensor,
    "target_logits": target_logits_tensor
}, fileName)


print(f"\nLogits saved successfully to {fileName}")


