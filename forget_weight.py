import torch
import numpy as np
from utils import files_to_dict
from transformers import AutoTokenizer
import torch.nn.functional as F
from transfer import plot_reinforced_logits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_target_logits():
    forget_set = " ".join(files_to_dict()["human.txt"])
    tokenizer = AutoTokenizer.from_pretrained("model/fine_tuned_gpt2_target")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = tokenizer.encode(forget_set, return_tensors='pt').to(device).squeeze(0).unique()
    logits = torch.load("logits_data.pt", weights_only=False)
    original_logits = torch.cat(logits["original_logits"])
    #target_logits = torch.cat(logits["new_logits"])
    # import pdb;pdb.set_trace()
    probability = 0.85
    decrease_factor = 0.5
    temperature = 1.5

    target_logits = original_logits.clone()
    for batch_idx in range(original_logits.shape[0]):
        softmax_probs = F.softmax(original_logits[batch_idx], dim=-1)
        print("Batch Idx: ",batch_idx)

        for seq_idx in range(original_logits.shape[1]):
            threshold = torch.quantile(softmax_probs[seq_idx], probability)
            for token_id in input_ids:
                if softmax_probs[seq_idx, token_id] > threshold:
                    original_logit_value = original_logits[batch_idx, seq_idx, token_id]
                    
                    if original_logit_value > 0:
                        target_logits[batch_idx, seq_idx, token_id] -= decrease_factor*original_logit_value
                    else:
                        target_logits[batch_idx, seq_idx, token_id] -= decrease_factor*abs(original_logit_value)
                        

    torch.save({"original_logits":logits["original_logits"], "target_logits":target_logits/temperature}, 'new_logits_data.pt')
    print("File saved")

    # Clear any remaining cache to prevent memory issues
    del original_logits, target_logits, input_ids
    gc.collect()
    torch.cuda.empty_cache()



def logits_by_percentile(original_logits, tokenizer, forget_texts, probability = 0.7, decrease_factor = 0.5):
    forget_set = " ".join(forget_texts)
    input_ids = tokenizer.encode(forget_set, return_tensors='pt').to(device).squeeze(0).unique()

    probability = probability
    decrease_factor = decrease_factor

    target_logits = original_logits.clone()
    for batch_idx in range(original_logits.shape[0]):
        softmax_probs = F.softmax(original_logits[batch_idx], dim=-1)
        print("Batch Idx: ",batch_idx)

        for seq_idx in range(original_logits.shape[1]):
            threshold = torch.quantile(softmax_probs[seq_idx], probability)
            for token_id in input_ids:
                if softmax_probs[seq_idx, token_id] > threshold:
                    original_logit_value = original_logits[batch_idx, seq_idx, token_id]
                    
                    if original_logit_value > 0:
                        target_logits[batch_idx, seq_idx, token_id] -= decrease_factor*original_logit_value
                    else:
                        target_logits[batch_idx, seq_idx, token_id] -= decrease_factor*abs(original_logit_value)
                 
    return target_logits/1.5


if __name__ == '__main__':
    create_target_logits()