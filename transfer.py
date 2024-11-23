import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def track_logit_changes(original_logits, new_logits, reinforced_logits, tokenizer,input_ids, tracked_changes):
    """
    Tracks logits changes for each token and stores original, new, and reinforced logits values.

    Args:
    - original_logits: Logits from the original model [batch_size, max_length, vocab_size].
    - new_logits: Modified logits after applying the unlearning process [batch_size, max_length, vocab_size].
    - reinforced_logits: Logits from the reinforce model [batch_size, max_length, vocab_size].
    - input_ids: Token IDs for the current batch [batch_size, max_length].
    - tokenizer: Tokenizer to decode the input tokens.
    - tracked_changes: Dictionary to track original, reinforced, and new logits for words across epochs.
    
    Returns:
    - Updated tracked_changes dictionary.
    """
    batch_size, seq_len, vocab_size = original_logits.shape

    for batch_idx in range(batch_size):
        for token_idx in range(seq_len):
            token_id = input_ids[batch_idx][token_idx].item()
            word = tokenizer.decode([token_id])

            # Ignore end-of-text token  
            #if word.strip() == "<|endoftext|>":
                #continue

            # Track original, new, and reinforced logits values for this word
            original_logit = original_logits[batch_idx, token_idx].max().item()
            new_logit = new_logits[batch_idx, token_idx].max().item()
            reinforced_logit = reinforced_logits[batch_idx, token_idx].max().item()

            #if abs(new_logit-original_logit) < 1e-5:
               # continue

            if word not in tracked_changes:
                tracked_changes[word] = {'original': [], 'new': [], 'reinforced': []}

            tracked_changes[word]['original'].append(original_logit)
            tracked_changes[word]['new'].append(new_logit)
            tracked_changes[word]['reinforced'].append(reinforced_logit)

    return tracked_changes


def plot_tracked_changes(tracked_changes, epoch):
    words = list(tracked_changes.keys())
    num_words = len(words)

    # Initialize arrays to hold logit values
    original_values = [tracked_changes[word]['original'][-1] for word in words]  # Get latest values (from last epoch)
    reinforced_values = [tracked_changes[word]['reinforced'][-1] for word in words]
    new_values = [tracked_changes[word]['new'][-1] for word in words]

    # Bar width
    bar_width = 0.2

    # X-axis locations for the groups
    r1 = np.arange(num_words)  # Locations for original logits
    r2 = [x + bar_width for x in r1]  # Locations for reinforced logits
    r3 = [x + bar_width for x in r2]  # Locations for new logits

    # Create the bar plot
    plt.figure(figsize=(20, 10))

    # Plot each bar group
    plt.bar(r1, original_values, color='blue', width=bar_width, edgecolor='grey', label='Original')
    plt.bar(r2, reinforced_values, color='orange', width=bar_width, edgecolor='grey', label='Reinforced')
    plt.bar(r3, new_values, color='green', width=bar_width, edgecolor='grey', label='New')

    # Add labels and title
    plt.xlabel('Words', fontweight='bold')
    plt.ylabel('Logits Value', fontweight='bold')
    plt.title(f'Logits Comparison (Epoch {epoch})')
    plt.xticks([r + bar_width for r in range(num_words)], words, rotation=45, ha='right')  # Rotate words for readability

    # Add legend
    plt.legend()

    # Save the plot
    plt.tight_layout()

    plt.savefig(f"logits_compare/softmax/epoch_{epoch}_logit_changes.png")
    plt.close()

    print(f"Saved token-level logit change plot for epoch {epoch}.")


def plot_reinforced_logits(target_logits, reinforce_logits, tokenizer,input_ids, epoch, title="Logits Comparison"):
   
    #ids = input_ids     
    #input_ids = ids[0]  
    _, input_ids = torch.max(target_logits, dim=-1)
   
    filtered_input_ids = [token_id for token_id in input_ids.view(-1) if tokenizer.convert_ids_to_tokens([token_id])[0] not in tokenizer.all_special_tokens]

    # Convert input_ids to a list of token IDs (assuming a batch size of 1 for simplicity)
    
    
    #input_ids = filtered_input_ids.squeeze().tolist()  # [seq_len]
    input_ids = filtered_input_ids

    # Number of tokens in the sequence
    num_tokens = len(input_ids)

    # Convert token IDs to words using the tokenizer
    token_words = [tokenizer.decode([token_id]).strip() for token_id in input_ids]
    
    # Collect logits for each token in the input sequence
    target_forget_logits = [target_logits[0, i, input_ids[i]].item() for i in range(num_tokens)]
    reinforce_forget_logits = [reinforce_logits[0, i, input_ids[i]].item() for i in range(num_tokens)]
    
    # Generate x-axis labels (the words corresponding to each token)
    x_labels = token_words
    #import pdb;pdb.set_trace()
    
    # Create a bar plot for the logits comparison
    x = np.arange(num_tokens)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot target model logits
    ax.bar(x - width/2, target_forget_logits, width, label='Target Model', color='blue')
    
    # Plot reinforced model logits
    ax.bar(x + width/2, reinforce_forget_logits, width, label='Froget Model', color='orange')
    
    # Add labels, title, and legend
    ax.set_ylabel('Logit Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()

    # Show plot
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"logits_compare/auxillary/epoch_{epoch}.png")
    plt.close()



import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def plot_reinforced_softmax_scores(target_logits, reinforce_logits, tokenizer, input_ids, epoch, title="Softmax Score Comparison"):
    # Filter out special tokens from input_ids
    filtered_input_ids = [token_id for token_id in input_ids if tokenizer.convert_ids_to_tokens([token_id])[0] not in tokenizer.all_special_tokens]

    # Apply softmax to obtain probabilities for target and reinforced logits
    target_softmax = F.softmax(target_logits, dim=-1)
    reinforce_softmax = F.softmax(reinforce_logits, dim=-1)

    # Convert input_ids to a list of token IDs
    input_ids = filtered_input_ids
    num_tokens = len(input_ids)

    # Convert token IDs to words using the tokenizer
    token_words = [tokenizer.decode([token_id]).strip() for token_id in input_ids]

    # Collect softmax scores for each token in the input sequence
    target_softmax_scores = [target_softmax[0, i, input_ids[i]].item() for i in range(num_tokens)]
    reinforce_softmax_scores = [reinforce_softmax[0, i, input_ids[i]].item() for i in range(num_tokens)]

    # Generate x-axis labels (the words corresponding to each token)
    x_labels = token_words
    x = np.arange(num_tokens)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot target model softmax scores
    ax.bar(x - width/2, target_softmax_scores, width, label='Target Model Softmax', color='blue')

    # Plot reinforced model softmax scores
    ax.bar(x + width/2, reinforce_softmax_scores, width, label='Reinforced Model Softmax', color='orange')

    # Add labels, title, and legend
    ax.set_ylabel('Softmax Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend()

    # Show plot
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"logits_compare/vanilla/epoch_{epoch}_softmax.png")
    plt.close()

















































