import torch
import torch.nn.functional as F

def sample_from_teacher_logits(teacher_logits, tokenizer, temperature=1.0, top_k=50):
    """
    Sample token-by-token from the teacher logits.

    teacher_logits: Tensor of shape [batch_size, seq_len, vocab_size]
    tokenizer: The same tokenizer used by the model
    temperature: Softens/sharpens the distribution when sampling
    top_k: If not None, we restrict sampling to the top_k tokens to reduce randomness

    Returns:
        A list of decoded strings (one per batch element).
    """
    batch_size, seq_len, vocab_size = teacher_logits.size()
    sampled_texts = []

    for b in range(batch_size):
        sampled_token_ids = []
        for s in range(seq_len):
            # 1) Get raw logits for this position
            logits = teacher_logits[b, s, :]

            # 2) Apply temperature
            logits = logits / temperature

            # 3) If top_k is specified, restrict distribution to top_k tokens
            if top_k is not None:
                top_vals, top_idx = torch.topk(logits, k=top_k)
                probs = F.softmax(top_vals, dim=-1)
                chosen_idx = torch.multinomial(probs, num_samples=1)
                token_id = top_idx[chosen_idx]
            else:
                # Sample from the full distribution
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)

            sampled_token_ids.append(token_id.item())

        # Decode the sequence of sampled tokens into text
        sampled_text = tokenizer.decode(sampled_token_ids, skip_special_tokens=True)
        sampled_texts.append(sampled_text)

    return sampled_texts

