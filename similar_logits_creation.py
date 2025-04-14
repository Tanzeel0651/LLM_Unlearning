from datasets import load_dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

class QuestionRetriever:
    def __init__(self, corpus_questions):
        """
        corpus_questions: a list of strings, each one a question in your dataset.
        """
        self.corpus_questions = corpus_questions

        # Fit TF–IDF on the entire corpus
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus_questions)

    def find_top_k_relevant(self, query, k=10):
        """
        Given a user query, return the top k most similar questions.
        """
        # Convert the query into the same TF–IDF vector space
        sentences = []
        score = []
        query_vec = self.vectorizer.transform([query])  # shape: [1, vocab_size]

        # Compute cosine similarity between the query vector and each question's vector
        similarities = cosine_similarity(self.doc_vectors, query_vec)  # shape: [num_docs, 1]
        similarities = similarities.flatten()  # shape: [num_docs]

        # Get the indices of the top k most similar questions
        top_k_indices = np.argsort(-similarities)[:k]

        # Return (question, similarity score) pairs
        for i in top_k_indices:
            sentences.append(self.corpus_questions[i])
            score.append(similarities[i])
        return sentences, score

def create_similar_questions():
    full_sentence = load_dataset("locuslab/TOFU", "retain90")["train"]["question"]
    forget_sentence = load_dataset("locuslab/TOFU", "forget10")["train"]["question"]
    
    retriever = QuestionRetriever(full_sentence)
    
    similar_question = {}
    scores = []
    for i, ques in enumerate(forget_sentence):
        top_sen, top_score = retriever.find_top_k_relevant(ques, k=2)
        top_sen = [sen+"[/INST]" for sen in top_sen]
        similar_question[ques+"[/INST]"] = top_sen
        scores.append(sum(top_score)/10)
        if i < 3:  # Print just a few examples to avoid flooding console
            print("Forget Question: ", ques)
            print("Similar: ", top_sen[:2])  # Show just first two similar questions
    
    return similar_question

def get_sequence_logits(model, tokenizer, text, max_length=600, max_new_tokens=200):
    """
    Example usage of the 'best generate' approach to avoid incomplete answers.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=500,  # Truncate extremely long prompts, if any
        return_tensors="pt",
        add_special_tokens=True,
        padding="max_length",
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Use your 'best' generate function
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        #max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Calculate input length to know where question ends and answer begins
    question_length = attention_mask.sum(dim=1)[0].item()
    
    # Get logits for the entire generated sequence
    all_attention_mask = (generated_ids != tokenizer.pad_token_id).long().to(model.device)
    with torch.no_grad():
        outputs = model(generated_ids, attention_mask=all_attention_mask)
        all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    return input_ids.squeeze(0), attention_mask.squeeze(0), generated_ids.squeeze(0), all_logits.squeeze(0), question_length

def create_entity_focused_dampening(similar_question_dict, model, tokenizer, max_length=600, 
                                   alpha=0.5, min_log_ratio=2.0):
    """
    Create dampened logit distributions focusing on entity names and content words.
    
    Args:
        similar_question_dict: Dictionary mapping forget questions to similar retain questions
        model: The language model
        tokenizer: The associated tokenizer
        max_length: Maximum sequence length
        alpha: Interpolation factor (1 = keep original, 0 = full replacement)
        min_log_ratio: Minimum log-ratio for token to be considered important
        
    Returns:
        dict: Mapping of input_ids to attention masks and dampened logits
    """
    result = {}
    
    # Create a list of common stop words and punctuation to ignore
    stop_words = set([
        "the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "when",
        "where", "how", "why", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may",
        "might", "must", "of", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
        "than", "too", "very", "s", "t", "just", "don", "now", "his", "her", "its",
        "by", "at", "he", "she", "it", "we", "they", "their", "our", "my", "your"
    ])
    
    # Extract entity names from questions to ensure they're targeted
    def extract_entity_names(question):
        """Extract potential entity names from a question."""
        # This is a simple approach - in practice, you might want to use NER
        words = question.split()
        potential_entities = []
        
        for word in words:
            # Look for capitalized words that aren't at the start of a sentence
            if word[0].isupper() and word.lower() not in stop_words and len(word) > 1:
                # Clean the word from punctuation
                clean_word = word.strip(".,?!:;\"'()[]{}")
                if clean_word:
                    potential_entities.append(clean_word.lower())
        
        return potential_entities
    
    for i, (forget_q, retain_q_list) in enumerate(similar_question_dict.items()):
        print(f"\nProcessing forget question {i+1}/{len(similar_question_dict)}")
        
        # Skip if no retain examples for this question
        if len(retain_q_list) == 0:
            print(f"Warning: no retain questions found for: {forget_q}")
            continue
        
        # Extract entity names from the forget question to prioritize
        entity_names = extract_entity_names(forget_q)
        print(f"Extracted entities: {entity_names}")
            
        # Get logits for the forget question
        f_input_ids, f_attention, f_gen_ids, f_logits, question_length = get_sequence_logits(
            model, tokenizer, forget_q, max_length=max_length
        )
        
        # Decode the question and generated answer for visualization
        forget_question = tokenizer.decode(f_input_ids[:question_length], skip_special_tokens=True)
        forget_answer = tokenizer.decode(f_gen_ids[question_length:], skip_special_tokens=True)
        print(f"Forget Q: {forget_question}")
        print(f"Generated A: {forget_answer}")
        
        # Get answer portion only (for target dampening)
        answer_logits = f_logits[question_length:]  # [answer_len, vocab_size]
        
        # Collect logits for each retain question
        retain_answer_logits_list = []
        
        for j, rq in enumerate(retain_q_list):
            # Process retain question
            r_input_ids, r_attention, r_gen_ids, r_logits, r_question_length = get_sequence_logits(
                model, tokenizer, rq, max_length=max_length
            )
            
            # Only keep answer portion of logits (after question)
            r_answer_logits = r_logits[r_question_length:]
            retain_answer_logits_list.append(r_answer_logits)
            
            # Print first retain example for visualization
            if j == 0:
                retain_question = tokenizer.decode(r_input_ids[:r_question_length], skip_special_tokens=True)
                retain_answer = tokenizer.decode(r_gen_ids[r_question_length:], skip_special_tokens=True)
                print(f"Similar Retain Q: {retain_question}")
                print(f"Generated A: {retain_answer}")
        
        # Handle sequences of different lengths by selecting min length
        answer_token_counts = [logits.size(0) for logits in retain_answer_logits_list]
        answer_token_counts.append(answer_logits.size(0))
        min_answer_length = min(answer_token_counts)
        
        # Truncate all sequences to minimum length for fair comparison
        truncated_forget_logits = answer_logits[:min_answer_length]
        truncated_retain_logits = [r_logits[:min_answer_length] for r_logits in retain_answer_logits_list]
        
        # Stack retain logits for easier processing
        retain_logits_stacked = torch.stack(truncated_retain_logits, dim=0)  # [num_retain, answer_len, vocab_size]
        
        # Average retain logits across examples
        avg_retain_logits = retain_logits_stacked.mean(dim=0)  # [answer_len, vocab_size]
        
        # Clone the original forget logits to create our dampened version
        dampened_logits = f_logits.clone()
        
        # Create placeholder for modified tokens to track changes
        modified_tokens = []
        
        # Process each position in the answer token by token
        for pos in range(min_answer_length):
            # Position relative to full sequence
            full_pos = question_length + pos
            
            # Get token at this position in the forget sequence
            token_id = f_gen_ids[full_pos].item()
            
            # Skip special tokens
            special_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id}
            if token_id in special_tokens:
                continue
                
            # Get the string representation of this token
            token_str = tokenizer.decode([token_id]).strip()
            
            # Skip very short tokens (likely punctuation or parts of tokenization)
            if len(token_str) <= 1:
                continue
                
            # Skip common stop words
            if token_str.lower() in stop_words:
                continue
            
            # Convert logits to probabilities with softmax
            f_probs = F.softmax(truncated_forget_logits[pos], dim=-1)
            r_probs = F.softmax(avg_retain_logits[pos], dim=-1)
            
            # Calculate log probabilities for more stable comparisons
            f_log_probs = torch.log(f_probs + 1e-10)  # add small epsilon to avoid log(0)
            r_log_probs = torch.log(r_probs + 1e-10)
            
            # Calculate log-ratio: log(p_f/p_r) = log(p_f) - log(p_r)
            log_ratio = f_log_probs - r_log_probs  # [vocab_size]
            current_log_ratio = log_ratio[token_id].item()
            
            # Determine if this token should be modified based on multiple criteria
            should_modify = False
            
            # 1. Check if token is part of an entity name from the question
            is_entity = False
            for entity in entity_names:
                if entity in token_str.lower() or token_str.lower() in entity:
                    is_entity = True
                    break
            
            # 2. Check if token has a high log ratio (much more likely in forget than retain)
            has_high_log_ratio = current_log_ratio > min_log_ratio
            
            # 3. Special handling for proper names (capitalized words)
            is_proper_name = token_str[0].isupper() and len(token_str) > 1
            
            # Determine whether to modify this token
            if is_entity:
                # Always modify entity names from the question
                should_modify = True
                modification_strength = 0.2  # Stronger modification (0.2 = 80% from retain)
            elif is_proper_name:
                # Likely a name or other proper noun
                should_modify = True
                modification_strength = 0.3  # Strong modification
            elif has_high_log_ratio:
                # Content word with high log ratio
                should_modify = True
                modification_strength = alpha  # Use default alpha
            
            # Apply the modification if needed
            if should_modify:
                # Record the original logit value
                orig_logit = dampened_logits[full_pos, token_id].item()
                
                # Apply interpolation between forget and retain logits
                # new_logit = modification_strength * forget_logit + (1-modification_strength) * retain_logit
                dampened_logits[full_pos, token_id] = (modification_strength * f_logits[full_pos, token_id] + 
                                                    (1-modification_strength) * avg_retain_logits[pos, token_id])
                
                new_logit = dampened_logits[full_pos, token_id].item()
                modification_type = "entity" if is_entity else "proper" if is_proper_name else "high-ratio"
                modified_tokens.append((token_str, pos, orig_logit, new_logit, 
                                      current_log_ratio, modification_type))
        
        # Report modifications
        if modified_tokens:
            print("\nTokens modified with entity-focused method:")
            for token, pos, old_val, new_val, ratio, mod_type in modified_tokens:
                print(f"  Position {pos}: Token '{token}' logit {old_val:.4f} → {new_val:.4f} (log-ratio: {ratio:.4f}, type: {mod_type})")
        else:
            print("\nNo tokens were modified")
        
        # Store the result for this question
        input_ids_tuple = tuple(f_input_ids.cpu().numpy().tolist())
        result[input_ids_tuple] = {
            "attention_mask": f_attention.cpu(),
            "dampened_logits": dampened_logits.cpu(),
            "modified_tokens": modified_tokens
        }
    
    return result

def main():
    # 1) Get similar questions for each forget question
    similar_question_dict = create_similar_questions()
    
    # 2) Load model and tokenizer
    model_id = "model/target_model/full_model_scratch"  # Replace with your model path
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 3) Create entity-focused dampened distributions
    data = create_entity_focused_dampening(
        similar_question_dict, 
        model, 
        tokenizer,
        max_length=512,
        alpha=0.5,         # Default interpolation factor (0.5 = 50% forget, 50% retain)
        min_log_ratio=1.0  # Minimum log-ratio for high-ratio tokens
    )
    
    # 4) Save the dampened distributions
    torch.save(data, "forget_dampened_data_entity_focused.pt")
    print(f"\nSaved entity-focused dampened distributions for {len(data)} forget questions.")
    
    # Print summary statistics
    total_tokens_modified = sum(len(item["modified_tokens"]) for item in data.values())
    total_questions = len(data)
    print(f"Total modified tokens: {total_tokens_modified}")
    print(f"Average modified tokens per question: {total_tokens_modified/total_questions:.2f}")

if __name__ == "__main__":
    main()