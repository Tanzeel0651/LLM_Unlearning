import torch
from torch.utils.data import Dataset
import datasets
from typing import List, Dict, Any
from utils import get_model_identifiers_from_yaml, add_dataset_index


class TextDatasetQA(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        model_family: str, 
        max_length: int = 512, 
        split: str = None, 
        question_key: str = 'question', 
        answer_key: str = 'answer', 
        total_length: int = None, 
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.split = split
        
        # Limit dataset size if specified
        if total_length:
            self.data = self.data.shuffle(seed=42).select(range(total_length))
        
        # Add index to dataset
        self.data = add_dataset_index(self.data)
        
        # Get model-specific configurations
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']

        # Ensure answers is a list
        if isinstance(answers, str):
            answers = [answers]

        # Prepare containers for different input formats
        pad_input_ids_list = []
        pad_attention_mask_list = []
        label_list = []

        for answer in answers:
            # Convert raw data to model-specific format
            (
                full_input_ids, 
                full_attention_mask, 
                labels
            ) = self._convert_raw_data_to_model_format(question, answer)

            pad_input_ids_list.append(full_input_ids)
            pad_attention_mask_list.append(full_attention_mask)
            label_list.append(labels)

        # Determine if this is a forget set sample
        is_forget = "forget" in str(self.split)

        # Return format based on return_question flag
        return {
            "input_ids": torch.stack(pad_input_ids_list).squeeze(),
            "attention_mask": torch.stack(pad_attention_mask_list).squeeze(),
            "labels": torch.stack(label_list).squeeze(),
            "idx": torch.tensor(indices),
            "is_forget": is_forget
        }

    def _convert_raw_data_to_model_format(self, question: str, answer: str):
        """
        Convert raw QA data to tokenized format
        
        Args:
            question (str): Input question
            answer (str): Input answer
        
        Returns:
            Tuple of tokenized inputs for different model configurations
        """
        # Extract model-specific tokens
        question_start_token = self.model_configs.get('question_start_tag', '')
        question_end_token = self.model_configs.get('question_end_tag', '')
        answer_token = self.model_configs.get('answer_tag', '')

        # Construct full text
        new_question = f"{question_start_token}{question}{question_end_token}"
        new_answer = f"{answer_token}{answer}"
        full_text = new_question + new_answer


        # Tokenize full QA sequence
        encoded_full = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )

        
        # Pad full sequence inputs
        pad_full_length = self.max_length - len(encoded_full.input_ids)
        full_input_ids = encoded_full.input_ids + [self.tokenizer.eos_token_id] * pad_full_length
        full_attention_mask = encoded_full.attention_mask + [0] * pad_full_length

        # Create labels, masking out question tokens
        num_question_tokens = len(self.tokenizer.tokenize(new_question, add_special_tokens=True))
        labels = full_input_ids.copy()
        
        # Mask out question tokens in labels
        for i in range(num_question_tokens):
            labels[i] = -100

        return (
            torch.tensor(full_input_ids),
            torch.tensor(full_attention_mask),
            torch.tensor(labels)
        )


class TextForgetDatasetQA(Dataset):
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        model_family: str, 
        max_length: int = 512, 
        split: str = None, 
        question_key: str = 'question', 
        answer_key: str = 'answer', 
        total_length: int = None, 
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.split = split
        
        # Limit dataset size if specified
        if total_length:
            self.data = self.data.shuffle(seed=42).select(range(total_length))
            
        # Add index to dataset
        self.data = add_dataset_index(self.data)
        
        # Get model-specific configurations
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']

        # Ensure answers is a list
        if isinstance(answers, str):
            answers = [answers]

        # Prepare containers for different input formats
        question_only_list = []
        question_only_attention = []

        for answer in answers:
            # Convert raw data to model-specific format
            (
                question_only, 
                question_attention
            ) = self._convert_raw_data_to_model_format(question, answer)

            question_only_list.append(question_only)
            question_only_attention.append(question_attention)

        # Determine if this is a forget set sample
        is_forget = "forget" in str(self.split)
        labels = torch.full(
            size = (self.max_length, len(question_only_list)),
            fill_value = -100,
            dtype = torch.long)

        # Return format based on return_question flag
        return {
            "input_ids": torch.stack(question_only_list).squeeze(),
            "attention_mask": torch.stack(question_only_attention).squeeze(),
            "labels": labels,
            "is_forget": is_forget
        }

    def _convert_raw_data_to_model_format(self, question: str, answer: str):
        # Extract model-specific tokens
        question_start_token = self.model_configs.get('question_start_tag', '')
        question_end_token = self.model_configs.get('question_end_tag', '')
        answer_token = self.model_configs.get('answer_tag', '')

        # Construct full text
        new_question = f"{question_start_token}{question}{question_end_token}"

        # Tokenize question only (for MLP input)
        encoded_question = self.tokenizer(
            new_question,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True
        )
        

        # Pad question inputs
        pad_question_length = self.max_length - len(encoded_question.input_ids)
        question_input_ids = encoded_question.input_ids + [self.tokenizer.eos_token_id] * pad_question_length
        question_attention_mask = encoded_question.attention_mask + [0] * pad_question_length


        return (
            torch.tensor(question_input_ids),
            torch.tensor(question_attention_mask)
        )

# Utility function for data collation
def mixed_data_collator(features):
    """
    Collate function for mixed training
    
    Args:
        features (List[Dict]): List of processed samples
    
    Returns:
        Dict of batched tensors
    """
    # Stack input_ids and attention_mask
    input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
    attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
    
    # Optional: add labels if needed
    labels = torch.stack([f["labels"] for f in features], dim=0)
    
    # Convert is_forget to tensor
    is_forget = torch.tensor([f["is_forget"] for f in features], dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "is_forget": is_forget
    }