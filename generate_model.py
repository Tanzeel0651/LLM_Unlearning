from transformers import AutoModelForCausalLM
import torch

def get_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    
    model = AutoModelForCausalLM.from_pretrained(
                model_id,
                #use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                torch_dtype=torch.bfloat16,
                trust_remote_code = True,
                force_download = True)
    print("Model: ",model_id)
    
    print(f"GENERATE MODEL: Embed tokens weight shape (on load): {model.model.embed_tokens.weight.shape}")
    return model
