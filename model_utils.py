from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

def get_target_modules(model_id: str):

    if "pythia" in model_id.lower():
        return [
            "dense_4h_to_h",
            "dense_h_to_4h",
            "query_key_value"
        ]
    elif "gpt-neo" in model_id.lower():
        return ["attn.c_attn", "attn.c_proj"]
    elif "mobilellm" in model_id.lower():
        return ["q_proj", "v_proj"]
    else:
        raise ValueError("Unknown model id.")
    
def load_model_and_tokenizer(model_id: str, freeze_embeddings: bool = True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if freeze_embeddings:
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = False

    #lora_config = LoraConfig(
    #    r=64,
    #   lora_alpha=32 * (64 ** 0.5),
    #   target_modules=get_target_modules(model_id),
    #    lora_dropout=0.05,
    #    bias="none",
    #    task_type="CAUSAL_LM"
    #)
    #model = get_peft_model(model, lora_config)
    #model.print_trainable_parameters(),
    
    return model, tokenizer

    

