from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
 
 
def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
 
    print(f"Loading the LoRA adapter from {lora_path}")
 
    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )
 
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)

apply_lora("/mnt/bn/minghui/models/llama2-7b-hf", "/mnt/bn/minghui/py-project/graduate_design/text_summarization/train-model/llama2-7b-hf-cnn-daily-mail-lora", "/mnt/bn/minghui/py-project/graduate_design/text_summarization/peft-dialogue-summary-lora")