import torch
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from datasets import Dataset, load_dataset
from datasets import load_dataset
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore")
dataset = load_dataset("/mnt/bn/minghui/py-project/graduate_design/datasets/cnn_daily_mail")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
model_path = "/mnt/bn/minghui/models/llama2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_instruction(dialogue: str, summary: str):
    return f"""### Instruction:
Summarize the following conversation.

### Input:
{dialogue.strip()}

### Summary:
{summary}
""".strip()

def generate_instruction_dataset(data_point):

    return {
        "article": data_point["article"],
        "highlights": data_point["highlights"],
        "text": format_instruction(data_point["article"],data_point["highlights"])
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_instruction_dataset).remove_columns(['id'])
    )

## APPLYING PREPROCESSING ON WHOLE DATASET
dataset["train"] = process_dataset(dataset["train"])
dataset["test"] = process_dataset(dataset["validation"])
dataset["validation"] = process_dataset(dataset["validation"])

# Select 1000 rows from the training split
train_data = dataset['train'].shuffle(seed=42).select([i for i in range(1000)])

# Select 100 rows from the test and validation splits
test_data = dataset['test'].shuffle(seed=42).select([i for i in range(100)])
validation_data = dataset['validation'].shuffle(seed=42).select([i for i in range(100)])

def test_inference():
    index = 2

    dialogue = test_data['article'][index]
    summary = test_data['highlights'][index]

    prompt = f"""
    Summarize the following conversation.

    ### Input:
    {dialogue}

    ### Summary:
    """

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
        )[0],
        skip_special_tokens=True
    )

    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

model.gradient_checkpointing_enable()
lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], #specific to Llama models.
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
OUTPUT_DIR = "llama2-docsum-adapter"
training_arguments = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_hf",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=4,
    evaluation_strategy="steps",
    eval_steps=0.2,
    warmup_ratio=0.05,
    save_strategy="epoch",
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=validation_data,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()
peft_model_path="./peft-dialogue-summary"
# {'train_runtime': 2922.7768, 'train_samples_per_second': 1.369, 'train_steps_per_second': 0.085, 'train_loss': 1.6268718430111486, 'epoch': 3.97}
trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)