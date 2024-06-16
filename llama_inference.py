import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from datasets import Dataset, load_dataset
warnings.filterwarnings("ignore")
rouge_metric = evaluate.load("rouge")
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"] 
dataset = load_dataset("/mnt/bn/minghui/py-project/graduate_design/datasets/cnn_daily_mail")
def format_instruction(dialogue: str, summary: str):
    return f"""### Instruction:
Summarize the following conversation.

### Input:
{dialogue.strip()}

### Summary:
{summary}
""".strip()

def chunks(list_of_elements, batch_size): 
	"""Yield successive batch-sized chunks from list_of_elements.""" 
	for i in range(0, len(list_of_elements), batch_size): 
	    yield list_of_elements[i : i + batch_size] 
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


# 参数
model_name_or_path = "/mnt/bn/minghui/py-project/graduate_design/text_summarization/peft-dialogue-summary-lora"  # PEFT训练后模型的路径
base_model_name_or_path = "/mnt/bn/minghui/models/llama2-7b-hf"  # 基础预训练模型的名称或路径

# 加载基础预训练模型及其对应的tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# 加载PEFT模型
peft_model = PeftModel.from_pretrained(model, model_name_or_path)

# 将PEFT模型设置为评估模式
peft_model.config.use_cache = True
peft_model.eval()

## APPLYING PREPROCESSING ON WHOLE DATASET
dataset["train"] = process_dataset(dataset["train"])
dataset["test"] = process_dataset(dataset["validation"])
dataset["validation"] = process_dataset(dataset["validation"])
# Select 1000 rows from the training split
train_data = dataset['train'].shuffle(seed=42).select([i for i in range(1000)])

# Select 100 rows from the test and validation splits
test_data = dataset['test'].shuffle(seed=42).select([i for i in range(100)])
validation_data = dataset['validation'].shuffle(seed=42).select([i for i in range(100)])

index = 51
rouge_metric = evaluate.load("rouge")
for index in range(0, 1):
    # dialogue = test_data['article'][index][:10000]
    # summary = test_data['highlights'][index]
    index = 52

    dialogue = train_data['article'][index][:10000]
    summary = train_data['highlights'][index]

    prompt = f"""
Summarize the following conversation.

### Input:
{dialogue}

### Summary:
"""
    # target_batches = list(summary) 
    # input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
    # outputs = peft_model.generate(input_ids=input_ids, max_new_tokens=1024)
    # output= tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    
    
    input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
    outputs = peft_model.generate(input_ids=input_ids, max_new_tokens=200)
    output= tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]


    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'TRAINED MODEL GENERATED TEXT :\n{output}')
#     decoded_summaries = [d.replace("<n>", " ") for d in output]
#     rouge_metric.add_batch(predictions=decoded_summaries, references=target_batches)
# score = rouge_metric.compute()
# rouge_dict = dict((rn, score[rn]) for rn in rouge_names) 
# print(rouge_dict)

    # dash_line = '-'.join('' for x in range(100))
# print(dash_line)
# print(f'INPUT PROMPT:\n{prompt}')
# print(dash_line)
# print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
# print(dash_line)
# print(f'TRAINED MODEL GENERATED TEXT :\n{output}')