import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import pandas as pd
import evaluate
from tqdm import tqdm
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM

rouge_metric = evaluate.load("rouge")
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"] 
dataset = load_dataset("/mnt/bn/minghui/py-project/graduate_design/datasets/cnn_daily_mail")
print(dataset)
sample_text = dataset["train"][1]["article"][:2000]
# We'll collect the generated summaries of each model in a dictionary 
# summaries = {}
# # baseline，直接提取前三句作为摘要的部分
def three_sentence_summary(text): 
	return "\n".join(sent_tokenize(text)[:3]) 
def evaluate_summaries_baseline(dataset, metric, column_text="article", column_summary="highlights"): 
    summaries = [three_sentence_summary(text) for text in dataset[column_text]] 
    metric.add_batch(predictions=summaries, references=dataset[column_summary]) 
    score = metric.compute() 
    return score

test_sampled = dataset["test"].shuffle(seed=42).select(range(1000)) 
# score = evaluate_summaries_baseline(test_sampled, rouge_metric) 
# rouge_dict = dict((rn, score[rn]) for rn in rouge_names) 
# rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"] 
# baseline{'rouge1': 0.3892491991935129, 'rouge2': 0.17144711241439836, 'rougeL': 0.2451290549077601, 'rougeLsum': 0.3542686419759322}
# pd.DataFrame.from_dict(rouge_dict, orient="index", columns=["baseline"]).T
# print(rouge_dict)
device = "cuda" if torch.cuda.is_available() else "cpu" 
def chunks(list_of_elements, batch_size): 
	"""Yield successive batch-sized chunks from list_of_elements.""" 
	for i in range(0, len(list_of_elements), batch_size): 
	    yield list_of_elements[i : i + batch_size] 
	
text_promt = """
## Role: 文本摘要高手
- 描述角色特质:专注于从大量文本中快速提取关键信息，生成简洁、准确、保留原文核心信息的摘要。
- 其他背景信息:无特定背景要求，适用于处理各种类型的文本摘要需求。

## Preferences:
- 倾向于生成简洁、清晰的摘要，避免冗余信息。
- 保持摘要的客观性和中立性。

## Profile:
- language: 中文
- description: 专门执行文本摘要任务，从原始文本中提取关键信息，生成简短、清晰的摘要。

## Goals:
- 从用户提供的文本中准确提取关键信息。
- 生成简洁、清晰且保留原文主旨的摘要。

## Constrains:
- 不添加个人观点或解释。
- 不包含超出原文内容的信息。

## Skills:
- 快速阅读和理解大量文本。
- 精准提取关键信息和核心观点。
- 简洁、准确的表达能力。

## Examples:
- 输入示例:一篇关于全球气候变化的新闻报道。
- 输出示例:新闻报道概述了全球气候变化的最新研究，指出温室气体排放是主要原因，呼吁国际社会采取行动。

## Workflow:
- 首先快速阅读全文，理解其主要内容。
- 然后识别并提取关键信息和核心观点。
- 最后以简洁、准确的语言组织摘要。

## OutputFormat:
- 以简洁、清晰的句子形式呈现摘要。

## Output STEP:
- 第一步:理解全文
     1）快速阅读全文，把握文章的主题、论点和结构。
     2）识别文章的重要信息和次要信息。
     3）注意文章的语气和目的。
- 第二步:提取关键信息
     1）确定文章的主要论点和证据。  
     2）提取文章的关键信息和数据。
     3）识别并记录文章中的特殊术语或概念。
- 第三步:组织摘要内容
     1）选择合适的句子结构，以简洁、直接的方式表达。
     2）确保摘要的连贯性和逻辑性。
     3）使用清晰、标准的语言，避免复杂或模糊的表达。
- 第四步:审阅和修改
     1）检查摘要的准确性和流畅性。
     2）确保摘要没有遗漏重要的信息。
     3）调整语言和句子结构，以提高摘要的质量。
- 第五步:最终检查
     1）确认摘要的长度适中，通常不超过原文的10%。
     2）确保摘要中没有主观评价或情感色彩。
     3）最后检查语法和拼写错误。
- 确保摘要准确反映原文的主旨和重点。

## Output Standard
- 内容要求
     - 精确性:摘要应准确反映原文的主要信息和核心观点，不添加个人解释或推测。
     - 简洁性:摘要应尽可能简洁，去除冗余信息，同时保留原文的必要细节。
     - 客观性:摘要应保持客观和中立，不包含主观评价或情感色彩。
     - 逻辑性:摘要的结构应清晰，信息组织合理，易于理解。
- 格式要求
     - 文本长度:摘要的长度应适中，通常不超过原文的10%。
     - 句子结构:使用简单、直接的句子结构，避免复杂的从句或长句。
     - 语言风格:使用清晰、标准的语言，避免使用专业术语或难懂的词汇，除非它们对于理解摘要至关重要。
- 输出步骤
     - 理解全文:快速阅读全文，把握文章的主题、论点和结构。
     - 提取关键信息:识别并提取文章的关键信息和核心观点。
     - 组织摘要内容:以简洁、准确的语言重新组织关键信息，形成摘要。
     - 审阅和修改:检查摘要的准确性和流畅性，进行必要的修改以确保质量。
- 示例
    - 输入文本:一篇关于最新人工智能研究的学术论文。
    - 输出摘要:学术论文探讨了人工智能在图像识别领域的最新进展，特别是深度学习技术的应用。研究指出，通过使用大规模数据集和更复杂的神经网络结构，目前的图像识别准确率显著提高，但在处理某些复杂场景时仍存在挑战。

## Initialization: 
作为文本摘要高手，我专注于从大量文本中快速提取关键信息，生成简洁、准确、保留原文核心信息的摘要。请按照格式【需要进行文本摘要的文本:###（这里填写需要进行文本摘要的文本。）##】提供需要进行摘要的文本。
"""
def evaluate_summaries_pegasus(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"): 
	article_batches = list(chunks(dataset[column_text], batch_size)) 
	target_batches = list(chunks(dataset[column_summary], batch_size)) 
	for article_batch, target_batch in tqdm( zip(article_batches, target_batches), total=len(article_batches)): 
		inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
		
		summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), length_penalty=0.8, num_beams=8, max_length=128) 
		decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries] 
		decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries] 
		metric.add_batch(predictions=decoded_summaries, references=target_batch) 
	score = metric.compute() 
	return score

def evaluate_summaries_llama(dataset, metric, model, tokenizer, batch_size=1, device=device, column_text="article", column_summary="highlights"):
     article_batches = list(chunks(dataset[column_text], batch_size))
     target_batches = list(chunks(dataset[column_summary], batch_size))
     tokenizer.pad_token = tokenizer.eos_token
     for article_batch, target_batch in tqdm( zip(article_batches, target_batches), total=len(article_batches)): 
          inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
          summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), length_penalty=0.8, num_beams=8, max_length=2048) 
          decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries] 
          decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries] 
          metric.add_batch(predictions=decoded_summaries, references=target_batch) 
     score = metric.compute() 
     return score

def evaluate_summaries_gpt2(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"): 
    article_batches = list(chunks(dataset[column_text], batch_size)) 
    target_batches = list(chunks(dataset[column_summary], batch_size)) 
    tokenizer.pad_token = tokenizer.eos_token
    for article_batch, target_batch in tqdm( zip(article_batches, target_batches), total=len(article_batches)):
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 确保模型配置参数和tokenizer一致
        model.config.max_length = 512
        tokenizer.model_max_length = 512
        # 将输入文本进行tokenize
        inputs = tokenizer(article_batch, max_length=model.config.max_length, truncation=True, padding="max_length",padding_side='left', return_tensors="pt").to(device)
        # 生成文本，确保使用 max_length 和 max_new_tokens 参数
        summaries = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
        # Generate summaries with GPT-2
        #  summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_length=128) 
        # Decode the generated summaries
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries] 
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries] 
        metric.add_batch(predictions=decoded_summaries, references=target_batch) 
    score = metric.compute() 
    return score




# model_ckpt = "/mnt/bn/minghui/models/pegasus/" 
model_ckpt = "/mnt/bn/minghui/py-project/graduate_design/text_summarization/train-model/llama2-7b-hf-cnn-daily-mail-lora"
device = 'cuda'
# 加载tokenizer和模型
model_type = "llama2"
if model_type == "gpt2":
     tokenizer = GPT2Tokenizer.from_pretrained(model_ckpt)
     model = GPT2LMHeadModel.from_pretrained(model_ckpt).to(device)
     score = evaluate_summaries_gpt2(test_sampled, rouge_metric, model, tokenizer, batch_size=8) 

elif model_type == "llama2":
	# 加载基础预训练模型及其对应的tokenizer
     model = AutoModelForCausalLM.from_pretrained(model_ckpt, device_map="auto")
     tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
     score = evaluate_summaries_llama(test_sampled, rouge_metric, model, tokenizer, batch_size=8) 

rouge_dict = dict((rn, score[rn]) for rn in rouge_names) 
print(rouge_dict)
# baseline的结果
# baseline{'rouge1': 0.3892491991935129, 'rouge2': 0.17144711241439836, 'rougeL': 0.2451290549077601, 'rougeLsum': 0.3542686419759322}
# Results of peagsus
# {'rouge1': 0.4346513644871044, 'rouge2': 0.21660620190711355, 'rougeL': 0.3124570029266593, 'rougeLsum': 0.3743885415137642}
# Bart-cnn的结果
# {'rouge1': 0.4226230554274447, 'rouge2': 0.2034613145893504, 'rougeL': 0.3009882051888342, 'rougeLsum': 0.36351185690351684}
# Bart的结果
# {'rouge1': 0.3641032541305088, 'rouge2': 0.15756653094534226, 'rougeL': 0.22737721992842427, 'rougeLsum': 0.2957963167136743}
# GPT2的结果
# {'rouge1': 0.1834744877147479, 'rouge2': 0.09893821236127415, 'rougeL': 0.12998709403876763, 'rougeLsum': 0.15835348451431946}
pd.DataFrame(rouge_dict, index=["pegasus"])