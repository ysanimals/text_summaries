from rouge import Rouge

generated_summary = """
Woman sexually assaulted, stabbed in park outside Gatlinburg .
$5,000 reward offered; suspect described as white male in his 40s .
Agents, rangers pore over leads in Friday's incident .
"""
reference_summary = """
Authorities search for man who sexually assaulted and stabbed hiker in Tennessee park.
National Park Service offers $5,000 reward for information.
Suspect was described as white male in his 40s, thin build, with tattoos.
"""

rouge = Rouge()

TOKENIZE_CHINESE = lambda x: ' '.join(x)

# from transformers import AutoTokenizer
# model_checkpoint = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# TOKENIZE_CHINESE = lambda x: ' '.join(
#     tokenizer.convert_ids_to_tokens(tokenizer(x).input_ids, skip_special_tokens=True)
# )

scores = rouge.get_scores(
    hyps=[generated_summary], 
    refs=[(reference_summary)]
)[0]
print('ROUGE:', scores)