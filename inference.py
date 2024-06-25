from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", device_map="auto", torch_dtype="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B", trust_remotecode = True)
inputs = tokenizer("有一个地方时间停滞不前。一个令人叹为观止的地方，但也", return_tensors="pt")
outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))