import json

# 假设文件路径
document_file_path = "../dataset/LCSTS/test/test.src.txt"
summary_file_path = "../dataset/LCSTS/test/test.tgt.txt"

# 读取两个文件的内容
with open(document_file_path, 'r', encoding='utf-8') as doc_file, \
     open(summary_file_path, 'r', encoding='utf-8') as sum_file:
    documents = doc_file.readlines()
    summaries = sum_file.readlines()

# 确保两个文件的行数相等
if len(documents) != len(summaries):
    raise ValueError("文件行数不匹配")

# 组装JSON对象
result_list = [{"document": document.strip(), "summary": summary.strip()}
               for document, summary in zip(documents, summaries)]

# 将结果转换为JSON格式的字符串
json_result = json.dumps(result_list, ensure_ascii=False, indent=4)

# 打印或保存结果
print(json_result)

# 如果需要保存结果到文件
with open('result.json', 'w', encoding='utf-8') as result_file:
    result_file.write(json_result)