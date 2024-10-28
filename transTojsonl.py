import json

# 读取JSON文件
with open(r'D:\Github\LLaVA\data\validation\dataset.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 写入问题文件（JSONL格式）
with open(r'D:\Github\LLaVA\data\validation\question.jsonl', 'w', encoding='utf-8') as question_file:
    for item in data:
        for conversation in item['conversations']:
            if conversation['from'] == 'human':
                question = {
                    "question_id": item['id'],
                    "image": item['image'],
                    "text": conversation['value']
                }
                question_file.write(json.dumps(question) + '\n')