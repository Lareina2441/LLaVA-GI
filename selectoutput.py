import json

# 文件路径
dataset_path = r'D:\Github\LLaVA-GI\data\validation\dataset.json'
answers_path = r'D:\Github\LLaVA-GI\data\validation\answer4med.jsonl'
output_path = r'D:\Github\LLaVA-GI\data\validation\output_results4-llavamed.json'

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None

def read_jsonl_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    except json.JSONDecodeError as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        return None

# 读取数据集文件
dataset = read_json_file(dataset_path)
if dataset is None:
    raise ValueError(f"Failed to read dataset file: {dataset_path}")

# 读取答案文件
answers = read_jsonl_file(answers_path)
if answers is None:
    raise ValueError(f"Failed to read answers file: {answers_path}")

# 创建一个映射，方便查找 machine_answer
answer_dict = {}
for entry in answers:
    key = (entry["question_id"], entry["prompt"])
    answer_dict[key] = entry["text"]

# 生成输出数据
output_data = []
for item in dataset:
    image_id = item["id"]
    image = item["image"]
    
    for conversation in item["conversations"]:
        if conversation["from"] == "human":
            question = conversation["value"]
            human_answer = next(
                (conv["value"] for conv in item["conversations"] if conv["from"] == "gpt"), ""
            )
            machine_answer = answer_dict.get((image_id, question), "")
            
            output_data.append({
                "image_id": image_id,
                "image": image,
                "question": question,
                "human_answer": human_answer,
                "machine_answer": machine_answer
            })

# 保存输出结果到新文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("匹配完成，结果已保存到", output_path)