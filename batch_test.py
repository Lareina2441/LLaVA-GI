import json
import subprocess
import os
import csv

# 读取JSON文件
with open('/home/louey/LLaVA/data/validation/dataset.json', 'r') as f:
    data = json.load(f)

# 初始化变量
results = []

# 遍历每个示例
for item in data:
    image_path = os.path.join("/home/louey/LLaVA/data/images", item['image'])
    question = item["conversations"][0]["value"]
    expected_output = item["conversations"][1]["value"]
    # 调用模型
    command = [
        "CUDA_VISIBLE_DEVICES=0", "python", "-m", "llava.serve.cli",
        "--model-path", "/home/louey/LLaVA/fted/llava-ftmodel_1",
        "--image-file", image_path,
        "--load-8bit"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    model_output = result.stdout.strip()
    # 收集结果
    results.append({
        "image_id": item['id'],
        "question": question,
        "expected_output": expected_output,
        "model_output": model_output
    })

with open('/home/louey/LLaVA/results.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'question', 'expected_output', 'model_output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in results:
        writer.writerow(result)




print("done")
