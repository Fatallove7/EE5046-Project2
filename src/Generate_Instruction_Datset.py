import os
import json
import numpy as np
import pandas as pd

ECG_TOKEN = "<|extra_0|>" 
INSTRUCTION_TEMPLATE = f"请仔细观察提供的{ECG_TOKEN}信号。你认为患者是否患有心房颤动（AF）？请直接回答‘是’或‘否’。"

ANSWER_AF = "有房颤。" # 对应标签 'A'
ANSWER_NORMAL = "无房颤。" # 对应标签 'N', 'O', '~'

def label_to_response(label_char):
    """将字符标签映射为指令回答文本"""
    # 'A' 被视为 1 (AF)，其他视为 0 (Non-AF)
    return ANSWER_AF if label_char == 'A' else ANSWER_NORMAL

def generate_instruction_metadata(base_dir, cv_indices, output_json_file):
    """
    根据 K-Fold CSV 文件生成多模态指令数据集元数据。
    Args:
        base_dir (str): '/Dataset' 的父目录路径。
        cv_indices (list): 想要包含的 K-Fold 折数列表 (例如 [0, 1, 2, 3, 4])。
    """
    instruction_data = []
    cv_path = os.path.join(base_dir, 'cv')

    for i in cv_indices:
        csv_file = os.path.join(cv_path, f'cv{i}.csv')
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found. Skipping fold {i}.")
            continue

        data = pd.read_csv(csv_file)
        for row in data.values:
            file_name = row[1] # 例如 'A0001'
            label_char = row[2] # 例如 'A' 或 'N'

            response = label_to_response(label_char)

            # 💥 修正 2：INSTRUCTION_TEMPLATE 现在已经包含了正确的 ECG_TOKEN
            # 移除多余的替换操作
            instruction_with_token = INSTRUCTION_TEMPLATE 

            # 构造完整的输入序列文本 (格式与 MultimodalDataset 中硬编码的分隔符一致)
            full_text = f"指令: {instruction_with_token}\n答案: {response}"

            entry = {
                "file_name": file_name,
                "instruction": instruction_with_token,
                "response": response,
                "full_text": full_text,# 完整文本用于 Tokenizer
                "label_char": label_char
            }
            instruction_data.append(entry)

    # 写入 JSON 文件
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(instruction_data, f, ensure_ascii=False, indent=4)

    print(f"指令数据集元数据已保存到 {output_json_file}")
    print(f"总样本数: {len(instruction_data)}")


# 使用:
if __name__ == '__main__':
    # 确保 BASE_PATH 是 /Dataset 的父目录
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Dataset'))
     
    # JSON 文件将保存在 /Dataset/MMID/
    OUTPUT_JSON_FILE = os.path.join(BASE_PATH, 'MMID/multimodal_instruction_data.json')
    output_dir = os.path.dirname(OUTPUT_JSON_FILE)
    os.makedirs(output_dir, exist_ok=True)

    # 包含所有 5 折的数据
    generate_instruction_metadata(BASE_PATH, cv_indices=[0, 1, 2, 3, 4], output_json_file=OUTPUT_JSON_FILE)