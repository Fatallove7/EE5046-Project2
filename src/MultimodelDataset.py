import torch
from torch.utils.data import Dataset
import json
import os
import scipy.io as io
import numpy as np
from scipy.interpolate import interp1d
from transformers import AutoTokenizer
# 假设 Config.py 中的常量可以被导入
from Config import FIXED_LENGTH, DOWNSAMPLE_RATE, AUGMENT_SETTING


# ------------------------------------------------------------------
# 关键辅助函数 (基于 ECG_dataset.py 逻辑)
# ------------------------------------------------------------------

def _get_ecg_data_raw(base_file, file_name):
    """根据文件名从 training2017 目录加载原始 .mat 信号"""
    # 路径: base_file ('../Dataset') + '/training2017/' + file_name + '.mat'
    mat_file = os.path.join(base_file, 'training2017', file_name + '.mat')
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"ECG .mat file not found: {mat_file}")
    # io.loadmat(mat_file)['val'] 返回的是一个数组，我们取第一个元素作为信号数据
    data_raw = io.loadmat(mat_file)['val']
    return data_raw[0]  # 返回一维的信号数组


# 复制 ECG_dataset 中的数据处理逻辑，作为独立的函数
def _add_noise(data):
    if np.random.rand() < 0.5:
        noise_level = 0.05
        noise = np.random.normal(0, noise_level, data.shape)
        data = data + noise
    return data


def _time_scaling(data):
    if np.random.rand() < 0.5:
        scale_factor = np.random.uniform(0.8, 1.2)
        old_len = data.shape[0]
        new_len = int(old_len * scale_factor)

        x_old = np.linspace(0, 1, old_len)
        x_new = np.linspace(0, 1, new_len)
        f = interp1d(x_old, data, kind='linear')
        data = f(x_new)
    return data


def _crop_padding(data, length, is_train, apply_augment):
    if data.shape[0] <= length:
        pad_len = length - data.shape[0]
        data = np.pad(data, (0, pad_len), 'constant')
    elif data.shape[0] > length:
        if is_train and apply_augment:
            # 训练时随机裁剪
            start = np.random.randint(0, data.shape[0] - length)
        else:
            # 测试或不增强时中心裁剪
            start = (data.shape[0] - length) // 2
        data = data[start:start + length]
    return data


def _data_process_full(data_raw, is_train, apply_augment):
    """整合所有数据预处理和增强步骤"""
    data = data_raw.copy()

    # 1. 降采样
    data = data[::DOWNSAMPLE_RATE]

    # 2. 时间缩放 (如果开启)
    if is_train and apply_augment:
        data = _time_scaling(data)

    # 3. 归一化
    data = data - data.mean()
    std = data.std()
    data = data / std

    # 4. 添加噪声 (如果开启)
    if is_train and apply_augment:
        data = _add_noise(data)

    # 5. 裁剪/填充到固定长度
    data = _crop_padding(data, FIXED_LENGTH, is_train, apply_augment)

    # 转换为 Tensor，并添加通道维度 [1, FIXED_LENGTH]
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)


# ------------------------------------------------------------------
# MultimodalDataset 类
# ------------------------------------------------------------------

class MultimodalDataset(Dataset):
    
    # ... (__init__ 方法参数保持不变) ...

    def __init__(self, json_path, data_dir, tokenizer, ecg_token="<ECG>", max_len=512, is_train=True,
                 augment=AUGMENT_SETTING):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.ecg_token = ecg_token
        self.max_len = max_len
        self.is_train = is_train
        self.augment = augment

        # 💥 修正 1：移除 add_tokens 块
        # 对于 Qwen，我们必须使用其内置的 Token (如 <|extra_0|>)，不能手动添加词汇。
        
        self.ecg_token_id = self.tokenizer.convert_tokens_to_ids(ecg_token)

        if self.ecg_token_id == self.tokenizer.unk_token_id:
            print(f"警告: ECG token '{ecg_token}' 不存在于词表中！")
            # 尝试使用 <|im_start|> 作为备用
            backup_token = "<|im_start|>"
            self.ecg_token = backup_token
            self.ecg_token_id = self.tokenizer.convert_tokens_to_ids(backup_token)
            print(f"使用备用 token: {backup_token} (ID: {self.ecg_token_id})")
        
        print(f"ECG token: '{self.ecg_token}' (ID: {self.ecg_token_id})")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        print(f"MultimodalDataset 初始化完成，共 {len(self.metadata)} 个样本。")

    def _find_answer_start(self, input_ids, tokenizer):
        """找到'答案:'在input_ids中的起始位置"""
        # 方法1：查找特定的token序列
        # 需要知道"答案:"在Qwen tokenizer中如何编码
        answer_tokens = tokenizer.encode("答案:", add_special_tokens=False)
        
        # 在input_ids中搜索这个序列
        for i in range(len(input_ids) - len(answer_tokens) + 1):
            if all(input_ids[i+j] == answer_tokens[j] for j in range(len(answer_tokens))):
                return i + len(answer_tokens)  # 返回"答案:"之后的位置
        
        # 如果找不到，返回-1
        return -1

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        file_name = item['file_name']

        # -------------------- A. 加载和处理 ECG 数据 --------------------
        data_raw = _get_ecg_data_raw(self.data_dir, file_name)
        ecg_data = _data_process_full(
            data_raw,
            is_train=self.is_train,
            apply_augment=self.augment
        )

        # -------------------- B. Tokenize 文本 --------------------
        full_text = item['full_text']
        final_input_text = f"{full_text}{self.tokenizer.eos_token}"

        # 修改这里：去掉 padding="max_length"，只做截断
        tokenized = self.tokenizer(
            final_input_text,
            max_length=self.max_len,
            truncation=True,  # 只做截断，不做填充
            return_tensors="pt",
            add_special_tokens=True
        )

        input_ids = tokenized['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # -------------------- C. 标签掩码 (Labels Masking) --------------------
        labels = input_ids.clone()
        
        ecg_positions = (input_ids == self.ecg_token_id).nonzero(as_tuple=True)[0]
        
        if len(ecg_positions) > 0:
            # 正常情况下只有一个ECG token
            ecg_pos = ecg_positions[0].item()
            
            # 找到"答案:"的位置（更可靠的方法）
            # 在tokenized后查找"答案"对应的token
            # 假设"答案:"对应的token序列是[token1, token2]
            answer_start_pos = self._find_answer_start(input_ids, self.tokenizer)
            
            if answer_start_pos > ecg_pos:
                # 掩码从开始到"答案:"之前的所有token
                labels[:answer_start_pos] = -100
            else:
                # 如果找不到"答案:"，至少掩码到ECG token之后
                labels[:ecg_pos + 1] = -100
        else:
           # 如果没有ECG token，使用fallback方法
            if "\n答案:" in full_text:
                prompt_text = full_text.split("\n答案:")[0] + "\n答案:"
                prompt_ids = self.tokenizer.encode(prompt_text)
                labels[:len(prompt_ids)] = -100

        if idx == 0:  # 只打印第一个样本
            print("\n=== 训练数据格式验证 ===")
            print(f"文件名: {file_name}")
            print(f"ECG数据形状: {ecg_data.shape}")
            print(f"完整文本: {full_text}")
            print(f"\nTokenized结果:")
            print(f"input_ids长度: {len(input_ids)}")
            print(f"input_ids: {input_ids.tolist()[:30]}...")
            print(f"\n标签掩码:")
            print(f"labels: {labels.tolist()[:30]}...")
            
            # 解码查看
            print(f"\n解码input_ids:")
            decoded_input = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            print(decoded_input[:200])
            
            print(f"\n解码labels（-100替换为[IGN]）:")
            labels_text = []
            for i, label in enumerate(labels):
                if label == -100:
                    labels_text.append("[IGN]")
                else:
                    labels_text.append(self.tokenizer.decode([label]))
            print(" ".join(labels_text[:50]))
            
            # 检查哪些位置计算损失
            loss_positions = (labels != -100).nonzero(as_tuple=True)[0]
            print(f"\n计算损失的位置（前10个）: {loss_positions[:10].tolist()}")
            print(f"这些位置的token: {[self.tokenizer.decode([input_ids[pos]]) for pos in loss_positions[:10]]}")
        
        return {
            "ecg_data": ecg_data,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }