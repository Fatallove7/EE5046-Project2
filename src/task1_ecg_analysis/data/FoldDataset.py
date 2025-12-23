import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio
from src.common.Config import DOWNSAMPLE_RATE, FIXED_LENGTH

class FoldDataset(Dataset):
    """处理从DataManager获取的数据数组"""
    
    def __init__(self, data_array, base_path, is_train=True, augment=False):
        """
        Args:
            data_array: numpy数组，每行是一个样本 [filename, label] 或 [index, filename, label]
            base_path: 数据集根目录
            is_train: 是否为训练集
            augment: 是否启用数据增强
        """
        self.data_array = data_array
        self.base_path = base_path
        self.is_train = is_train
        self.augment = augment
        
    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self, idx):
        # 获取文件名和标签
        record = self.data_array[idx]
        
        # 解析记录格式
        if len(record) >= 3:  # 有index列
            filename = record[1]
            label = record[2]
        else:  # 没有index列
            filename = record[0]
            label = record[1]
        
        # 加载.mat文件
        mat_path = os.path.join(self.base_path, 'training2017', f'{filename}.mat')
        try:
            data = scio.loadmat(mat_path)['val'][0]
        except Exception as e:
            print(f"错误加载 {mat_path}: {e}")
            data = np.zeros(FIXED_LENGTH)  # 返回默认值
        
        # 数据预处理（需要实现预处理逻辑）
        data = self._preprocess_data(data, self.is_train and self.augment)
        
        # 转换为张量
        data_tensor = torch.FloatTensor(data).unsqueeze(0)  # 添加通道维度
        
        # 标签转换
        label_tensor = torch.tensor([1.0]) if label == 'A' else torch.tensor([0.0])
        
        return data_tensor, label_tensor
    
    def _preprocess_data(self, data, apply_augment):
        """数据预处理（需要根据您的ECGDataset.py中的逻辑实现）"""
        # 这里需要实现与ECGDataset.py中相同的预处理逻辑
        # 包括降采样、归一化、数据增强、裁剪等
        
        # 由于篇幅限制，这里先实现一个简化版本
        # 您需要根据您的ECGDataset.py中的实际逻辑来完善这个函数
        
        # 1. 降采样
        if DOWNSAMPLE_RATE > 1:
            data = data[::DOWNSAMPLE_RATE]
        
        # 2. 归一化
        data = data - data.mean()
        std = data.std()
        if std > 0:
            data = data / std
        
        # 3. 裁剪/填充到固定长度
        L = len(data)
        if L < FIXED_LENGTH:
            data = np.pad(data, (0, FIXED_LENGTH - L), 'constant')
        elif L > FIXED_LENGTH:
            # 如果是训练集且启用增强，随机裁剪；否则中心裁剪
            if self.is_train and apply_augment:
                start = np.random.randint(0, L - FIXED_LENGTH)
            else:
                start = (L - FIXED_LENGTH) // 2
            data = data[start:start + FIXED_LENGTH]
        
        return data