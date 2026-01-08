import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio
from src.common.Config import DOWNSAMPLE_RATE, FIXED_LENGTH
from src.task1_ecg_analysis.data.FoldDataset import FoldDataset

class BalancedFoldDataset(FoldDataset):
    """平衡数据集，确保正负样本比例均衡"""
    
    def __init__(self, data_array, base_path, is_train=True, augment=False, 
                 augmentation_config=None, target_ratio=0.5):
        """
        Args:
            target_ratio: 目标正样本比例
        """
        super().__init__(data_array, base_path, is_train, augment, augmentation_config)
        
        self.target_ratio = target_ratio
        
        # 计算需要采样的数量
        self._create_balanced_indices()
    
    def _create_balanced_indices(self):
        """创建平衡的样本索引"""
        # 计算需要采样的正负样本数量
        total_samples = len(self.data_array) + len(self.augmented_samples)
        target_pos_count = int(total_samples * self.target_ratio)
        target_neg_count = total_samples - target_pos_count
        
        # 计算需要上采样/下采样的数量
        actual_pos_count = len(self.positive_indices) + len(self.augmented_samples)
        actual_neg_count = len(self.negative_indices)
        
        print(f"平衡前: 正样本={actual_pos_count}, 负样本={actual_neg_count}")
        print(f"目标: 正样本={target_pos_count}, 负样本={target_neg_count}")
        
        # 创建平衡索引
        self.balanced_indices = []
        
        # 对少数类上采样
        if actual_pos_count < target_pos_count:
            # 重复正样本索引
            repeat_times = target_pos_count // actual_pos_count + 1
            pos_indices = list(self.positive_indices)
            pos_indices.extend(range(len(self.data_array), 
                                   len(self.data_array) + len(self.augmented_samples)))
            
            for _ in range(repeat_times):
                self.balanced_indices.extend(pos_indices)
            
            # 截断到目标数量
            self.balanced_indices = self.balanced_indices[:target_pos_count]
        else:
            # 从正样本中随机选择
            pos_indices = list(self.positive_indices)
            pos_indices.extend(range(len(self.data_array), 
                                   len(self.data_array) + len(self.augmented_samples)))
            self.balanced_indices.extend(
                np.random.choice(pos_indices, target_pos_count, replace=False).tolist()
            )
        
        # 对多数类下采样
        if actual_neg_count > target_neg_count:
            # 从负样本中随机选择
            self.balanced_indices.extend(
                np.random.choice(self.negative_indices, target_neg_count, replace=False).tolist()
            )
        else:
            # 重复负样本索引
            repeat_times = target_neg_count // actual_neg_count + 1
            for _ in range(repeat_times):
                self.balanced_indices.extend(self.negative_indices)
            
            # 截断到目标数量
            neg_count = len(self.balanced_indices) - target_pos_count
            if neg_count > target_neg_count:
                self.balanced_indices = self.balanced_indices[:target_pos_count + target_neg_count]
        
        # 打乱顺序
        np.random.shuffle(self.balanced_indices)
        
        print(f"平衡后: 总样本={len(self.balanced_indices)}")
    
    def __len__(self):
        if self.is_train:
            return len(self.balanced_indices)
        return super().__len__()
    
    def __getitem__(self, idx):
        if self.is_train:
            # 使用平衡索引
            balanced_idx = self.balanced_indices[idx]
            
            # 判断是否是增强样本
            if balanced_idx >= len(self.data_array):
                # 增强样本
                aug_idx = balanced_idx - len(self.data_array)
                return super()._get_augmented_sample(aug_idx)
            else:
                # 原始样本
                return super().__getitem__(balanced_idx)
        else:
            # 验证/测试模式，不应用平衡采样
            return super().__getitem__(idx)