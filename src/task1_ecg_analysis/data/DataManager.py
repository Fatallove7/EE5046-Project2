import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class DataManager:
    """管理数据集的加载和划分"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.cv_path = os.path.join(base_path, 'cv')
        
    def load_cv_files(self, cv_indices):
        """
        加载指定的CV文件
        Args:
            cv_indices: 列表，如 [0,1,2,3]
        Returns:
            np.array: 包含所有数据的数组，每行格式 [filename, label] 或 [index, filename, label]
        """
        all_data = []
        for idx in cv_indices:
            csv_file = os.path.join(self.cv_path, f'cv{idx}.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                all_data.extend(df.values)
                print(f"从 cv{idx}.csv 加载了 {len(df)} 个样本")
            else:
                print(f"警告: 文件 {csv_file} 不存在")
        
        return np.array(all_data) if all_data else np.array([])
    
    def create_kfold_splits(self, train_cv_indices, k=5, random_seed=42):
        """
        在训练集上创建K折划分
        Args:
            train_cv_indices: 训练集CV索引，如 [0,1,2,3]
            k: 折数
        Returns:
            list: 每个元素是 (train_data, val_data) 的元组
        """
        # 1. 加载所有训练数据
        train_data = self.load_cv_files(train_cv_indices)
        if len(train_data) == 0:
            print("错误: 训练集数据为空")
            return []
        
        print(f"\n总训练数据: {len(train_data)} 个样本")
        
        # 2. 创建K折划分
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        
        # 获取所有文件名用于划分
        filenames = []
        for record in train_data:
            if len(record) >= 3:  # 格式: [index, filename, label]
                filenames.append(record[1])  # 文件名在第二列
            else:  # 格式: [filename, label]
                filenames.append(record[0])  # 文件名在第一列
        
        splits = []
        for train_idx, val_idx in kf.split(filenames):
            train_fold = train_data[train_idx]
            val_fold = train_data[val_idx]
            splits.append((train_fold, val_fold))
            print(f"  折划分: 训练集 {len(train_fold)} 样本, 验证集 {len(val_fold)} 样本")
        
        return splits