import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class DataManager:
    """管理数据集的加载和划分"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.cv_path = os.path.join(base_path, 'cv')
        
    def load_cv_data(self, cv_indices):
        """加载指定的CV文件数据"""
        all_records = []
        for i in cv_indices:
            csv_file = os.path.join(self.cv_path, f'cv{i}.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                # 假设CSV格式为: index,filename,label 或 filename,label
                all_records.extend(df.values)
            else:
                print(f"Warning: {csv_file} not found.")
        
        return np.array(all_records)
    
    def create_kfold_splits(self, train_indices, k=5, random_seed=42):
        """
        在训练集上创建K折划分
        返回: [(train_fold_indices, val_fold_indices), ...]
        """
        # 加载训练集数据
        train_data = self.load_cv_data(train_indices)
        
        # 创建K折划分
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        
        splits = []
        filenames = train_data[:, 1] if train_data.shape[1] > 2 else train_data[:, 0]
        labels = train_data[:, 2] if train_data.shape[1] > 2 else train_data[:, 1]
        
        for train_idx, val_idx in kf.split(filenames):
            # 获取每个折的训练集和验证集索引
            train_fold = train_data[train_idx]
            val_fold = train_data[val_idx]
            splits.append((train_fold, val_fold))
        
        return splits