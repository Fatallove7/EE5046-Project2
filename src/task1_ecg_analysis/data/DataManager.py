import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DataManager:
    """管理数据集的加载和划分"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.cv_path = os.path.join(base_path, 'cv')
        
    def load_cv_files(self, cv_indices):
        """加载指定CV索引的数据 - 修复版"""
        data_list = []
        
        for idx in cv_indices:
            csv_path = os.path.join(self.base_path, "cv", f"cv{idx}.csv")
            
            if not os.path.exists(csv_path):
                print(f"⚠️ 文件不存在: {csv_path}")
                continue
            
            try:
                # 方法1：尝试自动检测表头
                df = pd.read_csv(csv_path)
                
                # 调试：打印列名
                print(f"\n加载 cv{idx}.csv:")
                print(f"  形状: {df.shape}")
                print(f"  列名: {df.columns.tolist()}")
                
                # 显示前几行（对于调试）
                if len(df) > 0:
                    print(f"  前3行:")
                    for i in range(min(3, len(df))):
                        print(f"    行{i}: {df.iloc[i].tolist()}")
                
                # 查找正确的列
                if 'file_name' in df.columns and 'label' in df.columns:
                    # 情况1：有明确的file_name和label列
                    file_names = df['file_name'].astype(str).str.strip()
                    labels = df['label'].astype(str).str.strip().str.upper()
                elif len(df.columns) >= 3:
                    # 情况2：有三列，假设第二列是文件名，第三列是标签
                    file_names = df.iloc[:, 1].astype(str).str.strip()
                    labels = df.iloc[:, 2].astype(str).str.strip().str.upper()
                elif len(df.columns) == 2:
                    # 情况3：只有两列
                    file_names = df.iloc[:, 0].astype(str).str.strip()
                    labels = df.iloc[:, 1].astype(str).str.strip().str.upper()
                else:
                    print(f"❌ CSV格式异常: 只有{len(df.columns)}列")
                    continue
                
                # 标签转换
                binary_labels = []
                label_stats = {'A': 0, 'N': 0, 'O': 0, 'other': 0}
                
                for label in labels:
                    label_str = str(label).upper().strip()
                    
                    if label_str == 'A':
                        binary_labels.append(1)  # 房颤
                        label_stats['A'] += 1
                    elif label_str in ['N', 'O']:
                        binary_labels.append(0)  # 非房颤
                        label_stats[label_str] += 1
                    else:
                        binary_labels.append(0)  # 默认非房颤
                        label_stats['other'] += 1
                
                # 添加到数据列表
                for file_name, binary_label in zip(file_names, binary_labels):
                    data_list.append((file_name, binary_label))
                
                # 打印统计信息
                total = len(binary_labels)
                if total > 0:
                    print(f"  标签统计:")
                    for key, count in label_stats.items():
                        if count > 0:
                            print(f"    {key}: {count} ({count/total*100:.1f}%)")
                    print(f"  正样本比例: {sum(binary_labels)/total*100:.1f}%")
                else:
                    print("  警告: 没有加载到任何数据")
                
            except Exception as e:
                print(f"❌ 加载 {csv_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ 总共加载 {len(data_list)} 条记录")
        return data_list
    
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
    
        
        # 提取文件名和标签
        filenames = [item[0] for item in train_data]  # 文件名
        labels = [item[1] for item in train_data]     # 标签
        
        # 检查类别分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"类别分布: {dict(zip(unique_labels, counts))}")
        
        # 确保每个类别至少有k个样本
        if min(counts) < k:
            print(f"警告: 最少类别的样本数({min(counts)})少于折数({k})")
            print("考虑减少折数或增加数据")
            k = min(k, min(counts))  # 自动调整折数
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
        
        splits = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filenames, labels)):
            # 将 NumPy 数组转换为列表
            train_idx_list = train_idx.tolist() if isinstance(train_idx, np.ndarray) else list(train_idx)
            val_idx_list = val_idx.tolist() if isinstance(val_idx, np.ndarray) else list(val_idx)
            
            # 使用列表推导式提取数据
            train_fold = [train_data[i] for i in train_idx_list]
            val_fold = [train_data[i] for i in val_idx_list]
            
            splits.append((train_fold, val_fold))
            
            # 打印验证集类别分布
            val_labels = [item[1] for item in val_fold]
            val_unique, val_counts = np.unique(val_labels, return_counts=True)
            print(f"  折 {fold_idx+1}: 训练集 {len(train_fold)} 样本, 验证集 {len(val_fold)} 样本")
            print(f"      验证集类别分布: {dict(zip(val_unique, val_counts))}")
        
        return splits
    
    def augment_positive_samples(self, data_list, augmentation_factor=5):
        """对正样本进行数据增强"""
        # 分离正负样本
        positive_samples = [item for item in data_list if item[1] == 1]
        negative_samples = [item for item in data_list if item[1] == 0]
        
        print(f"数据增强前: 正样本={len(positive_samples)}, 负样本={len(negative_samples)}")
        
        if len(positive_samples) == 0:
            return data_list
        
        # 数据增强：复制并轻微修改正样本
        augmented_positives = []
        for i in range(augmentation_factor):
            for file_name, label in positive_samples:
                # 创建增强后的文件名（添加后缀）
                aug_file_name = f"{file_name}_aug{i}"
                augmented_positives.append((aug_file_name, label))
        
        # 合并原始负样本和增强后的正样本
        balanced_data = negative_samples + augmented_positives[:len(negative_samples)//10]  # 保持1:10比例
        
        print(f"数据增强后: 正样本={len([x for x in balanced_data if x[1]==1])}, 负样本={len([x for x in balanced_data if x[1]==0])}")
        
        return balanced_data
        
    def get_data_statistics(self, cv_indices):
        """获取数据统计信息"""
        data = self.load_cv_files(cv_indices)
        if not data:
            return {}
        
        labels = [label for _, label in data]
        labels_np = np.array(labels)
        
        stats = {
            'total_samples': len(data),
            'positive_samples': int(np.sum(labels_np)),
            'negative_samples': int(len(labels_np) - np.sum(labels_np)),
            'positive_ratio': float(np.mean(labels_np)),
            'negative_ratio': float(1 - np.mean(labels_np))
        }
        
        print(f"\n=== 数据统计 (CV{cv_indices}) ===")
        print(f"总样本数: {stats['total_samples']}")
        print(f"正样本数: {stats['positive_samples']} ({stats['positive_ratio']*100:.2f}%)")
        print(f"负样本数: {stats['negative_samples']} ({stats['negative_ratio']*100:.2f}%)")
        
        return stats