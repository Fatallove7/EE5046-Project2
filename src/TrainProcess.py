"""
ECG房颤检测训练主程序
支持超参数搜索、默认训练、完整训练和对比实验四种模式
"""

# ==================== 导入部分 ====================
import argparse
import json
import os
import sys
from datetime import datetime
from glob import glob
from sklearn.model_selection import KFold
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                           precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report)
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import seaborn as sns
import scipy.io as scio
from ECGDataset import ECG_dataset

# 自定义模块
from Config import (AUGMENT_SETTING, BATCH_SIZE, EARLY_STOP_PATIENCE,
                    EXPERIMENT_MODE, FIXED_LENGTH, INPUT_CHANNELS,
                    LEARNING_RATE, MIN_DELTA, NUM_EPOCHS, OUTPUT_CLASSES,
                    USE_STREAM2_SETTING, COMPARISON_MODE,
                    STREAM_COMPARISON_CONFIGS, AUGMENTATION_COMPARISON_CONFIGS,
                    DEFAULT_KERNEL_CONFIG, DOWNSAMPLE_RATE)
from TrainModel import Mscnn


# 设置字体格式
matplotlib.use('Agg')
# 设置中文字体
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12
})

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# =========================数据管理器，管理数据集的加载和划分=================================
class DataManager:
    """管理数据集的加载和划分"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.cv_path = os.path.join(base_path, 'cv')
        
    def load_cv_file(self, cv_index):
        """加载单个CV文件"""
        csv_file = os.path.join(self.cv_path, f'cv{cv_index}.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return df
        else:
            print(f"Warning: {csv_file} not found.")
            return pd.DataFrame()
        
    def load_multiple_cv_files(self, cv_indices):
        """加载多个CV文件并合并"""
        all_dfs = []
        for idx in cv_indices:
            df = self.load_cv_file(idx)
            if not df.empty:
                all_dfs.append(df)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Loaded {len(combined_df)} samples from CV files: {cv_indices}")
            return combined_df
        else:
            print("No data loaded.")
            return pd.DataFrame()
    
    def create_kfold_splits(self, data_df, k=5, random_seed=42):
        """
        创建K折交叉验证划分
        Returns: list of (train_indices, val_indices) tuples
        """
        if data_df.empty:
            return []
        
        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        
        # 获取文件名列表用于划分
        filenames = data_df.iloc[:, 1].values  # 假设第二列是文件名
        
        splits = []
        for train_idx, val_idx in kf.split(filenames):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def get_data_by_indices(self, data_df, indices):
        """根据索引获取数据子集"""
        if data_df.empty or len(indices) == 0:
            return pd.DataFrame()
        return data_df.iloc[indices].reset_index(drop=True)
    

# ==================== 自定义数据集类 ====================
class TemporaryECGDataset(Dataset):
    """临时数据集类，用于从DataFrame加载数据"""
    
    def __init__(self, data_df, base_path, is_train=True, augment=False):
        """
        Args:
            data_df: DataFrame包含数据
            base_path: 数据集根目录
            is_train: 是否为训练集
            augment: 是否启用数据增强
        """
        self.data_df = data_df
        self.base_path = base_path
        self.is_train = is_train
        self.augment = augment
        
    def __len__(self):
        return len(self.data_df)
    
    def _load_mat_data(self, filename):
        """加载.mat文件数据"""
        mat_path = os.path.join(self.base_path, 'training2017', f'{filename}.mat')
        try:
            data = scio.loadmat(mat_path)['val'][0]
            return data
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return np.zeros(1000)  # 返回默认值
    
    def _add_noise(self, data):
        """添加高斯噪声"""
        if np.random.rand() < 0.5:
            noise_level = 0.05
            noise = np.random.normal(0, noise_level, data.shape)
            data = data + noise
        return data
    
    def _time_scaling(self, data):
        """时间尺度缩放"""
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            old_len = len(data)
            new_len = int(old_len * scale_factor)
            
            # 使用线性插值
            x_old = np.linspace(0, 1, old_len)
            x_new = np.linspace(0, 1, new_len)
            data = np.interp(x_new, x_old, data)
        return data
    
    def _crop_padding(self, data, length, apply_augment):
        """裁剪或填充到固定长度"""
        L_raw = len(data)
        
        if L_raw <= length:
            # 填充
            pad_len = length - L_raw
            data = np.pad(data, (0, pad_len), 'constant')
        elif L_raw > length:
            # 裁剪
            max_start = L_raw - length
            
            if self.is_train and apply_augment:
                # 随机裁剪
                start = np.random.randint(0, max_start + 1)
            else:
                # 中心裁剪
                start = max_start // 2
                
            data = data[start:start + length]
        
        return data
    
    def _preprocess_data(self, data, apply_augment):
        """数据预处理流程"""
        # 1. 降采样
        if DOWNSAMPLE_RATE > 1:
            data = data[::DOWNSAMPLE_RATE]
        
        # 2. 时间缩放
        if self.is_train and apply_augment:
            data = self._time_scaling(data)
        
        # 3. 归一化
        data = data - data.mean()
        std = data.std()
        if std > 0:
            data = data / std
        
        # 4. 添加噪声
        if self.is_train and apply_augment:
            data = self._add_noise(data)
        
        # 5. 裁剪/填充
        data = self._crop_padding(data, FIXED_LENGTH, apply_augment)
        
        return data
    
    def __getitem__(self, idx):
        # 获取文件名和标签
        row = self.data_df.iloc[idx]
        
        # 假设CSV格式为: index, filename, label
        if len(row) >= 3:
            filename = row[1]
            label = row[2]
        else:
            # 如果没有index列
            filename = row[0]
            label = row[1]
        
        # 加载数据
        data = self._load_mat_data(filename)
        
        # 预处理
        if self.is_train and self.augment:
            data = self._preprocess_data(data, apply_augment=True)
        else:
            data = self._preprocess_data(data, apply_augment=False)
        
        # 转换为tensor
        data_tensor = torch.FloatTensor(data).unsqueeze(0)  # 添加通道维度
        
        # 标签转换
        if label == 'A':
            label_tensor = torch.tensor([1.0])
        else:
            label_tensor = torch.tensor([0.0])
        
        return data_tensor, label_tensor, filename
    
# ==================== 模型验证器 ====================
class ModelValidator:
    """模型验证相关功能"""
    
    @staticmethod
    def validate(model, criterion, dataloader, device):
        """
        验证函数：计算 Loss, Accuracy, AUC 并返回所有预测值
        返回: (avg_loss, accuracy, auc, labels, probabilities)
        """
        model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y, _ in dataloader:
                x = x.to(device).float()
                y = y.to(device).float()
                x = torch.reshape(x, (-1, 1, FIXED_LENGTH))
                
                probs = model(x)
                loss = criterion(probs, y)
                running_loss += loss.item()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        # 计算指标
        avg_loss = running_loss / len(dataloader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 预测类别
        preds = (all_probs >= 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # 防止只有一个类别报错

        # 计算precision, recall, f1
        try:
            precision = precision_score(all_labels, preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
        except:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        model.train()  # 恢复训练模式
        return avg_loss, acc, auc, all_labels, all_probs, precision, recall, f1

# =========================模型文件管理器，负责管理模型文件的保存和清理=================================
class ModelManager:
    @staticmethod
    def organize_model_files(experiment_base_dir, fold_results):
        """组织模型文件，创建统一的目录结构"""
        print("\n" + "="*80)
        print("📁 整理模型文件")
        print("="*80)
        
        # 1. 创建统一目录结构
        models_base_dir = os.path.join(experiment_base_dir, "Models")
        dirs_to_create = [
            os.path.join(models_base_dir, "Best_Models"),
            os.path.join(models_base_dir, "Checkpoints"),
            os.path.join(models_base_dir, "Final_Models"),
            os.path.join(models_base_dir, "Logs")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        # 2. 复制所有折的最佳模型到统一目录
        print("复制最佳模型到统一目录...")
        best_models_summary = []
        
        for fold_result in fold_results:
            fold = fold_result['fold']
            model_path = fold_result.get('best_model_path')
            
            if model_path and os.path.exists(model_path):
                # 复制到统一目录
                target_dir = os.path.join(models_base_dir, "Best_Models")
                target_path = os.path.join(target_dir, os.path.basename(model_path))
                
                try:
                    import shutil
                    shutil.copy2(model_path, target_path)
                    
                    best_models_summary.append({
                        'fold': fold,
                        'model_file': os.path.basename(model_path),
                        'loss': fold_result['best_loss'],
                        'accuracy': fold_result['final_val_acc']
                    })
                    
                    print(f"  ✅ 第{fold}折: {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"  ❌ 复制第{fold}折模型失败: {e}")
        
        # 3. 创建模型摘要文件
        summary_path = os.path.join(models_base_dir, "Logs", "model_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_folds': len(best_models_summary),
                'models': best_models_summary,
                'organized_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"📋 模型摘要已保存: {summary_path}")
        
        return models_base_dir
    
    @staticmethod
    def create_final_ensemble_model(fold_results, experiment_base_dir, kernel_config):
        """创建最终集成模型（可选）"""
        try:
            print("\n创建最终集成模型...")
            
            models_base_dir = os.path.join(experiment_base_dir, "Models", "Best_Models")
            model_files = glob(os.path.join(models_base_dir, "*.pth"))
            
            if len(model_files) < 3:  # 至少需要3个模型
                print("模型数量不足，跳过集成模型创建")
                return None
            
            # 创建模型配置摘要
            ensemble_config = {
                'type': 'ensemble',
                'num_models': len(model_files),
                'models': [os.path.basename(f) for f in model_files],
                'kernel_config': kernel_config,
                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            config_path = os.path.join(experiment_base_dir, "Models", "ensemble_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(ensemble_config, f, indent=2)
            
            print(f"✅ 集成模型配置已保存: {config_path}")
            
            # 创建一个轻量级的集成模型文件（包含所有模型路径）
            ensemble_model = {
                'ensemble_type': 'majority_voting',
                'model_paths': model_files,
                'weights': [1.0] * len(model_files),  # 平等权重
                'config': ensemble_config
            }
            
            ensemble_path = os.path.join(experiment_base_dir, "Models", "Final_Models", "ensemble_model.pth")
            torch.save(ensemble_model, ensemble_path)
            
            print(f"✅ 集成模型已保存: {ensemble_path}")
            
            return ensemble_path
            
        except Exception as e:
            print(f"创建集成模型失败: {e}")
            return None

# ==================== 工具函数模块 ====================
class TrainingUtils:
    """训练工具函数集合"""
    
    @staticmethod
    def save_loss(fold, value):
        """保存损失值到文件"""
        path = f'loss{fold}.txt'
        with open(path, mode='a+') as file:
            file.write(str(value) + '\n')
    
    @staticmethod
    def plot_roc_curve(labels, probs, epoch, title=None):
        """绘制ROC曲线并返回figure对象 - 使用英文"""
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)

        fig = plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')  # 改为英文
        plt.ylabel('True Positive Rate')   # 改为英文
        plt.title(title if title else f'ROC Curve (Epoch {epoch})')  # 改为英文
        plt.legend(loc="lower right")
        return fig, roc_auc
    
    @staticmethod
    def _generate_standard_model_name(self, fold, epoch, loss, is_best=False):
        """生成标准化的模型文件名"""
        # 基础组件
        kernel_tag = f"K{self.kernel_config['stream1_kernel']}{self.kernel_config['stream2_first_kernel']}"
        stream_tag = "S2" if self.use_stream2 else "S1"
        augment_tag = "Aug" if self.augment else "NoAug"
        config_tag = f"{self.config_name}" if self.config_name else ""
        
        # 构建文件名（按逻辑顺序）
        parts = [
            "Mscnn",
            kernel_tag,
            stream_tag,
            augment_tag,
            f"F{fold}",
            f"E{epoch:03d}",
            f"L{loss:.4f}".replace('.', 'p'),
            "BEST" if is_best else "",
            config_tag
        ]
        
        # 过滤空部分并连接
        filename = "_".join(filter(None, parts)) + ".pth"
        
        return filename
    
    
    @staticmethod
    def create_visualization_directory(base_dir, experiment_name):
        """创建可视化目录结构"""
        vis_dir = os.path.join(base_dir, "Visualizations", experiment_name)
        subdirs = ["Metrics", "ROC_Curves", "Comparison_Plots", "Confusion_Matrices"]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(vis_dir, subdir), exist_ok=True)
        
        return vis_dir


# ==================== 验证模块 ====================
class ModelValidator:
    """模型验证相关功能"""
    
    @staticmethod
    def validate(model, criterion, dataloader, device):
        """
        验证函数：计算 Loss, Accuracy, AUC 并返回所有预测值
        返回: (avg_loss, accuracy, auc, labels, probabilities)
        """
        model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, y, _ in dataloader:
                x = x.to(device).float()
                y = y.to(device).float()
                x = torch.reshape(x, (-1, 1, FIXED_LENGTH))
                
                probs = model(x)
                loss = criterion(probs, y)
                running_loss += loss.item()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        # 计算指标
        avg_loss = running_loss / len(dataloader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 预测类别
        preds = (all_probs >= 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5  # 防止只有一个类别报错

        # 计算precision, recall, f1
        try:
            precision = precision_score(all_labels, preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
        except:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        model.train()  # 恢复训练模式
        return avg_loss, acc, auc, all_labels, all_probs, precision, recall, f1
    
    @staticmethod
    def calculate_confusion_matrix(labels, probs, threshold=0.5):
        """计算混淆矩阵和详细指标"""
        
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds)
        
        # 计算性能指标
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        # 生成分类报告
        report = classification_report(labels, preds, target_names=['正常', '房颤'], output_dict=True, zero_division=0)

        return cm, {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'classification_report': report
        }


# ==================== 模型训练器模块 ====================
class ModelTrainer:
    """模型训练器，负责单次训练验证过程"""
    
    def __init__(self, base_path, kernel_config, batch_size, lr, 
                 use_stream2, augment, experiment_base_dir, config_name=None):
        self.base_path = base_path
        self.kernel_config = kernel_config
        self.batch_size = batch_size
        self.lr = lr
        self.use_stream2 = use_stream2
        self.augment = augment
        self.experiment_base_dir = experiment_base_dir
        self.config_name = config_name
        self.models_saved = 0
        self.best_model_path = None

        # 数据管理器
        self.data_manager = DataManager(base_path)
        
        # 训练状态
        self.best_f1 = 0.0  # F1分数越高越好
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        self.patience_counter = 0
        self.patience_counter_f1 = 0
        self.validation_fold = None  # 当前验证的折
        self.training_folds = None   # 当前训练的折列表
        
    def train_fold(self, train_folds, test_fold, num_epochs):
        """训练单个折"""
        self.validation_fold = test_fold
        self.training_folds = train_folds

        config_name_str = f" ({self.config_name})" if self.config_name else ""
        print(f"\n{'='*60}")
        print(f"训练配置{config_name_str}:")
        print(f"  Stream配置: {'Stream1+2' if self.use_stream2 else 'Stream1 Only'}")
        print(f"  卷积核: Stream1={self.kernel_config['stream1_kernel']}, "
              f"Stream2前4层={self.kernel_config['stream2_first_kernel']}")
        print(f"  数据增强: {'启用' if self.augment else '禁用'}")
        print(f"  批大小: {self.batch_size}, 学习率: {self.lr}")
        print(f"  测试折: cv{test_fold}")
        print(f"{'='*60}")
        
        # 1. 准备数据
        train_dataset = ECG_dataset(
            self.base_path, is_train=True, augment=self.augment, cv=train_folds
        )
        test_dataset = ECG_dataset(
            self.base_path, is_train=False, augment=False, cv=test_fold
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, drop_last=True
        )
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        # 2. 初始化模型和优化器
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 3. 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # 4. 设置TensorBoard日志
        log_dir = self._setup_logging(test_fold)
        writer = SummaryWriter(log_dir)
        
        # 5. 训练循环
        fold_results = self._training_loop(
            model, criterion, optimizer, scheduler, 
            train_loader, test_loader, writer, 
            test_fold, num_epochs
        )

        # 添加模型保存信息
        fold_results['best_model_path'] = self.best_model_path
        fold_results['total_models_saved'] = self.models_saved
        
        writer.close()
        return fold_results

    def cross_validate_on_train_set(self, train_indices, num_epochs, k_folds=5):
        """
        在训练集上进行K折交叉验证
        Args:
            train_indices: 训练集的CV索引列表，如[0,1,2,3]
            num_epochs: 每个折的训练epoch数
            k_folds: 交叉验证的折数
        Returns:
            平均验证准确率，各折结果
        """
        print(f"\n{'='*60}")
        print(f"在训练集上进行 {k_folds} 折交叉验证")
        print(f"训练集来源: CV{', '.join(map(str, train_indices))}")
        print(f"{'='*60}")
        
        # 创建K折划分
        kfold_splits = self.data_manager.create_kfold_splits(train_indices, k=k_folds)
        
        fold_results = []
        
        for fold_idx, (train_fold_data, val_fold_data) in enumerate(kfold_splits):
            print(f"\n--- 训练折 {fold_idx + 1}/{k_folds} ---")
            
            # 为这个折创建临时数据集
            train_dataset = self._create_dataset_from_data(train_fold_data, is_train=True)
            val_dataset = self._create_dataset_from_data(val_fold_data, is_train=False)
            
            # 训练和验证这个折
            fold_result = self._train_single_fold(
                train_dataset, val_dataset, 
                fold_idx, num_epochs
            )
            
            fold_results.append(fold_result)
        
        # 计算平均性能
        avg_metrics = self._compute_cv_metrics(fold_results)
        
        return avg_metrics, fold_results    
    
    def _train_single_fold(self, train_df, val_df, fold_idx, num_epochs):
        """训练单个折"""
        # 创建数据集
        train_dataset = TemporaryECGDataset(
            train_df, self.base_path, is_train=True, augment=self.augment
        )
        val_dataset = TemporaryECGDataset(
            val_df, self.base_path, is_train=False, augment=False
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        # 初始化模型
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 训练循环
        best_val_acc = 0
        best_val_auc = 0
        best_val_f1 = 0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(
                model, criterion, optimizer, train_loader
            )
            
            # 验证阶段
            val_loss, val_acc, val_auc, _, _, _, val_f1 = ModelValidator.validate(
                model, criterion, val_loader, device
            )
            
            # 早停检查
            if val_acc > best_val_acc + MIN_DELTA:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型（可选）
                if self.experiment_base_dir:
                    self._save_fold_model(model, fold_idx, epoch, val_acc)
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
                print(f"  Epoch {epoch}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  早停触发于第 {epoch} 轮")
                break
        
        return {
            'fold': fold_idx,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch
        }
    
    def _setup_logging(self, test_fold):
        """设置TensorBoard日志目录"""
        config_name_suffix = f"_{self.config_name}" if self.config_name else ""
        log_dir_relative = (
            f"K{self.kernel_config['stream1_kernel']}_{self.kernel_config['stream2_first_kernel']}_"
            f"BS{self.batch_size}_LR{self.lr}_S{'2' if self.use_stream2 else '1'}_"
            f"{'Aug' if self.augment else 'NoAug'}{config_name_suffix}/Fold_{test_fold}"
        )
        log_dir = os.path.join(
            self.experiment_base_dir, "TensorBoard_Logs", log_dir_relative
        )
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def _training_loop(self, model, criterion, optimizer, scheduler,
                      train_loader, test_loader, writer, test_fold, num_epochs):
        """训练循环主逻辑"""  

        print(f"\n🛡️ 双重早停策略已启用:")
        print(f"  - MIN_DELTA: {MIN_DELTA}")
        print(f"  - EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}")
        print(f"  - 监控指标: F1分数 + 损失")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch  # 更新当前epoch
            print(f'\nFold {test_fold} - Epoch {epoch}/{num_epochs}')
            print(f'📊 早停计数器: {self.patience_counter}/{EARLY_STOP_PATIENCE}')
            print(f'🏆 最佳F1分数: {self.best_f1:.4f} (Epoch {self.best_epoch})')
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(
                model, criterion, optimizer, train_loader, epoch
            )
            
            # 验证阶段
            val_loss, val_acc, val_auc, val_labels, val_probs, val_precision, val_recall, val_f1 = ModelValidator.validate(
            model, criterion, test_loader, device
        )
            
            print(f"Fold {test_fold} - Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
            
            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 记录到TensorBoard
            self._log_to_tensorboard(
            writer, epoch, train_loss, train_acc, 
            val_loss, val_acc, val_auc, val_precision, val_recall, val_f1, val_labels, val_probs
        )
            
           # 检查是否保存最佳模型 - 使用F1分数
            if self._should_save_model(val_loss, val_f1):
                self.best_f1 = val_f1
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter_f1 = 0  # 重置F1耐心计数器
                self._save_best_model(
                    model, val_loss, val_f1, epoch, test_fold, 
                    self.batch_size, self.lr
                )
            else:
                self.patience_counter_f1 += 1

            # 早停检查
            if self._should_early_stop(current_val_f1=val_f1):
                break
        
        return {
            'fold': test_fold,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_val_acc': val_acc,
            'final_val_auc': val_auc,
            'final_val_precision': val_precision,
            'final_val_recall': val_recall,
            'final_val_f1': val_f1,
            'val_labels': val_labels,
            'val_probs': val_probs
    }
    
    def _train_epoch(self, model, criterion, optimizer, train_loader):
        """训练单个epoch"""
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        for x, y, _ in train_loader:
            x = x.to(device).float()
            x = torch.reshape(x, (-1, 1, FIXED_LENGTH))
            y = y.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集预测结果
            preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(y.detach().cpu().numpy().flatten())
        
        # 计算训练准确率
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)
        
        return avg_loss, train_acc
    
    def _log_to_tensorboard(self, writer, epoch, train_loss, train_acc,
                           val_loss, val_acc,val_auc, val_precision, val_recall, val_f1, val_labels, val_probs):
        """记录训练信息到TensorBoard - 修改后版本"""
        # 标量记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Metric/AUC', val_auc, epoch)
        writer.add_scalar('Metric/Precision', val_precision, epoch)
        writer.add_scalar('Metric/Recall', val_recall, epoch)
        writer.add_scalar('Metric/F1', val_f1, epoch)
    
        # ROC曲线记录（每10个epoch记录一次以节省空间）
        if epoch % 10 == 0 or epoch == 1:
            fig, _ = TrainingUtils.plot_roc_curve(val_labels, val_probs, epoch)
            writer.add_figure('ROC_Curve', fig, epoch)
            plt.close(fig)
    
    def _should_save_model(self, val_loss,val_f1):
        """
        判断是否应该保存模型
        标准：F1分数有显著提升
        """
        if val_f1 > self.best_f1 + MIN_DELTA:
            print(f"🎯 F1分数从 {self.best_f1:.4f} 改进到 {val_f1:.4f}，保存模型")
            self.best_f1 = val_f1
            self.best_loss = val_loss  # 同时记录最佳损失
            self.patience_counter = 0
            return True
        return False
    
    def _save_best_model(self, model, val_loss, val_f1, epoch, fold_idx, 
                     is_cv_fold=True, batch_size=None, lr=None):
        """
        通用模型保存方法，支持两种模式：
        - is_cv_fold=True: 保存交叉验证的折模型
        - is_cv_fold=False: 保存最终模型
        """
        if is_cv_fold:
            # 保存交叉验证折模型
            model_filename = f"cv_fold{fold_idx}_epoch{epoch}_f1{val_f1:.4f}.pth"
            save_dir = os.path.join(self.experiment_base_dir, "CV_Fold_Models")
        else:
            # 保存最终模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"final_model_{timestamp}_f1{val_f1:.4f}.pth"
            save_dir = os.path.join(self.experiment_base_dir, "Final_Models")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, model_filename)
        
        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'fold_idx': fold_idx if is_cv_fold else None,
            'config': {
                'kernel_config': self.kernel_config,
                'batch_size': batch_size or self.batch_size,
                'lr': lr or self.lr,
                'use_stream2': self.use_stream2,
                'augment': self.augment,
                'is_cv_fold': is_cv_fold
            }
        }, save_path)
        
        print(f"模型已保存: {save_path}")
        return save_path

    def _generate_standard_model_name(self, fold, epoch, loss, f1_score=None, is_best=False):
        """生成标准化的模型文件名"""
        # 基础组件
        kernel_tag = f"K{self.kernel_config['stream1_kernel']}{self.kernel_config['stream2_first_kernel']}"
        stream_tag = "S2" if self.use_stream2 else "S1"
        augment_tag = "Aug" if self.augment else "NoAug"
        config_tag = f"{self.config_name}" if self.config_name else ""

        # 确保epoch是整数
        epoch_int = int(epoch)  # 强制转换为整数
        
        # 构建文件名（按逻辑顺序）
        parts = [
            "Mscnn",
            kernel_tag,
            stream_tag,
            augment_tag,
            f"F{fold}",
            f"E{epoch_int:03d}",
            f"L{loss:.4f}".replace('.', 'p'),
        ]
        
        # 如果有F1分数，添加到文件名中
        if f1_score is not None:
            parts.append(f"F1{f1_score:.4f}".replace('.', 'p'))

        if is_best:
            parts.append("BEST")
        
        parts.append(config_tag)
        
        # 过滤空部分并连接
        filename = "_".join(filter(None, parts)) + ".pth"

        # 过滤空部分并连接
        filename = "_".join(filter(None, parts)) + ".pth"
        
        return filename
    
    def _clean_old_models_fold(self, fold_models_dir, fold):
        """清理旧的同折模型文件 - 只保留最新最佳"""
        try:
            # 获取目录中所有模型文件
            pattern = os.path.join(fold_models_dir, f"*_F{fold}_*.pth")
            old_files = glob(pattern)
            
            if len(old_files) > 0:
                print(f"🔄 清理第{fold}折旧模型，保留最新最佳...")
                for f in old_files:
                    try:
                        os.remove(f)
                        print(f"   🗑️ 删除: {os.path.basename(f)}")
                    except Exception as e:
                        print(f"   清理失败: {e}")
        except Exception as e:
            print(f"⚠️ 清理旧模型时出错: {e}")
        
    
    def _should_early_stop(self, current_val_f1):
        """
        单指标早停策略：仅监控F1分数
        
        Args:
            current_val_f1: 当前验证F1分数
        
        Returns:
            bool: 是否触发早停
        """
        
        # 检查F1分数是否有改进
        if current_val_f1 > self.best_f1 + MIN_DELTA:
            print(f"🎯 F1分数改进: {self.best_f1:.4f} → {current_val_f1:.4f}，重置耐心计数器")
            self.best_f1 = current_val_f1
            self.best_epoch = self.current_epoch  # 需要保存当前epoch
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            print(f"🔄 F1分数无改进 ({self.best_f1:.4f})，耐心计数器: {self.patience_counter}/{EARLY_STOP_PATIENCE}")
            
            # 检查是否达到早停条件
            if self.patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n🚨 早停触发!")
                print(f"   连续 {EARLY_STOP_PATIENCE} 个epoch F1分数没有显著改进")
                print(f"   最佳epoch: {self.best_epoch}, 最佳F1分数: {self.best_f1:.4f}")
                print(f"   最终F1分数: {current_val_f1:.4f}")
                return True
            
            return False

# ==================== 完整训练模块 ====================
class CompleteTrainer:
    """使用最佳配置进行完整训练"""
    
    @staticmethod
    def train_with_best_config(base_path, best_config_data, num_epochs):
        """
        使用搜索得到的最佳配置进行完整训练（5折交叉验证）
        """
        print("=" * 80)
        print("使用最佳配置进行完整训练")
        print("=" * 80)
        
        kernel_config = best_config_data['kernel_config']
        batch_size = best_config_data['batch_size']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"/home/xusi/Logs/FinalTraining/Results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"最佳配置: {kernel_config['name']}, 批大小: {batch_size}")
        print(f"将进行完整的5折交叉验证，每个折训练{num_epochs}个epoch")
        print(f"📁 结果目录: {results_dir}")
        
        # K-Fold训练
        K = 5
        fold_results = []
        
        for k in range(K):
            test_fold = k
            train_folds = [i for i in range(5) if i != k]
            
            print(f"\n{'='*60}")
            print(f"📈 训练折 {k + 1}/{K} (测试折: cv{test_fold})")
            print(f"{'='*60}")
            
            trainer = ModelTrainer(
                base_path=base_path,
                kernel_config=kernel_config,
                batch_size=batch_size,
                lr=LEARNING_RATE,
                use_stream2=USE_STREAM2_SETTING,
                augment=AUGMENT_SETTING,
                experiment_base_dir=results_dir,
                config_name=kernel_config['name']
            )
            
            result = trainer.train_fold(
                train_folds=train_folds,
                test_fold=test_fold,
                num_epochs=num_epochs
            )
            
            fold_results.append(result)
            
            print(f"\n📊 第{k+1}折结果:")
            print(f"  ✅ 最佳Loss: {result['best_loss']:.4f}")
            print(f"  ✅ 验证准确率: {result['final_val_acc']:.4f}")
            print(f"  ✅ AUC: {result['final_val_auc']:.4f}")
            print(f"  ✅ 精确率: {result['final_val_precision']:.4f}")
            print(f"  ✅ 召回率: {result['final_val_recall']:.4f}")
            print(f"  ✅ F1分数: {result['final_val_f1']:.4f}")
            print(f"  💾 模型保存: {result.get('best_model_path', 'N/A')}")

        # 组织模型文件
        models_dir = ModelManager.organize_model_files(results_dir, fold_results)

        # 可选：创建集成模型
        ensemble_path = ModelManager.create_final_ensemble_model(fold_results, results_dir, kernel_config)
        
        # 计算完整训练的平均结果
        avg_metrics = CompleteTrainer._compute_final_metrics(fold_results)
        
        # 保存结果
        final_results, simplified_results = CompleteTrainer._save_final_results(
            kernel_config, batch_size, fold_results, 
            avg_metrics, results_dir, ensemble_path
        )

        # 打印详细的文件位置信息
        CompleteTrainer._print_file_locations(results_dir, final_results, simplified_results)
        
        return avg_metrics,final_results,simplified_results
    
    @staticmethod
    def _print_file_locations(results_dir, final_results, simplified_results):
        """打印文件保存位置信息"""
        print("\n" + "="*80)
        print("📁 文件保存位置")
        print("="*80)
        
        print(f"🏠 主结果目录: {results_dir}")
        
        # 检查并列出所有重要文件
        important_files = {
            "完整评估结果": os.path.join(results_dir, "final_results.json"),
            "LLM兼容结果": os.path.join(results_dir, "cnn_evaluation_results.json"),
            "模型目录": os.path.join(results_dir, "Models"),
            "TensorBoard日志": os.path.join(results_dir, "TensorBoard_Logs"),
            "最佳模型": os.path.join(results_dir, "Models", "Best_Models")
        }
        
        for desc, path in important_files.items():
            if os.path.exists(path):
                file_type = "📁 目录" if os.path.isdir(path) else "📄 文件"
                print(f"{file_type} {desc}: {path}")
            else:
                print(f"❌ 缺失: {desc} ({path})")
        
        print(f"\n📊 CNN模型评估指标:")
        print(f"  准确率: {final_results['accuracy']:.4f}")
        print(f"  精确率: {final_results['precision']:.4f}")
        print(f"  召回率: {final_results['recall']:.4f}")
        print(f"  F1分数: {final_results['f1']:.4f}")
        print(f"  AUC: {final_results['auc']:.4f}")
        
        print(f"\n💡 LLM对比文件: {os.path.join(results_dir, 'cnn_evaluation_results.json')}")
        print("   可以直接在LLM评估代码中使用此文件进行对比")
    
    @staticmethod
    def _compute_final_metrics(fold_results):
        """计算最终指标"""
        avg_loss = np.mean([r['best_loss'] for r in fold_results])
        avg_acc = np.mean([r['final_val_acc'] for r in fold_results])
        avg_auc = np.mean([r['final_val_auc'] for r in fold_results])
        avg_precision = np.mean([r['final_val_precision'] for r in fold_results])
        avg_recall = np.mean([r['final_val_recall'] for r in fold_results])
        avg_f1 = np.mean([r['final_val_f1'] for r in fold_results])
        
        std_acc = np.std([r['final_val_acc'] for r in fold_results])
        std_auc = np.std([r['final_val_auc'] for r in fold_results])
        std_precision = np.std([r['final_val_precision'] for r in fold_results])
        std_recall = np.std([r['final_val_recall'] for r in fold_results])
        std_f1 = np.std([r['final_val_f1'] for r in fold_results])
        
        print("\n" + "=" * 80)
        print("完整训练结果汇总:")
        print("=" * 80)
        print(f"平均Loss: {avg_loss:.4f}")
        print(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"平均AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"平均精确率: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"平均召回率: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"平均F1分数: {avg_f1:.4f} ± {std_f1:.4f}")
        
        for k, result in enumerate(fold_results):
            print(f"折 {k}: 准确率={result['final_val_acc']:.4f}, "
                  f"AUC={result['final_val_auc']:.4f}, "
                  f"精确率={result['final_val_precision']:.4f}, "
                  f"召回率={result['final_val_recall']:.4f}, "
                  f"F1={result['final_val_f1']:.4f}")
        
        return {
            'avg_loss': float(avg_loss),
            'avg_accuracy': float(avg_acc),
            'std_accuracy': float(std_acc),
            'avg_auc': float(avg_auc),
            'std_auc': float(std_auc),
            'avg_precision': float(avg_precision),
            'std_precision': float(std_precision),
            'avg_recall': float(avg_recall),
            'std_recall': float(std_recall),
            'avg_f1': float(avg_f1),
            'std_f1': float(std_f1)
        }
    
    @staticmethod
    def _save_final_results(kernel_config, batch_size, fold_results, 
                           avg_metrics, results_dir, ensemble_path=None):
        """保存最终训练结果 - 生成兼容LLM评估的JSON格式"""
        
        # 创建与LLM评估兼容的JSON结构
        final_results = {
            'best_config': {
                'kernel_config': {
                    'name': kernel_config['name'],
                    'stream1_kernel': kernel_config['stream1_kernel'],
                    'stream2_first_kernel': kernel_config['stream2_first_kernel'],
                    'ch_in': 1,
                    'ch_out': 1,
                    'use_stream2': USE_STREAM2_SETTING
                },
                'batch_size': batch_size
            },
            
            # 主要性能指标（与LLM评估相同）
            'accuracy': avg_metrics['avg_accuracy'],
            'precision': avg_metrics['avg_precision'],
            'recall': avg_metrics['avg_recall'],
            'f1': avg_metrics['avg_f1'],
            'auc': avg_metrics['avg_auc'],
            
            # 详细统计信息
            'final_metrics': avg_metrics,
            
            # 交叉验证结果（每个fold）
            'cross_validation_results': {
                f'cv{k}': {
                    'accuracy': float(r['final_val_acc']),
                    'precision': float(r['final_val_precision']),
                    'recall': float(r['final_val_recall']),
                    'f1': float(r['final_val_f1']),
                    'auc': float(r['final_val_auc'])
                } for k, r in enumerate(fold_results)
            },
            
            # 原始fold结果（保持原有格式）
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_loss': float(r['best_loss']),
                    'best_epoch': r['best_epoch'],
                    'final_val_acc': float(r['final_val_acc']),
                    'final_val_auc': float(r['final_val_auc']),
                    'final_val_precision': float(r['final_val_precision']),
                    'final_val_recall': float(r['final_val_recall']),
                    'final_val_f1': float(r['final_val_f1'])
                } for r in fold_results
            ],
            
            'training_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_name': 'ECG_CNN',
            'dataset': 'training2017'
        }
        
        # 保存两个版本：完整版本和简化版本（用于LLM对比）
        results_path = os.path.join(results_dir, 'final_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        # 保存简化版本
        simplified_results = {
            'best_config': final_results['best_config'],
            'accuracy': final_results['accuracy'],
            'precision': final_results['precision'],
            'recall': final_results['recall'],
            'f1': final_results['f1'],
            'auc': final_results['auc'],
            'cross_validation_results': final_results['cross_validation_results'],
            'model_files_location': os.path.join(results_dir, "Models")
        }
        
        simplified_path = os.path.join(results_dir, 'cnn_evaluation_results.json')
        with open(simplified_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n完整训练结果已保存: {results_path}")
        print(f"LLM兼容结果已保存: {simplified_path}")
        
        return final_results,simplified_path

    
# ==================== 超参数搜索模块 ====================
class HyperparameterSearcher:
    """超参数搜索器"""
    
    # 搜索配置
    KERNEL_CONFIGS = [
        {'name': 'MS-CNN(3,9)', 'stream1_kernel': 3, 'stream2_first_kernel': 9},
        {'name': 'MS-CNN(3,7)', 'stream1_kernel': 3, 'stream2_first_kernel': 7},
        {'name': 'MS-CNN(3,5)', 'stream1_kernel': 3, 'stream2_first_kernel': 5},
        {'name': 'MS-CNN(3,3)', 'stream1_kernel': 3, 'stream2_first_kernel': 3},
    ]
    
    BATCH_SIZES = [32, 64, 128]
    
    def __init__(self, base_path):
        self.base_path = base_path
    
    def search(self, num_folds=3, num_epochs_search=15):
        """执行超参数搜索"""
        print("=" * 80)
        print("开始智能超参数搜索")
        print(f"将测试 {len(self.KERNEL_CONFIGS)} 种卷积核配置 × "
              f"{len(self.BATCH_SIZES)} 种批大小")
        print(f"共 {len(self.KERNEL_CONFIGS) * len(self.BATCH_SIZES)} 种组合")
        print(f"将在 {num_folds} 折数据上快速评估，"
              f"每种组合训练 {num_epochs_search} 个epoch")
        print("=" * 80)
        
        # 创建结果存储目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        search_dir = f"/home/xusi/Logs/HyperparamSearch/Results_{timestamp}"
        os.makedirs(search_dir, exist_ok=True)
        
        all_results = {}
        
        # 循环所有超参数组合
        for kernel_config in self.KERNEL_CONFIGS:
            for batch_size in self.BATCH_SIZES:
                config_results = self._evaluate_config(
                    kernel_config, batch_size, 
                    num_folds, num_epochs_search, search_dir
                )
                
                # 计算平均性能
                avg_metrics = self._compute_average_metrics(config_results)
                
                all_results[avg_metrics['config_name']] = {
                    'kernel_config': kernel_config,
                    'batch_size': batch_size,
                    **avg_metrics,
                    'fold_results': config_results
                }
        
        # 保存和分析结果
        self._save_and_analyze_results(all_results, search_dir)
        return all_results, search_dir
    
    def _evaluate_config(self, kernel_config, batch_size, 
                        num_folds, num_epochs_search, search_dir):
        """评估单个配置"""
        config_name = f"{kernel_config['name']}_BS{batch_size}"
        print(f"\n{'='*60}")
        print(f"测试配置: {config_name}")
        print(f"卷积核: Stream1={kernel_config['stream1_kernel']}, "
              f"Stream2前4层={kernel_config['stream2_first_kernel']}")
        print(f"批大小: {batch_size}")
        print(f"{'='*60}")
        
        config_results = []
        
        for fold_idx in range(num_folds):
            test_fold = fold_idx
            train_folds = [i for i in range(5) if i != test_fold]
            
            print(f"\n折 {fold_idx + 1}/{num_folds}")
            
            # 训练和验证
            trainer = ModelTrainer(
                base_path=self.base_path,
                kernel_config=kernel_config,
                batch_size=batch_size,
                lr=LEARNING_RATE,
                use_stream2=USE_STREAM2_SETTING,
                augment=AUGMENT_SETTING,
                experiment_base_dir=search_dir
            )
            
            result = trainer.train_fold(
                train_folds=train_folds,
                test_fold=test_fold,
                num_epochs=num_epochs_search
            )
            
            config_results.append(result)
            
            print(f"  最佳Loss: {result['best_loss']:.4f}, "
                  f"验证准确率: {result['final_val_acc']:.4f}, "
                  f"AUC: {result['final_val_auc']:.4f}")
        
        return config_results
    
    def _compute_average_metrics(self, config_results):
        """计算平均指标"""
        avg_loss = np.mean([r['best_loss'] for r in config_results])
        avg_acc = np.mean([r['final_val_acc'] for r in config_results])
        avg_auc = np.mean([r['final_val_auc'] for r in config_results])
        
        return {
            'config_name': f"{config_results[0]['fold']}_avg",
            'avg_loss': float(avg_loss),
            'avg_accuracy': float(avg_acc),
            'avg_auc': float(avg_auc)
        }
    
    def _save_and_analyze_results(self, all_results, search_dir):
        """保存和分析搜索结果"""
        self._save_results(all_results, search_dir)
        self._create_visual_report(all_results, search_dir)
        self._find_best_configuration(all_results, search_dir)
    
    def _save_results(self, all_results, search_dir):
        """保存结果到文件"""
        # JSON格式详细结果
        detailed_path = os.path.join(search_dir, 'detailed_results.json')
        with open(detailed_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # CSV格式摘要
        summary_data = []
        for config_name, data in all_results.items():
            summary_data.append({
                'Config': config_name,
                'Kernel': f"{data['kernel_config']['stream1_kernel']}_"
                         f"{data['kernel_config']['stream2_first_kernel']}",
                'Batch_Size': data['batch_size'],
                'Avg_Loss': data['avg_loss'],
                'Avg_Accuracy': data['avg_accuracy'],
                'Avg_AUC': data['avg_auc']
            })
        
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(search_dir, 'summary.csv')
        df.to_csv(summary_path, index=False)
        
        print(f"\n搜索结果已保存到: {search_dir}")
    
    def _create_visual_report(self, all_results, search_dir):
        """创建可视化报告"""
        try:
            config_names = list(all_results.keys())
            accuracies = [all_results[name]['avg_accuracy'] for name in config_names]
            auc_scores = [all_results[name]['avg_auc'] for name in config_names]
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # 准确率柱状图
            x = range(len(config_names))
            axes[0].bar(x, accuracies, color='skyblue', edgecolor='black')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('超参数搜索结果 - 准确率')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(config_names, rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # AUC柱状图
            axes[1].bar(x, auc_scores, color='lightgreen', edgecolor='black')
            axes[1].set_xlabel('配置')
            axes[1].set_ylabel('AUC')
            axes[1].set_title('超参数搜索结果 - AUC')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(config_names, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(search_dir, 'search_results_chart.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存: {chart_path}")
            
        except ImportError:
            print("注意: matplotlib未安装，跳过图表生成。")
    
    def _find_best_configuration(self, all_results, search_dir):
        """找出最佳超参数配置"""
        sorted_configs = sorted(
            all_results.items(),
            key=lambda x: x[1]['avg_accuracy'],
            reverse=True
        )
        
        print("\n" + "=" * 80)
        print("超参数搜索排名结果:")
        print("=" * 80)
        
        best_config_name, best_config_data = sorted_configs[0]
        
        for rank, (config_name, data) in enumerate(sorted_configs, 1):
            marker = " ★" if rank == 1 else ""
            print(f"{rank:2d}. {config_name:30s} "
                  f"准确率: {data['avg_accuracy']:.4f} | "
                  f"AUC: {data['avg_auc']:.4f} | "
                  f"Loss: {data['avg_loss']:.4f}{marker}")
        
        print("\n" + "=" * 80)
        print("最佳配置:")
        print("=" * 80)
        print(f"配置名称: {best_config_name}")
        print(f"卷积核: Stream1={best_config_data['kernel_config']['stream1_kernel']}, "
              f"Stream2前4层={best_config_data['kernel_config']['stream2_first_kernel']}")
        print(f"批大小: {best_config_data['batch_size']}")
        print(f"平均准确率: {best_config_data['avg_accuracy']:.4f}")
        print(f"平均AUC: {best_config_data['avg_auc']:.4f}")
        print(f"平均Loss: {best_config_data['avg_loss']:.4f}")
        
        # 保存最佳配置
        best_config_path = os.path.join(search_dir, 'best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump({
                'best_config_name': best_config_name,
                'best_config_data': best_config_data,
                'search_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        return best_config_name, best_config_data



# ==================== 对比实验模块 ====================
class ComparisonExperiment:
    """对比实验控制器"""
    
    def __init__(self, base_path, comparison_mode='stream'):
        """
        初始化对比实验
        Args:
            base_path: 数据路径
            comparison_mode: 'stream'对比stream配置, 'augment'对比数据增强
        """
        self.base_path = base_path
        self.comparison_mode = comparison_mode
        
        # 根据对比模式选择配置
        if comparison_mode == 'stream':
            self.comparison_configs = STREAM_COMPARISON_CONFIGS
            self.experiment_name = "Stream_Comparison"
            self.title_prefix = "Stream配置对比: "
        elif comparison_mode == 'augment':
            self.comparison_configs = AUGMENTATION_COMPARISON_CONFIGS
            self.experiment_name = "Augmentation_Comparison"
            self.title_prefix = "数据增强对比: "
        else:
            raise ValueError(f"未知的对比模式: {comparison_mode}")
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"/home/xusi/Logs/Comparison/{self.experiment_name}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 创建子目录
        self.models_dir = os.path.join(self.results_dir, "Models")
        self.metrics_dir = os.path.join(self.results_dir, "Metrics")
        self.plots_dir = os.path.join(self.results_dir, "Plots")
        self.tensorboard_dir = os.path.join(self.results_dir, "TensorBoard")
        
        for dir_path in [self.models_dir, self.metrics_dir, self.plots_dir, self.tensorboard_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 存储结果
        self.comparison_results = {}
        
        print(f"对比实验目录: {self.results_dir}")
    
    def run_comparison(self, num_folds=5, num_epochs=NUM_EPOCHS):
        """运行对比实验"""
        print("\n" + "="*80)
        print(f"开始对比实验: {self.title_prefix}")
        print("="*80)
        
        # 运行每种配置
        for config in self.comparison_configs:
            config_name = config['name']
            print(f"\n{'='*60}")
            print(f"训练配置: {config_name}")
            print(f"描述: {config['description']}")
            print(f"{'='*60}")
            
            # 训练该配置
            fold_results = self._train_configuration(
                config=config,
                num_folds=num_folds,
                num_epochs=num_epochs
            )
            
            # 计算平均指标
            avg_metrics = self._calculate_average_metrics(fold_results)
            
            # 存储结果
            self.comparison_results[config_name] = {
                'config': config,
                'fold_results': fold_results,
                'average_metrics': avg_metrics
            }
        
        # 生成对比分析报告
        self._generate_comparison_report()
        
        # 生成简单的可视化对比结果
        self._create_comparison_visualizations()
        
        # 保存完整的实验结果
        self._save_experiment_results()
        
        print(f"\n对比实验完成！结果保存在: {self.results_dir}")
        
        return self.comparison_results
    
    def _train_configuration(self, config, num_folds, num_epochs):
        """训练特定配置（5折交叉验证）"""
        fold_results = []
        
        for fold_idx in range(num_folds):
            print(f"\n--- 训练折 {fold_idx+1}/{num_folds} ---")
            
            # 准备训练和验证折
            test_fold = fold_idx
            train_folds = [i for i in range(num_folds) if i != fold_idx]
            
            # 根据对比模式设置参数
            if self.comparison_mode == 'stream':
                use_stream2 = config['use_stream2']
                augment = AUGMENT_SETTING  # 使用全局设置
            else:  # augment模式
                use_stream2 = USE_STREAM2_SETTING  # 使用全局设置
                augment = config['augment']
            
            # 创建训练器
            trainer = ModelTrainer(
                base_path=self.base_path,
                kernel_config=DEFAULT_KERNEL_CONFIG,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                use_stream2=use_stream2,
                augment=augment,
                experiment_base_dir=self.results_dir,
                config_name=config['name']
            )
            
            # 训练单个折
            fold_result = trainer.train_fold(
                train_folds=train_folds,
                test_fold=test_fold,
                num_epochs=num_epochs
            )
            
            fold_results.append(fold_result)
            
            print(f"折 {fold_idx+1} 结果: "
                  f"准确率={fold_result['final_val_acc']:.4f}, "
                  f"AUC={fold_result['final_val_auc']:.4f}")
        
        return fold_results
    
    def _calculate_average_metrics(self, fold_results):
        """计算平均指标"""
        accuracies = [r['final_val_acc'] for r in fold_results]
        auc_scores = [r['final_val_auc'] for r in fold_results]
        losses = [r['best_loss'] for r in fold_results]
        
        return {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_auc': float(np.mean(auc_scores)),
            'std_auc': float(np.std(auc_scores)),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
            'accuracy_95ci': [
                float(np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))),
                float(np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)))
            ],
            'auc_95ci': [
                float(np.mean(auc_scores) - 1.96 * np.std(auc_scores) / np.sqrt(len(auc_scores))),
                float(np.mean(auc_scores) + 1.96 * np.std(auc_scores) / np.sqrt(len(auc_scores)))
            ]
        }
    
    def _save_config_results(self, config_name, fold_results, avg_metrics):
        """保存单个配置的结果"""
        # 保存详细结果
        config_result = {
            'config_name': config_name,
            'average_metrics': avg_metrics,
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_loss': float(r['best_loss']),
                    'best_epoch': r['best_epoch'],
                    'final_val_acc': float(r['final_val_acc']),
                    'final_val_auc': float(r['final_val_auc'])
                } for r in fold_results
            ],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        result_file = os.path.join(self.metrics_dir, f"{config_name}_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(config_result, f, indent=2, ensure_ascii=False)
    
    def _generate_comparison_report(self):
        """生成对比分析报告"""
        print("\n" + "=" * 80)
        print(f"对比实验结果报告: {self.title_prefix}")
        print("=" * 80)
        
        report_data = []
        
        for config_name, results in self.comparison_results.items():
            metrics = results['average_metrics']
            config = results['config']
            
            print(f"\n配置: {config_name}")
            print(f"描述: {config['description']}")
            print(f"准确率: {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f}")
            print(f"AUC: {metrics['mean_auc']:.4f} ± {metrics['std_auc']:.4f}")
            print(f"损失: {metrics['mean_loss']:.4f} ± {metrics['std_loss']:.4f}")
            print(f"准确率95%置信区间: [{metrics['accuracy_95ci'][0]:.4f}, {metrics['accuracy_95ci'][1]:.4f}]")
            print(f"AUC 95%置信区间: [{metrics['auc_95ci'][0]:.4f}, {metrics['auc_95ci'][1]:.4f}]")
            
            # 收集数据用于表格
            report_data.append({
                'Configuration': config_name,
                'Description': config['description'],
                'Accuracy (mean±std)': f"{metrics['mean_accuracy']:.4f}±{metrics['std_accuracy']:.4f}",
                'AUC (mean±std)': f"{metrics['mean_auc']:.4f}±{metrics['std_auc']:.4f}",
                'Loss (mean±std)': f"{metrics['mean_loss']:.4f}±{metrics['std_loss']:.4f}",
                'Accuracy_Mean': metrics['mean_accuracy'],
                'AUC_Mean': metrics['mean_auc']
            })
        
        # 计算改进百分比（如果有两个配置）
        if len(self.comparison_results) == 2:
            config_names = list(self.comparison_results.keys())
            config1_name = config_names[0]
            config2_name = config_names[1]
            
            metrics1 = self.comparison_results[config1_name]['average_metrics']
            metrics2 = self.comparison_results[config2_name]['average_metrics']
            
            acc_improvement = ((metrics2['mean_accuracy'] - metrics1['mean_accuracy']) 
                              / metrics1['mean_accuracy'] * 100)
            auc_improvement = ((metrics2['mean_auc'] - metrics1['mean_auc']) 
                              / metrics1['mean_auc'] * 100)
            
            print(f"\n{'='*60}")
            print(f"性能改进分析")
            print(f"{'='*60}")
            print(f"{config2_name} 相对于 {config1_name}:")
            print(f"准确率改进: {acc_improvement:.2f}%")
            print(f"AUC改进: {auc_improvement:.2f}%")
            
            if acc_improvement > 0:
                print(f"✓ {config2_name} 在准确率上表现更好")
            else:
                print(f"✗ {config2_name} 在准确率上没有改进")
        
        # 保存报告为CSV
        df_report = pd.DataFrame(report_data)
        report_path = os.path.join(self.metrics_dir, "comparison_report.csv")
        df_report.to_csv(report_path, index=False, encoding='utf-8')
        
        # 保存详细报告为JSON
        detailed_report = {
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comparison_mode': self.comparison_mode,
            'comparison_results': {
                name: {
                    'config': results['config'],
                    'average_metrics': results['average_metrics']
                } for name, results in self.comparison_results.items()
            },
            'experiment_config': {
                'num_folds': 5,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'num_epochs': NUM_EPOCHS,
                'kernel_config': DEFAULT_KERNEL_CONFIG
            }
        }
        
        json_path = os.path.join(self.metrics_dir, "detailed_comparison_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
    
    def _create_comparison_visualizations(self):
        """创建简化的对比可视化图表"""
        print("\n生成对比可视化图表...")
    
        # 1. 准确率对比柱状图
        self._create_simple_accuracy_comparison_chart()
        
        # 2. AUC对比柱状图
        self._create_simple_auc_comparison_chart()
        
        # 3. 损失对比柱状图
        self._create_simple_loss_comparison_chart()
        
        print(f"可视化图表已保存到: {self.plots_dir}")
    
    def _create_simple_accuracy_comparison_chart(self):

        """Create accuracy comparison bar chart in English"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        config_names = []
        mean_accuracies = []
        std_accuracies = []
        colors = []
        descriptions = []
        
        for config_name, results in self.comparison_results.items():
            config = results['config']
            metrics = results['average_metrics']
            
            config_names.append(config_name)
            mean_accuracies.append(metrics['mean_accuracy'])
            std_accuracies.append(metrics['std_accuracy'])
            colors.append(config['color'])
            descriptions.append(config['description'])
        
        # Draw bar chart
        bars = ax.bar(config_names, mean_accuracies, yerr=std_accuracies,
                    capsize=10, color=colors, edgecolor='black', linewidth=1.5,
                    alpha=0.8)
        
        # Add value labels
        for bar, mean, std in zip(bars, mean_accuracies, std_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        
        # Set title based on comparison mode
        if self.comparison_mode == 'stream':
            ax.set_title('Accuracy Comparison: Stream1 Only vs Stream1+Stream2', 
                        fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Accuracy Comparison: No Augmentation vs With Augmentation', 
                        fontsize=14, fontweight='bold', pad=20)
        
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Add legend with descriptions
        from matplotlib.patches import Patch
        legend_elements = []
        for i, config_name in enumerate(config_names):
            legend_elements.append(
                Patch(facecolor=colors[i], edgecolor='black', alpha=0.8,
                    label=f"{config_name}: {descriptions[i]}")
            )
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, "accuracy_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
  
    
    def _create_simple_auc_comparison_chart(self):
        """Create AUC comparison bar chart in English"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        config_names = []
        mean_aucs = []
        std_aucs = []
        colors = []
        
        for config_name, results in self.comparison_results.items():
            config = results['config']
            metrics = results['average_metrics']
            
            config_names.append(config_name)
            mean_aucs.append(metrics['mean_auc'])
            std_aucs.append(metrics['std_auc'])
            colors.append(config['color'])
        
        # Draw bar chart
        bars = ax.bar(config_names, mean_aucs, yerr=std_aucs,
                    capsize=10, color=colors, edgecolor='black', linewidth=1.5,
                    alpha=0.8)
        
        # Add value labels
        for bar, mean, std in zip(bars, mean_aucs, std_aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        
        # Set title based on comparison mode
        if self.comparison_mode == 'stream':
            ax.set_title('AUC Comparison: Stream1 Only vs Stream1+Stream2', 
                        fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('AUC Comparison: No Augmentation vs With Augmentation', 
                        fontsize=14, fontweight='bold', pad=20)
        
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, "auc_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_simple_loss_comparison_chart(self):
        """创建损失对比柱状图"""
        fig, ax = plt.subplots(figsize=(8, 6))
    
        config_names = []
        mean_losses = []
        std_losses = []
        colors = []
        
        for config_name, results in self.comparison_results.items():
            config = results['config']
            metrics = results['average_metrics']
            
            config_names.append(config_name)
            mean_losses.append(metrics['mean_loss'])
            std_losses.append(metrics['std_loss'])
            colors.append(config['color'])
        
        # Draw bar chart
        bars = ax.bar(config_names, mean_losses, yerr=std_losses,
                    capsize=10, color=colors, edgecolor='black', linewidth=1.5,
                    alpha=0.8)
        
        # Add value labels
        for bar, mean, std in zip(bars, mean_losses, std_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        
        # Set title based on comparison mode
        if self.comparison_mode == 'stream':
            ax.set_title('Loss Comparison: Stream1 Only vs Stream1+Stream2', 
                        fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title('Loss Comparison: No Augmentation vs With Augmentation', 
                        fontsize=14, fontweight='bold', pad=20)
        
        # Calculate appropriate y-axis limit
        max_loss = max(mean_losses) + max(std_losses) if std_losses else max(mean_losses)
        ax.set_ylim([0, max_loss * 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save chart
        save_path = os.path.join(self.plots_dir, "loss_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_experiment_results(self):
        """保存完整的实验结果"""
        # 保存完整的对比结果
        complete_results = {
            'experiment_info': {
                'name': self.experiment_name,
                'mode': self.comparison_mode,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'base_path': self.base_path
            },
            'configurations': {
                name: {
                    'config': results['config'],
                    'average_metrics': results['average_metrics']
                } for name, results in self.comparison_results.items()
            },
            'detailed_results': self.comparison_results
        }
        
        complete_results_path = os.path.join(self.results_dir, "complete_comparison_results.json")
        with open(complete_results_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"完整实验结果已保存: {complete_results_path}")


# ==================== 主程序模块 ====================
class TrainingPipeline:
    """训练管道主控制器"""
    
    def __init__(self):
        self.base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../Dataset')
        )
    
    def run(self):
        """运行训练管道"""
        parser = argparse.ArgumentParser(description='ECG房颤检测训练脚本')
        parser.add_argument('--mode', type=str, default=EXPERIMENT_MODE,
                            choices=['search', 'train', 'full', 'compare'],
                            help='运行模式: search=超参数搜索, train=默认训练, full=最佳配置完整训练, compare=对比实验')
        parser.add_argument('--use_best_config', action='store_true',
                            help='使用之前搜索得到的最佳配置')
        parser.add_argument('--best_config_path', type=str,
                            default='/home/xusi/Logs/HyperparamSearch/latest_best_config.json',
                            help='最佳配置文件路径')
        parser.add_argument('--compare_mode', type=str, default=COMPARISON_MODE,
                            choices=['stream', 'augment'],
                            help='对比实验模式: stream=对比Stream配置, augment=对比数据增强')
        
        args = parser.parse_args()
        
        # 执行对应模式
        if args.mode == 'search':
            self._run_search_mode()
        elif args.mode == 'train':
            self._run_train_mode()
        elif args.mode == 'full' or args.use_best_config:
            self._run_full_mode(args)
        elif args.mode == 'compare':
            self._run_compare_mode(args)
        else:
            print(f"未知模式: {args.mode}")
    
    def _run_search_mode(self):
        """运行超参数搜索模式"""
        print("\n模式: 超参数搜索")
        print(f"FIXED_LENGTH = {FIXED_LENGTH} ({FIXED_LENGTH / 300:.1f}秒 @300Hz)")
        
        # 这里需要导入HyperparameterSearcher类（假设它在同一个文件中）
        searcher = HyperparameterSearcher(self.base_path)
        search_results, search_dir = searcher.search(
            num_folds=3,
            num_epochs_search=15
        )
        
        # 更新最新最佳配置
        latest_best_config = os.path.join(search_dir, 'best_config.json')
        if os.path.exists(latest_best_config):
            import shutil
            shutil.copy2(latest_best_config, 
                        '/home/xusi/Logs/HyperparamSearch/latest_best_config.json')
            print(f"\n已更新最新最佳配置: /home/xusi/Logs/HyperparamSearch/latest_best_config.json")
    
    def _run_train_mode(self):
        """运行默认训练模式"""
        print("\n模式: 使用默认配置训练")
        print(f"卷积核: MS-CNN(3,7), 批大小: {BATCH_SIZE}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_base_dir = os.path.join(
            "/home/xusi/Logs/DefaultTraining", f"Run_{timestamp}"
        )
        os.makedirs(experiment_base_dir, exist_ok=True)
        
        # 使用默认配置
        default_kernel_config = DEFAULT_KERNEL_CONFIG
        
        # 5折交叉验证
        K = 5
        fold_results = []
        
        for k in range(K):
            test_fold = k
            train_folds = [i for i in range(5) if i != k]
            
            trainer = ModelTrainer(
                base_path=self.base_path,
                kernel_config=default_kernel_config,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                use_stream2=USE_STREAM2_SETTING,
                augment=AUGMENT_SETTING,
                experiment_base_dir=experiment_base_dir
            )
            
            result = trainer.train_fold(
                train_folds=train_folds,
                test_fold=test_fold,
                num_epochs=NUM_EPOCHS
            )
            
            fold_results.append(result)
            print(f"折 {k + 1} 完成: "
                  f"准确率={result['final_val_acc']:.4f}, "
                  f"AUC={result['final_val_auc']:.4f}")
        
        # 计算平均结果
        avg_acc = np.mean([r['final_val_acc'] for r in fold_results])
        avg_auc = np.mean([r['final_val_auc'] for r in fold_results])
        
        print(f"\n默认配置训练完成!")
        print(f"平均准确率: {avg_acc:.4f}")
        print(f"平均AUC: {avg_auc:.4f}")
    
    def _run_full_mode(self, args):
        """运行完整训练模式"""
        print("\n模式: 使用最佳配置进行完整训练")

        best_config_path = "/home/xusi/Logs/HyperparamSearch/latest_best_config.json"

        # 加载最佳配置
        if os.path.exists(best_config_path):
            with open(best_config_path, 'r') as f:
                best_config_data = json.load(f)
            
            best_config_name = best_config_data['best_config_name']
            best_config_details = best_config_data['best_config_data']
            
            print(f"加载最佳配置: {best_config_name}")
            print(f"搜索时间: {best_config_data.get('search_timestamp', '未知')}")
            
            # 使用最佳配置进行完整训练
            avg_metrics, final_results,simplified_path = CompleteTrainer.train_with_best_config(
                self.base_path,
                best_config_details,
                num_epochs=NUM_EPOCHS
            )
        
            # 打印与LLM评估兼容的结果
            print("\n" + "=" * 80)
            print("CNN模型评估结果（LLM兼容格式）:")
            print("=" * 80)
            print(f"平均准确率: {final_results['accuracy']:.4f}")
            print(f"平均精确率: {final_results['precision']:.4f}")
            print(f"平均召回率: {final_results['recall']:.4f}")
            print(f"平均F1分数: {final_results['f1']:.4f}")
            print(f"平均AUC: {final_results['auc']:.4f}")

            print(f"\n💡 将此文件路径用于LLM对比:")
            print(f"   CNN_BASELINE_RESULTS = '{simplified_path}'")
        
            print("\n交叉验证结果:")
            for fold_name, metrics in final_results['cross_validation_results'].items():
                print(f"  {fold_name}: 准确率={metrics['accuracy']:.4f}, "
                    f"精确率={metrics['precision']:.4f}, "
                    f"召回率={metrics['recall']:.4f}, "
                    f"F1={metrics['f1']:.4f}, "
                    f"AUC={metrics['auc']:.4f}")
        else:
            print(f"错误: 找不到最佳配置文件 {best_config_path}")
            print("请先运行超参数搜索模式: python Train_Process.py --mode search")

    def _run_compare_mode(self, args):
        """运行对比实验模式"""
        print("\n模式: 对比实验")
        
        # 创建对比实验
        comparison_experiment = ComparisonExperiment(
            base_path=self.base_path,
            comparison_mode=args.compare_mode
        )
        
        # 运行对比实验
        results = comparison_experiment.run_comparison(
            num_folds=5,
            num_epochs=NUM_EPOCHS
        )
        
        print(f"\n对比实验完成！")
        print(f"对比模式: {args.compare_mode}")
        print(f"结果目录: {comparison_experiment.results_dir}")
        
        # 输出主要发现
        print("\n" + "="*60)
        print("主要发现:")
        print("="*60)
        
        config_names = list(results.keys())
        if len(config_names) >= 2:
            config1 = config_names[0]
            config2 = config_names[1]
            
            metrics1 = results[config1]['average_metrics']
            metrics2 = results[config2]['average_metrics']
            
            acc_diff = metrics2['mean_accuracy'] - metrics1['mean_accuracy']
            auc_diff = metrics2['mean_auc'] - metrics1['mean_auc']
            
            print(f"1. {config2} 相对于 {config1}:")
            print(f"   准确率差异: {acc_diff:+.4f} ({acc_diff/metrics1['mean_accuracy']*100:+.1f}%)")
            print(f"   AUC差异: {auc_diff:+.4f} ({auc_diff/metrics1['mean_auc']*100:+.1f}%)")
            
            if acc_diff > 0 and auc_diff > 0:
                print(f"   ✓ {config2} 在两项指标上均表现更好")
            elif acc_diff > 0:
                print(f"   ⚠ {config2} 在准确率上表现更好，但AUC略差")
            elif auc_diff > 0:
                print(f"   ⚠ {config2} 在AUC上表现更好，但准确率略差")
            else:
                print(f"   ✗ {config2} 在两项指标上均未表现出优势")
        
        print("\n详细对比结果请查看生成的图表和报告文件。")


# ==================== 程序入口 ====================
if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run()
    print("\n训练完成!")