"""
ECG房颤检测训练主程序 - 修复版（添加模型保存功能）
"""

# ==================== 解决导入问题 ====================
import sys
import os

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
print(f"当前文件: {current_file}")

# 根据你的项目结构计算项目根目录
# 当前文件: EE5046_Projects/src/task1_ecg_analysis/training/TrainProcess.py
# 项目根目录应该是: EE5046_Projects (上三级目录)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)
print(f"项目根目录已添加到Python路径: {project_root}")


# ==================== 导入部分 ====================
import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, 
                           precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm

# 自定义模块
from src.common.Config import (AUGMENT_SETTING, BATCH_SIZE, EARLY_STOP_PATIENCE,
                    EXPERIMENT_MODE, FIXED_LENGTH, INPUT_CHANNELS,
                    LEARNING_RATE, MIN_DELTA, NUM_EPOCHS, OUTPUT_CLASSES,
                    USE_STREAM2_SETTING, COMPARISON_MODE,
                    STREAM_COMPARISON_CONFIGS, AUGMENTATION_COMPARISON_CONFIGS,
                    DEFAULT_KERNEL_CONFIG)
from src.task1_ecg_analysis.data.DataManager import DataManager
from src.task1_ecg_analysis.data.FoldDataset import FoldDataset
from src.task1_ecg_analysis.visualization.TrainingVisualizer import TrainingVisualizer
from TrainModel import Mscnn


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 新增：综合评分计算器 ====================
class CompositeScoreCalculator:
    """计算综合评分，考虑多个指标"""
    
    # 默认权重配置
    DEFAULT_WEIGHTS = {
        'accuracy': 0.35,
        'auc': 0.30,
        'f1': 0.25,
        'stability': 0.10  # 稳定性分数（基于方差）
    }
    
    @staticmethod
    def calculate_composite_score(metrics, weights=None, fold_results=None):
        """
        计算综合评分
        Args:
            metrics: 包含单个评估指标的字典，如 {'accuracy': 0.85, 'auc': 0.90, 'f1': 0.82}
            weights: 各指标的权重，默认为DEFAULT_WEIGHTS
            fold_results: 交叉验证的详细结果（用于计算稳定性）
        Returns:
            composite_score: 综合评分
            breakdown: 各指标贡献明细
        """
        if weights is None:
            weights = CompositeScoreCalculator.DEFAULT_WEIGHTS
        
        # 确保所有需要的指标都存在
        required_metrics = ['accuracy', 'auc', 'f1']
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"缺少必要指标: {metric}")
        
        # 计算稳定性分数（如果有fold_results）
        stability_score = 1.0  # 默认值
        if fold_results is not None and len(fold_results) > 1:
            # 提取各折的准确率
            accuracies = [fold['best_val_acc'] for fold in fold_results]
            # 稳定性分数 = 1 - 变异系数（归一化方差）
            cv = np.std(accuracies) / (np.mean(accuracies) + 1e-8)  # 变异系数
            stability_score = max(0, 1 - cv)  # 确保在0-1之间
        
        # 计算加权综合评分
        composite_score = 0
        breakdown = {}
        
        for metric, weight in weights.items():
            if metric == 'stability':
                score = stability_score
            elif metric in metrics:
                score = metrics[metric]
            else:
                score = 0.5  # 默认值
            
            contribution = score * weight
            composite_score += contribution
            breakdown[metric] = {
                'score': score,
                'weight': weight,
                'contribution': contribution
            }
        
        return composite_score, breakdown
    
    @staticmethod
    def normalize_metrics(metrics, ideal_values=None):
        """归一化指标到0-1范围"""
        if ideal_values is None:
            ideal_values = {
                'accuracy': 1.0,
                'auc': 1.0,
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0
            }
        
        normalized = {}
        for metric, value in metrics.items():
            if metric in ideal_values:
                # 简单线性归一化
                normalized[metric] = min(value / ideal_values[metric], 1.0)
            else:
                normalized[metric] = value
        
        return normalized


# ==================== 模型文件管理器（保持不变） ====================
class ModelFileManager:
    """管理模型文件的保存和加载"""
    
    @staticmethod
    def create_experiment_dir(base_dir, experiment_type, config_name=""):
        """创建实验目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config_name:
            dir_name = f"{experiment_type}_{config_name}_{timestamp}"
        else:
            dir_name = f"{experiment_type}_{timestamp}"
        
        exp_dir = os.path.join(base_dir, dir_name)
        subdirs = [
            "models",           # 保存模型文件
            "logs",            # 训练日志
            "configs",         # 配置文件
            "metrics",         # 性能指标
            "visualizations"   # 可视化图表
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
        
        print(f"📁 创建实验目录: {exp_dir}")
        return exp_dir
    
    @staticmethod
    def save_model(model, save_path, metadata=None):
        """保存模型和元数据"""
        model_state = {
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'config', {}),
            'save_time': datetime.now().isoformat()
        }
        
        if metadata:
            model_state.update(metadata)
        
        torch.save(model_state, save_path)
        print(f"💾 模型已保存: {save_path}")
    
    @staticmethod
    def generate_model_name(config, fold=None, epoch=None, metric=None, composite_score=None):
        """生成模型文件名"""
        parts = []
        
        # 基础信息
        kernel_config = config.get('kernel_config', {})
        if kernel_config.get('name'):
            parts.append(kernel_config['name'])
        else:
            parts.append(f"K{kernel_config.get('stream1_kernel', '?')}")
            parts.append(f"S2{kernel_config.get('stream2_first_kernel', '?')}")
        
        # 训练配置
        parts.append(f"BS{config.get('batch_size', BATCH_SIZE)}")
        parts.append(f"LR{config.get('lr', LEARNING_RATE)}")
        
        # 训练状态
        if fold is not None:
            parts.append(f"F{fold}")
        if epoch is not None:
            parts.append(f"E{epoch}")
        
        # 性能指标
        if composite_score is not None:
            parts.append(f"CS{composite_score:.4f}".replace('.', 'p'))
        elif metric is not None:
            parts.append(f"A{metric:.4f}".replace('.', 'p'))
        
        # 时间戳
        parts.append(datetime.now().strftime("%m%d%H%M"))
        
        return "_".join(parts) + ".pth"
    
    @staticmethod
    def save_metrics(metrics, save_path):
        """保存性能指标"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"📊 指标已保存: {save_path}")
    
    @staticmethod
    def save_config(config, save_path):
        """保存配置"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"⚙️ 配置已保存: {save_path}")


# ==================== 改进的模型训练器模块 ====================
class ModelTrainer:
    """模型训练器，负责单次训练验证过程"""
    
    def __init__(self, base_path, kernel_config, batch_size, lr, 
                 use_stream2, augment, experiment_dir, config_name=None,
                 composite_weights=None):
        self.base_path = base_path
        self.kernel_config = kernel_config
        self.batch_size = batch_size
        self.lr = lr
        self.use_stream2 = use_stream2
        self.augment = augment
        self.experiment_dir = experiment_dir
        self.config_name = config_name
        self.file_manager = ModelFileManager()
        self.visualizer = TrainingVisualizer()
        
        # 综合评分权重
        self.composite_weights = composite_weights or CompositeScoreCalculator.DEFAULT_WEIGHTS
        

        print(f"初始化DataManager，数据集路径: {self.base_path}")
        if not os.path.exists(self.base_path):
            print(f"❌ 数据集路径不存在: {self.base_path}")
            print("请检查Dataset目录是否正确放置")
            sys.exit(1)
        
        self.data_manager = DataManager(base_path)
        # 数据管理器
        
        # 训练状态
        self.best_val_acc = 0
        self.best_val_auc = 0
        self.best_val_f1 = 0
        self.best_composite_score = 0
        self.best_model_state = None
        self.best_epoch = 0
        self.early_stop_counter = 0

    def cross_validate_on_train_set(self, train_cv_indices, num_epochs, k_folds=5, save_models=True):
        """
        在训练集上进行K折交叉验证
        """
        print(f"\n{'='*60}")
        print(f"在训练集上进行 {k_folds} 折交叉验证")
        print(f"{'='*60}")
        
        # 创建K折划分
        kfold_splits = self.data_manager.create_kfold_splits(train_cv_indices, k_folds)
        if not kfold_splits:
            print("错误: 无法创建K折划分")
            return {}, []
        
        fold_results = []
        fold_models = []

        # 使用进度条显示折的训练进度
        fold_pbar = self.visualizer.create_progress_bar(k_folds, "交叉验证进度")
        
        # 训练每一折
        for fold_idx, (train_data, val_data) in enumerate(kfold_splits):
            print(f"\n--- 第 {fold_idx + 1}/{k_folds} 折 ---")
            
            # 训练当前折
            fold_result, fold_model = self._train_single_fold(
                train_data, val_data, fold_idx, num_epochs, save_models
            )
            
            fold_results.append(fold_result)
            fold_models.append(fold_model)

            # 更新进度条
            fold_pbar.update(1)
            
            # 使用综合评分进行评价
            metrics = {
                'accuracy': fold_result['best_val_acc'],
                'auc': fold_result['best_val_auc'],
                'f1': fold_result['best_val_f1']
            }
            composite_score, breakdown = CompositeScoreCalculator.calculate_composite_score(metrics)
            
            print(f"折 {fold_idx + 1} 结果:")
            print(f"  验证准确率: {fold_result['best_val_acc']:.4f}")
            print(f"  AUC: {fold_result['best_val_auc']:.4f}")
            print(f"  F1分数: {fold_result['best_val_f1']:.4f}")
            print(f"  综合评分: {composite_score:.4f}")

        fold_pbar.close()
        
        # 计算平均指标
        avg_metrics = self._compute_average_metrics(fold_results)
        
        # 计算平均综合评分
        all_metrics = [{
            'accuracy': r['best_val_acc'],
            'auc': r['best_val_auc'],
            'f1': r['best_val_f1']
        } for r in fold_results]
        
        avg_composite_score = np.mean([
            CompositeScoreCalculator.calculate_composite_score(m)[0] for m in all_metrics
        ])
        avg_metrics['avg_composite_score'] = float(avg_composite_score)
        
        # 保存交叉验证结果
        if self.experiment_dir:
            cv_results = {
                'avg_metrics': avg_metrics,
                'fold_results': fold_results,
                'config': self._get_config_dict(),
                'timestamp': datetime.now().isoformat(),
                'composite_score_weights': self.composite_weights
            }
            
            results_path = os.path.join(self.experiment_dir, "metrics", "cross_validation_results.json")
            self.file_manager.save_metrics(cv_results, results_path)
            
            # 保存最佳模型（基于综合评分最高的折）
            if fold_models and save_models:
                # 计算每折的综合评分
                composite_scores = []
                for r in fold_results:
                    metrics = {
                        'accuracy': r['best_val_acc'],
                        'auc': r['best_val_auc'],
                        'f1': r['best_val_f1']
                    }
                    score, _ = CompositeScoreCalculator.calculate_composite_score(metrics)
                    composite_scores.append(score)
                
                # 选择综合评分最高的折
                best_fold_idx = np.argmax(composite_scores)
                best_model = fold_models[best_fold_idx]
                best_fold_result = fold_results[best_fold_idx]
                best_composite_score = composite_scores[best_fold_idx]
                
                model_name = self.file_manager.generate_model_name(
                    self._get_config_dict(),
                    fold=best_fold_idx,
                    epoch=best_fold_result['best_epoch'],
                    composite_score=best_composite_score
                )
                
                model_path = os.path.join(self.experiment_dir, "models", model_name)
                metadata = {
                    'fold': best_fold_idx,
                    'val_acc': best_fold_result['best_val_acc'],
                    'val_auc': best_fold_result['best_val_auc'],
                    'val_f1': best_fold_result['best_val_f1'],
                    'composite_score': best_composite_score,
                    'epoch': best_fold_result['best_epoch']
                }
                self.file_manager.save_model(best_model, model_path, metadata)
        
        print(f"\n{'='*60}")
        print(f"{k_folds}折交叉验证结果汇总:")
        print(f"{'='*60}")
        print(f"平均验证准确率: {avg_metrics['avg_val_acc']:.4f} ± {avg_metrics['std_val_acc']:.4f}")
        print(f"平均AUC: {avg_metrics['avg_val_auc']:.4f} ± {avg_metrics['std_val_auc']:.4f}")
        print(f"平均F1分数: {avg_metrics['avg_val_f1']:.4f} ± {avg_metrics['std_val_f1']:.4f}")
        print(f"平均综合评分: {avg_metrics['avg_composite_score']:.4f}")

        # 可视化交叉验证结果
        if self.experiment_dir:
            self._visualize_cv_results(fold_results, avg_metrics)
        
        return avg_metrics, fold_results
    
    def train_final_model(self, train_cv_indices, num_epochs, save_model=True, val_ratio=0.2):
        """
        使用全部训练集训练最终模型，包含验证集
        Args:
            train_cv_indices: 训练集CV索引
            num_epochs: 训练轮数
            save_model: 是否保存模型
            val_ratio: 从训练集中划分验证集的比例
        """
        print(f"\n使用全部训练集训练最终模型")
        print(f"训练集: CV{', '.join(map(str, train_cv_indices))}")
        print(f"验证集比例: {val_ratio:.1%}")
        
        # 加载训练集数据
        train_data = self.data_manager.load_cv_files(train_cv_indices)
        if len(train_data) == 0:
            print("错误: 训练集数据为空")
            return None, {}
        
        # 划分训练集和验证集
        total_size = len(train_data)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        # 随机划分
        torch.manual_seed(42)  # 确保可重复性
        train_subset, val_subset = random_split(train_data, [train_size, val_size])
        
        print(f"训练样本数: {train_size}, 验证样本数: {val_size}")
        
        # 创建数据集
        train_dataset = FoldDataset(
            list(train_subset), self.base_path, is_train=True, augment=self.augment
        )
        val_dataset = FoldDataset(
            list(val_subset), self.base_path, is_train=False, augment=False
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=0
        )
        
        # 初始化模型
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        # 损失函数和优化器
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 训练循环
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        val_f1s = []
        
        best_train_acc = 0
        best_val_acc = 0
        best_val_auc = 0
        best_val_f1 = 0
        best_composite_score = 0
        best_model_state = None
        best_epoch = 0
        early_stop_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for x, y in train_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 收集预测结果
                preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
                train_preds.extend(preds.flatten())
                train_labels.extend(y.detach().cpu().numpy().flatten())
            
            # 计算训练指标
            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc, val_auc, val_labels, val_probs, val_precision, val_recall, val_f1 = self._validate_model(
                model, criterion, val_loader
            )
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_aucs.append(val_auc)
            val_f1s.append(val_f1)
            
            # 计算综合评分
            val_metrics = {
                'accuracy': val_acc,
                'auc': val_auc,
                'f1': val_f1
            }
            composite_score, breakdown = CompositeScoreCalculator.calculate_composite_score(val_metrics)
            
            # 检查是否是最佳模型
            is_best = False
            if composite_score > best_composite_score + MIN_DELTA:
                is_best = True
                best_composite_score = composite_score
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0  # 重置早停计数器
            else:
                early_stop_counter += 1
            
            # 更新训练最佳准确率
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            
            # 打印进度
            if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
                print(f"  Epoch {epoch}/{num_epochs}:")
                print(f"    训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"    验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
                print(f"    综合评分: {composite_score:.4f}")
            
            # 早停检查
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"  ⚠️ 早停触发于epoch {epoch}，连续{EARLY_STOP_PATIENCE}个epoch验证集无显著提升")
                break
        
        print(f"  最佳验证综合评分: {best_composite_score:.4f} (Epoch {best_epoch})")
        
        # 加载最佳模型状态
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 保存最终模型
        if save_model and self.experiment_dir:
            model_name = self.file_manager.generate_model_name(
                self._get_config_dict(),
                epoch=best_epoch,
                composite_score=best_composite_score
            )
            
            model_path = os.path.join(self.experiment_dir, "models", "final_" + model_name)
            metadata = {
                'best_train_acc': best_train_acc,
                'best_val_acc': best_val_acc,
                'best_val_auc': best_val_auc,
                'best_val_f1': best_val_f1,
                'best_composite_score': best_composite_score,
                'avg_train_loss': np.mean(train_losses),
                'num_epochs': epoch,  # 实际训练的epoch数（可能因早停而小于num_epochs）
                'best_epoch': best_epoch,
                'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE
            }
            self.file_manager.save_model(model, model_path, metadata)
        
        train_metrics = {
            'final_train_loss': train_losses[-1],
            'final_train_acc': train_accs[-1],
            'best_train_acc': best_train_acc,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'best_val_f1': best_val_f1,
            'best_composite_score': best_composite_score,
            'avg_train_loss': np.mean(train_losses),
            'avg_val_loss': np.mean(val_losses),
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE
        }
        
        # 保存训练指标
        if self.experiment_dir:
            metrics_path = os.path.join(self.experiment_dir, "metrics", "final_training_metrics.json")
            self.file_manager.save_metrics(train_metrics, metrics_path)
        
        return model, train_metrics
    
    def _train_single_fold(self, train_data, val_data, fold_idx, num_epochs, save_model=True, min_epochs=10):
        """训练单个折，包含早停机制"""
        # 创建数据集
        train_dataset = FoldDataset(
            train_data, self.base_path, is_train=True, augment=self.augment
        )
        val_dataset = FoldDataset(
            val_data, self.base_path, is_train=False, augment=False
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
        
        # 损失函数和优化器
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 训练状态
        best_val_acc = 0
        best_val_auc = 0
        best_val_f1 = 0
        best_composite_score = 0
        best_epoch = 0
        best_model_state = None
        early_stop_counter = 0
        
        # 用于可视化的历史记录
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        val_f1s = []

        # 创建epoch进度条
        epoch_pbar = tqdm(
            range(1, num_epochs + 1),
            desc=f"折 {fold_idx+1} 训练进度",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=1.0
        )

        # 训练循环
        for epoch in epoch_pbar:
            # 训练
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            # 创建批次进度条 - 使用不同的position
            batch_pbar = tqdm(
                enumerate(train_loader, 1),
                total=len(train_loader),
                desc="批次训练",
                position=1,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
                maxinterval=1.0,
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
            )
            for batch_idx, (x, y) in batch_pbar:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 收集预测结果
                preds = (outputs.detach().cpu().numpy() >= 0.5).astype(int)
                train_preds.extend(preds.flatten())
                train_labels.extend(y.detach().cpu().numpy().flatten())

                # 更新批次进度条 - 使用更详细的格式
                avg_loss_so_far = train_loss / batch_idx
                batch_pbar.set_postfix({
                    'batch': f'{batch_idx}/{len(train_loader)}',
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss_so_far:.4f}'
                })

            batch_pbar.close()
            
            # 计算训练准确率
            train_acc = accuracy_score(train_labels, train_preds)
            avg_train_loss = train_loss / len(train_loader)

            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            # 验证 - 修复这里的解包问题
            val_metrics = self._validate_model(model, criterion, val_loader)
            val_loss, val_acc, val_auc, _, _, val_precision, val_recall, val_f1 = val_metrics

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_aucs.append(val_auc)
            val_f1s.append(val_f1)
            
            # 计算综合评分
            val_metrics_dict = {
                'accuracy': val_acc,
                'auc': val_auc,
                'f1': val_f1
            }
            composite_score, breakdown = CompositeScoreCalculator.calculate_composite_score(val_metrics_dict)
            
            # 检查是否是最佳模型
            is_best = False
            if composite_score > best_composite_score + MIN_DELTA:
                is_best = True
                best_composite_score = composite_score
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_val_f1 = val_f1
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0  # 重置早停计数器
            else:
                early_stop_counter += 1

            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'best': '★' if is_best else ''
            })
            epoch_pbar.update(1)
            
            # 早停检查（至少训练min_epochs个epoch）
            if epoch >= min_epochs and early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"    ⏹️ 早停触发于epoch {epoch}")
                break

        epoch_pbar.close()
        
        print(f"  最佳验证综合评分: {best_composite_score:.4f} (Epoch {best_epoch})")
        
        # 加载最佳模型状态
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 保存当前折的最佳模型
        if save_model and self.experiment_dir:
            model_name = self.file_manager.generate_model_name(
                self._get_config_dict(),
                fold=fold_idx,
                epoch=best_epoch,
                composite_score=best_composite_score
            )
            
            model_path = os.path.join(self.experiment_dir, "models", f"fold{fold_idx}_" + model_name)
            metadata = {
                'fold': fold_idx,
                'val_acc': best_val_acc,
                'val_auc': best_val_auc,
                'val_f1': best_val_f1,
                'composite_score': best_composite_score,
                'epoch': best_epoch,
                'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE
            }
            self.file_manager.save_model(model, model_path, metadata)
        
        fold_result = {
            'fold': fold_idx,
            'best_val_acc': float(best_val_acc),
            'best_val_auc': float(best_val_auc),
            'best_val_f1': float(best_val_f1),
            'best_composite_score': float(best_composite_score),
            'best_epoch': best_epoch,
            'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE,
            'total_epochs': epoch
        }
        
        return fold_result, model
    
    def _validate_model(self, model, criterion, val_loader):
        """验证模型"""
        model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                probs = model(x)
                loss = criterion(probs, y)
                running_loss += loss.item()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())
        
        # 计算指标
        avg_loss = running_loss / len(val_loader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        preds = (all_probs >= 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        try:
            precision = precision_score(all_labels, preds, average='binary', zero_division=0)
            recall = recall_score(all_labels, preds, average='binary', zero_division=0)
            f1 = f1_score(all_labels, preds, average='binary', zero_division=0)
        except:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        model.train()
        return avg_loss, acc, auc, all_labels, all_probs, precision, recall, f1
    
    def _compute_average_metrics(self, fold_results):
        if not fold_results:
            return {
                'avg_val_acc': 0.0,
                'std_val_acc': 0.0,
                'avg_val_auc': 0.0,
                'std_val_auc': 0.0,
                'avg_val_f1': 0.0,
                'std_val_f1': 0.0,
                'num_folds': 0
            }
    
        val_accs = [r['best_val_acc'] for r in fold_results]
        val_aucs = [r['best_val_auc'] for r in fold_results]
        val_f1s = [r['best_val_f1'] for r in fold_results]
        
        return {
            'avg_val_acc': float(np.mean(val_accs)),
            'std_val_acc': float(np.std(val_accs)),
            'avg_val_auc': float(np.mean(val_aucs)),
            'std_val_auc': float(np.std(val_aucs)),
            'avg_val_f1': float(np.mean(val_f1s)),
            'std_val_f1': float(np.std(val_f1s)),
            'num_folds': len(fold_results)
        }
    
    def evaluate_on_test_set(self, test_cv_indices, model, save_results=True):
        """在测试集上评估模型"""
        print(f"\n在测试集上评估模型")
        print(f"测试集: CV{', '.join(map(str, test_cv_indices))}")
        
        # 加载测试集数据
        test_data = self.data_manager.load_cv_files(test_cv_indices)
        if len(test_data) == 0:
            print("错误: 测试集数据为空")
            return {}
        
        # 创建数据集
        test_dataset = FoldDataset(
            test_data, self.base_path, is_train=False, augment=False
        )
        
        # 创建数据加载器
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        # 评估
        criterion = torch.nn.BCELoss()
        test_loss, test_acc, test_auc, _, _, test_precision, test_recall, test_f1 = self._validate_model(
            model, criterion, test_loader
        )
        
        # 计算综合评分
        test_metrics = {
            'accuracy': test_acc,
            'auc': test_auc,
            'f1': test_f1
        }
        test_composite_score, breakdown = CompositeScoreCalculator.calculate_composite_score(test_metrics)
        
        test_results = {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_composite_score': test_composite_score,
            'test_loss': test_loss,
            'evaluation_time': datetime.now().isoformat(),
            'score_breakdown': breakdown
        }
        
        print(f"测试集结果:")
        print(f"  准确率: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"  综合评分: {test_composite_score:.4f}")
        print(f"  损失: {test_loss:.4f}")
        
        # 保存评估结果
        if save_results and self.experiment_dir:
            results_path = os.path.join(self.experiment_dir, "metrics", "test_evaluation.json")
            self.file_manager.save_metrics(test_results, results_path)
        
        return test_results
    
    def _visualize_cv_results(self, fold_results, avg_metrics):
        """可视化交叉验证结果"""
        if not self.experiment_dir:
            return
        
        # 绘制各折性能对比
        fold_accs = [r['best_val_acc'] for r in fold_results]
        fold_aucs = [r['best_val_auc'] for r in fold_results]
        fold_f1s = [r['best_val_f1'] for r in fold_results]
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 各折准确率
        axes[0].bar(range(1, len(fold_accs) + 1), fold_accs, color='skyblue', edgecolor='black')
        axes[0].axhline(y=avg_metrics['avg_val_acc'], color='red', linestyle='--', 
                       label=f'平均值: {avg_metrics["avg_val_acc"]:.4f}')
        axes[0].set_title('各折验证准确率', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('折数', fontsize=12)
        axes[0].set_ylabel('准确率', fontsize=12)
        axes[0].set_xticks(range(1, len(fold_accs) + 1))
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 各折AUC
        axes[1].bar(range(1, len(fold_aucs) + 1), fold_aucs, color='lightgreen', edgecolor='black')
        axes[1].axhline(y=avg_metrics['avg_val_auc'], color='red', linestyle='--', 
                       label=f'平均值: {avg_metrics["avg_val_auc"]:.4f}')
        axes[1].set_title('各折AUC分数', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('折数', fontsize=12)
        axes[1].set_ylabel('AUC', fontsize=12)
        axes[1].set_xticks(range(1, len(fold_aucs) + 1))
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 各折F1分数
        axes[2].bar(range(1, len(fold_f1s) + 1), fold_f1s, color='lightcoral', edgecolor='black')
        axes[2].axhline(y=avg_metrics['avg_val_f1'], color='red', linestyle='--', 
                       label=f'平均值: {avg_metrics["avg_val_f1"]:.4f}')
        axes[2].set_title('各折F1分数', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('折数', fontsize=12)
        axes[2].set_ylabel('F1分数', fontsize=12)
        axes[2].set_xticks(range(1, len(fold_f1s) + 1))
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        vis_path = os.path.join(self.experiment_dir, "visualizations", "cv_results_comparison.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 交叉验证结果可视化已保存: {vis_path}")

    def _get_config_dict(self):
        """获取当前训练配置"""
        return {
            'kernel_config': self.kernel_config,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'use_stream2': self.use_stream2,
            'augment': self.augment,
            'config_name': self.config_name,
            'composite_weights': self.composite_weights
        }


# ==================== 改进的超参数搜索模块 ====================
class HyperparameterSearcher:
    """超参数搜索器，使用综合评分选择最佳配置"""
    
    # 搜索配置
    KERNEL_CONFIGS = [
        {'name': 'MS-CNN(3,9)', 'stream1_kernel': 3, 'stream2_first_kernel': 9},
        {'name': 'MS-CNN(3,7)', 'stream1_kernel': 3, 'stream2_first_kernel': 7},
        {'name': 'MS-CNN(3,5)', 'stream1_kernel': 3, 'stream2_first_kernel': 5},
        {'name': 'MS-CNN(3,3)', 'stream1_kernel': 3, 'stream2_first_kernel': 3},
    ]
    
    BATCH_SIZES = [32, 64, 128]
    
    def __init__(self, base_path, composite_weights=None):
        self.base_path = base_path
        self.file_manager = ModelFileManager()
        self.composite_weights = composite_weights
        self.visualizer = TrainingVisualizer()
    
    def search(self, num_epochs_search=20):
        """执行超参数搜索，使用综合评分评估配置"""
        print("=" * 80)
        print("超参数搜索模式（使用综合评分和早停机制）")
        print("=" * 80)
        
        # 创建搜索目录
        search_dir = self.file_manager.create_experiment_dir(
            "/home/xusi/EE5046_Projects/Task1_Results/HyperparamSearch",
            "HyperparamSearch"
        )
        
        # 保存搜索配置
        search_config = {
            'kernel_configs': self.KERNEL_CONFIGS,
            'batch_sizes': self.BATCH_SIZES,
            'num_epochs': num_epochs_search,
            'learning_rate': LEARNING_RATE,
            'use_stream2': USE_STREAM2_SETTING,
            'augment': AUGMENT_SETTING,
            'composite_weights': self.composite_weights,
            'search_time': datetime.now().isoformat()
        }
        
        config_path = os.path.join(search_dir, "configs", "search_config.json")
        self.file_manager.save_config(search_config, config_path)
        
        best_composite_score = 0
        best_config = None
        all_results = {}
        
        # 计算总配置数
        total_configs = len(self.KERNEL_CONFIGS) * len(self.BATCH_SIZES)
        
        # 创建总体搜索进度条
        config_pbar = self.visualizer.create_progress_bar(total_configs, "超参数搜索进度")
        config_idx = 0

        # 遍历所有超参数组合
        for kernel_config in self.KERNEL_CONFIGS:
            for batch_size in self.BATCH_SIZES:
                config_name = f"{kernel_config['name']}_BS{batch_size}"
                print(f"\n{'='*50}")
                print(f"测试配置: {config_name}")
                print(f"{'='*50}")
                
                # 为每个配置创建子目录
                config_dir = os.path.join(search_dir, "configs", config_name)
                subdirs = [
                    "models",           # 保存模型文件
                    "logs",            # 训练日志
                    "configs",         # 配置文件
                    "metrics",         # 性能指标
                    "visualizations"   # 可视化图表
                ]
                for subdir in subdirs:
                    os.makedirs(os.path.join(config_dir, subdir), exist_ok=True)
                
                # 创建训练器
                trainer = ModelTrainer(
                    base_path=self.base_path,
                    kernel_config=kernel_config,
                    batch_size=batch_size,
                    lr=LEARNING_RATE,
                    use_stream2=USE_STREAM2_SETTING,
                    augment=AUGMENT_SETTING,
                    experiment_dir=config_dir,
                    config_name=config_name,
                    composite_weights=self.composite_weights
                )
                
                # 在训练集上进行5折交叉验证
                train_indices = [0, 1, 2, 3]
                avg_metrics, fold_results = trainer.cross_validate_on_train_set(
                    train_indices, num_epochs_search, k_folds=5
                )

                avg_composite_score = avg_metrics.get('avg_composite_score', 0)
                
                # 更新进度条信息
                config_pbar.set_postfix({
                    'config': config_name,
                    'score': f"{avg_composite_score:.4f}",
                    'best': '★' if avg_composite_score > best_composite_score else ''
                })

                
                # 保存配置结果
                config_result = {
                    'config_name': config_name,
                    'kernel_config': kernel_config,
                    'batch_size': batch_size,
                    'avg_accuracy': avg_metrics['avg_val_acc'],
                    'avg_auc': avg_metrics['avg_val_auc'],
                    'avg_f1': avg_metrics['avg_val_f1'],
                    'avg_composite_score': avg_composite_score,
                    'std_accuracy': avg_metrics['std_val_acc'],
                    'std_auc': avg_metrics['std_val_auc'],
                    'std_f1': avg_metrics['std_val_f1'],
                    'fold_results': fold_results,
                    'directory': config_dir
                }
                
                result_path = os.path.join(config_dir, f"{config_name}_results.json")
                self.file_manager.save_metrics(config_result, result_path)
                
                all_results[config_name] = config_result
                
                # 更新最佳配置（基于综合评分）
                if avg_composite_score > best_composite_score + MIN_DELTA:
                    best_composite_score = avg_composite_score
                    best_config = config_result
                    print(f"  🎯 新的最佳配置!")

                # 更新进度条
                config_pbar.update(1)

        config_pbar.close()
        
        # 保存所有结果和最佳配置
        self._save_search_results(all_results, best_config, search_dir)

        # 可视化搜索结果
        self._visualize_search_results(all_results, search_dir)
        
        return best_config, search_dir
    
    def _save_search_results(self, all_results, best_config, search_dir):
        """保存搜索结果"""
        # 保存所有结果
        all_results_path = os.path.join(search_dir, "metrics", "all_search_results.json")
        self.file_manager.save_metrics(all_results, all_results_path)
        
        # 按综合评分排序
        sorted_configs = sorted(
            all_results.items(),
            key=lambda x: x[1]['avg_composite_score'],
            reverse=True
        )
        
        # 创建排名报告
        ranking_report = {
            'search_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_configs_tested': len(all_results),
            'ranking': []
        }
        
        print(f"\n{'='*80}")
        print("超参数搜索排名结果（按综合评分）:")
        print(f"{'='*80}")
        
        for rank, (config_name, data) in enumerate(sorted_configs, 1):
            marker = " 🏆" if rank == 1 else ""
            print(f"{rank:2d}. {config_name:25s} "
                  f"综合评分: {data['avg_composite_score']:.4f} | "
                  f"准确率: {data['avg_accuracy']:.4f} | "
                  f"AUC: {data['avg_auc']:.4f} | "
                  f"F1: {data['avg_f1']:.4f}{marker}")
            
            ranking_report['ranking'].append({
                'rank': rank,
                'config_name': config_name,
                'avg_composite_score': data['avg_composite_score'],
                'avg_accuracy': data['avg_accuracy'],
                'avg_auc': data['avg_auc'],
                'avg_f1': data['avg_f1']
            })
        
        # 保存最佳配置
        best_config_data = {
            'best_config_name': best_config['config_name'],
            'best_config_data': best_config,
            'search_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'selection_criteria': 'composite_score',
            'composite_weights': self.composite_weights
        }
        
        best_config_path = os.path.join(search_dir, "configs", "best_config.json")
        self.file_manager.save_config(best_config_data, best_config_path)
        
        # 同时保存到标准位置以便完整训练使用
        latest_best_path = "/home/xusi/EE5046_Projects/Task1_Results/HyperparamSearch/latest_best_config.json"
        self.file_manager.save_config(best_config_data, latest_best_path)
        
        # 保存排名报告
        ranking_path = os.path.join(search_dir, "metrics", "ranking_report.json")
        self.file_manager.save_metrics(ranking_report, ranking_path)
        
        print(f"\n{'='*80}")
        print("超参数搜索完成!")
        print(f"{'='*80}")
        print(f"最佳配置: {best_config['config_name']}")
        print(f"综合评分: {best_config['avg_composite_score']:.4f}")
        print(f"平均准确率: {best_config['avg_accuracy']:.4f}")
        print(f"平均AUC: {best_config['avg_auc']:.4f}")
        print(f"平均F1分数: {best_config['avg_f1']:.4f}")
        print(f"结果目录: {search_dir}")
        print(f"最佳配置已保存: {latest_best_path}")


    def _visualize_search_results(self, all_results, search_dir):
        """可视化超参数搜索结果"""
        if not all_results:
            return
        
        # 提取配置名称和指标
        config_names = []
        composite_scores = []
        accuracies = []
        aucs = []
        f1s = []
        
        for config_name, data in all_results.items():
            config_names.append(config_name)
            composite_scores.append(data['avg_composite_score'])
            accuracies.append(data['avg_accuracy'])
            aucs.append(data['avg_auc'])
            f1s.append(data['avg_f1'])
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 综合评分排名
        sorted_indices = np.argsort(composite_scores)[::-1]
        sorted_names = [config_names[i] for i in sorted_indices]
        sorted_scores = [composite_scores[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_scores)))
        bars1 = axes[0, 0].bar(range(len(sorted_scores)), sorted_scores, color=colors)
        axes[0, 0].set_title('超参数配置综合评分排名', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('配置排名', fontsize=12)
        axes[0, 0].set_ylabel('综合评分', fontsize=12)
        axes[0, 0].set_xticks(range(len(sorted_scores)))
        axes[0, 0].set_xticklabels([f'#{i+1}' for i in range(len(sorted_scores))], rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, score in zip(bars1, sorted_scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 准确率对比
        axes[0, 1].bar(range(len(config_names)), accuracies, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('各配置平均准确率', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('配置', fontsize=12)
        axes[0, 1].set_ylabel('准确率', fontsize=12)
        axes[0, 1].set_xticks(range(len(config_names)))
        axes[0, 1].set_xticklabels(config_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # AUC对比
        axes[1, 0].bar(range(len(config_names)), aucs, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('各配置平均AUC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('配置', fontsize=12)
        axes[1, 0].set_ylabel('AUC', fontsize=12)
        axes[1, 0].set_xticks(range(len(config_names)))
        axes[1, 0].set_xticklabels(config_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # F1分数对比
        axes[1, 1].bar(range(len(config_names)), f1s, color='lightcoral', edgecolor='black')
        axes[1, 1].set_title('各配置平均F1分数', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('配置', fontsize=12)
        axes[1, 1].set_ylabel('F1分数', fontsize=12)
        axes[1, 1].set_xticks(range(len(config_names)))
        axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        vis_path = os.path.join(search_dir, "visualizations", "search_results_comparison.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 超参数搜索结果可视化已保存: {vis_path}")


# ==================== 完整训练模块（修改版） ====================
class CompleteTrainer:
    """使用最佳配置进行完整训练"""
    
    @staticmethod
    def train_with_best_config(base_path, best_config_data):
        """
        完整训练流程：
        1. 在训练集（CV0~CV3）上使用最佳超参数进行5折交叉验证
        2. 使用全部训练集训练最终模型（包含验证集和早停）
        3. 在测试集（CV4）上最终评估
        """
        print("=" * 80)
        print("完整训练模式（使用综合评分和早停机制）")
        print("=" * 80)
        
        # 创建实验目录
        file_manager = ModelFileManager()
        experiment_dir = file_manager.create_experiment_dir(
            "/home/xusi/EE5046_Projects/Task1_Results/CompleteTraining",
            "CompleteTraining",
            f"{best_config_data['kernel_config']['name']}_BS{best_config_data['batch_size']}"
        )
        
        # 1. 在训练集上进行5折交叉验证
        print("\n步骤1: 在训练集（CV0~CV3）上进行5折交叉验证")
        
        # 使用最佳配置中的综合评分权重（如果有）
        composite_weights = best_config_data.get('composite_weights')
        
        trainer = ModelTrainer(
            base_path=base_path,
            kernel_config=best_config_data['kernel_config'],
            batch_size=best_config_data['batch_size'],
            lr=LEARNING_RATE,
            use_stream2=USE_STREAM2_SETTING,
            augment=AUGMENT_SETTING,
            experiment_dir=experiment_dir,
            config_name="BestConfig",
            composite_weights=composite_weights
        )
        
        train_indices = [0, 1, 2, 3]
        cv_metrics, cv_results = trainer.cross_validate_on_train_set(
            train_indices, NUM_EPOCHS, k_folds=5
        )
        
        print(f"\n5折交叉验证结果:")
        print(f"平均准确率: {cv_metrics['avg_val_acc']:.4f} ± {cv_metrics['std_val_acc']:.4f}")
        print(f"平均AUC: {cv_metrics['avg_val_auc']:.4f} ± {cv_metrics['std_val_auc']:.4f}")
        print(f"平均F1分数: {cv_metrics['avg_val_f1']:.4f} ± {cv_metrics['std_val_f1']:.4f}")
        print(f"平均综合评分: {cv_metrics.get('avg_composite_score', 0):.4f}")
        
        # 2. 使用全部训练集训练最终模型（包含验证集和早停）
        print("\n步骤2: 使用全部训练集（CV0~CV3）训练最终模型（包含验证集）")
        final_model, train_metrics = trainer.train_final_model(train_indices, NUM_EPOCHS)
        
        # 3. 在测试集（CV4）上评估
        print("\n步骤3: 在测试集（CV4）上评估最终模型")
        test_results = trainer.evaluate_on_test_set([4], final_model)
        
        print(f"\n最终测试结果:")
        print(f"测试集准确率: {test_results['test_acc']:.4f}")
        print(f"测试集AUC: {test_results['test_auc']:.4f}")
        print(f"测试集F1分数: {test_results['test_f1']:.4f}")
        print(f"测试集综合评分: {test_results['test_composite_score']:.4f}")
        
        # 保存完整训练摘要
        summary = {
            'cross_validation': cv_metrics,
            'final_training': train_metrics,
            'test_evaluation': test_results,
            'best_config': best_config_data,
            'complete_training_time': datetime.now().isoformat(),
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'min_delta': MIN_DELTA,
            'composite_weights': composite_weights
        }
        
        summary_path = os.path.join(experiment_dir, "metrics", "complete_training_summary.json")
        file_manager.save_metrics(summary, summary_path)
        
        return cv_metrics, test_results, experiment_dir


# ==================== 主程序模块（修改版） ====================
class TrainingPipeline:
    """训练管道主控制器"""
    
    def __init__(self):
        # 数据集路径
        self.base_path = "/home/xusi/EE5046_Projects/Dataset"

        # 检查路径是否存在
        if not os.path.exists(self.base_path):
            print(f"❌ Dataset目录不存在: {self.base_path}")
            print("请确保Dataset目录在指定位置")
            sys.exit(1)
        
        # 检查cv目录
        cv_dir = os.path.join(self.base_path, "cv")
        if not os.path.exists(cv_dir):
            print(f"❌ cv目录不存在: {cv_dir}")
            print("请确保Dataset目录包含cv子目录")
            sys.exit(1)
        
        # 检查training2017目录
        training_dir = os.path.join(self.base_path, "training2017")
        if not os.path.exists(training_dir):
            print(f"❌ training2017目录不存在: {training_dir}")
            print("请确保Dataset目录包含training2017子目录")
            sys.exit(1)
        
        print(f"✅ 使用数据集目录: {self.base_path}")
        print(f"✅ cv目录: {cv_dir}")
        print(f"✅ training2017目录: {training_dir}")
        
        # 检查CSV文件
        for i in range(5):
            csv_file = os.path.join(cv_dir, f"cv{i}.csv")
            if os.path.exists(csv_file):
                print(f"✅ 找到文件: cv{i}.csv")
            else:
                print(f"⚠️ 警告: 文件 cv{i}.csv 不存在")

        self.train_indices = [0, 1, 2, 3]
        self.test_indices = [4]
        
        # 自定义综合评分权重（可根据任务调整）
        self.custom_weights = {
            'accuracy': 0.40,  # 提高准确率权重
            'auc': 0.35,       # AUC权重
            'f1': 0.20,        # F1分数权重
            'stability': 0.05  # 稳定性权重
        }
    
    def run(self):
        """运行训练管道"""
        parser = argparse.ArgumentParser(description='ECG房颤检测训练脚本（改进版）')
        parser.add_argument('--mode', type=str, default=EXPERIMENT_MODE,
                            choices=['search', 'train', 'full', 'compare'],
                            help='运行模式')
        parser.add_argument('--compare_mode', type=str, default=COMPARISON_MODE,
                            choices=['stream', 'augment'],
                            help='对比实验模式')
        parser.add_argument('--weights', type=str, default='default',
                            choices=['default', 'accuracy_focus', 'balanced', 'auc_focus'],
                            help='综合评分权重策略')
        
        parser.add_argument('--no_progress', action='store_true',
                            help='禁用进度条显示')
        
        args = parser.parse_args()

        if args.no_progress:
            from tqdm import tqdm
            tqdm.__init__ = lambda self, *args, **kwargs: None
        
        print(f"\n数据集划分策略:")
        print(f"训练集: CV{', '.join(map(str, self.train_indices))}")
        print(f"测试集: CV{', '.join(map(str, self.test_indices))}")
        print(f"早停耐心值: {EARLY_STOP_PATIENCE}, 最小提升: {MIN_DELTA}")
        
        # 根据权重策略选择权重
        if args.weights == 'accuracy_focus':
            weights = {'accuracy': 0.50, 'auc': 0.30, 'f1': 0.15, 'stability': 0.05}
        elif args.weights == 'auc_focus':
            weights = {'accuracy': 0.30, 'auc': 0.50, 'f1': 0.15, 'stability': 0.05}
        elif args.weights == 'balanced':
            weights = {'accuracy': 0.35, 'auc': 0.35, 'f1': 0.25, 'stability': 0.05}
        else:
            weights = None  # 使用默认权重
        
        if weights:
            print(f"综合评分权重: {weights}")
        
        if args.mode == 'search':
            self._run_search_mode(weights)
        elif args.mode == 'train':
            self._run_train_mode(weights)
        elif args.mode == 'full':
            self._run_full_mode(args, weights)
        elif args.mode == 'compare':
            self._run_compare_mode(args, weights)
    
    def _run_search_mode(self, weights):
        """运行超参数搜索模式"""
        print("\n模式: 超参数搜索（使用综合评分）")
        
        searcher = HyperparameterSearcher(self.base_path, composite_weights=weights)
        best_config, search_dir = searcher.search(num_epochs_search=30)
        
        print(f"\n超参数搜索完成!")
        print(f"最佳配置: {best_config['config_name']}")
        print(f"综合评分: {best_config['avg_composite_score']:.4f}")
        print(f"结果目录: {search_dir}")
    
    def _run_train_mode(self, weights):
        """运行默认训练模式"""
        print("\n模式: 默认配置训练（使用综合评分和早停）")
        
        # 创建实验目录
        file_manager = ModelFileManager()
        experiment_dir = file_manager.create_experiment_dir(
            "/home/xusi/EE5046_Projects/Task1_Results/DefaultTraining",
            "DefaultTraining",
            f"{DEFAULT_KERNEL_CONFIG['name']}_BS{BATCH_SIZE}"
        )
        
        trainer = ModelTrainer(
            base_path=self.base_path,
            kernel_config=DEFAULT_KERNEL_CONFIG,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            use_stream2=USE_STREAM2_SETTING,
            augment=AUGMENT_SETTING,
            experiment_dir=experiment_dir,
            config_name="DefaultConfig",
            composite_weights=weights
        )
        
        # 在训练集上进行5折交叉验证
        cv_metrics, cv_results = trainer.cross_validate_on_train_set(
            self.train_indices, NUM_EPOCHS, k_folds=5
        )
        
        print(f"\n默认训练完成!")
        print(f"平均综合评分: {cv_metrics.get('avg_composite_score', 0):.4f}")
        print(f"结果目录: {experiment_dir}")
    
    def _run_full_mode(self, args, weights):
        """运行完整训练模式"""
        print("\n模式: 使用最佳配置进行完整训练")
        
        best_config_path = "/home/xusi/EE5046_Projects/Task1_Results/HyperparamSearch/latest_best_config.json"
        
        if os.path.exists(best_config_path):
            with open(best_config_path, 'r') as f:
                best_config_data = json.load(f)
            
            print(f"加载最佳配置: {best_config_data['best_config_name']}")
            print(f"选择标准: {best_config_data.get('selection_criteria', 'accuracy')}")
            
            # 使用最佳配置进行完整训练
            cv_metrics, test_results, experiment_dir = CompleteTrainer.train_with_best_config(
                self.base_path, best_config_data['best_config_data']
            )
            
            print(f"\n完整训练完成!")
            print(f"交叉验证平均综合评分: {cv_metrics.get('avg_composite_score', 0):.4f}")
            print(f"测试集综合评分: {test_results['test_composite_score']:.4f}")
            print(f"结果目录: {experiment_dir}")
        else:
            print(f"错误: 找不到最佳配置文件 {best_config_path}")
            print("请先运行超参数搜索模式: python TrainProcess.py --mode search")
    
    def _run_compare_mode(self, args, weights):
        """运行对比实验模式"""
        print("\n模式: 对比实验")
        print("注意: 对比实验模式暂未实现综合评分，使用原有逻辑")
        
        # 对比实验模块需要相应修改，这里暂时跳过
        print("对比实验模式暂未更新，请使用原有版本")
        return


# ==================== 程序入口 ====================
if __name__ == '__main__':
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置matplotlib中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行训练管道
    try:
        pipeline = TrainingPipeline()
        pipeline.run()
        print("\n🎉 训练完成!")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()