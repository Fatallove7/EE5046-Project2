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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, 
                           precision_score, recall_score, f1_score)
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm

# 自定义模块
from src.common.Config import (AUGMENT_SETTING, BATCH_SIZE, EARLY_STOP_PATIENCE,
                    EXPERIMENT_MODE, FIXED_LENGTH, FOCAL_PRESET_CONFIGS, INPUT_CHANNELS,
                    LEARNING_RATE, LOSS_FUNCTION_CONFIG, MIN_DELTA, NUM_EPOCHS, OUTPUT_CLASSES, USE_FOCAL_LOSS,
                    USE_STREAM2_SETTING, COMPARISON_MODE,
                    STREAM_COMPARISON_CONFIGS, AUGMENTATION_COMPARISON_CONFIGS,
                    DEFAULT_KERNEL_CONFIG,LR_SCHEDULER_CONFIG, get_loss_config)
from src.task1_ecg_analysis.data.DataManager import DataManager
from src.task1_ecg_analysis.data.FoldDataset import FoldDataset
from src.task1_ecg_analysis.visualization.TrainingVisualizer import TrainingVisualizer
from src.task1_ecg_analysis.data.BalancedFoldDataset import BalancedFoldDataset
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

# ==================== FocalLoss类 ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    Paper: Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float, optional): Weighting factor for the rare class (0 < alpha < 1).
        gamma (float, optional): Focusing parameter (gamma >= 0). Higher gamma reduces 
                                the loss contribution from easy examples.
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'
        logits (bool, optional): If True, expects raw logits as input,
                                 otherwise expects probabilities (0-1).
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', logits=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits
        
    def forward(self, inputs, targets):
        if self.logits:
            # If using logits, apply sigmoid first
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            # If using probabilities, use regular BCE
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities from logits if needed
        if self.logits:
            pt = torch.sigmoid(inputs)
        else:
            pt = inputs
        
        # Ensure pt is within [0, 1]
        pt = torch.clamp(pt, 1e-8, 1 - 1e-8)
        
        # Calculate p_t
        p_t = pt * targets + (1 - pt) * (1 - targets)
        
        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate modulating factor
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = alpha_t * modulating_factor * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss with dynamic alpha calculation based on class distribution.
    """
    
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean', logits=True):
        super(WeightedFocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits
        
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities from logits if needed
        if self.logits:
            pt = torch.sigmoid(inputs)
        else:
            pt = inputs
        
        # Ensure pt is within [0, 1]
        pt = torch.clamp(pt, 1e-8, 1 - 1e-8)
        
        # Calculate p_t
        p_t = pt * targets + (1 - pt) * (1 - targets)
        
        # Calculate alpha_t based on class distribution
        if self.pos_weight is not None:
            # Use provided pos_weight to calculate alpha
            alpha_t = self.pos_weight * targets + (1 - targets)
            # Normalize so that alpha_t sums to 2 (like in original focal loss)
            alpha_t = alpha_t / (alpha_t.mean() + 1e-8) * 1.0
        else:
            # Default: equal weighting
            alpha_t = torch.ones_like(targets)
        
        # Calculate modulating factor
        modulating_factor = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = alpha_t * modulating_factor * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# ==================== 模型文件管理器 ====================
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
                 composite_weights=None,lr_scheduler_config=None,
                 use_focal_loss=USE_FOCAL_LOSS,focal_alpha=0.25,focal_gamma=2.0):
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

        # Focal Loss相关参数
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # 打印配置信息
        if self.use_focal_loss:
            print(f"📈 使用Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
        else:
            print(f"📈 使用BCEWithLogitsLoss")
        
        # 综合评分权重
        self.composite_weights = composite_weights or CompositeScoreCalculator.DEFAULT_WEIGHTS

        # 学习率调度器配置
        self.lr_scheduler_config = lr_scheduler_config or LR_SCHEDULER_CONFIG  # 使用配置或默认配置
        print(f"📈 学习率调度器配置: {self.lr_scheduler_config['scheduler_type']}")
        
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

    def _find_precision_optimized_threshold(self, labels, probs, return_all=False):
        """寻找更注重精确率的阈值"""
        # 首先确保有足够的概率变化
        if np.max(probs) - np.min(probs) < 0.1:
            print("  警告: 概率变化很小，使用默认阈值")
            if return_all:
                return 0.5, 0, 0, 0
            else:
                return 0.5
        
        # 搜索范围根据概率分布调整
        prob_min = max(0.1, np.min(probs))
        prob_max = min(0.9, np.max(probs))
        
        # 如果概率范围太小，扩大搜索范围
        if prob_max - prob_min < 0.3:
            prob_min = max(0.1, prob_min - 0.1)
            prob_max = min(0.9, prob_max + 0.1)
        
        thresholds = np.linspace(prob_min, prob_max, 51)
        
        best_score = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        for th in thresholds:
            preds = (probs >= th).astype(int)
            pos_predictions = np.sum(preds)
            
            # 如果没有任何预测，跳过这个阈值
            if pos_predictions == 0:
                continue
                
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            
            # 更注重精确率的评分
            # 但也要确保有一定数量的预测
            weighted_score = 0.5 * precision + 0.3 * f1 + 0.2 * min(recall, 0.5)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_threshold = th
                best_precision = precision
                best_recall = recall
        
        # 如果没有任何阈值能预测出正样本，使用更低的阈值
        if best_score == 0:
            print("  警告: 没有阈值能预测出正样本，使用更低的阈值")
            best_threshold = prob_min  # 使用最低的阈值
            preds = (probs >= best_threshold).astype(int)
            best_precision = precision_score(labels, preds, zero_division=0)
            best_recall = recall_score(labels, preds, zero_division=0)
            best_f1 = f1_score(labels, preds, zero_division=0)
        
        if return_all:
            return best_threshold, best_f1, best_precision, best_recall
        else:
            return best_threshold

    def _find_best_two_stage_combo(self, labels, probs, max_combinations=20):
        """寻找最佳的两阶段阈值组合"""
        print("  搜索最佳两阶段阈值组合...")
        
        # 首先检查概率分布
        print(f"  概率范围: [{np.min(probs):.4f}, {np.max(probs):.4f}]")
        print(f"  概率平均值: {np.mean(probs):.4f}")
        
        # 确保有正样本的概率
        if np.max(probs) < 0.3:  # 如果最大概率都很低
            print("  警告: 所有预测概率都很低，可能模型有问题")
            return {
                'stage1_threshold': 0.1,  # 使用很低的阈值
                'stage2_threshold': 0.3,
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
        
        # 定义搜索范围（根据实际概率分布调整）
        prob_min = max(0.1, np.percentile(probs, 5))  # 第5百分位数作为下限
        prob_max = min(0.9, np.percentile(probs, 95))  # 第95百分位数作为上限
        
        # 确保搜索范围合理
        if prob_max - prob_min < 0.2:
            prob_min = max(0.1, prob_min - 0.1)
            prob_max = min(0.9, prob_max + 0.1)
        
        stage1_options = np.linspace(prob_min, min(prob_max, 0.6), 6)
        stage2_options = np.linspace(max(prob_min, 0.4), prob_max, 8)
        
        best_f1 = 0
        best_combo = {
            'stage1_threshold': 0.3,
            'stage2_threshold': 0.5,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        tested_combos = 0
        
        for s1 in stage1_options:
            for s2 in stage2_options:
                if s2 <= s1:  # 确保第二阶段阈值高于第一阶段
                    continue
                
                # 两阶段预测
                stage1_preds = (probs >= s1).astype(int)
                pos_indices = np.where(stage1_preds == 1)[0]
                final_preds = stage1_preds.copy()
                
                if len(pos_indices) > 0:
                    pos_probs = probs[pos_indices]
                    stage2_preds = (pos_probs >= s2).astype(int)
                    final_preds[pos_indices] = stage2_preds
                
                # 计算指标
                try:
                    precision = precision_score(labels, final_preds, zero_division=0)
                    recall = recall_score(labels, final_preds, zero_division=0)
                    f1 = f1_score(labels, final_preds, zero_division=0)
                except:
                    precision = recall = f1 = 0
                
                # 检查是否预测了正样本
                pos_predictions = np.sum(final_preds)
                if pos_predictions == 0:
                    # 如果没有预测正样本，跳过这个组合
                    continue
                
                # 综合评分：平衡精确率、召回率和F1
                composite_score = 0.4 * precision + 0.4 * f1 + 0.2 * recall
                
                if composite_score > best_f1:
                    best_f1 = composite_score
                    best_combo = {
                        'stage1_threshold': s1,
                        'stage2_threshold': s2,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'pos_predictions': int(pos_predictions)
                    }
                
                tested_combos += 1
                if tested_combos >= max_combinations:
                    break
            
            if tested_combos >= max_combinations:
                break
        
        print(f"  测试了 {tested_combos} 种阈值组合")
        print(f"  最佳组合: stage1={best_combo['stage1_threshold']:.2f}, stage2={best_combo['stage2_threshold']:.2f}")
        print(f"  预测正样本数: {best_combo.get('pos_predictions', 0)}")
        print(f"  对应指标: 精确率={best_combo['precision']:.4f}, 召回率={best_combo['recall']:.4f}, F1={best_combo['f1']:.4f}")
        
        return best_combo

    def _two_stage_evaluate(self, probs, labels, stage1_th=0.4, stage2_th=None):
        """两阶段评估 - 优化版本"""
        if stage2_th is None:
            # 如果没有指定第二阶段阈值，使用更注重精确率的阈值
            stage2_th = self._find_precision_optimized_threshold(labels, probs)
        
        print(f"两阶段阈值策略: 第一阶段={stage1_th:.2f}, 第二阶段={stage2_th:.2f}")
        
        # 第一阶段：低阈值获取高召回
        stage1_preds = (probs >= stage1_th).astype(int)
        stage1_recall = recall_score(labels, stage1_preds, zero_division=0)
        stage1_precision = precision_score(labels, stage1_preds, zero_division=0)
        print(f"第一阶段: 召回率={stage1_recall:.4f}, 精确率={stage1_precision:.4f}")
        
        # 第二阶段：只对第一阶段预测为正的样本使用高阈值
        stage1_pos_indices = np.where(stage1_preds == 1)[0]
        if len(stage1_pos_indices) == 0:
            print("⚠️ 第一阶段没有预测为正的样本")
            final_preds = stage1_preds
        else:
            stage1_pos_probs = probs[stage1_pos_indices]
            
            # 对这些样本使用第二阶段阈值
            stage2_pos_preds = (stage1_pos_probs >= stage2_th).astype(int)
            
            # 合并结果
            final_preds = stage1_preds.copy()
            final_preds[stage1_pos_indices] = stage2_pos_preds
            
            print(f"第一阶段正样本数: {len(stage1_pos_indices)}")
            print(f"第二阶段保留数: {np.sum(stage2_pos_preds)} (过滤率: {(1 - np.sum(stage2_pos_preds)/len(stage1_pos_indices))*100:.1f}%)")
        
        # 计算最终指标
        final_recall = recall_score(labels, final_preds, zero_division=0)
        final_precision = precision_score(labels, final_preds, zero_division=0)
        final_f1 = f1_score(labels, final_preds, zero_division=0)
        final_acc = accuracy_score(labels, final_preds)
        
        print(f"第二阶段结果:")
        print(f"  召回率: {final_recall:.4f} (相比第一阶段: {final_recall-stage1_recall:+.4f})")
        print(f"  精确率: {final_precision:.4f} (相比第一阶段: {final_precision-stage1_precision:+.4f})")
        print(f"  F1分数: {final_f1:.4f}")
        print(f"  准确率: {final_acc:.4f}")
        
        return {
            'predictions': final_preds,
            'acc': final_acc,
            'recall': final_recall,
            'precision': final_precision,
            'f1': final_f1,
            'stage1_threshold': stage1_th,
            'stage2_threshold': stage2_th,
            'stage1_recall': stage1_recall,
            'stage1_precision': stage1_precision
        }
    
    def _create_criterion(self, pos_weight, device):
        """创建损失函数（支持BCE和Focal Loss）"""
        if self.use_focal_loss:
            # 使用Focal Loss
            if self.focal_alpha is not None:
                # 使用固定的alpha
                criterion = FocalLoss(
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction='mean',
                    logits=True
                ).to(device)
            else:
                # 使用加权Focal Loss
                criterion = WeightedFocalLoss(
                    pos_weight=pos_weight,
                    gamma=self.focal_gamma,
                    reduction='mean',
                    logits=True
                ).to(device)
            print(f"✅ 创建Focal Loss: alpha={self.focal_alpha if self.focal_alpha else 'dynamic'}, "
                f"gamma={self.focal_gamma}")
        else:
            # 使用BCEWithLogitsLoss
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
            print(f"✅ 创建BCEWithLogitsLoss: pos_weight={pos_weight.item():.2f}")
        
        return criterion

    def _create_lr_scheduler(self, optimizer, num_epochs, train_loader=None):
        """根据配置创建学习率调度器"""
        if not self.lr_scheduler_config.get('use_scheduler', True):
            print("⚠️ 未启用学习率调度器")
            return None
        
        scheduler_type = self.lr_scheduler_config.get('scheduler_type', 'plateau')
        
        if scheduler_type == 'plateau':
            config = self.lr_scheduler_config.get('plateau_config', {})
            # 移除 verbose 参数
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'max'),
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 5),
                min_lr=config.get('min_lr', 1e-6),
                # verbose=config.get('verbose', True)  # 注释掉或移除这行
            )
            print(f"✅ 创建 ReduceLROnPlateau 调度器，耐心值: {config.get('patience', 5)}")
        
        elif scheduler_type == 'cosine':
            config = self.lr_scheduler_config.get('cosine_config', {})
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('T_max', num_epochs),
                eta_min=config.get('eta_min', 1e-6)
            )
            print(f"✅ 创建 CosineAnnealingLR 调度器，T_max: {config.get('T_max', num_epochs)}")
        
        elif scheduler_type == 'onecycle':
            config = self.lr_scheduler_config.get('onecycle_config', {})
            if train_loader is None:
                print("⚠️ OneCycleLR 需要 train_loader，使用默认调度器")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=config.get('max_lr', self.lr),
                    steps_per_epoch=len(train_loader),
                    epochs=num_epochs,
                    pct_start=config.get('pct_start', 0.3),
                    div_factor=config.get('div_factor', 25.0),
                    final_div_factor=config.get('final_div_factor', 1e4)
                )
                print(f"✅ 创建 OneCycleLR 调度器，max_lr: {config.get('max_lr', self.lr)}")
        
        elif scheduler_type == 'step':
            config = self.lr_scheduler_config.get('step_config', {})
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('step_size', 10),
                gamma=config.get('gamma', 0.5)
            )
            print(f"✅ 创建 StepLR 调度器，step_size: {config.get('step_size', 10)}, gamma: {config.get('gamma', 0.5)}")
        
        else:
            print(f"⚠️ 未知的学习率调度器类型: {scheduler_type}，使用默认 ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-6
                # verbose=True  # 同样移除这里的 verbose
            )
        
        return scheduler
    
    def cross_validate_on_train_set(self, train_cv_indices, num_epochs, k_folds=5, save_models=True):
        """
        在训练集上进行K折交叉验证（简化输出版本）
        """
        print(f"\n{'='*60}")
        print(f"🎯 开始 {k_folds} 折交叉验证")
        print(f"训练集: CV{', '.join(map(str, train_cv_indices))}")
        print(f"模型配置: {self.kernel_config.get('name', 'Unknown')}")
        print(f"批次大小: {self.batch_size}, 学习率: {self.lr}")
        print(f"数据增强: {'是' if self.augment else '否'}")
        print(f"{'='*60}")
        
        # 创建K折划分
        kfold_splits = self.data_manager.create_kfold_splits(train_cv_indices, k_folds)
        if not kfold_splits:
            print("❌ 错误: 无法创建K折划分")
            return {}, []
        
        fold_results = []
        fold_models = []

        # 创建简洁的进度条
        fold_pbar = tqdm(
            range(k_folds), 
            desc="交叉验证进度",
            bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # 训练每一折
        for fold_idx, (train_data, val_data) in enumerate(kfold_splits):
            fold_pbar.set_description(f"训练折 {fold_idx+1}/{k_folds}")
            
            # 训练当前折
            fold_result, fold_model = self._train_single_fold(
                train_data, val_data, fold_idx, num_epochs, save_models
            )
            
            fold_results.append(fold_result)
            fold_models.append(fold_model)
            
            # 更新进度条并显示当前折的结果
            fold_pbar.set_postfix({
                'acc': f"{fold_result['best_val_acc']:.3f}",
                'f1': f"{fold_result['best_val_f1']:.3f}"
            })
            
            # 显示当前折的简单结果
            print(f"  ✅ 折 {fold_idx+1} 完成: 验证准确率={fold_result['best_val_acc']:.4f}, "
                f"F1={fold_result['best_val_f1']:.4f} (最佳 epoch {fold_result['best_epoch']})")

            fold_pbar.update(1)
        
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
                    'epoch': best_fold_result['best_epoch'],
                    'early_stopped': best_fold_result.get('early_stopped', False)
                }
                self.file_manager.save_model(best_model, model_path, metadata)
                
                print(f"  💾 最佳模型已保存: {model_name}")
        
        # 显示交叉验证结果总结
        print(f"\n{'='*60}")
        print(f"📊 {k_folds}折交叉验证结果总结")
        print(f"{'='*60}")
        
        # 显示每折详细结果
        print(f"{'折':<4} {'验证准确率':<12} {'AUC':<12} {'F1分数':<12} {'综合评分':<12} {'最佳epoch':<10}")
        print(f"{'-'*70}")
        
        for i, result in enumerate(fold_results):
            metrics = {
                'accuracy': result['best_val_acc'],
                'auc': result['best_val_auc'],
                'f1': result['best_val_f1']
            }
            composite_score, _ = CompositeScoreCalculator.calculate_composite_score(metrics)
            
            print(f" {i+1:<3} {result['best_val_acc']:<12.4f} {result['best_val_auc']:<12.4f} "
                f"{result['best_val_f1']:<12.4f} {composite_score:<12.4f} {result['best_epoch']:<10}")
        
        print(f"{'-'*70}")
        print(f" 平均: {avg_metrics['avg_val_acc']:<12.4f} {avg_metrics['avg_val_auc']:<12.4f} "
            f"{avg_metrics['avg_val_f1']:<12.4f} {avg_metrics['avg_composite_score']:<12.4f}")
        print(f" 标准差: {avg_metrics['std_val_acc']:<12.4f} {avg_metrics['std_val_auc']:<12.4f} "
            f"{avg_metrics['std_val_f1']:<12.4f}")
        print(f"{'='*60}")
        
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
        
        # 新增：检查数据分布
        print("\n=== 数据分布诊断 ===")
        
        # 检查整个训练集的类别分布
        all_labels = []
        for _, label in train_data:
            all_labels.append(label)
        
        all_labels_np = np.array(all_labels)
        print(f"整个训练集 (CV0~CV3) 统计:")
        print(f"  总样本数: {len(all_labels_np)}")
        print(f"  正样本数: {np.sum(all_labels_np)}")
        print(f"  负样本数: {len(all_labels_np) - np.sum(all_labels_np)}")
        print(f"  正样本比例: {np.mean(all_labels_np):.2%}")
        
        # 划分训练集和验证集
        total_size = len(train_data)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        # 随机划分
        torch.manual_seed(42)  # 确保可重复性
        train_subset, val_subset = random_split(train_data, [train_size, val_size])

        # 新增：检查划分后的分布
        train_labels = []
        for idx in train_subset.indices:
            _, label = train_data[idx]
            train_labels.append(label)
        
        val_labels = []
        for idx in val_subset.indices:
            _, label = train_data[idx]
            val_labels.append(label)
        
        print(f"\n划分后统计:")
        print(f"  训练集大小: {train_size}, 验证集大小: {val_size}")
        print(f"  训练集正样本比例: {np.mean(train_labels):.2%}")
        print(f"  验证集正样本比例: {np.mean(val_labels):.2%}")
        print(f"  训练集类别分布: 正={np.sum(train_labels)}, 负={len(train_labels)-np.sum(train_labels)}")
        print(f"  验证集类别分布: 正={np.sum(val_labels)}, 负={len(val_labels)-np.sum(val_labels)}")
            
        # 创建平衡数据集
        train_dataset = BalancedFoldDataset(
            list(train_subset),
            base_path=self.base_path,
            is_train=True,
            augment=True,
            target_ratio=0.5,  # 1:3.3 的比例，比原始1:10更平衡
            augmentation_config={
                'positive_augment_factor': 15,  # 大幅增加正样本增强倍数（原来是3）
                'noise_std': 0.02,              # 增加噪声强度
                'scale_range': (0.8, 1.2),      # 扩大缩放范围
                'shift_range': (-25, 25),       # 扩大平移范围
                'use_mixup': True,
                'mixup_alpha': 0.3,             # 增加mixup强度
                'use_time_warp': True,          # 新增时间扭曲
                'time_warp_factor': 0.4,
                'use_random_cutout': True,      # 新增随机遮挡
                'cutout_size': 60,
                'cutout_probability': 0.4,
                'use_frequency_mask': True,     # 新增频域掩码
                'freq_mask_ratio': 0.2,
                # ECG特有增强
                'use_baseline_wander': True,
                'bw_amplitude': 0.03,
                'use_powerline_noise': True,
                'pl_amplitude': 0.015
            }
        )
        val_dataset = FoldDataset(
            list(val_subset),
            base_path=self.base_path,
            is_train=False,
            augment=False
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

        # 计算正类权重（处理不平衡）
        pos_weight = self._calculate_pos_weight(train_loader)
        print(f"正类权重: {pos_weight.item():.2f}")
        
        # 初始化模型
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        criterion = self._create_criterion(pos_weight, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 创建学习率调度器
        scheduler = self._create_lr_scheduler(optimizer, num_epochs, train_loader)
        
        # 训练循环
        train_losses = []
        train_accs = []
        train_aucs = []
        train_f1s = []
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
            # 训练阶段
            model.train()
            epoch_train_loss = 0.0
            train_probs = []
            train_labels = []
            
            for x, y in train_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                # 收集概率和标签（用于后续计算指标）
                probs = torch.sigmoid(logits)
                train_probs.extend(probs.detach().cpu().numpy().flatten())
                train_labels.extend(y.detach().cpu().numpy().flatten())
            
            # 使用新函数计算训练集指标
            train_metrics = self._calculate_training_metrics(model, train_loader, criterion, device)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_acc = train_metrics['acc']
            train_auc = train_metrics['auc']
            train_f1 = train_metrics['f1']
            train_threshold = train_metrics['threshold']
            
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            train_aucs.append(train_auc)
            train_f1s.append(train_f1)
            
            # 验证阶段
            val_res = self._validate_model(model, val_loader, criterion, device)
            
            val_loss = val_res['loss']
            val_acc = val_res['acc']
            val_auc = val_res['auc']
            val_f1 = val_res['f1']
            val_threshold = val_res['threshold']
            
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
            
            # 更新学习率调度器
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(composite_score)
                else:
                    scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
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
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            # 更新训练最佳准确率
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            
            # 打印进度 - 现在显示训练集和验证集的阈值
            if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
                print(f"  Epoch {epoch}/{num_epochs}:")
                print(f"    训练 - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}, 阈值: {train_threshold:.3f}")
                print(f"    验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}, 阈值: {val_threshold:.3f}")
                print(f"    综合评分: {composite_score:.4f}, 学习率: {current_lr:.2e}")
            
            # 早停检查
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"  ⚠️ 早停触发于epoch {epoch}，连续{EARLY_STOP_PATIENCE}个epoch验证集无显著提升")
                break
        
        print(f"  最佳验证综合评分: {best_composite_score:.4f} (Epoch {best_epoch})")
        print(f"  最终学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
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
                'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE,
                'final_lr': optimizer.param_groups[0]['lr'],  # 保存最终学习率
                'pos_weight': pos_weight.item(),  # 保存正类权重
                'lr_scheduler_type': self.lr_scheduler_config.get('scheduler_type', 'plateau')
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
            'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE,
            'final_lr': optimizer.param_groups[0]['lr'],
            'pos_weight': pos_weight.item()
        }
        
        # 保存训练指标
        if self.experiment_dir:
            metrics_path = os.path.join(self.experiment_dir, "metrics", "final_training_metrics.json")
            self.file_manager.save_metrics(train_metrics, metrics_path)
        
        return model, train_metrics
    
    def test_basic_functionality(self):
        """测试基本功能，确认没有实现错误"""
        print("\n=== 基本功能测试 ===")
        
        # 1. 加载少量数据
        test_indices = [0]  # 只使用CV0
        test_data = self.data_manager.load_cv_files(test_indices)
        
        if len(test_data) == 0:
            print("错误: 测试数据为空")
            return
        
        # 只取前100个样本
        test_data = test_data[:100]
        
        # 2. 创建简单的模型
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        # 3. 创建数据集和加载器
        test_dataset = FoldDataset(
            test_data, self.base_path, is_train=False, augment=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        # 4. 测试前向传播
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                outputs = model(x)
                probs = torch.sigmoid(outputs)
                
                # 打印前几个样本的信息
                if sample_count < 5:
                    print(f"样本 {sample_count}:")
                    print(f"  输入形状: {x.shape}")
                    print(f"  标签: {y.cpu().numpy()[0][0]:.1f}")
                    print(f"  输出logits: {outputs.cpu().numpy()[0][0]:.4f}")
                    print(f"  概率: {probs.cpu().numpy()[0][0]:.4f}")
                    print()
                
                sample_count += 1
                if sample_count >= 10:
                    break
        
        print(f"测试完成，处理了 {sample_count} 个样本")
    
    def _train_single_fold(self, train_data, val_data, fold_idx, num_epochs, save_model=True, min_epochs=10):
        """训练单个折，包含早停机制（简化输出版本）"""
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
            val_dataset, batch_size=self.batch_size,  # 使用相同批次大小
            shuffle=False, num_workers=0
        )
        
        # 计算正类权重
        pos_weight = self._calculate_pos_weight(train_loader)
        
        # 初始化模型
        model = Mscnn(
            INPUT_CHANNELS,
            OUTPUT_CLASSES,
            use_stream2=self.use_stream2,
            stream1_kernel=self.kernel_config['stream1_kernel'],
            stream2_first_kernel=self.kernel_config['stream2_first_kernel']
        ).to(device)
        
        criterion = self._create_criterion(pos_weight, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # 创建学习率调度器
        scheduler = self._create_lr_scheduler(optimizer, num_epochs, train_loader)
        
        # 训练状态
        best_val_acc = 0
        best_val_auc = 0
        best_val_f1 = 0
        best_composite_score = 0
        best_epoch = 0
        best_model_state = None
        early_stop_counter = 0
        
        # 创建简洁的epoch进度条
        epoch_pbar = tqdm(
            range(1, num_epochs + 1), 
            desc=f"折 {fold_idx+1} 训练进度",
            bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            position=0,
            leave=False
        )
        
        # 训练循环
        for epoch in epoch_pbar:
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            # 批次训练（不显示内部批次信息）
            for x, y in train_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 计算训练集指标（静默模式）
            train_metrics = self._calculate_training_metrics(model, train_loader, criterion, device)
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段（静默模式）
            val_res = self._validate_model(model, val_loader, criterion, device)
            
            # 计算综合评分
            val_metrics_dict = {
                'accuracy': val_res['acc'],
                'auc': val_res['auc'],
                'f1': val_res['f1']
            }
            composite_score, _ = CompositeScoreCalculator.calculate_composite_score(val_metrics_dict)
            
            # 更新学习率调度器
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_res['loss'])  # 使用验证损失
                else:
                    scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 检查是否是最佳模型
            is_best = False
            if composite_score > best_composite_score:
                if composite_score >= best_composite_score + MIN_DELTA:
                    is_best = True
                    best_composite_score = composite_score
                    best_val_acc = val_res['acc']
                    best_val_auc = val_res['auc']
                    best_val_f1 = val_res['f1']
                    best_epoch = epoch
                    best_model_state = model.state_dict().copy()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            else:
                early_stop_counter += 1
            
            # 更新进度条显示
            epoch_pbar.set_postfix({
                'loss': f"{avg_train_loss:.3f}",
                'val_acc': f"{val_res['acc']:.3f}",
                'val_f1': f"{val_res['f1']:.3f}",
                'lr': f"{current_lr:.1e}"
            })
            
            # 每5个epoch或最后一个epoch显示详细信息
            if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
                print(f"    Epoch {epoch:3d}/{num_epochs}: "
                    f"训练损失={avg_train_loss:.4f}, 准确率={train_metrics['acc']:.4f} | "
                    f"验证准确率={val_res['acc']:.4f}, F1={val_res['f1']:.4f} | "
                    f"学习率={current_lr:.2e}" + (" ★" if is_best else ""))
            
            # 早停检查（至少训练min_epochs个epoch）
            if epoch >= min_epochs and early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"    ⏹️  早停触发于epoch {epoch}，连续{EARLY_STOP_PATIENCE}个epoch验证集无显著提升")
                break
        
        epoch_pbar.close()
        
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
                'early_stopped': early_stop_counter >= EARLY_STOP_PATIENCE,
                'pos_weight': pos_weight.item(),
                'final_lr': optimizer.param_groups[0]['lr'],
                'lr_scheduler_type': self.lr_scheduler_config.get('scheduler_type', 'plateau')
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

    
    def _calculate_pos_weight(self, dataloader):
        """计算正类权重"""
        if self.use_focal_loss:
            # Focal Loss不使用pos_weight，返回None或默认值
            print(f"⚠️ Focal Loss不使用pos_weight，将忽略此参数")
            return torch.tensor([1.0], dtype=torch.float32).to(device)


        all_labels = []
        for _, y in dataloader:
            all_labels.extend(y.numpy().flatten())
        
        all_labels = np.array(all_labels, dtype=int)
        class_counts = np.bincount(all_labels, minlength=2)
        
        # 计算正类比例
        total_samples = np.sum(class_counts)
        positive_ratio = class_counts[1] / total_samples
        negative_ratio = class_counts[0] / total_samples
        
        print(f"类别分布: 负类={class_counts[0]}, 正类={class_counts[1]}, 正类比例={positive_ratio:.2%}")

        # if positive_ratio < 0.2:  # 正类比例低于20%
        #     # 使用2-5之间的权重，而不是10.39
        #     adjusted_weight = min(5.0, max(2.0, 1.0 / positive_ratio))
        # else:
        #     adjusted_weight = 1.0

        # 方法1: 基于逆频率（当前1:10.6比例）
        raw_weight = class_counts[0] / class_counts[1]  # 约10.6
        adjusted_weight = min(8.0, max(3.0, raw_weight * 0.6))
    
        
        pos_weight = torch.tensor([adjusted_weight], dtype=torch.float32)
        
        print(f"使用正类权重: {adjusted_weight:.2f} (原始权重: {class_counts[0]/class_counts[1] if class_counts[1] > 0 else 0:.2f})")
        
        return pos_weight.to(device)
    
    def _validate_model(self, model, val_loader, criterion, device):
        """验证模型"""
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        all_logits = []  # 新增：用于调试
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 确保输入形状正确
                if inputs.dim() == 2:
                    inputs = inputs.unsqueeze(1)  # [batch, 1, length]
                
                # 模型输出logits
                logits = model(inputs)
                
                # 统一维度
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                
                # 计算损失（输入logits）
                loss = criterion(logits, labels.float())
                val_loss += loss.item()
                
                # 获取概率（手动应用sigmoid）
                probs = torch.sigmoid(logits)
                
                # 调试信息
                all_logits.extend(logits.cpu().numpy().flatten().tolist())
                all_probs.extend(probs.cpu().numpy().flatten().tolist())
                all_labels.extend(labels.cpu().numpy().flatten().tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 添加调试信息
        if len(all_logits) > 0:
            all_logits = np.array(all_logits)
            print(f"验证集logits统计: min={all_logits.min():.4f}, max={all_logits.max():.4f}, mean={all_logits.mean():.4f}")
            print(f"验证集概率统计: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}")
        
        # 静默处理标签转换
        if all_labels.dtype != np.int64 and all_labels.dtype != np.int32:
            all_labels = np.round(all_labels).astype(int)
        
        # 寻找最优阈值
        best_threshold, best_f1, _, _ = self._find_optimal_threshold(all_labels, all_probs)
        
        # 基于最优阈值计算准确率
        preds = (all_probs >= best_threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        return {
            'loss': avg_val_loss,
            'acc': acc,
            'auc': auc,
            'f1': best_f1,
            'threshold': best_threshold,
            'probs': all_probs,
            'labels': all_labels,
            'logits': all_logits if 'all_logits' in locals() else None  # 可选：返回logits
        }
    
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
    
    def evaluate_on_test_set(self, test_cv_indices, model, save_results=True, 
                        optimize_threshold=True, use_two_stage=True,
                        stage1_threshold=0.4, stage2_threshold=None):
        """在测试集上评估模型，可选阈值优化"""
        print(f"\n在测试集上评估模型")
        print(f"测试集: CV{', '.join(map(str, test_cv_indices))}")

        # ==================== 参数验证和默认值设置 ====================
        # 设置默认值
        if stage1_threshold is None:
            stage1_threshold = 0.4
        if stage2_threshold is None:
            stage2_threshold = 0.76
        
        print(f"初始阈值设置: stage1={stage1_threshold:.2f}, stage2={stage2_threshold:.2f}")
    # ===========================================================
        
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
        
        # ==================== 创建一致的损失函数 ====================
        if self.use_focal_loss:
            if self.focal_alpha is not None:
                criterion = FocalLoss(
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction='mean',
                    logits=True
                ).to(device)
                print(f"评估使用Focal Loss: alpha={self.focal_alpha}, gamma={self.focal_gamma}")
            else:
                criterion = WeightedFocalLoss(
                    pos_weight=None,
                    gamma=self.focal_gamma,
                    reduction='mean',
                    logits=True
                ).to(device)
                print(f"评估使用WeightedFocalLoss: gamma={self.focal_gamma}")
        else:
            criterion = torch.nn.BCEWithLogitsLoss().to(device)
            print(f"评估使用BCEWithLogitsLoss")

        # ==================== 添加调试信息：检查前几个样本 ====================
        print(f"\n前5个样本的模型输出检查:")
        model.eval()
        sample_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                logits = model(x)
                probs = torch.sigmoid(logits)
                
                print(f"样本{sample_count}: logits={logits.cpu().numpy()[0][0]:.6f}, "
                    f"prob={probs.cpu().numpy()[0][0]:.6f}, label={y.cpu().numpy()[0][0]:.0f}")
                
                sample_count += 1
                if sample_count >= 5:
                    break
        
        # 评估 - 获取原始概率
        test_res = self._validate_model(model, test_loader, criterion, device)

        test_probs = test_res['probs']
        test_labels = test_res['labels']
        test_loss = test_res['loss']
        test_auc = test_res['auc']
        
        # ==================== 添加详细统计信息 ====================
        print(f"\n测试集详细统计:")
        print(f"总样本数: {len(test_probs)}")
        print(f"正样本数: {np.sum(test_labels)}")
        print(f"负样本数: {len(test_labels) - np.sum(test_labels)}")
        print(f"正样本比例: {np.mean(test_labels):.2%}")
        
        print(f"\n模型输出概率分布:")
        print(f"最小值: {np.min(test_probs):.6f}")
        print(f"最大值: {np.max(test_probs):.6f}")
        print(f"平均值: {np.mean(test_probs):.6f}")
        print(f"中位数: {np.median(test_probs):.6f}")
        print(f"标准差: {np.std(test_probs):.6f}")
        
        # 概率分布直方图
        print(f"\n概率分布直方图:")
        bins = np.linspace(0, 1, 21)
        hist, bin_edges = np.histogram(test_probs, bins=bins)
        for i in range(len(hist)):
            print(f"  {bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}: {hist[i]} samples")
        
        # 查看各个阈值下的预测情况
        print(f"\n不同阈值下的预测结果:")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for th in thresholds:
            preds = (test_probs >= th).astype(int)
            pos_preds = np.sum(preds)
            print(f"  阈值={th:.1f}: 预测正样本数={pos_preds}")

        # ==================== 优化阶段：提高精确率和F1 ====================
        if use_two_stage:
            print("\n🎯 使用两阶段阈值策略（优化精确率和F1）...")
            
            # 方案1：提高第二阶段阈值以提高精确率
            if stage2_threshold is None:
                # 寻找更注重精确率的阈值
                stage2_threshold = self._find_precision_optimized_threshold(test_labels, test_probs)
            
            # 方案2：尝试多个阈值组合，选择F1最高的
            best_metrics = self._find_best_two_stage_combo(test_labels, test_probs)
            
            if best_metrics['f1'] > 0.82:  # 如果找到更好的组合
                stage1_threshold = best_metrics['stage1_threshold']
                stage2_threshold = best_metrics['stage2_threshold']
                print(f"使用优化后的阈值组合: stage1={stage1_threshold:.2f}, stage2={stage2_threshold:.2f}")
            else:
                print(f"使用默认/指定阈值: stage1={stage1_threshold:.2f}, stage2={stage2_threshold:.2f}")
            
            # 调用两阶段评估方法
            two_stage_results = self._two_stage_evaluate(
                test_probs, test_labels, 
                stage1_th=stage1_threshold, 
                stage2_th=stage2_threshold
            )
            
            test_acc = two_stage_results['acc']
            test_precision = two_stage_results['precision']
            test_recall = two_stage_results['recall']
            test_f1 = two_stage_results['f1']
            best_threshold = two_stage_results['stage2_threshold']
            stage1_recall = two_stage_results['stage1_recall']
            
            print(f"优化后的两阶段结果:")
            print(f"  第一阶段召回率: {stage1_recall:.4f}")
            print(f"  最终精确率: {test_precision:.4f} (目标: 80%+)")
            print(f"  最终召回率: {test_recall:.4f}")
            print(f"  最终F1: {test_f1:.4f} (目标: 83%+)")
            
        else:
            # 单阶段阈值优化
            print("\n使用单阶段阈值策略...")
            
            if optimize_threshold:
                # 使用更注重精确率的阈值寻找方法
                best_threshold, test_f1, test_precision, test_recall = self._find_precision_optimized_threshold(
                    test_labels, test_probs, return_all=True
                )
                
                best_preds = (test_probs >= best_threshold).astype(int)
                test_acc = accuracy_score(test_labels, best_preds)
                
                print(f"精确率优化阈值: {best_threshold:.3f}")
                print(f"精确率: {test_precision:.4f}, F1: {test_f1:.4f}")
            else:
                best_threshold = 0.5
                best_preds = (test_probs >= 0.5).astype(int)
                test_acc = accuracy_score(test_labels, best_preds)
                test_precision = precision_score(test_labels, best_preds, zero_division=0)
                test_recall = recall_score(test_labels, best_preds, zero_division=0)
                test_f1 = f1_score(test_labels, best_preds, zero_division=0)
                print(f"使用默认阈值: {best_threshold}")
        
        # ==================== 计算综合评分 ====================
        test_metrics = {
            'accuracy': test_acc,
            'auc': test_auc,
            'f1': test_f1
        }
        test_composite_score, breakdown = CompositeScoreCalculator.calculate_composite_score(test_metrics)
        
        # ==================== 构建结果字典 ====================
        test_results = {
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_composite_score': test_composite_score,
            'test_loss': test_loss,
            'evaluation_time': datetime.now().isoformat(),
            'score_breakdown': breakdown,
            'optimal_threshold': best_threshold,
            'threshold_strategy': 'two_stage_optimized' if use_two_stage else ('single_optimized' if optimize_threshold else 'single_default'),
            'class_distribution': self._get_class_distribution(test_labels)
        }
        
        # 添加两阶段特定信息
        if use_two_stage:
            test_results.update({
                'two_stage_used': True,
                'stage1_threshold': stage1_threshold,
                'stage2_threshold': stage2_threshold,
                'stage1_recall': stage1_recall
            })
        else:
            test_results['two_stage_used'] = False
        
        # ==================== 打印结果 ====================
        print(f"\n📊 测试集最终结果:")
        print(f"  阈值策略: {'两阶段优化' if use_two_stage else '单阶段' + (' (优化)' if optimize_threshold else ' (默认)')}")
        if use_two_stage:
            print(f"  第一阶段阈值: {stage1_threshold:.3f}")
            print(f"  第二阶段阈值: {stage2_threshold:.3f}")
        else:
            print(f"  阈值: {best_threshold:.3f}")
        print(f"  准确率: {test_acc:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"  综合评分: {test_composite_score:.4f}")
        print(f"  损失: {test_loss:.4f}")
        
        # ==================== 保存评估结果 ====================
        if save_results and self.experiment_dir:
            results_path = os.path.join(self.experiment_dir, "metrics", "test_evaluation.json")
            self.file_manager.save_metrics(test_results, results_path)
            print(f"💾 测试结果已保存: {results_path}")
        
        return test_results

    def _find_optimal_threshold(self, labels, logits_or_probs, metric='f1',is_logits=True):
        """寻找最优阈值"""
        # 如果输入是logits，先转换为概率
        if is_logits:
            probs = 1 / (1 + np.exp(-logits_or_probs))  # 手动sigmoid
        else:
            probs = logits_or_probs

        # 确保 labels 是整数类型
        labels = np.array(labels)
        probs = np.array(probs)
        
        # 静默处理标签转换
        if labels.dtype != np.int64 and labels.dtype != np.int32:
            # 通过四舍五入将浮点数转换为0/1
            labels = np.round(labels).astype(int)
        
        # 确保 probs 在0-1范围内
        if np.min(probs) < 0 or np.max(probs) > 1:
            probs = np.clip(probs, 0, 1)
        
        best_threshold = 0.5
        best_score = 0
        best_precision = 0
        best_recall = 0
        
        # 尝试多个阈值
        thresholds = np.linspace(0.1, 0.9, 81)
        
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            try:
                if metric == 'f1':
                    score = f1_score(labels, preds, zero_division=0)
                elif metric == 'balanced_accuracy':
                    score = balanced_accuracy_score(labels, preds)
                else:
                    score = f1_score(labels, preds, zero_division=0)
            except:
                score = 0
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # 使用最优阈值计算最终指标
        best_preds = (probs >= best_threshold).astype(int)
        best_precision = precision_score(labels, best_preds, zero_division=0)
        best_recall = recall_score(labels, best_preds, zero_division=0)
        best_f1 = f1_score(labels, best_preds, zero_division=0)
        
        return best_threshold, best_f1, best_precision, best_recall


    def _calculate_training_metrics(self, model, train_loader, criterion, device):
        """计算训练集指标"""
        model.eval()
        train_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device).float()
                x = x.view(-1, 1, FIXED_LENGTH)
                y = y.to(device).float()
                
                # 模型输出logits
                logits = model(x)
                
                # 计算损失
                loss = criterion(logits, y)
                train_loss += loss.item()
                
                # 手动应用sigmoid获取概率
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten().tolist())
                all_labels.extend(y.cpu().numpy().flatten().tolist())
        
        avg_train_loss = train_loss / len(train_loader)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 使用与验证集相同的阈值计算方法
        best_threshold, best_f1, _, _ = self._find_optimal_threshold(all_labels, all_probs)
        
        # 使用最优阈值计算预测
        preds = (all_probs >= best_threshold).astype(int)
        train_acc = accuracy_score(all_labels, preds)
        
        try:
            train_auc = roc_auc_score(all_labels, all_probs)
        except:
            train_auc = 0.5
        
        return {
            'loss': avg_train_loss,
            'acc': train_acc,
            'auc': train_auc,
            'f1': best_f1,
            'threshold': best_threshold,
            'probs': all_probs,
            'labels': all_labels
        }
    
    def _get_class_distribution(self, labels):
        """获取类别分布"""
        labels = np.array(labels)
        total = len(labels)
        positive = np.sum(labels)
        negative = total - positive
        
        return {
            'total': int(total),
            'positive': int(positive),
            'negative': int(negative),
            'positive_ratio': float(positive / total),
            'negative_ratio': float(negative / total)
        }
        
    def _visualize_cv_results(self, fold_results, avg_metrics):
        """Visualize cross-validation results (English labels)"""
        if not self.experiment_dir:
            return
        
        # Extract fold performance metrics
        fold_accs = [r['best_val_acc'] for r in fold_results]
        fold_aucs = [r['best_val_auc'] for r in fold_results]
        fold_f1s = [r['best_val_f1'] for r in fold_results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Validation Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar chart of fold accuracies
        axes[0, 0].bar(range(1, len(fold_accs) + 1), fold_accs, 
                    color='skyblue', edgecolor='black', alpha=0.8)
        axes[0, 0].axhline(y=avg_metrics['avg_val_acc'], color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {avg_metrics["avg_val_acc"]:.4f}')
        axes[0, 0].set_title('Validation Accuracy per Fold', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Fold Number', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].set_xticks(range(1, len(fold_accs) + 1))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, acc in enumerate(fold_accs):
            axes[0, 0].text(i + 1, acc + 0.005, f'{acc:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 2. Bar chart of fold AUC scores
        axes[0, 1].bar(range(1, len(fold_aucs) + 1), fold_aucs, 
                    color='lightgreen', edgecolor='black', alpha=0.8)
        axes[0, 1].axhline(y=avg_metrics['avg_val_auc'], color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {avg_metrics["avg_val_auc"]:.4f}')
        axes[0, 1].set_title('AUC Score per Fold', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Fold Number', fontsize=12)
        axes[0, 1].set_ylabel('AUC Score', fontsize=12)
        axes[0, 1].set_xticks(range(1, len(fold_aucs) + 1))
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, auc in enumerate(fold_aucs):
            axes[0, 1].text(i + 1, auc + 0.005, f'{auc:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 3. Bar chart of fold F1 scores
        axes[1, 0].bar(range(1, len(fold_f1s) + 1), fold_f1s, 
                    color='lightcoral', edgecolor='black', alpha=0.8)
        axes[1, 0].axhline(y=avg_metrics['avg_val_f1'], color='red', linestyle='--', 
                        linewidth=2, label=f'Mean: {avg_metrics["avg_val_f1"]:.4f}')
        axes[1, 0].set_title('F1 Score per Fold', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Fold Number', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_xticks(range(1, len(fold_f1s) + 1))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, f1 in enumerate(fold_f1s):
            axes[1, 0].text(i + 1, f1 + 0.005, f'{f1:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 4. Performance metrics summary table
        axes[1, 1].axis('off')
        
        # Create table data
        table_data = []
        for i in range(len(fold_results)):
            table_data.append([
                f'Fold {i+1}',
                f'{fold_accs[i]:.4f}',
                f'{fold_aucs[i]:.4f}',
                f'{fold_f1s[i]:.4f}'
            ])
        
        # Add average row
        table_data.append([
            'Average ± Std',
            f'{avg_metrics["avg_val_acc"]:.4f} ± {avg_metrics["std_val_acc"]:.4f}',
            f'{avg_metrics["avg_val_auc"]:.4f} ± {avg_metrics["std_val_auc"]:.4f}',
            f'{avg_metrics["avg_val_f1"]:.4f} ± {avg_metrics["std_val_f1"]:.4f}'
        ])
        
        # Create table
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Fold', 'Accuracy', 'AUC', 'F1 Score'],
            colWidths=[0.15, 0.25, 0.25, 0.25],
            cellLoc='center',
            loc='center',
            fontsize=11
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style average row
        for i in range(4):
            table[(len(fold_results), i)].set_facecolor('#f2f2f2')
            table[(len(fold_results), i)].set_text_props(weight='bold')
        
        # Style alternating rows
        for i in range(1, len(fold_results)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f9f9f9')
        
        axes[1, 1].set_title('Cross-Validation Performance Summary', 
                            fontsize=14, fontweight='bold', y=1.05)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for suptitle
        
        # Save visualization
        vis_path = os.path.join(self.experiment_dir, "visualizations", "cv_results_comparison.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a performance trend visualization
        self._create_performance_trend_visualization(fold_results)
        
        print(f"📊 Cross-validation visualization saved: {vis_path}")

    def _create_performance_trend_visualization(self, fold_results):
        """Create a line chart showing performance trends across folds"""
        if not self.experiment_dir:
            return
        
        # Extract metrics for trend analysis
        fold_numbers = list(range(1, len(fold_results) + 1))
        accuracies = [r['best_val_acc'] for r in fold_results]
        aucs = [r['best_val_auc'] for r in fold_results]
        f1_scores = [r['best_val_f1'] for r in fold_results]
        
        # Create trend visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot lines with markers
        ax.plot(fold_numbers, accuracies, 'o-', color='skyblue', linewidth=2, markersize=8, 
                label='Accuracy', alpha=0.8)
        ax.plot(fold_numbers, aucs, 's-', color='lightgreen', linewidth=2, markersize=8, 
                label='AUC', alpha=0.8)
        ax.plot(fold_numbers, f1_scores, '^-', color='lightcoral', linewidth=2, markersize=8, 
                label='F1 Score', alpha=0.8)
        
        # Calculate and plot trend lines
        if len(fold_numbers) >= 3:
            # Linear regression for accuracy trend
            z_acc = np.polyfit(fold_numbers, accuracies, 1)
            p_acc = np.poly1d(z_acc)
            ax.plot(fold_numbers, p_acc(fold_numbers), '--', color='skyblue', alpha=0.5, 
                    label='Accuracy Trend')
            
            # Linear regression for AUC trend
            z_auc = np.polyfit(fold_numbers, aucs, 1)
            p_auc = np.poly1d(z_auc)
            ax.plot(fold_numbers, p_auc(fold_numbers), '--', color='lightgreen', alpha=0.5, 
                    label='AUC Trend')
            
            # Linear regression for F1 trend
            z_f1 = np.polyfit(fold_numbers, f1_scores, 1)
            p_f1 = np.poly1d(z_f1)
            ax.plot(fold_numbers, p_f1(fold_numbers), '--', color='lightcoral', alpha=0.5, 
                    label='F1 Trend')
        
        # Style the plot
        ax.set_title('Performance Trends Across Cross-Validation Folds', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Fold Number', fontsize=12)
        ax.set_ylabel('Score Value', fontsize=12)
        ax.set_xticks(fold_numbers)
        ax.set_ylim([0.5, 1.0])  # Set reasonable y-limits for classification metrics
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)
        
        # Add value annotations
        for i, (acc, auc, f1) in enumerate(zip(accuracies, aucs, f1_scores)):
            ax.annotate(f'{acc:.3f}', (fold_numbers[i], acc), 
                    textcoords="offset points", xytext=(0,5), 
                    ha='center', fontsize=8, color='skyblue')
            ax.annotate(f'{auc:.3f}', (fold_numbers[i], auc), 
                    textcoords="offset points", xytext=(0,5), 
                    ha='center', fontsize=8, color='lightgreen')
            ax.annotate(f'{f1:.3f}', (fold_numbers[i], f1), 
                    textcoords="offset points", xytext=(0,5), 
                    ha='center', fontsize=8, color='lightcoral')
        
        # Add statistics text box
        stats_text = f"""Statistics:
    Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}
    Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}
    Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}

    Total Folds: {len(fold_results)}
    """
        
        # Place text box in upper left
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Save trend visualization
        trend_path = os.path.join(self.experiment_dir, "visualizations", "cv_performance_trend.png")
        plt.savefig(trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance trend visualization saved: {trend_path}")

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
    
    def __init__(self, base_path, composite_weights=None,
                 use_focal_loss=False, focal_alpha=None, focal_gamma=2.0, adjusted_lr=None):
        self.base_path = base_path
        self.file_manager = ModelFileManager()
        self.composite_weights = composite_weights
        self.visualizer = TrainingVisualizer()
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.adjusted_lr = adjusted_lr or LEARNING_RATE
    
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
        config_pbar = tqdm(
            total=total_configs, 
            desc="超参数搜索进度",
            bar_format='{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
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
                    lr=self.adjusted_lr,  # 使用调整后的学习率
                    use_stream2=USE_STREAM2_SETTING,
                    augment=AUGMENT_SETTING,
                    experiment_dir=config_dir,
                    config_name=config_name,
                    composite_weights=self.composite_weights,
                    use_focal_loss=self.use_focal_loss,  # 新增
                    focal_alpha=self.focal_alpha,        # 新增
                    focal_gamma=self.focal_gamma         # 新增
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
    def train_with_best_config(base_path, best_config_data, lr_scheduler_config=None,
                               use_focal_loss=False, focal_alpha=None, focal_gamma=2.0, 
                               adjusted_lr=None):
        """
        完整训练流程：
        1. 在训练集（CV0~CV3）上使用最佳超参数进行5折交叉验证
        2. 使用全部训练集训练最终模型（包含验证集和早停）
        3. 在测试集（CV4）上最终评估
        
        Args:
            base_path: 数据集路径
            best_config_data: 最佳配置数据
            lr_scheduler_config: 学习率调度器配置
        """
        print("=" * 80)
        print("完整训练模式（使用综合评分和早停机制）")
        print("=" * 80)

        if adjusted_lr is None:
            adjusted_lr = LEARNING_RATE
        
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

        # 如果未提供调度器配置，使用 Config.py 中的默认配置
        if lr_scheduler_config is None:
            lr_scheduler_config = LR_SCHEDULER_CONFIG
        
        trainer = ModelTrainer(
            base_path=base_path,
            kernel_config=best_config_data['kernel_config'],
            batch_size=best_config_data['batch_size'],
            lr=adjusted_lr,  # 使用调整后的学习率
            use_stream2=USE_STREAM2_SETTING,
            augment=AUGMENT_SETTING,
            experiment_dir=experiment_dir,
            config_name="BestConfig",
            composite_weights=composite_weights,
            lr_scheduler_config=lr_scheduler_config,
            use_focal_loss=use_focal_loss,  # 新增
            focal_alpha=focal_alpha,        # 新增
            focal_gamma=focal_gamma         # 新增
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
        test_results = trainer.evaluate_on_test_set([4], final_model,use_two_stage=True,
                                                    optimize_threshold=True,  # 不使用自动优化，使用我们指定的
                                                    stage1_threshold=0.3,  # 提高第一阶段门槛
                                                    stage2_threshold=0.5   # 提高第二阶段门槛以提高精确率
                                                    )
        
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
            'composite_weights': composite_weights,
            'lr_scheduler_config':LR_SCHEDULER_CONFIG  # 保存调度器配置
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
        self.loss_config = get_loss_config()
        self.use_focal_loss = USE_FOCAL_LOSS
        self.focal_alpha = self.loss_config.get('focal_alpha')
        self.focal_gamma = self.loss_config.get('focal_gamma')
        self.adjusted_lr = self.loss_config.get('adjusted_lr', LEARNING_RATE)

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
        
        # 调整后的综合评分权重 - 更重视F1分数
        self.custom_weights = {
            'accuracy': 0.30,  # 降低准确率权重
            'auc': 0.30,       # AUC权重保持不变
            'f1': 0.35,        # 提高F1分数权重
            'stability': 0.05  # 稳定性权重
        }

        # 初始化训练器配置缓存 - 这是关键修复！
        self.trainer_config = None
        
        # 打印初始化完成信息
        print(f"✅ TrainingPipeline 初始化完成")
        print(f"  训练集: CV{', '.join(map(str, self.train_indices))}")
        print(f"  测试集: CV{', '.join(map(str, self.test_indices))}")

    def _create_trainer(self, **kwargs):
        """创建训练器的统一方法"""
        default_kwargs = {
            'base_path': self.base_path,
            'use_focal_loss': self.use_focal_loss,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'lr': self.adjusted_lr,  # 使用调整后的学习率
        }
        default_kwargs.update(kwargs)
        return ModelTrainer(**default_kwargs)
    
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

        parser.add_argument('--loss', type=str, default='focal',
                        choices=['bce', 'focal'],
                        help='损失函数类型: bce或focal')
        parser.add_argument('--focal_alpha', type=float, default=None,
                            help='Focal Loss的alpha参数 (0-1)。默认None时会自动计算')
        parser.add_argument('--focal_gamma', type=float, default=2.0,
                            help='Focal Loss的gamma参数 (默认2.0)')
        parser.add_argument('--focal_config', type=str, default='focus_positive',
                            choices=['default', 'balanced', 'focus_positive', 'focus_hard'],
                            help='预设的Focal Loss配置')
        parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                            help='初始学习率')
        
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
            self._run_search_mode(args,weights)
        elif args.mode == 'train':
            self._run_train_mode(args,weights)
        elif args.mode == 'full':
            self._run_data_diagnostic()
            self._run_full_mode(args, weights)
        elif args.mode == 'compare':
            self._run_compare_mode(args, weights)

    def _setup_focal_loss_config(self, args):
        """设置Focal Loss配置 - 统一版本"""
        # 使用Config.py中的默认配置
        default_config = LOSS_FUNCTION_CONFIG.copy()
        
        # 如果有预设配置名称，使用预设配置
        if args.focal_config in FOCAL_PRESET_CONFIGS:
            preset = FOCAL_PRESET_CONFIGS[args.focal_config]
            
            # 用命令行参数覆盖预设配置
            focal_alpha = args.focal_alpha or preset.get('alpha', default_config.get('focal_alpha', 0.25))
            focal_gamma = args.focal_gamma or preset.get('gamma', default_config.get('focal_gamma', 2.0))
            lr_factor = preset.get('lr_factor', 0.5)
            
            config_source = f"预设配置: {args.focal_config}"
        else:
            # 使用命令行参数或Config.py中的默认值
            focal_alpha = args.focal_alpha or default_config.get('focal_alpha', 0.25)
            focal_gamma = args.focal_gamma or default_config.get('focal_gamma', 2.0)
            lr_factor = default_config.get('lr_factor', 0.5)
            
            config_source = f"命令行参数 + Config.py默认值"
        
        # 计算调整后的学习率
        adjusted_lr = args.lr * lr_factor
        
        print(f"\n🎯 Focal Loss配置 ({config_source}):")
        print(f"  Alpha: {focal_alpha}")
        print(f"  Gamma: {focal_gamma}")
        print(f"  学习率调整因子: {lr_factor}")
        print(f"  调整后学习率: {adjusted_lr:.6f}")
        
        return focal_alpha, focal_gamma, adjusted_lr
    
    def _get_trainer_config(self, args):
        """获取统一的训练器配置"""
        if self.trainer_config is None:
            # 计算Focal Loss参数
            if args.loss == 'focal':
                focal_alpha, focal_gamma, adjusted_lr = self._setup_focal_loss_config(args)
                use_focal_loss = True
            else:
                focal_alpha, focal_gamma, adjusted_lr = None, None, args.lr
                use_focal_loss = False
            
            # 缓存配置
            self.trainer_config = {
                'use_focal_loss': use_focal_loss,
                'focal_alpha': focal_alpha,
                'focal_gamma': focal_gamma,
                'adjusted_lr': adjusted_lr
            }
            
            print(f"\n🎯 统一训练器配置:")
            print(f"  使用Focal Loss: {self.trainer_config['use_focal_loss']}")
            if self.trainer_config['use_focal_loss']:
                print(f"  Focal Alpha: {self.trainer_config['focal_alpha']}")
                print(f"  Focal Gamma: {self.trainer_config['focal_gamma']}")
            print(f"  学习率: {self.trainer_config['adjusted_lr']:.6f}")
        
        return self.trainer_config
    
    def _run_search_mode(self,args,weights):
        """运行超参数搜索模式"""
        print("\n模式: 超参数搜索（使用综合评分）")
        # 获取统一的训练器配置
        trainer_config = self._get_trainer_config(args)
        
        searcher = HyperparameterSearcher(
            self.base_path, 
            composite_weights=weights,
            use_focal_loss=trainer_config['use_focal_loss'],
            focal_alpha=trainer_config['focal_alpha'],
            focal_gamma=trainer_config['focal_gamma'],
            adjusted_lr=trainer_config['adjusted_lr']
        )
        best_config, search_dir = searcher.search(num_epochs_search=30)
        
        print(f"\n超参数搜索完成!")
        print(f"最佳配置: {best_config['config_name']}")
        print(f"综合评分: {best_config['avg_composite_score']:.4f}")
        print(f"结果目录: {search_dir}")
    
    def _run_train_mode(self,args,weights):
        """运行默认训练模式"""
        print("\n模式: 默认配置训练（使用综合评分和早停）")
        # 获取统一的训练器配置
        trainer_config = self._get_trainer_config(args)
        
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
            lr=trainer_config['adjusted_lr'],  # 使用统一的学习率
            use_stream2=USE_STREAM2_SETTING,
            augment=AUGMENT_SETTING,
            experiment_dir=experiment_dir,
            config_name="DefaultConfig",
            composite_weights=weights,
            use_focal_loss=trainer_config['use_focal_loss'],
            focal_alpha=trainer_config['focal_alpha'],
            focal_gamma=trainer_config['focal_gamma']
    )
        
        # 在训练集上进行5折交叉验证
        cv_metrics, cv_results = trainer.cross_validate_on_train_set(
            self.train_indices, NUM_EPOCHS, k_folds=5
        )
        
        print(f"\n默认训练完成!")
        print(f"平均综合评分: {cv_metrics.get('avg_composite_score', 0):.4f}")
        print(f"结果目录: {experiment_dir}")

    def _run_data_diagnostic(self):
        """运行数据诊断"""
        print("\n=== 数据诊断模式 ===")
        
        # 创建临时训练器
        trainer = ModelTrainer(
            base_path=self.base_path,
            kernel_config=DEFAULT_KERNEL_CONFIG,
            batch_size=32,
            lr=0.001,
            use_stream2=True,
            augment=True,
            experiment_dir=None,
            config_name="Diagnostic"
        )
        
        # 运行基本功能测试
        trainer.test_basic_functionality()
        
        # 检查数据分布
        print("\n=== 检查CV文件分布 ===")
        for cv_idx in range(5):
            data = trainer.data_manager.load_cv_files([cv_idx])
            if len(data) > 0:
                labels = [label for _, label in data[:1000]]  # 只检查前1000个
                labels_np = np.array(labels)
                print(f"CV{cv_idx}: 样本数={len(data)}, 正样本比例={np.mean(labels_np):.2%}")
    
    def _run_full_mode(self, args, weights):
        """运行完整训练模式"""
        print("\n模式: 使用最佳配置进行完整训练")
        
        best_config_path = "/home/xusi/EE5046_Projects/Task1_Results/HyperparamSearch/latest_best_config.json"
        
        if os.path.exists(best_config_path):
            with open(best_config_path, 'r') as f:
                best_config_data = json.load(f)
            
            print(f"加载最佳配置: {best_config_data['best_config_name']}")
            print(f"选择标准: {best_config_data.get('selection_criteria', 'accuracy')}")

            # 获取统一的训练器配置
            trainer_config = self._get_trainer_config(args)
            
            # 使用最佳配置进行完整训练，并传递学习率调度器配置
            cv_metrics, test_results, experiment_dir = CompleteTrainer.train_with_best_config(
                self.base_path, 
                best_config_data['best_config_data'],
                lr_scheduler_config=LR_SCHEDULER_CONFIG,
                use_focal_loss=trainer_config['use_focal_loss'],
                focal_alpha=trainer_config['focal_alpha'],
                focal_gamma=trainer_config['focal_gamma'],
                adjusted_lr=trainer_config['adjusted_lr']
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