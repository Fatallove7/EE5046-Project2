from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


class TrainingVisualizer:
    """训练过程可视化"""
    
    @staticmethod
    def create_progress_bar(total, desc, position=0):
        """创建进度条"""
        return tqdm(total=total, desc=desc, position=position, leave=False, 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
        """绘制训练历史图表"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练历史图表已保存: {save_path}")
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, title, save_path):
        """绘制指标对比雷达图"""
        labels = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # 如果是两个指标，使用柱状图
        if len(labels) <= 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            bars = ax.bar(labels, values, color=colors, edgecolor='black')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
        else:
            # 多个指标使用雷达图
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]  # 闭合图形
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 指标对比图已保存: {save_path}")
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc, save_path):
        """绘制ROC曲线"""
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📉 ROC曲线已保存: {save_path}")