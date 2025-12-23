import json
import os
import matplotlib.pyplot as plt
import numpy as np

save_dir = "/home/xusi/EE5046_Projects/Figures"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
json_path = "/home/xusi/Logs/FinalTraining/Results_20251217_131604/final_results.json"
# 读取JSON文件
with open(json_path, 'r') as f:
    data = json.load(f)

# ==================== 第一部分：各个fold的指标对比 ====================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Performance Metrics Across 5 Folds', fontsize=16, fontweight='bold')

# 提取fold数据
fold_nums = [f'Fold {i}' for i in range(5)]
losses = [fold['best_loss'] for fold in data['fold_results']]
accuracies = [fold['final_val_acc'] for fold in data['fold_results']]
aucs = [fold['final_val_auc'] for fold in data['fold_results']]
f1_scores = [fold['final_val_f1'] for fold in data['fold_results']]

# 颜色设置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 子图1: Loss对比
bars1 = axes1[0, 0].bar(fold_nums, losses, color=colors, edgecolor='black')
axes1[0, 0].set_title('Validation Loss per Fold', fontweight='bold')
axes1[0, 0].set_ylabel('Loss', fontweight='bold')
axes1[0, 0].set_ylim(0, max(losses) * 1.1)
axes1[0, 0].grid(True, alpha=0.3, axis='y')

# 在每个柱子上添加数值
for bar in bars1:
    height = bar.get_height()
    axes1[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

# 子图2: Accuracy对比
bars2 = axes1[0, 1].bar(fold_nums, accuracies, color=colors, edgecolor='black')
axes1[0, 1].set_title('Accuracy per Fold', fontweight='bold')
axes1[0, 1].set_ylabel('Accuracy', fontweight='bold')
axes1[0, 1].set_ylim(0.9, 1.0)  # 根据数据范围调整
axes1[0, 1].grid(True, alpha=0.3, axis='y')

for bar in bars2:
    height = bar.get_height()
    axes1[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# 子图3: AUC对比
bars3 = axes1[1, 0].bar(fold_nums, aucs, color=colors, edgecolor='black')
axes1[1, 0].set_title('AUC per Fold', fontweight='bold')
axes1[1, 0].set_ylabel('AUC', fontweight='bold')
axes1[1, 0].set_xlabel('Fold Number', fontweight='bold')
axes1[1, 0].set_ylim(0.94, 1.0)  # 根据数据范围调整
axes1[1, 0].grid(True, alpha=0.3, axis='y')

for bar in bars3:
    height = bar.get_height()
    axes1[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# 子图4: F1 Score对比
bars4 = axes1[1, 1].bar(fold_nums, f1_scores, color=colors, edgecolor='black')
axes1[1, 1].set_title('F1 Score per Fold', fontweight='bold')
axes1[1, 1].set_ylabel('F1 Score', fontweight='bold')
axes1[1, 1].set_xlabel('Fold Number', fontweight='bold')
axes1[1, 1].set_ylim(0.6, 0.85)  # 根据数据范围调整
axes1[1, 1].grid(True, alpha=0.3, axis='y')

for bar in bars4:
    height = bar.get_height()
    axes1[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
fold_performance_path = os.path.join(save_dir, 'fold_performance_comparison.png')
plt.savefig(fold_performance_path, dpi=300, bbox_inches='tight')
plt.show()

# ==================== 第二部分：Final Metrics统计 ====================
fig2, ax2 = plt.subplots(figsize=(12, 6))

# 提取final_metrics数据
final_metrics = data['final_metrics']
avg_metrics = ['avg_loss', 'avg_accuracy', 'avg_auc', 'avg_precision', 'avg_recall', 'avg_f1']
std_metrics = ['std_accuracy', 'std_auc', 'std_precision', 'std_recall', 'std_f1']

# 对应的显示名称
metric_names = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score']

# 准备数据
values = [final_metrics[metric] for metric in avg_metrics]
# 注意：avg_loss没有对应的std，所以我们需要特殊处理
errors = []
for i, metric in enumerate(avg_metrics):
    if metric == 'avg_loss':
        errors.append(0)  # loss没有标准差
    else:
        # 找到对应的std指标
        std_key = metric.replace('avg_', 'std_')
        errors.append(final_metrics.get(std_key, 0))

# 创建柱状图
x_pos = np.arange(len(metric_names))
bars = ax2.bar(x_pos, values, yerr=errors, capsize=5, 
               color='#2ca02c', edgecolor='black', alpha=0.8, error_kw=dict(ecolor='red', lw=1.5))

# 设置图表属性
ax2.set_title('Final Metrics with Standard Deviation', fontsize=14, fontweight='bold')
ax2.set_xlabel('Metrics', fontweight='bold')
ax2.set_ylabel('Value', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(metric_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(values) * 1.2)  # 留出空间显示数值

# 在每个柱子上添加数值和误差范围
for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
    height = bar.get_height()
    if metric_names[i] == 'Loss':
        text = f'{value:.3f}'
    else:
        text = f'{value:.4f}\n(±{error:.3f})' if error > 0 else f'{value:.4f}'
    
    ax2.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
             text, ha='center', va='bottom', fontsize=9)

# 添加水平参考线（平均线）
mean_value = np.mean([v for i, v in enumerate(values) if i != 0])  # 排除loss
ax2.axhline(y=mean_value, color='blue', linestyle='--', alpha=0.5, 
            label=f'Avg (exc. loss): {mean_value:.4f}')

# 添加图例
ax2.legend()

plt.tight_layout()
final_metrics_path = os.path.join(save_dir, 'final_metrics_summary.png')
plt.savefig(final_metrics_path, dpi=300, bbox_inches='tight')
plt.show()

# ==================== 第三部分：额外的综合分析图 ====================
# 创建综合对比图：各个fold的四个指标对比
fig3, ax3 = plt.subplots(figsize=(12, 6))

# 设置宽度
x = np.arange(len(fold_nums))
width = 0.2

# 绘制分组柱状图
rects1 = ax3.bar(x - 1.5*width, losses, width, label='Loss', color='#d62728')
rects2 = ax3.bar(x - 0.5*width, accuracies, width, label='Accuracy', color='#1f77b4')
rects3 = ax3.bar(x + 0.5*width, aucs, width, label='AUC', color='#2ca02c')
rects4 = ax3.bar(x + 1.5*width, f1_scores, width, label='F1 Score', color='#ff7f0e')

# 设置图表属性
ax3.set_title('Comprehensive Performance Comparison Across Folds', fontsize=14, fontweight='bold')
ax3.set_xlabel('Fold Number', fontweight='bold')
ax3.set_ylabel('Value', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(fold_nums)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 自动调整y轴范围
all_values = losses + accuracies + aucs + f1_scores
ax3.set_ylim(min(all_values)*0.95, max(all_values)*1.05)

plt.tight_layout()
comprehensive_path = os.path.join(save_dir, 'comprehensive_comparison.png')
plt.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图表已保存到: {save_dir}")
print(f"1. Fold性能对比图: {fold_performance_path}")
print(f"2. 最终指标汇总图: {final_metrics_path}")
print(f"3. 综合分析图: {comprehensive_path}")