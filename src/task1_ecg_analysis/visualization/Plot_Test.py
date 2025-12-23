import json
from src.task1_ecg_analysis.models.ECGAnalyzer import ECGAnalyzer
from src.common.Config import DATASET_PATH

analyzer = ECGAnalyzer(DATASET_PATH)

indices_A = analyzer.get_indices_by_label('A')
indices_O = analyzer.get_indices_by_label('O')
indices_N = analyzer.get_indices_by_label('N')
indices_Tilde = analyzer.get_indices_by_label('~') # 噪声标签

print(f"A (房颤) 样本数量: {len(indices_A)}")
print(f"O (其他) 样本数量: {len(indices_O)}")
print(f"N (正常) 样本数量: {len(indices_N)}")
print(f"~ (噪声) 样本数量: {len(indices_Tilde)}")

# 1. 绘制原始信号分析图 (随机选择一个样本)
analyzer.plot_analysis(save_plot=True, idx=indices_Tilde[3])

# # 2. 绘制数据增强对比图 (随机选择一个样本)
# analyzer.plot_augmentation_comparison(save_plot=True)

# # 3. 绘制特定样本的增强对比图 (例如索引 100)
# fig, original, augmented = analyzer.plot_augmentation_comparison(save_plot=True, idx=100)
# print("绘图完成，增强前后的数据已保存在变量 original_data 和 augmented_data 中。")

# 绘制json文件
# json_path = "/home/xusi/Logs/FinalTraining/Results_20251212_121215/final_results.json"
# save_path = "/home/xusi/EE5046_Projects/Figures/5-Folds-json-plots"
# try:
#     # 生成完整分析图
#     summary = analyzer.plot_cnn_results_from_json(
#         json_file_path=json_path,
#         save_path=save_path  # 保存为PNG文件
#     )
        
#     print("\n" + "="*50)
#     print("CNN模型性能分析摘要")
#     print("="*50)
#     print(f"模型配置: {summary['model_config']['kernel_name']}")
#     print(f"平均准确率: {summary['performance']['avg_accuracy']:.4f}")
#     print(f"平均AUC: {summary['performance']['avg_auc']:.4f}")
#     print(f"平均损失: {summary['performance']['avg_loss']:.4f}")
        
#     # 生成简化版对比图（适合PPT）
#     stats = analyzer.plot_simple_comparison(
#         json_file_path=json_path,
#         save_path=save_path
#     )
        
#     print("\n" + "="*50)
#     print("关键统计数据")
#     print("="*50)
#     print(f"平均准确率: {stats['avg_accuracy']:.4f} (±{stats['accuracy_std']:.4f})")
#     print(f"平均AUC: {stats['avg_auc']:.4f} (±{stats['auc_std']:.4f})")
#     print(f"模型: {stats['kernel_name']}")
        
# except FileNotFoundError:
#     print(f"错误: 未找到JSON文件: {json_path}")
#     print("请检查文件路径是否正确")
# except json.JSONDecodeError:
#     print(f"错误: JSON文件格式不正确: {json_path}")
# except Exception as e:
#     print(f"生成图表时出错: {e}")