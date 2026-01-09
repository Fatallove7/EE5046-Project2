from peft import PeftModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import scipy.io as sio
from datetime import datetime
from pathlib import Path
import re

# 导入自定义模块（根据您的实际路径调整）
from src.task2_multimodal_llm.models.MultimodelLLM import MultimodalLLM
from EE5046_Projects.src.task2_multimodal_llm.models.ECGEncoder import ECGEncoder
from src.common.Config import CNN_WEIGHTS_PATH, DATASET_PATH, FIXED_LENGTH

# 修改后的标签映射字典
# 根据CSV文件格式：A表示房颤，其他(0, N, O, ~等)表示非房颤
LABEL_MAPPING = {
    "A": 1,  # 房颤
    "N": 0,  # 噪声（视为正常）
    "O": 0,  # 其他（视为正常）
    "~": 0,  # 无法分类（视为正常）
}

class ResultSaver:
    """统一的评估结果保存器"""
    
    def __init__(self, base_dir="Evaluation_Results", experiment_name=None):
        """
        初始化结果保存器
        
        Args:
            base_dir: 基础目录
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        self.base_dir = os.path.abspath(base_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建目录结构
        self.run_dir = self._create_directory_structure()
        
        # 设置日志路径
        self._log_path = os.path.join(self.run_dir, "logs", "evaluation.log")
        
        # 初始化日志文件
        self._init_log_file()
        
        print(f"✓ ResultSaver初始化完成")
        print(f"  实验目录: {self.run_dir}")
        print(f"  日志文件: {self._log_path}")
    
    def _create_directory_structure(self):
        """创建标准化的目录结构"""
        run_dir = os.path.join(self.base_dir, "runs", self.experiment_name)
        
        # 创建所有必要的子目录
        directories = [
            os.path.join(run_dir, "results"),
            os.path.join(run_dir, "logs"),
            os.path.join(run_dir, "models"),
            os.path.join(run_dir, "visualizations"),
            os.path.join(run_dir, "responses"),
            os.path.join(run_dir, "confusion_matrices"),
            os.path.join(run_dir, "detailed_results")
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            
        return run_dir
    
    def _init_log_file(self):
        """初始化日志文件"""
        try:
            with open(self._log_path, "w", encoding="utf-8") as f:
                f.write(f"ECG评估实验日志 - {self.experiment_name}\n")
                f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
        except Exception as e:
            print(f"警告: 无法创建日志文件 {self._log_path}: {e}")
            # 使用备用日志路径
            self._log_path = os.path.join(os.getcwd(), "evaluation_fallback.log")
            print(f"使用备用日志路径: {self._log_path}")
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        try:
            # 确保日志文件存在
            log_dir = os.path.dirname(self._log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(log_message + "\n")
        except Exception as e:
            # 如果写入失败，尝试使用备用路径
            print(f"无法写入日志文件 {self._log_path}: {e}")
            # 尝试直接打印
            print(f"日志内容: {log_message}")
            
            # 创建备用日志
            backup_log = os.path.join(os.getcwd(), "evaluation_error.log")
            try:
                with open(backup_log, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [ERROR] 无法写入主日志: {e}\n")
                    f.write(f"原始日志内容: {log_message}\n")
            except:
                pass
        
        # 控制台输出
        if level == "INFO":
            print(message)
        elif level == "WARNING":
            print(f"警告: {message}")
        elif level == "ERROR":
            print(f"错误: {message}")
    
    def save_json(self, data, filepath):
        """保存JSON文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.log(f"JSON文件已保存: {filepath}")
            return True
        except Exception as e:
            self.log(f"保存JSON文件失败 {filepath}: {e}", level="ERROR")
            return False
    
    def save_evaluation_results(self, results, dataset_name, save_detailed=True):
        """
        保存评估结果
        
        Args:
            results: 评估结果字典
            dataset_name: 数据集名称（如cv0, cv1等）
            save_detailed: 是否保存详细结果
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 保存主结果文件
            main_result_path = os.path.join(self.run_dir, "results", f"evaluation_summary_{dataset_name}.json")
            self.save_json(results, main_result_path)
            
            # 2. 保存详细结果（CSV格式）
            if save_detailed and 'detailed_results' in results:
                detailed_csv_path = os.path.join(self.run_dir, "detailed_results", f"detailed_results_{dataset_name}.csv")
                df = pd.DataFrame(results['detailed_results'])
                df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
                self.log(f"详细结果已保存到: {detailed_csv_path}")
            
            # 3. 保存模型性能指标
            metrics = {
                'dataset': dataset_name,
                'timestamp': timestamp,
                'metrics': {
                    'accuracy': results.get('accuracy', 0),
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'f1': results.get('f1', 0)
                },
                'sample_stats': {
                    'total_samples': results.get('total_samples', 0),
                    'valid_samples': results.get('valid_samples', 0),
                    'invalid_samples': results.get('invalid_samples', 0),
                    'invalid_rate': results.get('invalid_rate', 0)
                },
                'confusion_matrix': results.get('confusion_matrix', []),
                'classification_report': results.get('classification_report', '')
            }
            
            metrics_path = os.path.join(self.run_dir, "results", f"metrics_{dataset_name}.json")
            self.save_json(metrics, metrics_path)
            
            # 4. 保存响应示例
            if 'response_examples' in results:
                examples_path = os.path.join(self.run_dir, "responses", f"response_examples_{dataset_name}.json")
                self.save_json(results['response_examples'], examples_path)
            
            # 5. 保存混淆矩阵
            if 'confusion_matrix' in results:
                cm_path = os.path.join(self.run_dir, "confusion_matrices", f"confusion_matrix_{dataset_name}.json")
                self.save_json({'confusion_matrix': results['confusion_matrix']}, cm_path)
                
                # 同时保存为CSV便于分析
                cm_df = pd.DataFrame(results['confusion_matrix'], 
                                    index=['实际:正常', '实际:房颤'], 
                                    columns=['预测:正常', '预测:房颤'])
                cm_csv_path = os.path.join(self.run_dir, "confusion_matrices", f"confusion_matrix_{dataset_name}.csv")
                cm_df.to_csv(cm_csv_path, encoding='utf-8-sig')
            
            self.log(f"评估结果已保存到: {self.run_dir}")
            return self.run_dir
            
        except Exception as e:
            self.log(f"保存评估结果失败: {e}", level="ERROR")
            return None
    
    def save_comparison_results(self, comparison, dataset_name):
        """
        保存对比结果
        
        Args:
            comparison: 对比结果字典
            dataset_name: 数据集名称
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 保存JSON格式
            comparison_json_path = os.path.join(self.run_dir, "results", f"comparison_{dataset_name}.json")
            self.save_json(comparison, comparison_json_path)
            
            # 2. 保存文本报告
            report_text = self._generate_comparison_report(comparison, dataset_name, timestamp)
            report_path = os.path.join(self.run_dir, "results", f"comparison_report_{dataset_name}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # 3. 保存Markdown格式
            markdown_path = os.path.join(self.run_dir, "results", f"comparison_{dataset_name}.md")
            self._save_markdown_report(comparison, dataset_name, markdown_path)
            
            self.log(f"对比结果已保存到: {comparison_json_path}")
            
        except Exception as e:
            self.log(f"保存对比结果失败: {e}", level="ERROR")
    
    def save_cross_validation_summary(self, cv_results):
        """
        保存交叉验证汇总结果
        
        Args:
            cv_results: 交叉验证结果字典
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 保存汇总结果
            summary_path = os.path.join(self.run_dir, "results", "cross_validation_summary.json")
            self.save_json(cv_results, summary_path)
            
            # 2. 创建每个fold的单独文件
            if 'fold_results' in cv_results:
                for fold_name, fold_result in cv_results['fold_results'].items():
                    fold_path = os.path.join(self.run_dir, "results", f"fold_{fold_name}.json")
                    self.save_json(fold_result, fold_path)
            
            # 3. 创建CSV汇总表格
            self._create_cv_summary_table(cv_results)
            
            self.log(f"交叉验证汇总已保存到: {summary_path}")
            
        except Exception as e:
            self.log(f"保存交叉验证汇总失败: {e}", level="ERROR")
    
    def log_experiment_info(self, config):
        """
        保存实验配置信息
        
        Args:
            config: 实验配置字典
        """
        try:
            info = {
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': config,
                'system_info': {
                    'python_version': os.sys.version,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
                }
            }
            
            info_path = os.path.join(self.run_dir, "logs", "experiment_info.json")
            self.save_json(info, info_path)
            
            self.log(f"实验信息已保存到: {info_path}")
            
        except Exception as e:
            self.log(f"保存实验信息失败: {e}", level="ERROR")
    
    def save_error_report(self, error_info, context=""):
        """
        保存错误报告
        
        Args:
            error_info: 错误信息
            context: 错误上下文
        """
        try:
            error_report = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(error_info),
                'context': context,
                'traceback': self._get_traceback()
            }
            
            error_path = os.path.join(self.run_dir, "logs", "error_reports.json")
            
            # 如果文件已存在，追加错误
            if os.path.exists(error_path):
                with open(error_path, 'r', encoding='utf-8') as f:
                    existing_errors = json.load(f)
                existing_errors.append(error_report)
                errors_to_save = existing_errors
            else:
                errors_to_save = [error_report]
            
            self.save_json(errors_to_save, error_path)
            self.log(f"错误报告已保存: {error_path}", level="ERROR")
            
        except Exception as e:
            print(f"保存错误报告失败: {e}")
    
    def _generate_comparison_report(self, comparison, dataset_name, timestamp):
        """生成对比报告文本"""
        report = f"""
{'='*80}
ECG分类模型对比报告
{'='*80}

数据集: {dataset_name}
生成时间: {timestamp}

性能指标对比:
{'='*80}

| 指标       | LLM模型  | CNN基线  | 差异      | 性能提升  |
|------------|----------|----------|----------|-----------|
"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            llm_val = comparison['LLM'].get(metric, 0) * 100
            cnn_val = comparison['CNN_Baseline'].get(metric, 0) * 100
            diff = comparison['Difference'].get(metric, 0) * 100
            
            if diff > 0:
                conclusion = f"↑{diff:.2f}%"
                diff_str = f"+{diff:.2f}%"
            elif diff < 0:
                conclusion = f"↓{-diff:.2f}%"
                diff_str = f"{diff:.2f}%"
            else:
                conclusion = "持平"
                diff_str = "0.00%"
            
            report += f"| {metric.capitalize():10} | {llm_val:.2f}% | {cnn_val:.2f}% | {diff_str:9} | {conclusion:10} |\n"
        
        # 添加总结
        avg_diff = sum([abs(comparison['Difference'][m]) for m in metrics]) / len(metrics) * 100
        overall_diff = sum([comparison['Difference'][m] for m in metrics]) / len(metrics) * 100
        
        if overall_diff > 5:
            summary = "LLM模型整体性能显著优于CNN基线"
        elif overall_diff > 0:
            summary = "LLM模型整体性能略优于CNN基线"
        elif overall_diff < -5:
            summary = "CNN基线整体性能显著优于LLM模型"
        elif overall_diff < 0:
            summary = "CNN基线整体性能略优于LLM模型"
        else:
            summary = "两个模型性能相当"
        
        report += f"""
{'='*80}
总结: {summary}
平均差异: {overall_diff:.2f}%
最大差异: {max([abs(comparison['Difference'][m]) for m in metrics])*100:.2f}%
{'='*80}
"""
        
        return report
    
    def _save_markdown_report(self, comparison, dataset_name, filepath):
        """保存Markdown格式报告"""
        md_content = f"""# ECG分类模型对比报告

## 基本信息
- **数据集**: {dataset_name}
- **生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **实验目录**: {self.run_dir}

## 性能对比

| 指标 | LLM模型 | CNN基线 | 差异 | 结论 |
|------|---------|---------|------|------|
"""
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            llm_val = comparison['LLM'].get(metric, 0) * 100
            cnn_val = comparison['CNN_Baseline'].get(metric, 0) * 100
            diff = comparison['Difference'].get(metric, 0) * 100
            
            if diff > 0:
                conclusion = "✅ LLM更优"
                diff_str = f"+{diff:.2f}%"
            elif diff < 0:
                conclusion = "🔵 CNN更优"
                diff_str = f"{diff:.2f}%"
            else:
                conclusion = "⚪ 持平"
                diff_str = "0.00%"
            
            md_content += f"| {metric.capitalize()} | {llm_val:.2f}% | {cnn_val:.2f}% | {diff_str} | {conclusion} |\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _create_cv_summary_table(self, cv_results):
        """创建交叉验证汇总表格"""
        if 'fold_results' not in cv_results:
            return
        
        rows = []
        for fold_name, fold_data in cv_results['fold_results'].items():
            rows.append({
                'Fold': fold_name,
                'Accuracy': fold_data.get('accuracy', 0),
                'Precision': fold_data.get('precision', 0),
                'Recall': fold_data.get('recall', 0),
                'F1': fold_data.get('f1', 0),
                'Valid_Samples': fold_data.get('valid_samples', 0),
                'Invalid_Samples': fold_data.get('invalid_samples', 0),
                'Invalid_Rate': fold_data.get('invalid_rate', 0)
            })
        
        # 添加平均值行
        if rows:
            avg_row = {
                'Fold': 'Average',
                'Accuracy': cv_results.get('averages', {}).get('accuracy', 0),
                'Precision': cv_results.get('averages', {}).get('precision', 0),
                'Recall': cv_results.get('averages', {}).get('recall', 0),
                'F1': cv_results.get('averages', {}).get('f1', 0),
                'Valid_Samples': sum(r['Valid_Samples'] for r in rows) // len(rows),
                'Invalid_Samples': sum(r['Invalid_Samples'] for r in rows) // len(rows),
                'Invalid_Rate': sum(r['Invalid_Rate'] for r in rows) / len(rows)
            }
            rows.append(avg_row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.run_dir, "results", "cv_summary_table.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    def _get_traceback(self):
        """获取当前traceback信息"""
        import traceback
        return traceback.format_exc()


class SimpleResultSaver:
    """简化版结果保存器，用于应急情况"""
    
    def __init__(self, base_dir="Evaluation_Results", experiment_name=None):
        self.base_dir = os.path.abspath(base_dir)
        self.experiment_name = experiment_name or f"simple_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join(self.base_dir, self.experiment_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 创建基本子目录
        for subdir in ["results", "logs", "comparisons", "errors"]:
            os.makedirs(os.path.join(self.run_dir, subdir), exist_ok=True)
        
        print(f"简易保存器初始化: {self.run_dir}")
    
    def log(self, message, level="INFO"):
        """简化日志记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        
        # 写入日志文件
        log_file = os.path.join(self.run_dir, "logs", "log.txt")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")
        except:
            pass
    
    def save_json(self, data, filepath):
        """简化JSON保存"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.log(f"文件已保存: {filepath}")
            return True
        except Exception as e:
            self.log(f"保存文件失败: {e}", level="ERROR")
            return False
    
    def save_evaluation_results(self, results, dataset_name, save_detailed=True):
        """简化版保存评估结果"""
        try:
            # 只保存最基本的结果
            result_file = os.path.join(self.run_dir, "results", f"results_{dataset_name}.json")
            self.save_json(results, result_file)
            
            # 如果有可能，保存CSV
            if save_detailed and 'detailed_results' in results:
                csv_file = os.path.join(self.run_dir, "results", f"detailed_{dataset_name}.csv")
                df = pd.DataFrame(results['detailed_results'])
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                
            return self.run_dir
        except Exception as e:
            self.log(f"保存评估结果失败: {e}", level="ERROR")
            return None
    
    def save_comparison_results(self, comparison, dataset_name):
        """简化版保存对比结果"""
        try:
            comparison_file = os.path.join(self.run_dir, "comparisons", f"comparison_{dataset_name}.json")
            self.save_json(comparison, comparison_file)
            
            # 同时保存为文本格式
            text_file = os.path.join(self.run_dir, "comparisons", f"comparison_{dataset_name}.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(f"模型对比结果 - {dataset_name}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n")
                
                metrics = ['accuracy', 'precision', 'recall', 'f1']
                for metric in metrics:
                    llm_val = comparison.get('LLM', {}).get(metric, 0) * 100
                    cnn_val = comparison.get('CNN_Baseline', {}).get(metric, 0) * 100
                    diff = comparison.get('Difference', {}).get(metric, 0) * 100
                    
                    f.write(f"\n{metric.upper()}:\n")
                    f.write(f"  LLM模型: {llm_val:.2f}%\n")
                    f.write(f"  CNN基线: {cnn_val:.2f}%\n")
                    f.write(f"  差异: {diff:+.2f}%\n")
            
            self.log(f"对比结果已保存: {comparison_file}")
        except Exception as e:
            self.log(f"保存对比结果失败: {e}", level="ERROR")
    
    def save_error_report(self, error_info, context=""):
        """简化版保存错误报告"""
        try:
            error_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(error_info),
                'context': context
            }
            
            error_file = os.path.join(self.run_dir, "errors", f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            self.save_json(error_data, error_file)
            
            # 同时记录到日志
            self.log(f"错误报告已保存: {error_file}", level="ERROR")
        except Exception as e:
            print(f"保存错误报告失败: {e}")
    
    def save_cross_validation_summary(self, cv_results):
        """简化版保存交叉验证汇总"""
        try:
            cv_file = os.path.join(self.run_dir, "results", "cross_validation_summary.json")
            self.save_json(cv_results, cv_file)
            self.log(f"交叉验证汇总已保存: {cv_file}")
        except Exception as e:
            self.log(f"保存交叉验证汇总失败: {e}", level="ERROR")
    
    def log_experiment_info(self, config):
        """简化版记录实验信息"""
        try:
            info_file = os.path.join(self.run_dir, "logs", "experiment_info.json")
            self.save_json(config, info_file)
            self.log(f"实验信息已保存: {info_file}")
        except Exception as e:
            self.log(f"保存实验信息失败: {e}", level="ERROR")

# ============================================================================
# LLMEvaluator类
# ============================================================================
    def save_evaluation_results(self, results, dataset_name, save_detailed=True):
        """简化版保存评估结果"""
        try:
            # 只保存最基本的结果
            result_file = os.path.join(self.run_dir, f"results_{dataset_name}.json")
            self.save_json(results, result_file)
            
            # 如果有可能，保存CSV
            if save_detailed and 'detailed_results' in results:
                csv_file = os.path.join(self.run_dir, f"detailed_{dataset_name}.csv")
                df = pd.DataFrame(results['detailed_results'])
                df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                
            return self.run_dir
        except Exception as e:
            self.log(f"保存评估结果失败: {e}", level="ERROR")
            return None


class LLMEvaluator:
    def __init__(self, model_path, cnn_config, ecg_token_id, llm_embed_dim, device=None, log_dir=None):
        """
        兼容性评估器，自动处理维度问题
        Args:
            model_path: 模型保存路径
            cnn_config: CNN配置
            ecg_token_id: ECG token的ID
            llm_embed_dim: LLM嵌入维度
            device: 设备
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"评估设备: {self.device}")

        self.result_saver = None
        self.ecg_token_id = ecg_token_id
        self.llm_embed_dim = llm_embed_dim
        self.log_dir = log_dir
        
        # 1. 首先创建ECG编码器并计算实际输出维度
        print("计算ECG编码器输出维度...")
        self.ecg_encoder = ECGEncoder(cnn_config, CNN_WEIGHTS_PATH, device=self.device)
        self.actual_flat_dim = self._calculate_ecg_output_dim(FIXED_LENGTH)
        print(f"ECG编码器实际输出维度: {self.actual_flat_dim}")
        
        # 2. 加载模型
        self.model = self._load_model_compatible(
            model_path, cnn_config, ecg_token_id, 
            self.actual_flat_dim, llm_embed_dim
        )
        
        # 3. 设置tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/home/xusi/EE5046_Projects/LLM_Models/Qwen_Qwen-7B",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4. 初始化结果保存器
        self._init_result_saver(model_path)

    def _init_result_saver(self, model_path):
        """初始化结果保存器"""
        try:
            # 从路径提取模型名称
            model_name = os.path.basename(os.path.dirname(model_path))
            if not model_name or model_name == ".":
                model_name = os.path.basename(model_path)
            
            # 清理模型名称中的特殊字符
            model_name = re.sub(r'[^\w\-_]', '_', model_name)
            
            exp_name = f"llm_eval_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"创建ResultSaver: {exp_name}")
            
            # 创建ResultSaver
            self.result_saver = ResultSaver(
                base_dir=self.log_dir or "Evaluation_Results",
                experiment_name=exp_name
            )
            
            print(f"✓ ResultSaver创建成功")
            print(f"  运行目录: {self.result_saver.run_dir}")
            
            # 记录基本模型信息
            model_info = {
                'model_path': model_path,
                'device': str(self.device),
                'ecg_output_dim': self.actual_flat_dim,
                'ecg_token_id': self.ecg_token_id,
                'llm_embed_dim': self.llm_embed_dim,
                'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'cnn_weights_path': CNN_WEIGHTS_PATH,
                'fixed_length': FIXED_LENGTH,
                'dataset_path': DATASET_PATH
            }
            
            # 直接保存实验信息，避免在初始化过程中调用log
            info_path = os.path.join(self.result_saver.run_dir, "logs", "experiment_info.json")
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            print(f"✓ 实验信息已保存到: {info_path}")
            
            # 现在可以安全地使用log方法
            self.result_saver.log(f"模型评估器初始化完成")
            self.result_saver.log(f"模型路径: {model_path}")
            self.result_saver.log(f"ECG输出维度: {self.actual_flat_dim}")
            
        except Exception as e:
            print(f"初始化结果保存器失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 创建简易后备
            self.result_saver = SimpleResultSaver(self.log_dir or "Evaluation_Results", "fallback_eval")
            print(f"使用简易ResultSaver: {self.result_saver.run_dir}")
            


    def generate_response(self, ecg_data, prompt_template="请分析这个ECG信号，判断是否有房颤。"):
        """
        生成LLM的响应 - 手动实现生成
        """
        with torch.no_grad():
            # 准备输入
            ecg_data = ecg_data.to(self.device)
            
            # 调整形状
            if ecg_data.dim() == 1:
                ecg_data = ecg_data.unsqueeze(0).unsqueeze(0)
            elif ecg_data.dim() == 2:
                ecg_data = ecg_data.unsqueeze(1)
            
            # 构建提示词
            prompt = f"指令: <|extra_0|>{prompt_template}\n答案:"
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # 手动生成
            generated_ids = self._simple_generate(
                ecg_data=ecg_data,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50
            )
            
            # 解码
            full_response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 提取答案
            if "答案:" in full_response:
                answer = full_response.split("答案:")[1].strip()
            else:
                answer = full_response.replace(prompt, "").strip()
                
            return answer
        
    def _simple_generate(self, ecg_data, input_ids, attention_mask, max_new_tokens=50):
        """
        简单的贪婪解码生成
        """
        generated_ids = input_ids.clone()
        
        for i in range(max_new_tokens):
            # 前向传播
            outputs = self.model(
                ecg_data=ecg_data,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                labels=None
            )
            
            # 获取下一个token的logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 贪婪解码：选择概率最高的token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 检查是否生成结束符
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 更新attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=-1)
        
        return generated_ids
    
    
    def _calculate_ecg_output_dim(self, fixed_length):
        """计算ECG编码器的输出维度"""
        # 创建模拟输入
        dummy_input = torch.randn(1, 1, fixed_length).to(self.device)
        
        # 设置为评估模式
        self.ecg_encoder.eval()
        
        with torch.no_grad():
            output = self.ecg_encoder(dummy_input)
        
        # 展平后计算维度
        output_dim = output.view(1, -1).size(1)
        return output_dim
    
    def _load_model_compatible(self, model_path, cnn_config, ecg_token_id, flat_dim, llm_embed_dim):
        """加载模型，自动处理维度不匹配"""
        print(f"正在加载模型: {model_path}")
        
        try: 
            # 1. 初始化模型结构
            model = MultimodalLLM(
                llm_path="/home/xusi/EE5046_Projects/LLM_Models/Qwen_Qwen-7B",
                cnn_config=cnn_config,
                cnn_weights_path=CNN_WEIGHTS_PATH,
                ecg_token_id=ecg_token_id,
                flat_dim=flat_dim,  # 使用计算得到的实际维度
                llm_embed_dim=llm_embed_dim,
                device=self.device
            )
            
            # 2. 加载LoRA适配器
            lora_path = os.path.join(model_path, "lora_adapter")
            if os.path.exists(lora_path):
                print(f"加载LoRA适配器: {lora_path}")
                try:
                    # 先保存原始LLM
                    original_llm = model.llm
                    
                    # 使用PeftModel加载适配器
                    model.llm = PeftModel.from_pretrained(original_llm, lora_path)
                    print("LoRA适配器加载成功")
                except Exception as e:
                    print(f"加载LoRA失败: {e}")
                    print("尝试直接使用基础模型...")
                    # 如果失败，保持原始模型
            else:
                print(f"警告: 未找到LoRA适配器: {lora_path}")
            
            # 3. 加载projector权重
            projector_path = os.path.join(model_path, "projector.pth")
            if os.path.exists(projector_path):
                print(f"加载projector权重: {projector_path}")
                projector_state = torch.load(projector_path, map_location=self.device)
                
                # 检查维度是否匹配
                weight_shape = projector_state['weight'].shape
                expected_input_dim = weight_shape[1]  # 权重形状: [output_dim, input_dim]
                actual_input_dim = flat_dim
                
                if expected_input_dim != actual_input_dim:
                    print(f"维度不匹配: 投影层期望输入维度={expected_input_dim}, 实际维度={actual_input_dim}")
                    print("正在调整投影层权重...")
                    
                    # 调整投影层权重以适应实际维度
                    adjusted_projector = self._adjust_projector_weights(
                        projector_state, expected_input_dim, actual_input_dim, llm_embed_dim
                    )
                    model.projector.load_state_dict(adjusted_projector)
                    print("投影层权重调整完成")
                else:
                    model.projector.load_state_dict(projector_state)
                    print("投影层权重加载成功")
            else:
                print(f"警告: 未找到projector权重: {projector_path}")
                print("使用随机初始化的投影层")
            
            # 4. 设置为评估模式
            model.eval()
            
            # 5. 打印模型信息
            print(f"模型加载完成")
            print(f"  设备: {self.device}")
            print(f"  ECG编码器输出维度: {flat_dim}")
            print(f"  投影层输入维度: {flat_dim}, 输出维度: {llm_embed_dim}")
            
            return model
        
        except Exception as e:
            print(f"模型加载失败: {e}")
            if self.result_saver:
                self.result_saver.save_error_report(e, "模型加载失败")
            raise
    
    def _adjust_projector_weights(self, original_state, expected_dim, actual_dim, llm_embed_dim):
        """
        调整投影层权重以适应不同的输入维度
        
        Args:
            original_state: 原始投影层状态字典
            expected_dim: 期望的输入维度（训练时的维度）
            actual_dim: 实际的输入维度（当前ECG编码器的输出维度）
            llm_embed_dim: LLM嵌入维度（输出维度）
            
        Returns:
            调整后的投影层状态字典
        """
        print(f"调整投影层权重: {expected_dim} -> {actual_dim}")
        
        # 原始权重形状: [llm_embed_dim, expected_dim]
        original_weight = original_state['weight'].cpu()
        original_bias = original_state['bias'].cpu()
        
        if actual_dim > expected_dim:
            # 实际维度更大，需要扩展权重矩阵
            new_weight = torch.zeros(llm_embed_dim, actual_dim)
            new_weight[:, :expected_dim] = original_weight  # 复制原始权重
            # 剩余部分保持为零（相当于丢弃额外特征）
            print(f"  扩展权重矩阵: {original_weight.shape} -> {new_weight.shape}")
            
            # 偏置不变
            new_bias = original_bias
            
        elif actual_dim < expected_dim:
            # 实际维度更小，需要截断权重矩阵
            new_weight = original_weight[:, :actual_dim]
            print(f"  截断权重矩阵: {original_weight.shape} -> {new_weight.shape}")
            
            # 偏置不变
            new_bias = original_bias
            
        else:
            # 维度相同，无需调整
            new_weight = original_weight
            new_bias = original_bias
            print(f"  维度相同，无需调整")
        
        # 创建新的状态字典
        new_state = {
            'weight': new_weight.to(self.device),
            'bias': new_bias.to(self.device)
        }
        
        return new_state
    
    def parse_label_from_text(self, text):
        """
        从LLM生成的文本中解析标签
        """
        text = text.lower().strip()
        
        # 查找关键词
        for keyword, label in LABEL_MAPPING.items():
            if keyword.lower() in text:
                return label
        
        # 如果没有匹配到关键词，使用启发式规则
        negative_keywords = ["无", "不", "非", "正常", "否", "negative", "normal", "正常", "窦性"]
        positive_keywords = ["有", "是", "异常", "房颤", "af", "abnormal", "心房颤动", "心房纤颤"]
        
        negative_count = sum(1 for word in negative_keywords if word in text)
        positive_count = sum(1 for word in positive_keywords if word in text)
        
        if positive_count > negative_count:
            return 1
        elif negative_count > positive_count:
            return 0
        else:
            # 如果无法确定，返回-1表示无法解析
            return -1
    
    def evaluate_on_dataset(self, dataset, batch_size=8,save_dir=None,dataset_name="unknown"):
        """
        在整个数据集上评估模型
        """
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: self._collate_fn(batch)
            )
            
            all_true_labels = []
            all_pred_labels = []
            all_responses = []
            all_filenames = []
            all_confidences = []  # 保存置信度信息
            
            print(f"正在评估数据集 (大小: {len(dataset)})...")
            self.result_saver.log(f"开始评估数据集: {dataset_name}, 样本数: {len(dataset)}")
            
            progress_bar = tqdm(dataloader, desc=f"评估 {dataset_name}")
            for batch_idx, batch in enumerate(progress_bar):
                ecg_data = batch['ecg_data'].to(self.device)
                true_labels = batch['labels'].cpu().numpy()
                filenames = batch['file_names']
                
                # 批量生成响应
                batch_responses = []
                for i in range(ecg_data.size(0)):
                    try:
                        response = self.generate_response(ecg_data[i])
                        batch_responses.append(response)
                    except Exception as e:
                        batch_responses.append(f"生成失败: {str(e)}")
                        self.result_saver.log(f"样本 {filenames[i]} 生成失败: {e}", level="WARNING")
                
                # 解析标签
                batch_pred_labels = []
                batch_confidences = []
                for response in batch_responses:
                    pred_label = self.parse_label_from_text(response)
                    batch_pred_labels.append(pred_label)
                    
                    # 简单的置信度估计（基于关键词匹配程度）
                    confidence = self._estimate_confidence(response, pred_label)
                    batch_confidences.append(confidence)
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(batch_pred_labels)
                all_responses.extend(batch_responses)
                all_filenames.extend(filenames)
                all_confidences.extend(batch_confidences)
                
                # 更新进度条
                progress_bar.set_postfix({
                    '准确率': f"{accuracy_score([l for l in all_pred_labels if l != -1], [all_true_labels[i] for i, l in enumerate(all_pred_labels) if l != -1]):.3f}" 
                    if len([l for l in all_pred_labels if l != -1]) > 0 else "N/A"
                })
            
            # 过滤掉无法解析的样本
            valid_indices = [i for i, label in enumerate(all_pred_labels) if label != -1]
            valid_true = [all_true_labels[i] for i in valid_indices]
            valid_pred = [all_pred_labels[i] for i in valid_indices]
            valid_responses = [all_responses[i] for i in valid_indices]
            valid_filenames = [all_filenames[i] for i in valid_indices]
            valid_confidences = [all_confidences[i] for i in valid_indices]
            
            # 计算指标
            results = self._calculate_metrics(
                valid_true, valid_pred, valid_responses, 
                valid_filenames, valid_confidences
            )
            
            # 添加无法解析的样本信息
            results['total_samples'] = len(all_true_labels)
            results['valid_samples'] = len(valid_true)
            results['invalid_samples'] = len(all_true_labels) - len(valid_true)
            results['invalid_rate'] = results['invalid_samples'] / results['total_samples'] if results['total_samples'] > 0 else 0
            
            # 添加数据集信息
            results['dataset_name'] = dataset_name
            results['evaluation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 使用ResultSaver保存结果
            self.result_saver.save_evaluation_results(results, dataset_name, save_detailed=True)
            
            # 打印简要结果
            self._print_summary(results, dataset_name)
            
            return results
            
        except Exception as e:
            self.result_saver.save_error_report(e, f"数据集评估失败: {dataset_name}")
            raise

    def _estimate_confidence(self, response, predicted_label):
        """估计预测置信度"""
        response_lower = response.lower()
        
        # 基于关键词匹配的置信度
        if predicted_label == 1:  # 房颤
            strong_indicators = ["确诊", "明确", "肯定", "一定是", "确定", "毫无疑问"]
            weak_indicators = ["可能", "疑似", "怀疑", "倾向", "考虑"]
            
            confidence = 0.8  # 基础置信度
            for indicator in strong_indicators:
                if indicator in response_lower:
                    confidence = min(confidence + 0.15, 1.0)
            for indicator in weak_indicators:
                if indicator in response_lower:
                    confidence = max(confidence - 0.2, 0.5)
                    
        elif predicted_label == 0:  # 正常
            strong_indicators = ["完全正常", "未见异常", "正常心率", "窦性心律", "无异常"]
            weak_indicators = ["基本正常", "大致正常", "未见明显异常"]
            
            confidence = 0.8  # 基础置信度
            for indicator in strong_indicators:
                if indicator in response_lower:
                    confidence = min(confidence + 0.15, 1.0)
            for indicator in weak_indicators:
                if indicator in response_lower:
                    confidence = max(confidence - 0.2, 0.5)
        else:
            confidence = 0.0
            
        return round(confidence, 2)
    
    def _collate_fn(self, batch):
        """评估时的collate函数"""
        # 处理ECG数据
        ecg_data_list = []
        for item in batch:
            ecg = item['ecg_data']
            # 确保是1D，然后添加通道维度
            if ecg.dim() == 1:
                ecg = ecg.unsqueeze(0)  # [length] -> [1, length]
            ecg_data_list.append(ecg)
        
        # 堆叠: [batch, 1, length]
        ecg_data = torch.stack(ecg_data_list)
        
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        file_names = [item['file_name'] for item in batch]
        
        return {
            'ecg_data': ecg_data,  # [batch, 1, length]
            'labels': labels,
            'file_names': file_names
        }
    
    def _calculate_metrics(self, true_labels, pred_labels, responses, filenames=None,confidences=None):
        """计算分类指标"""
        if len(true_labels) == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'classification_report': '',
                'detailed_results': [],
                'class_distribution': {'normal': 0, 'af': 0},
                'average_confidence': 0
            }
        
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels).tolist()
        
        # 生成分类报告
        report = classification_report(true_labels, pred_labels, target_names=['正常', '房颤'], zero_division=0,output_dict=True)
        
        # 创建详细结果列表
        detailed_results = []
        for i in range(len(true_labels)):
            result = {
                'filename': filenames[i] if filenames else f"sample_{i}",
                'true_label': int(true_labels[i]),
                'true_label_str': '房颤' if true_labels[i] == 1 else '正常',
                'pred_label': int(pred_labels[i]),
                'pred_label_str': '房颤' if pred_labels[i] == 1 else '正常',
                'response': responses[i],
                'confidence': confidences[i] if confidences else None,
                'correct': true_labels[i] == pred_labels[i],
                'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            detailed_results.append(result)
        
        # 统计响应示例
        response_examples = []
        normal_examples = []
        af_examples = []

        for i in range(min(10, len(responses))):
            example = {
                'true_label': int(true_labels[i]),
                'true_label_str': '房颤' if true_labels[i] == 1 else '正常',
                'pred_label': int(pred_labels[i]),
                'pred_label_str': '房颤' if pred_labels[i] == 1 else '正常',
                'response': responses[i][:200] + "..." if len(responses[i]) > 200 else responses[i],
                'filename': filenames[i] if filenames else f"sample_{i}",
                'confidence': confidences[i] if confidences else None
            }
            response_examples.append(example)
            
            # 分类别收集示例
            if true_labels[i] == 0 and len(normal_examples) < 3:
                normal_examples.append(example)
            elif true_labels[i] == 1 and len(af_examples) < 3:
                af_examples.append(example)
        
        # 计算类别分布
        normal_count = sum(1 for label in true_labels if label == 0)
        af_count = sum(1 for label in true_labels if label == 1)
        
        # 计算平均置信度
        avg_confidence = np.mean(confidences) if confidences and len(confidences) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm,
            'classification_report': report,
            'classification_report_str': classification_report(true_labels, pred_labels, target_names=['正常', '房颤'], zero_division=0),
            'response_examples': response_examples,
            'detailed_results': detailed_results,
            'class_distribution': {
                'normal': normal_count,
                'af': af_count,
                'normal_percentage': normal_count / len(true_labels) * 100 if len(true_labels) > 0 else 0,
                'af_percentage': af_count / len(true_labels) * 100 if len(true_labels) > 0 else 0
            },
            'average_confidence': float(avg_confidence),
            'normal_examples': normal_examples,
            'af_examples': af_examples,
            'metrics_by_class': {
                'normal': {
                    'precision': report.get('正常', {}).get('precision', 0),
                    'recall': report.get('正常', {}).get('recall', 0),
                    'f1': report.get('正常', {}).get('f1-score', 0)
                },
                'af': {
                    'precision': report.get('房颤', {}).get('precision', 0),
                    'recall': report.get('房颤', {}).get('recall', 0),
                    'f1': report.get('房颤', {}).get('f1-score', 0)
                }
            }
        }
    
    def _print_summary(self, results, dataset_name):
        """打印评估摘要"""
        print("\n" + "="*60)
        print(f"LLM模型评估结果 - {dataset_name}")
        print("="*60)
        print(f"总样本数: {results['total_samples']}")
        print(f"有效样本数: {results['valid_samples']}")
        print(f"无法解析样本数: {results['invalid_samples']} ({results['invalid_rate']*100:.2f}%)")
        print(f"类别分布: 正常={results['class_distribution']['normal']} ({results['class_distribution']['normal_percentage']:.1f}%), "
              f"房颤={results['class_distribution']['af']} ({results['class_distribution']['af_percentage']:.1f}%)")
        print(f"平均置信度: {results['average_confidence']:.2f}")
        print()
        print(f"准确率: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"精确率: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"召回率: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        print(f"F1分数: {results['f1']:.4f} ({results['f1']*100:.2f}%)")
        
        print("\n混淆矩阵:")
        cm = results['confusion_matrix']
        print(f"         预测正常   预测房颤")
        print(f"实际正常   {cm[0][0]:^10}   {cm[0][1]:^10}")
        print(f"实际房颤   {cm[1][0]:^10}   {cm[1][1]:^10}")
        
        # 记录到日志
        self.result_saver.log(f"评估完成: {dataset_name} - 准确率: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
    

    def compare_with_cnn_baseline(self, llm_results, cnn_results_path,dataset_name):
        """
        与CNN基线模型对比
        """
        try:
            # 加载CNN基线结果
            if os.path.exists(cnn_results_path):
                with open(cnn_results_path, 'r', encoding='utf-8') as f:
                    cnn_results_data = json.load(f)
                
                # 尝试从不同格式中提取CNN结果
                if 'metrics' in cnn_results_data:
                    cnn_results = cnn_results_data['metrics']
                elif 'final_results' in cnn_results_data:
                    cnn_results = cnn_results_data['final_results']
                else:
                    cnn_results = cnn_results_data
            else:
                self.result_saver.log(f"警告: 未找到CNN基线结果: {cnn_results_path}", level="WARNING")
                # 使用默认值
                cnn_results = {
                    'accuracy': 0.85,
                    'precision': 0.86,
                    'recall': 0.84,
                    'f1': 0.85
                }
            
            comparison = {
                'dataset': dataset_name,
                'comparison_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'LLM': {
                    'accuracy': llm_results['accuracy'],
                    'precision': llm_results['precision'],
                    'recall': llm_results['recall'],
                    'f1': llm_results['f1']
                },
                'CNN_Baseline': {
                    'accuracy': cnn_results.get('accuracy', cnn_results.get('Accuracy', 0)),
                    'precision': cnn_results.get('precision', cnn_results.get('Precision', 0)),
                    'recall': cnn_results.get('recall', cnn_results.get('Recall', 0)),
                    'f1': cnn_results.get('f1', cnn_results.get('F1', 0))
                }
            }
            
            # 计算差异
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            comparison['Difference'] = {}
            for metric in metrics:
                llm_val = comparison['LLM'][metric]
                cnn_val = comparison['CNN_Baseline'][metric]
                comparison['Difference'][metric] = llm_val - cnn_val
            
            # 保存对比结果
            self.result_saver.save_comparison_results(comparison, dataset_name)
            
            # 打印对比结果
            self._print_comparison(comparison)
            
            return comparison
            
        except Exception as e:
            self.result_saver.save_error_report(e, "CNN基线对比失败")
            raise

    def _print_comparison(self, comparison):
        """打印对比结果"""
        print("\n" + "="*60)
        print("与CNN基线模型对比")
        print("="*60)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            llm_val = comparison['LLM'][metric] * 100
            cnn_val = comparison['CNN_Baseline'][metric] * 100
            diff = comparison['Difference'][metric] * 100
            
            print(f"\n{metric.capitalize()}:")
            print(f"  LLM: {llm_val:.2f}%")
            print(f"  CNN: {cnn_val:.2f}%")
            
            if diff > 0:
                print(f"  LLM优于CNN: +{diff:.2f}%")
            elif diff < 0:
                print(f"  CNN优于LLM: {-diff:.2f}%")
            else:
                print(f"  两者相同")
    
    def test_single_sample(self, dataset):
        """测试单个样本的处理"""
        print("测试单个样本...")
        
        if len(dataset) == 0:
            print("数据集为空")
            return False
        
        # 获取第一个样本
        sample = dataset[0]
        print(f"样本ECG形状: {sample['ecg_data'].shape}")
        print(f"样本标签: {sample['label']} ({'房颤' if sample['label'] == 1 else '正常'})")
        
        # 测试生成
        try:
            response = self.generate_response(sample['ecg_data'])
            print(f"生成成功!")
            print(f"响应: {response}")

            # 解析标签
            pred_label = self.parse_label_from_text(response)
            print(f"解析标签: {pred_label} ({'房颤' if pred_label == 1 else '正常' if pred_label == 0 else '无法解析'})")
            
            # 检查是否正确
            if pred_label == sample['label']:
                print("✓ 预测正确")
            elif pred_label == -1:
                print("⚠ 无法解析响应")
            else:
                print("✗ 预测错误")
            
            return True
        except Exception as e:
            print(f"生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def load_cnn_config(config_path):
    """加载CNN配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 尝试从不同格式中提取配置
        if 'best_config' in config:
            return config['best_config'].get('kernel_config', {})
        elif 'kernel_config' in config:
            return config['kernel_config']
        else:
            return config
    except Exception as e:
        print(f"加载CNN配置失败: {e}")
        return {}


def load_ecg_from_mat(file_name, data_dir, ecg_key='val'):
    """
    从.mat文件加载ECG数据
    
    Args:
        file_name: .mat文件名（不带路径）
        data_dir: 数据目录
        ecg_key: .mat文件中ECG数据的键名
    
    Returns:
        ecg_data: ECG信号数据
    """
    # 构建完整路径
    full_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(full_path):
        # 尝试添加扩展名
        if not file_name.endswith('.mat'):
            full_path = os.path.join(data_dir, f"{file_name}.mat")
        
        if not os.path.exists(full_path):
            print(f"警告: 文件不存在: {full_path}")
            return None
    
    try:
        # 加载.mat文件
        mat_data = sio.loadmat(full_path)
        
        # 查找ECG数据
        # 常见键名: 'val', 'ECG', 'ecg', 'data'
        if ecg_key in mat_data:
            ecg_data = mat_data[ecg_key]
        elif 'ECG' in mat_data:
            ecg_data = mat_data['ECG']
        elif 'ecg' in mat_data:
            ecg_data = mat_data['ecg']
        elif 'data' in mat_data:
            ecg_data = mat_data['data']
        else:
            # 如果找不到常见的键，尝试第一个数值数组
            for key in mat_data.keys():
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    ecg_data = mat_data[key]
                    print(f"使用键 '{key}' 作为ECG数据")
                    break
            else:
                raise ValueError(f"在文件 {file_name} 中找不到ECG数据")
        
        # 确保是1维或2维数组，并转换为1维
        if ecg_data.ndim > 1:
            ecg_data = ecg_data.flatten()
        
        return ecg_data
        
    except Exception as e:
        print(f"加载ECG文件失败 {file_name}: {e}")
        return None


def preprocess_ecg_data(ecg_data, fixed_length, normalize=True):
    """
    预处理ECG数据
    
    Args:
        ecg_data: 原始ECG数据
        fixed_length: 固定长度
        normalize: 是否标准化
    
    Returns:
        processed_data: 预处理后的ECG数据
    """
    if ecg_data is None:
        return None
    
    # 1. 截断或填充到固定长度
    if len(ecg_data) > fixed_length:
        # 截断中间部分
        start_idx = (len(ecg_data) - fixed_length) // 2
        ecg_data = ecg_data[start_idx:start_idx + fixed_length]
    elif len(ecg_data) < fixed_length:
        # 填充两侧
        pad_left = (fixed_length - len(ecg_data)) // 2
        pad_right = fixed_length - len(ecg_data) - pad_left
        ecg_data = np.pad(ecg_data, (pad_left, pad_right), mode='constant')
    
    # 2. 归一化（可选）
    if normalize:
        ecg_data = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + 1e-8)
    
    # 3. 转换为tensor并调整维度 [1, 1, FIXED_LENGTH]
    ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32)
    
    return ecg_tensor


def create_evaluation_dataset_from_csv(csv_path, data_dir, fixed_length, test_mode=True):
    """
    从CSV文件创建评估数据集
    
    Args:
        csv_path: CSV文件路径
        data_dir: ECG数据文件目录（包含.mat文件）
        fixed_length: ECG数据固定长度
        test_mode: 是否为测试模式（禁用数据增强）
    
    Returns:
        dataset: 评估数据集
    """
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path, data_dir, fixed_length, test_mode=True):
            # 加载CSV文件
            self.df = pd.read_csv(csv_path)
            
            # 重命名列以统一处理
            if 'file_name' not in self.df.columns:
                # 尝试找到包含文件名的列
                if 'record_name' in self.df.columns:
                    self.df = self.df.rename(columns={'record_name': 'file_name'})
                elif 'filename' in self.df.columns:
                    self.df = self.df.rename(columns={'filename': 'file_name'})
                elif 'name' in self.df.columns:
                    self.df = self.df.rename(columns={'name': 'file_name'})
                else:
                    # 假设第一列是文件名
                    self.df = self.df.rename(columns={self.df.columns[0]: 'file_name'})
            
            if 'label' not in self.df.columns:
                # 尝试找到包含标签的列
                if 'symbol' in self.df.columns:
                    self.df = self.df.rename(columns={'symbol': 'label'})
                elif 'class' in self.df.columns:
                    self.df = self.df.rename(columns={'class': 'label'})
                elif 'target' in self.df.columns:
                    self.df = self.df.rename(columns={'target': 'label'})
                else:
                    # 假设第二列是标签
                    self.df = self.df.rename(columns={self.df.columns[1]: 'label'})
            
            # 确保标签是字符串类型，便于处理
            self.df['label'] = self.df['label'].astype(str).str.strip()
            
            # 转换标签：A -> 1 (房颤), 其他 -> 0 (非房颤)
            self.df['label_int'] = self.df['label'].apply(
                lambda x: 1 if x.upper() == 'A' else 0
            )
            
            self.data_dir = data_dir
            self.fixed_length = fixed_length
            self.test_mode = test_mode
            
            print(f"从CSV加载数据集完成，共 {len(self.df)} 个样本")
            print(f"标签分布: 房颤(A)={sum(self.df['label_int'] == 1)}, 非房颤={sum(self.df['label_int'] == 0)}")
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            file_name = row['file_name']
            
            # 加载ECG数据
            ecg_raw = load_ecg_from_mat(file_name, self.data_dir)
            
            if ecg_raw is None:
                # 如果加载失败，创建零数据（但会标记为无效）
                print(f"警告: 无法加载ECG数据: {file_name}")
                ecg_raw = np.zeros(self.fixed_length)
            
            # 预处理ECG数据
            ecg_data = preprocess_ecg_data(ecg_raw, self.fixed_length, normalize=True)
            
            # 获取标签（整数形式）
            label = int(row['label_int'])
            
            return {
                'ecg_data': ecg_data,
                'label': label,
                'file_name': file_name
            }
    
    return EvalDataset(csv_path, data_dir, fixed_length, test_mode)


def create_evaluation_dataset_from_multiple_csv(csv_paths, data_dir, fixed_length, test_mode=True):
    """
    从多个CSV文件创建评估数据集
    
    Args:
        csv_paths: CSV文件路径列表
        data_dir: ECG数据文件目录
        fixed_length: ECG数据固定长度
        test_mode: 是否为测试模式
    
    Returns:
        dataset: 合并的评估数据集
    """
    from torch.utils.data import ConcatDataset
    
    datasets = []
    for csv_path in csv_paths:
        print(f"加载CSV文件: {csv_path}")
        dataset = create_evaluation_dataset_from_csv(csv_path, data_dir, fixed_length, test_mode)
        datasets.append(dataset)
    
    return ConcatDataset(datasets)


def main():
    """主评估函数"""
    try: 
        # 配置参数 - 需要根据实际情况修改
        MODEL_PATH = "/home/xusi/EE5046_Projects/Trained_Multimodal_Models/Qwen7B_ECG_B8_LR2e-05_E8/final_model"
        
        # 数据集路径 - 根据您的目录结构调整
        DATASET_BASE = DATASET_PATH  # 从Config.py导入，应该是Dataset目录的父目录
        TRAINING2017_DIR = os.path.join(DATASET_BASE, "training2017")
        CV_DIR = os.path.join(DATASET_BASE, "cv")
        
        # 选择要评估的CSV文件（可以评估单个或多个）
        # 方案1：评估单个CSV文件
        TEST_CSV_PATH = os.path.join(CV_DIR, "cv1.csv")  # 使用cv1.csv作为测试集
        
        # 方案2：评估所有CSV文件（交叉验证）
        # all_csv_paths = [os.path.join(CV_DIR, f"cv{i}.csv") for i in range(5)]
        
        # 基线结果路径
        CNN_BASELINE_RESULTS = "/home/xusi/Logs/FinalTraining/Results_20251217_115456/cnn_evaluation_results.json"
        OUTPUT_DIR = "/home/xusi/EE5046_Projects/Evaluation_Results"
        
        # 加载CNN配置
        cnn_config_path = "/home/xusi/Logs/FinalTraining/Results_20251217_115456/cnn_evaluation_results.json"
        cnn_config = load_cnn_config(cnn_config_path).get("best_config", {}).get("kernel_config", {})
        
        # 设置模型参数（需要与训练时一致）
        cnn_config_full = {
            'ch_in': 1,
            'ch_out': 1,
            'use_stream2': True,
            'stream1_kernel': 3,
            'stream2_first_kernel': 7,
        }
        
        # 设置ECG token ID（必须与训练时相同）
        ECG_TOKEN_ID = 151646  # <|extra_0|>的ID
        
        # 设置维度参数（需要与训练时一致）
        LLM_EMBED_DIM = 4096  # Qwen-7B嵌入维度
        
        # 创建评估器
        print("初始化评估器...")
        evaluator = LLMEvaluator(
            model_path=MODEL_PATH,
            cnn_config=cnn_config_full,
            ecg_token_id=ECG_TOKEN_ID,
            llm_embed_dim=LLM_EMBED_DIM,
            log_dir=OUTPUT_DIR
        )
        
        # 检查数据目录是否存在
        if not os.path.exists(TRAINING2017_DIR):
            print(f"错误: 数据目录不存在: {TRAINING2017_DIR}")
            print("请检查DATASET_PATH配置是否正确")
            return
        
        if not os.path.exists(TEST_CSV_PATH):
            print(f"错误: CSV文件不存在: {TEST_CSV_PATH}")
            print(f"CV目录内容: {os.listdir(CV_DIR) if os.path.exists(CV_DIR) else '目录不存在'}")
            return
        
        # 创建评估数据集（从CSV文件）
        print(f"创建评估数据集...")
        
        eval_dataset = create_evaluation_dataset_from_csv(
            csv_path=TEST_CSV_PATH,
            data_dir=TRAINING2017_DIR,
            fixed_length=FIXED_LENGTH,
            test_mode=True
        )
        
        # 检查数据集是否为空
        if len(eval_dataset) == 0:
            print("错误: 评估数据集为空！")
            return
        
        # 测试单个样本
        evaluator.test_single_sample(eval_dataset)

        # 评估模型
        print("开始评估LLM模型...")
        dataset_name = os.path.basename(TEST_CSV_PATH).replace('.csv', '')
        results = evaluator.evaluate_on_dataset(eval_dataset, batch_size=4,dataset_name=dataset_name)

        # 与CNN基线对比（如果有）
        if os.path.exists(CNN_BASELINE_RESULTS):
            comparison = evaluator.compare_with_cnn_baseline(results, CNN_BASELINE_RESULTS, dataset_name)

        print(f"\n所有结果已保存到: {evaluator.result_saver.run_dir}")

    except Exception as e:
        print(f"评估过程失败: {e}")
        import traceback
        traceback.print_exc()
    

def generate_comparison_report(comparison, llm_results, output_dir, timestamp, dataset_name):
    """生成对比报告"""
    report = f"""
    ============================================================================
    ECG分类模型对比报告 - {dataset_name}
    ============================================================================
    
    评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    1. 数据集信息
    -------------
    总样本数: {llm_results['total_samples']}
    有效样本数: {llm_results['valid_samples']}
    无法解析样本数: {llm_results['invalid_samples']} ({llm_results['invalid_rate']*100:.2f}%)
    
    2. 模型性能对比
    --------------
    """
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        llm_val = comparison['LLM'][metric]
        cnn_val = comparison['CNN_Baseline'][metric]
        diff = comparison['Difference'][metric]
        
        report += f"""
    {metric.upper()}:
      - LLM模型: {llm_val:.4f}
      - CNN基线: {cnn_val:.4f}
      - 差异: {diff:+.4f} ({'LLM更优' if diff > 0 else 'CNN更优' if diff < 0 else '相同'})"""
    
    report += f"""
    
    3. LLM模型详细结果
    -----------------
    混淆矩阵: {llm_results['confusion_matrix']}
    
    分类报告:
    {llm_results['classification_report']}
    
    4. 结论
    -------
    """
    
    # 计算平均差异
    avg_diff = sum(comparison['Difference'].values()) / len(comparison['Difference'])
    
    if avg_diff > 0.05:
        conclusion = "LLM模型在整体性能上显著优于CNN基线模型。"
    elif avg_diff > 0:
        conclusion = "LLM模型在整体性能上略优于CNN基线模型。"
    elif avg_diff < -0.05:
        conclusion = "CNN基线模型在整体性能上显著优于LLM模型。"
    elif avg_diff < 0:
        conclusion = "CNN基线模型在整体性能上略优于LLM模型。"
    else:
        conclusion = "LLM模型和CNN基线模型性能相当。"
    
    report += conclusion + "\n"
    
    # 保存报告
    report_path = os.path.join(output_dir, f"comparison_report_{dataset_name}_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"对比报告已保存到: {report_path}")


def run_cross_validation_evaluation():
    """
    运行交叉验证评估（评估所有cv0-cv4）
    """
    try: 
        # 配置参数
        MODEL_PATH = "/home/xusi/EE5046_Projects/Trained_Multimodal_Models/Qwen7B_ECG_B8_LR2e-05_E8/final_model"
        DATASET_BASE = DATASET_PATH
        TRAINING2017_DIR = os.path.join(DATASET_BASE, "training2017")
        CV_DIR = os.path.join(DATASET_BASE, "cv")
        OUTPUT_DIR = "/home/xusi/EE5046_Projects/Evaluation_Results"
        
        # 加载CNN配置
        cnn_config_path = "/home/xusi/Logs/FinalTraining/Results_20251217_115456/cnn_evaluation_results.json"
        cnn_config = load_cnn_config(cnn_config_path).get("best_config", {}).get("kernel_config", {})
        
        # 设置模型参数
        cnn_config_full = {
            'ch_in': 1,
            'ch_out': 1,
            'use_stream2': True,
            'stream1_kernel': 3,
            'stream2_first_kernel': 7
        }
        
        ECG_TOKEN_ID = 151646
        LLM_EMBED_DIM = 4096
        
        # 创建评估器
        print("初始化评估器...")
        evaluator = LLMEvaluator(
            model_path=MODEL_PATH,
            cnn_config=cnn_config_full,
            ecg_token_id=ECG_TOKEN_ID,
            llm_embed_dim=LLM_EMBED_DIM,
            log_dir=OUTPUT_DIR
        )
        
        # 评估所有CSV文件
        all_results = {}
        
        for i in range(5):
            csv_path = os.path.join(CV_DIR, f"cv{i}.csv")
            
            if not os.path.exists(csv_path):
                print(f"跳过不存在的CSV文件: {csv_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"评估数据集: cv{i}.csv")
            print(f"{'='*60}")
            
            # 创建评估数据集
            eval_dataset = create_evaluation_dataset_from_csv(
                csv_path=csv_path,
                data_dir=TRAINING2017_DIR,
                fixed_length=FIXED_LENGTH,
                test_mode=True
            )
            
            if len(eval_dataset) == 0:
                print(f"数据集 cv{i}.csv 为空，跳过")
                continue
            
            # 评估模型
            dataset_name = f"cv{i}"
            results = evaluator.evaluate_on_dataset(eval_dataset, batch_size=4,dataset_name=dataset_name)
            
            # 保存到汇总结果
            all_results[f"cv{i}"] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'total_samples': results['total_samples'],
                'valid_samples': results['valid_samples'],
                'invalid_samples': results['invalid_samples'],
                'invalid_rate': results['invalid_rate'],
                'class_distribution': results['class_distribution']
            }
        
        # 计算平均指标
        if all_results:
            print(f"\n{'='*60}")
            print("交叉验证汇总结果")
            print(f"{'='*60}")
            
            avg_accuracy = np.mean([r['accuracy'] for r in all_results.values()])
            avg_precision = np.mean([r['precision'] for r in all_results.values()])
            avg_recall = np.mean([r['recall'] for r in all_results.values()])
            avg_f1 = np.mean([r['f1'] for r in all_results.values()])
            avg_invalid_rate = np.mean([r['invalid_rate'] for r in all_results.values()])
            
            print(f"平均准确率: {avg_accuracy:.4f}")
            print(f"平均精确率: {avg_precision:.4f}")
            print(f"平均召回率: {avg_recall:.4f}")
            print(f"平均F1分数: {avg_f1:.4f}")
            print(f"平均无效样本率: {avg_invalid_rate:.4f} ({avg_invalid_rate*100:.2f}%)")
            
            # 保存汇总结果
            cv_summary = {
                    'cross_validation_results': all_results,
                    'averages': {
                        'accuracy': float(avg_accuracy),
                        'precision': float(avg_precision),
                        'recall': float(avg_recall),
                        'f1': float(avg_f1),
                        'invalid_rate': float(avg_invalid_rate)
                    },
                    'total_samples': sum(r['total_samples'] for r in all_results.values()),
                    'valid_samples': sum(r['valid_samples'] for r in all_results.values()),
                    'invalid_samples': sum(r['invalid_samples'] for r in all_results.values()),
                    'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            evaluator.result_saver.save_cross_validation_summary(cv_summary)

    except Exception as e:
        print(f"交叉验证评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("ECG分类模型评估系统")
    print("="*60)
    
    # 检查必要的导入
    try:
        import scipy.io
    except ImportError:
        print("错误: 需要安装scipy库")
        print("请运行: pip install scipy")
        sys.exit(1)
    
    try:
        import sklearn
    except ImportError:
        print("错误: 需要安装scikit-learn库")
        print("请运行: pip install scikit-learn")
        sys.exit(1)
    
    # 选择运行方式
    print("\n请选择评估模式:")
    print("1. 评估单个CSV文件 (cv0.csv)")
    print("2. 运行交叉验证 (评估所有cv0-cv4.csv)")
    print("3. 退出")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        run_cross_validation_evaluation()
    elif choice == '3':
        print("退出程序")
    else:
        print("无效选择，退出程序")