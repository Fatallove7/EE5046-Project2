import os
import pandas as pd
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d  # 用于 time_scaling 依赖
import torch  # 用于 ECG_dataset 依赖
import json
import matplotlib.pyplot as plt
import numpy as np

# 导入您自定义的模块
# 请确保以下导入路径对于您运行的脚本是正确的
from ECGDataset import ECG_dataset
from Config import FIXED_LENGTH, DOWNSAMPLE_RATE


# 注意：为了让 ECGAnalyzer 独立工作，ECG_dataset 的依赖（如 time_scaling, add_noise）
# 必须能被访问到。由于这些方法在 ECG_dataset 内部，我们通过实例化它来使用。

class ECGAnalyzer:
    def __init__(self, base_path):
        """
        初始化ECG分析和可视化工具
        :param base_path: 数据集根目录 (例如 '../Dataset')
        """
        self.base_path = base_path
        self.file_map = []  # 用于存储所有的 [index, filename, label]
        self.save_dir = os.path.join(self.base_path, "Fig")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Figures will be saved to: {os.path.abspath(self.save_dir)}")

        # --- 整合自 ECGPlotter ---
        self.data_processor = ECG_dataset(
            base_file=base_path,
            is_train=True,
            augment=True,
            cv=0  # 只需要一个有效的 cv 即可初始化
        )
        # -------------------------

        # 1. 读取 cv0.csv 到 cv4.csv 获取所有文件的索引
        cv_path = os.path.join(base_path, 'cv')
        for i in range(5):
            csv_file = os.path.join(cv_path, f'cv{i}.csv')
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                self.file_map.extend(df.values)
            else:
                print(f"Warning: {csv_file} not found.")

        self.file_map = np.array(self.file_map)
        print(f"Total records loaded: {len(self.file_map)}")
        
        # 现在 file_map 有数据了，再创建标签映射 ✅
        self.label_indices = self._create_label_map()  # 移动到数据加载之后

    def _create_label_map(self):
        """
        内部方法：创建标签到全局索引的映射。
        """
        label_map = {'A': [], 'O': [], 'N': [], '~': []}

        # self.file_map 结构: [[index, filename, label], ...]
        # 我们关注的是原始的全局索引 i 和标签 record[2]
        for i, record in enumerate(self.file_map):
            label = record[2]  # 标签在数组的第三列
            if label in label_map:
                label_map[label].append(i)

        # 将列表转换为 NumPy 数组，方便后续随机抽取
        for label, indices in label_map.items():
            label_map[label] = np.array(indices)

        return label_map

    def get_indices_by_label(self, label):
        """
        根据标签（'A', 'O', 'N', '~'）获取所有对应的全局索引。

        Args:
            label (str): 目标类别标签。

        Returns:
            np.ndarray: 包含该标签所有记录的全局索引数组。
        """
        if label in self.label_indices:
            return self.label_indices[label]
        else:
            print(f"Warning: Label '{label}' not found in dataset.")
            return np.array([])

    def get_random_idx_by_label(self, label):
        """
        获取指定标签下的一个随机全局索引。
        """
        indices = self.get_indices_by_label(label)
        if len(indices) > 0:
            return np.random.choice(indices)
        else:
            return None

    def load_raw_data(self, idx):
        """
        根据索引读取原始 .mat 文件
        """
        if idx >= len(self.file_map):
            raise IndexError("Index out of bounds")

        record = self.file_map[idx]
        file_name = record[1]
        label = record[2]

        mat_path = os.path.join(self.base_path, 'training2017', f'{file_name}.mat')

        try:
            data = scio.loadmat(mat_path)
            signal = data['val'].squeeze()
            return signal, label, file_name
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            return None, None, None

    # =========================================================
    # 方法一：原始数据分析绘图 (时域/频域/时频)
    # =========================================================
    def plot_analysis(self, idx=None, fs=300, save_plot=True):
        """
        绘制原始数据的时域、频域、时频图
        """
        if idx is None:
            idx = np.random.randint(0, len(self.file_map))

        signal, label, filename = self.load_raw_data(idx)

        if signal is None:
            return

        N = len(signal)
        duration = N / fs

        # ... (绘图逻辑保持不变) ...
        fig = plt.figure(figsize=(15, 10))
        plt.suptitle(f"Raw ECG Analysis (File: {filename}, Label: {label})", fontsize=16)

        # 1. 时域图 (Time Domain)
        ax1 = fig.add_subplot(3, 1, 1)
        t = np.linspace(0, duration, N)
        ax1.plot(t, signal, color='black', linewidth=0.8)
        ax1.set_title(f"Time Domain (Length: {N} points, Duration: {duration:.2f}s)", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (mV)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, duration])

        # 2. 频域图 (Frequency Domain)
        ax2 = fig.add_subplot(3, 1, 2)
        yf = fft(signal)
        xf = fftfreq(N, 1 / fs)
        idx_half = N // 2
        freqs = xf[:idx_half]
        magnitudes = 2.0 / N * np.abs(yf[:idx_half])

        ax2.plot(freqs, magnitudes, color='tab:blue')
        ax2.set_title("Frequency Domain (FFT Spectrum)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_xlim([0, 100])
        ax2.grid(True, alpha=0.3)

        # 3. 时频图 (Spectrogram)
        ax3 = fig.add_subplot(3, 1, 3)
        Pxx, freqs_spec, bins, im = ax3.specgram(signal, NFFT=256, Fs=fs, noverlap=128, cmap='jet')
        ax3.set_title("Time-Frequency Domain (Spectrogram)", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Frequency (Hz)")
        ax3.set_ylim([0, 100])

        cbar = plt.colorbar(im, ax=ax3, orientation='vertical')
        cbar.set_label('Intensity (dB)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plot:
            # 格式：[Label]_[Filename]_Analysis.png
            save_name = f"{label}_{filename}_Analysis.png"
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path)
            print(f"Analysis figure saved to: {save_path}")
        plt.show()

    # =========================================================
    # 方法二：数据增强对比绘图 (整合自 ECGPlotter)
    # =========================================================
    def plot_augmentation_comparison(self, idx=None, show_plot=True, save_plot=True):
        """
        加载指定文件，处理并绘制原始波形与增强后的波形对比图。

        :param idx: 数据索引。
        :param show_plot: 是否显示绘图窗口。
        """
        if idx is None:
            idx = np.random.randint(0, len(self.file_map))

        data_raw, label, file_name = self.load_raw_data(idx)

        if data_raw is None:
            return None, None, None

        # 1. 获取增强前的数据（关闭增强）
        # 调用 data_processor 的 data_process 方法，关闭增强
        data_original = self.data_processor.data_process(data_raw.copy(), apply_augment=False)

        # 2. 获取增强后的数据（开启增强）
        # 调用 data_processor 的 data_process 方法，开启增强
        data_augmented = self.data_processor.data_process(data_raw.copy(), apply_augment=True)

        # 3. 绘图对比
        time_points = np.arange(FIXED_LENGTH)
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        plt.suptitle(f"Augmentation Comparison (File: {file_name}, Label: {label})", fontsize=16)

        # 绘制原始波形
        axes[0].plot(time_points, data_original, label='Original Processed (No Augmentation)', color='blue')
        axes[0].set_title('Processed Waveform (No Augmentation)', fontsize=14)
        axes[0].set_ylabel('Amplitude (Normalized)')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # 绘制增强后的波形
        axes[1].plot(time_points, data_augmented, label='Augmented Waveform', color='red', alpha=0.7)
        axes[1].set_title(f'Augmented Waveform (Time Scaling + Noise)', fontsize=14)
        axes[1].set_xlabel(f'Time Steps (Length: {FIXED_LENGTH}, Downsampled by {DOWNSAMPLE_RATE})')
        axes[1].set_ylabel('Amplitude (Normalized)')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_plot:
            # 格式：[Label]_[Filename]_AugCompare.png
            save_name = f"{label}_{file_name}_AugCompare.png"
            save_path = os.path.join(self.save_dir, save_name)
            fig.savefig(save_path)
            print(f"Augmentation comparison figure saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig, data_original, data_augmented



    def plot_cnn_results_from_json(json_file_path, save_path=None):
        """
        从JSON文件加载CNN训练结果并绘制柱状图
        
        Args:
            json_file_path: JSON文件路径
            save_path: 图片保存路径，如果为None则显示图片
        """
        # 1. 加载JSON数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 提取数据
        # 基础配置
        kernel_config = data.get('best_config', {}).get('kernel_config', {})
        kernel_name = kernel_config.get('name', 'Unknown')
        stream1_kernel = kernel_config.get('stream1_kernel', 'N/A')
        stream2_kernel = kernel_config.get('stream2_first_kernel', 'N/A')
        batch_size = data.get('best_config', {}).get('batch_size', 'N/A')
        
        # 最终指标
        final_metrics = data.get('final_metrics', {})
        avg_accuracy = final_metrics.get('avg_accuracy', 0)
        avg_auc = final_metrics.get('avg_auc', 0)
        avg_loss = final_metrics.get('avg_loss', 0)
        
        # 各fold结果
        fold_results = data.get('fold_results', [])
        
        # 3. 准备绘图数据
        fold_numbers = []
        fold_accuracies = []
        fold_aucs = []
        fold_losses = []
        best_epochs = []
        
        for fold in fold_results:
            fold_numbers.append(f"Fold {fold['fold']}")
            fold_accuracies.append(fold.get('final_val_acc', 0))
            fold_aucs.append(fold.get('final_val_auc', 0))
            fold_losses.append(fold.get('best_loss', 0))
            best_epochs.append(fold.get('best_epoch', 0))
        
        # 4. 创建图形和子图
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'CNN Model Performance Analysis - {kernel_name}', fontsize=16, fontweight='bold')
        
        # 子图1: 各Fold准确率对比
        ax1 = plt.subplot(2, 3, 1)
        x_pos = np.arange(len(fold_numbers))
        bars1 = ax1.bar(x_pos, fold_accuracies, color='skyblue', alpha=0.8)
        ax1.axhline(y=avg_accuracy, color='red', linestyle='--', linewidth=2, 
                    label=f'Avg: {avg_accuracy:.4f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy per Fold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(fold_numbers)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 子图2: 各Fold AUC对比
        ax2 = plt.subplot(2, 3, 2)
        bars2 = ax2.bar(x_pos, fold_aucs, color='lightgreen', alpha=0.8)
        ax2.axhline(y=avg_auc, color='red', linestyle='--', linewidth=2,
                    label=f'Avg: {avg_auc:.4f}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('AUC')
        ax2.set_title('AUC per Fold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(fold_numbers)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 子图3: 各Fold最佳损失对比
        ax3 = plt.subplot(2, 3, 3)
        bars3 = ax3.bar(x_pos, fold_losses, color='lightcoral', alpha=0.8)
        ax3.axhline(y=avg_loss, color='blue', linestyle='--', linewidth=2,
                    label=f'Avg: {avg_loss:.4f}')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Loss')
        ax3.set_title('Best Loss per Fold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(fold_numbers)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 子图4: 各Fold最佳Epoch对比
        ax4 = plt.subplot(2, 3, 4)
        colors = plt.cm.viridis(np.linspace(0, 1, len(best_epochs)))
        bars4 = ax4.bar(x_pos, best_epochs, color=colors, alpha=0.8)
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Best Epoch')
        ax4.set_title('Best Training Epoch per Fold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(fold_numbers)
        ax4.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 子图5: 准确率和AUC的联合对比
        ax5 = plt.subplot(2, 3, 5)
        width = 0.35
        bars5a = ax5.bar(x_pos - width/2, fold_accuracies, width, 
                        color='skyblue', alpha=0.8, label='Accuracy')
        bars5b = ax5.bar(x_pos + width/2, fold_aucs, width, 
                        color='lightgreen', alpha=0.8, label='AUC')
        ax5.set_xlabel('Fold')
        ax5.set_ylabel('Score')
        ax5.set_title('Accuracy vs AUC per Fold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(fold_numbers)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 在柱子上添加数值标签（简化版，避免重叠）
        for i, (acc, auc) in enumerate(zip(fold_accuracies, fold_aucs)):
            ax5.text(i - width/2, acc + 0.005, f'{acc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
            ax5.text(i + width/2, auc + 0.005, f'{auc:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 子图6: 模型配置和总体统计信息（文本信息）
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')  # 关闭坐标轴
        
        # 构建信息文本
        info_text = f"""
        Model Configuration:
        --------------------
        Kernel Name: {kernel_name}
        Stream1 Kernel: {stream1_kernel}
        Stream2 First Kernel: {stream2_kernel}
        Batch Size: {batch_size}
        
        Overall Performance:
        --------------------
        Average Accuracy: {avg_accuracy:.4f}
        Standard Deviation: {final_metrics.get('std_accuracy', 0):.4f}
        
        Average AUC: {avg_auc:.4f}
        Standard Deviation: {final_metrics.get('std_auc', 0):.4f}
        
        Average Loss: {avg_loss:.4f}
        
        Training Info:
        --------------
        Timestamp: {data.get('training_timestamp', 'Unknown')}
        Total Folds: {len(fold_results)}
        """
        
        ax6.text(0.05, 0.95, info_text, fontsize=10, family='monospace',
                verticalalignment='top', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. 调整布局并保存/显示
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        # 6. 返回统计摘要
        summary = {
            'model_config': {
                'kernel_name': kernel_name,
                'stream1_kernel': stream1_kernel,
                'stream2_kernel': stream2_kernel,
                'batch_size': batch_size
            },
            'performance': {
                'avg_accuracy': avg_accuracy,
                'avg_auc': avg_auc,
                'avg_loss': avg_loss,
                'fold_accuracies': fold_accuracies,
                'fold_aucs': fold_aucs,
                'fold_losses': fold_losses
            }
        }
        
        return summary

    def plot_simple_comparison(json_file_path, save_path=None):
        """
        简化的对比图表，适合PPT展示
        
        Args:
            json_file_path: JSON文件路径
            save_path: 图片保存路径
        """
        # 加载JSON数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取数据
        kernel_config = data.get('best_config', {}).get('kernel_config', {})
        kernel_name = kernel_config.get('name', 'Unknown')
        
        final_metrics = data.get('final_metrics', {})
        avg_accuracy = final_metrics.get('avg_accuracy', 0)
        avg_auc = final_metrics.get('avg_auc', 0)
        
        fold_results = data.get('fold_results', [])
        fold_numbers = [f"Fold {fold['fold']}" for fold in fold_results]
        fold_accuracies = [fold.get('final_val_acc', 0) for fold in fold_results]
        fold_aucs = [fold.get('final_val_auc', 0) for fold in fold_results]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：各Fold准确率
        x_pos = np.arange(len(fold_numbers))
        width = 0.4
        
        bars1 = ax1.bar(x_pos - width/2, fold_accuracies, width, 
                        color='#2E86AB', alpha=0.8, label='Fold Accuracy')
        ax1.axhline(y=avg_accuracy, color='#A23B72', linestyle='--', 
                    linewidth=2.5, label=f'Average: {avg_accuracy:.4f}')
        
        ax1.set_xlabel('Cross-Validation Fold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title(f'Accuracy per Fold\n({kernel_name})', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(fold_numbers)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 右图：各Fold AUC
        bars2 = ax2.bar(x_pos - width/2, fold_aucs, width, 
                        color='#18A999', alpha=0.8, label='Fold AUC')
        ax2.axhline(y=avg_auc, color='#A23B72', linestyle='--', 
                    linewidth=2.5, label=f'Average: {avg_auc:.4f}')
        
        ax2.set_xlabel('Cross-Validation Fold', fontsize=12)
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.set_title(f'AUC per Fold\n({kernel_name})', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(fold_numbers)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"简化对比图已保存到: {save_path}")
        else:
            plt.show()
        
        # 返回关键统计数据
        return {
            'kernel_name': kernel_name,
            'avg_accuracy': avg_accuracy,
            'avg_auc': avg_auc,
            'accuracy_std': final_metrics.get('std_accuracy', 0),
            'auc_std': final_metrics.get('std_auc', 0)
        }