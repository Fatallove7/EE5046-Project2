import os
import random  # 添加导入
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scio
from src.common.Config import DOWNSAMPLE_RATE, FIXED_LENGTH

class FoldDataset(Dataset):
    """处理从DataManager获取的数据数组，添加不平衡数据处理功能"""
    
    def __init__(self, data, base_path, is_train=True, augment=False, 
                 augmentation_config=None):
        """
        Args:
            data: 支持多种格式:
                  - List[Tuple[filename, label]]
                  - List[Tuple[index, filename, label]]
                  - np.array of shape (n_samples, 2) or (n_samples, 3)
            base_path: 数据集根目录
            is_train: 是否为训练集
            augment: 是否启用数据增强
            augmentation_config: 数据增强配置
        """
        # 统一数据格式为numpy数组
        self.data_array = self._standardize_data_format(data)
        
        self.base_path = base_path
        self.is_train = is_train
        self.augment = augment
        
        # 默认数据增强配置
        self.augmentation_config = augmentation_config or {
            'positive_augment_factor': 5,
            'noise_std': 0.01,
            'scale_range': (0.9, 1.1),
            'shift_range': (-10, 10),
            'warp_range': (0.95, 1.05),
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'use_cutmix': False,
            'positive_only': True
        }
        
        # 初始化增强样本列表
        self.augmented_samples = []
        
        # 分离正负样本索引
        self._init_sample_indices()
        
        # 如果启用了数据增强，创建增强样本
        if self.is_train and self.augment and len(self.positive_indices) > 0:
            self._create_augmented_samples()

    def _standardize_data_format(self, data):
        """将不同格式的数据统一为numpy数组"""
        if data is None or len(data) == 0:
            return np.array([], dtype=object)
        
        if isinstance(data, np.ndarray):
            return data
        
        # 转换为列表的列表
        data_list = []
        for item in data:
            if isinstance(item, (list, tuple, np.ndarray)):
                data_list.append(list(item))
            else:
                # 如果是单个元素，包装成列表
                data_list.append([item])
        
        return np.array(data_list, dtype=object)
    
    def _init_sample_indices(self):
        """初始化正负样本索引"""
        self.positive_indices = []
        self.negative_indices = []
        
        for idx in range(len(self.data_array)):
            record = self.data_array[idx]
            label = self._extract_label_from_record(record)
            
            if label == 1:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        print(f"FoldDataset: 总样本={len(self.data_array)}, "
              f"正样本={len(self.positive_indices)}, "
              f"负样本={len(self.negative_indices)}")
    
    def _extract_label_from_record(self, record):
        """从记录中提取标签"""
        # 根据记录长度确定标签位置
        if len(record) >= 3:  # [index, filename, label]
            label = record[2]
        else:  # [filename, label]
            label = record[1]
        
        # 转换为二进制标签
        if isinstance(label, (int, float, np.integer, np.floating)):
            return 1 if float(label) > 0.5 else 0
        elif isinstance(label, str):
            return 1 if label.upper() == 'A' else 0
        else:
            return 0
    
    def _extract_filename_from_record(self, record):
        """从记录中提取文件名"""
        if len(record) >= 3:  # [index, filename, label]
            return str(record[1])
        else:  # [filename, label]
            return str(record[0])
    
    def __len__(self):
        """返回数据集总长度"""
        # 如果启用了数据增强，返回原始长度+增强样本长度
        if self.is_train and self.augment:
            return len(self.data_array) + len(self.augmented_samples)
        return len(self.data_array)
    
    def __getitem__(self, idx):
        """获取指定索引的数据样本"""
        # 判断是否在增强样本范围内
        if self.is_train and self.augment and idx >= len(self.data_array):
            # 返回增强样本
            aug_idx = idx - len(self.data_array)
            if aug_idx < len(self.augmented_samples):
                return self._get_augmented_sample(aug_idx)
        
        # 获取原始样本
        record = self.data_array[idx]
        filename = self._extract_filename_from_record(record)
        label = self._extract_label_from_record(record)
        
        # 加载.mat文件
        mat_path = os.path.join(self.base_path, 'training2017', f'{filename}.mat')
        try:
            data = scio.loadmat(mat_path)['val'][0]
        except Exception as e:
            print(f"错误加载 {mat_path}: {e}")
            data = np.zeros(FIXED_LENGTH)
        
        # 数据预处理
        data = self._preprocess_data(data)
        
        # 应用在线数据增强（仅对正样本且在训练时）
        if self.is_train and self.augment and label == 1:
            data = self._apply_online_augmentation(data)
        
        # 转换为张量
        data_tensor = torch.FloatTensor(data).unsqueeze(0)
        label_tensor = torch.tensor([float(label)])
        
        return data_tensor, label_tensor
    
    def _preprocess_data(self, data):
        """数据预处理"""
        # 1. 降采样
        if DOWNSAMPLE_RATE > 1:
            data = data[::DOWNSAMPLE_RATE]
        
        # 2. 归一化
        data = data - data.mean()
        std = data.std()
        if std > 0:
            data = data / std
        else:
            data = data * 0  # 如果标准差为0，设为0
        
        # 3. 裁剪/填充到固定长度
        L = len(data)
        if L < FIXED_LENGTH:
            # 填充
            pad_width = FIXED_LENGTH - L
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            data = np.pad(data, (left_pad, right_pad), 'constant')
        elif L > FIXED_LENGTH:
            # 如果是训练集且启用增强，随机裁剪；否则中心裁剪
            if self.is_train and self.augment:
                start = np.random.randint(0, L - FIXED_LENGTH)
            else:
                start = (L - FIXED_LENGTH) // 2
            data = data[start:start + FIXED_LENGTH]
        
        return data
    
    def _create_augmented_samples(self):
        """创建增强样本（离线增强）"""
        config = self.augmentation_config
        
        # 只对正样本进行离线增强
        for i in range(config['positive_augment_factor']):
            for pos_idx in self.positive_indices:
                record = self.data_array[pos_idx]
                filename = self._extract_filename_from_record(record)
                label = self._extract_label_from_record(record)
                
                # 创建增强版本的文件名
                aug_filename = f"{filename}_aug{i}"
                
                # 保存增强信息
                self.augmented_samples.append({
                    'original_idx': pos_idx,
                    'filename': aug_filename,
                    'label': label,
                    'augmentation_type': 'offline'
                })
        
        print(f"创建了 {len(self.augmented_samples)} 个离线增强样本")
    
    def _get_augmented_sample(self, aug_idx):
        """获取增强样本"""
        aug_info = self.augmented_samples[aug_idx]
        original_idx = aug_info['original_idx']
        label = aug_info['label']
        
        # 获取原始数据
        record = self.data_array[original_idx]
        filename = self._extract_filename_from_record(record)
        
        # 加载原始数据
        mat_path = os.path.join(self.base_path, 'training2017', f'{filename}.mat')
        try:
            data = scio.loadmat(mat_path)['val'][0]
        except Exception as e:
            print(f"错误加载 {mat_path}: {e}")
            data = np.zeros(FIXED_LENGTH)
        
        # 预处理
        data = self._preprocess_data(data)
        
        # 应用更强的增强（因为是离线增强，可以应用多种增强组合）
        data = self._apply_strong_augmentation(data)
        
        # 转换为张量
        data_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        # 标签转换
        label_val = float(label) if isinstance(label, (int, float)) else 1.0
        label_tensor = torch.tensor([label_val])
        
        return data_tensor, label_tensor
    
    def _apply_online_augmentation(self, data):
        """应用在线数据增强（训练时实时增强）"""
        if len(data) == 0:
            return data
        
        augmented = data.copy()
        
        # 随机选择一种或多种增强方式
        aug_methods = []
        
        # 添加噪声
        if random.random() < 0.3:  # 30%概率
            aug_methods.append('noise')
        
        # 缩放
        if random.random() < 0.3:  # 30%概率
            aug_methods.append('scale')
        
        # 时间平移
        if random.random() < 0.2:  # 20%概率
            aug_methods.append('shift')
        
        # 时间扭曲
        if random.random() < 0.1:  # 10%概率
            aug_methods.append('warp')
        
        # 如果选择了增强方法，按顺序应用
        if aug_methods:
            for method in aug_methods:
                if method == 'noise' and self.augmentation_config.get('noise_std', 0.01) > 0:
                    augmented = self._add_noise(augmented)
                elif method == 'scale':
                    augmented = self._scale_signal(augmented)
                elif method == 'shift':
                    augmented = self._shift_signal(augmented)
                elif method == 'warp':
                    augmented = self._warp_signal(augmented)
        
        return augmented
    
    def _apply_strong_augmentation(self, data):
        """应用更强的数据增强（用于离线增强）"""
        if len(data) == 0:
            return data
        
        augmented = data.copy()
        
        # 应用所有增强方式（组合增强）
        config = self.augmentation_config
        
        # 1. 添加噪声
        if config.get('noise_std', 0.01) > 0:
            augmented = self._add_noise(augmented)
        
        # 2. 缩放
        augmented = self._scale_signal(augmented)
        
        # 3. 时间平移（50%概率）
        if random.random() < 0.5:
            augmented = self._shift_signal(augmented)
        
        # 4. 时间扭曲（30%概率）
        if random.random() < 0.3:
            augmented = self._warp_signal(augmented)
        
        return augmented
    
    def _add_noise(self, signal_data):
        """添加高斯噪声"""
        noise_std = self.augmentation_config.get('noise_std', 0.01)
        noise = np.random.normal(0, noise_std, signal_data.shape)
        return signal_data + noise
    
    def _scale_signal(self, signal_data):
        """随机缩放信号"""
        scale_range = self.augmentation_config.get('scale_range', (0.9, 1.1))
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return signal_data * scale
    
    def _shift_signal(self, signal_data):
        """随机平移信号"""
        shift_range = self.augmentation_config.get('shift_range', (-10, 10))
        shift = np.random.randint(shift_range[0], shift_range[1] + 1)
        
        if shift == 0:
            return signal_data
        
        result = np.zeros_like(signal_data)
        if shift > 0:
            # 向右平移
            result[shift:] = signal_data[:-shift]
            result[:shift] = signal_data[-shift:]  # 循环平移
        else:
            # 向左平移
            shift = abs(shift)
            result[:-shift] = signal_data[shift:]
            result[-shift:] = signal_data[:shift]  # 循环平移
        
        return result
    
    def _warp_signal(self, signal_data):
        """时间扭曲（插值）"""
        warp_range = self.augmentation_config.get('warp_range', (0.95, 1.05))
        scale = np.random.uniform(warp_range[0], warp_range[1])
        
        orig_len = len(signal_data)
        new_len = int(orig_len * scale)
        
        # 创建原始坐标和新坐标
        orig_x = np.linspace(0, orig_len - 1, orig_len)
        new_x = np.linspace(0, orig_len - 1, new_len)
        
        # 线性插值
        warped = np.interp(new_x, orig_x, signal_data)
        
        # 调整回原始长度
        if new_len < orig_len:
            # 填充
            warped = np.pad(warped, (0, orig_len - new_len), 'constant')
        elif new_len > orig_len:
            # 裁剪
            start = (new_len - orig_len) // 2
            warped = warped[start:start + orig_len]
        
        return warped
    
    def apply_mixup(self, data1, label1, data2, label2):
        """应用Mixup增强"""
        if not self.augmentation_config.get('use_mixup', True):
            return data1, label1
        
        alpha = self.augmentation_config.get('mixup_alpha', 0.2)
        
        # 生成混合系数
        lam = np.random.beta(alpha, alpha)
        
        # 混合数据
        mixed_data = lam * data1 + (1 - lam) * data2
        
        # 混合标签（对于多标签可能需要不同的处理，这里是二分类）
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_data, mixed_label