import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scipy.io as io
import torch
from src.common.Config import FIXED_LENGTH, DOWNSAMPLE_RATE, AUGMENT_SETTING
from scipy.interpolate import interp1d


class ECG_dataset(Dataset):

    def __init__(self, base_file=None, cv=0, is_train=True, augment=AUGMENT_SETTING,data_list=None):
        # specify annotation file for dataset
        self.is_train = is_train
        # 是否启用数据增强
        self.augment = augment
        self.base_file = base_file
        if data_list is not None:
            # 将data_list转换为与原有格式一致
            formatted_data = []
            for item in data_list:
                # item应该是(filename, label)
                formatted_data.append([0, item[0], item[1]])
            self.file = np.array(formatted_data)
            print(f"ECG_dataset Loaded: {len(self.file)} samples from provided data_list")
        else:
            if isinstance(cv, int):
                cv_indices = [cv]
            elif isinstance(cv, list):
                cv_indices = cv
            else:
                raise ValueError("CV parameter must be an integer (for test fold) or a list of integers (for train folds).")

            # ------------------------------------------------------------------
            # 替换原有的加载/拼接逻辑，直接加载指定的 CSV 文件
            # ------------------------------------------------------------------
            all_records = []
            cv_path = os.path.join(base_file, 'cv')
            loaded_cvs = []

            for i in cv_indices:
                csv_file = os.path.join(cv_path, f'cv{i}.csv')

                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file)
                    # 将数据帧的值（记录）添加到 all_records 中
                    all_records.extend(data.values)
                    loaded_cvs.append(str(i))
                else:
                    # 警告：文件未找到
                    print(f"Warning: {csv_file} not found for K-Fold loading.")

                # 将所有的记录合并为最终的 file 数组
            self.file = np.array(all_records)

            # 打印加载摘要，方便调试
            print(f"ECG_dataset Loaded: {len(self.file)} samples from CV folds: [{', '.join(loaded_cvs)}]")

    def __len__(self):
        return self.file.shape[0]

    def load_data(self, file_name, label):
        mat_file = self.base_file + '/training2017/' + file_name + '.mat'
        data = io.loadmat(mat_file)['val']
        # if label == 'N':
        #     one_hot = torch.tensor([1, 0, 0, 0])
        # elif label == 'O':
        #     one_hot = torch.tensor([0, 1, 0, 0])
        # elif label == 'A':
        #     one_hot = torch.tensor([0, 0, 1, 0])
        # elif label == '~':
        #     one_hot = torch.tensor([0, 0, 0, 1])

        # 修改为二分类标签：AF=1, Non-AF=0
        if label == 'A':
            one_hot = torch.tensor([1.0])  # 必须是浮点数
        else:
            one_hot = torch.tensor([0.0])
        return data, one_hot

    # 添加高斯噪声
    def add_noise(self, data):
        # 50% 的概率添加噪声
        if np.random.rand() < 0.5:
            noise_level = 0.05  # 噪声强度，可调整
            noise = np.random.normal(0, noise_level, data.shape)
            data = data + noise
        return data

    # 增加随机时间尺度放缩
    def time_scaling(self, data):
        # 50% 的概率进行缩放
        if np.random.rand() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)  # 随机缩放因子 0.8~1.2
            old_len = data.shape[0]
            new_len = int(old_len * scale_factor)

            # 使用线性插值重新采样
            x_old = np.linspace(0, 1, old_len)
            x_new = np.linspace(0, 1, new_len)
            f = interp1d(x_old, data, kind='linear')
            data = f(x_new)
        return data

    # 实现随机裁切功能
    def crop_padding(self, data, length, apply_augment):
        L_raw = data.shape[0]
        
        if L_raw <= length:
            # 填充 0
            pad_len = length - L_raw
            # np.pad 接受一个元组 (before, after)，对于一维数组是 (pad_len_start, pad_len_end)
            data = np.pad(data, (0, pad_len), 'constant')
            
        elif L_raw > length:
            # 裁剪 (Cropping)
            max_start = L_raw - length
            
            if self.is_train and apply_augment:
                # 训练集 + 增强开启: 随机裁切
                start = np.random.randint(0, max_start + 1)
            else:
                # 验证集/测试集/增强关闭: 确定性裁切 (中心裁切)
                start = max_start // 2  # 整数除法，取中心点
                
            data = data[start:start + length]

        return data
    
    def data_process(self, data, apply_augment):
        # 1. 降采样
        data = data[::DOWNSAMPLE_RATE]

        # Task 1.5: 在裁剪前进行时间缩放 (如果开启)(暂时关闭时间缩放)
        if self.is_train and apply_augment:
            data = self.time_scaling(data)

        # 2. 归一化
        data = data - data.mean()
        # 防止除以 0
        std = data.std()
        data = data / std

        # Task 1.5: 在归一化后添加噪声 (如果开启)
        # 放在归一化后，噪声的大小(0.05)就相对标准差(1.0)有了明确的物理意义
        if self.is_train and apply_augment:
            data = self.add_noise(data)

        # 3. 裁剪/填充到固定长度
        data = self.crop_padding(data, FIXED_LENGTH,apply_augment)
        return data

    def __getitem__(self, idx):
        file_name = self.file[idx][1]
        label = self.file[idx][2]
        data_raw, one_hot = self.load_data(file_name, label)

        # 1. 获取增强前的数据（关闭增强）
        # 注意：这里需要传入 data_raw[0] 的副本，否则 data_process 会修改原始数据
        data_original = self.data_process(data_raw[0].copy(), apply_augment=False)

        # 2. 获取增强后的数据（开启增强，如果 is_train=True 且 self.augment=True）
        # 这里的 self.augment 决定了数据增强是否启用
        if self.is_train and self.augment:
            # 同样传入副本
            data_augmented = self.data_process(data_raw[0].copy(), apply_augment=True)
        else:
            # 如果不增强，则返回原始数据（或者您也可以直接返回 data_original）
            data_augmented = data_original

            # 仅返回增强后的数据用于训练/验证，但您可以利用 data_original 和 data_augmented 进行绘图
        return data_augmented, one_hot, file_name