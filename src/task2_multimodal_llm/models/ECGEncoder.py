# ECGEncoder.py
import torch
import torch.nn as nn
from src.task1_ecg_analysis.training.TrainModel import Mscnn  # 导入任务 1 的 CNN 定义

class ECGEncoder(Mscnn):
    """
    基于 Mscnn，只执行特征提取，且完全冻结。
    """

    def __init__(self, cnn_config, weights_path, device=None):
        """
        Args:
            cnn_config: CNN 配置字典
            weights_path: 预训练权重路径
            device: 设备 ('cpu', 'cuda', 'cuda:0' 等)，如果为 None 则自动选择
        """
        # 实例化 Mscnn，但只关注特征提取部分
        super(ECGEncoder, self).__init__(**cnn_config)
        
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"ECGEncoder 将在设备上运行: {self.device}")

        # 1. 加载任务 1 训练好的权重
        # 使用 map_location 将权重加载到指定设备
        state_dict = torch.load(weights_path, map_location=self.device)
        
        # 检查权重是否与模型架构匹配
        self.load_state_dict(state_dict)

        # 2. 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 3. 设置为评估模式
        self.eval()

        # 4. 将模型移动到指定设备
        self.to(self.device)

    def forward(self, x):
        # 确保输入在正确的设备上
        if x.device != self.device:
            x = x.to(self.device)
        
        # 复制 Mscnn forward 中特征提取和融合的部分
        # (确保 x 的形状是 [N, 1, FIXED_LENGTH])
        with torch.no_grad():
            x1 = self.s1_pool5(self.s1_conv5(self.s1_pool4(self.s1_conv4(
                self.s1_pool3(self.s1_conv3(self.s1_pool2(self.s1_conv2(self.s1_pool1(self.s1_conv1(x))))))))))
            x1_flat = x1.view(x1.size(0), -1)

            if self.use_stream2:
                x2 = self.s2_pool5(self.s2_conv5(self.s2_pool4(self.s2_conv4(
                    self.s2_pool3(self.s2_conv3(self.s2_pool2(self.s2_conv2(self.s2_pool1(self.s2_conv1(x))))))))))
                x2_flat = x2.view(x2.size(0), -1)
                merge = torch.cat([x1_flat, x2_flat], dim=1)  # Z_v
            else:
                merge = x1_flat

            return merge  # 返回特征向量 Z_v