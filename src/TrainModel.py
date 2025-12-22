# %%
import torch
import torch.nn as nn
from Config import FIXED_LENGTH, INPUT_CHANNELS, OUTPUT_CLASSES

# %%
# In this section, we will apply an CNN to extract features and implement a classification task.
# Firstly, we should build the model by PyTorch. We provide a baseline model here.
# You can use your own model for better performance
class Doubleconv_33(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_33, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_35(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_35, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_37(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_37, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=7),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Tripleconv(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size):
        super(Tripleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class MLP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, input):
        return self.fc(input)


    
class Mscnn(nn.Module):
    def _create_conv_block(self, ch_in, ch_out, kernel_size, triple=False):
        """创建卷积块，支持不同大小的卷积核"""
        if triple:
            return nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size),
                nn.ReLU(inplace=True)
            )


    def __init__(self, ch_in, ch_out, use_stream2=True, stream1_kernel=3, stream2_first_kernel=7):
        """
        灵活的MS-CNN模型
        Args:
            stream1_kernel: Stream1所有层的卷积核大小（论文固定为3）
            stream2_first_kernel: Stream2前4层的卷积核大小（5,7,9）
        """
        super(Mscnn, self).__init__()
        self.use_stream2 = use_stream2
        self.stream1_kernel = stream1_kernel
        self.stream2_first_kernel = stream2_first_kernel

        print(f"初始化MS-CNN: Stream1_kernel={stream1_kernel}, "
              f"Stream2_first_kernel={stream2_first_kernel}")
        # 默认卷积核大小
        # ==================== Stream 1 Definition ====================
        if stream1_kernel == 3:
            self.s1_conv1 = Doubleconv_33(ch_in, 64)
            self.s1_conv2 = Doubleconv_33(64, 128)
        elif stream1_kernel == 5:
            self.s1_conv1 = Doubleconv_35(ch_in, 64)
            self.s1_conv2 = Doubleconv_35(64, 128)
        elif stream1_kernel == 7:
            self.s1_conv1 = Doubleconv_37(ch_in, 64)
            self.s1_conv2 = Doubleconv_37(64, 128)
        else:
            # 自定义卷积核
            self.s1_conv1 = self._create_conv_block(ch_in, 64, stream1_kernel)
            self.s1_conv2 = self._create_conv_block(64, 128, stream1_kernel)

        self.s1_pool1 = nn.MaxPool1d(3, stride=3)
        self.s1_pool2 = nn.MaxPool1d(3, stride=3)

        # Stream1的后面三层（固定使用Tripleconv，但Tripleconv使用kernel=3）
        self.s1_conv3 = Tripleconv(128, 256,kernel_size=3)
        self.s1_pool3 = nn.MaxPool1d(2, stride=2)
        self.s1_conv4 = Tripleconv(256, 512,kernel_size=3)
        self.s1_pool4 = nn.MaxPool1d(2, stride=2)
        self.s1_conv5 = Tripleconv(512, 512,kernel_size=3)
        self.s1_pool5 = nn.MaxPool1d(2, stride=2)

        # ==================== Stream 2 Definition ====================
        if self.use_stream2:
            # 前4层使用指定的卷积核大小
            if stream2_first_kernel == 3:
                self.s2_conv1 = Doubleconv_33(ch_in, 64)
                self.s2_conv2 = Doubleconv_33(64, 128)
                self.s2_conv3 = Tripleconv(128, 256,kernel_size=3)
                self.s2_conv4 = Tripleconv(256, 512,kernel_size=3)
            elif stream2_first_kernel == 5:
                self.s2_conv1 = Doubleconv_35(ch_in, 64)
                self.s2_conv2 = Doubleconv_35(64, 128)
                self.s2_conv3 = Tripleconv(128, 256,kernel_size=5)  # 注意：这里需要修改Tripleconv以支持kernel=5
                self.s2_conv4 = Tripleconv(256, 512,kernel_size=5)
            elif stream2_first_kernel == 7:
                self.s2_conv1 = Doubleconv_37(ch_in, 64)
                self.s2_conv2 = Doubleconv_37(64, 128)
                self.s2_conv3 = Tripleconv(128, 256,kernel_size=7)  # 注意：这里需要修改Tripleconv以支持kernel=7
                self.s2_conv4 = Tripleconv(256, 512,kernel_size=7)
            elif stream2_first_kernel == 9:
                self.s2_conv1 = self._create_conv_block(ch_in, 64, 9)
                self.s2_conv2 = self._create_conv_block(64, 128, 9)
                self.s2_conv3 = self._create_conv_block(128, 256, 9, triple=True)
                self.s2_conv4 = self._create_conv_block(256, 512, 9, triple=True)
            else:
                raise ValueError(f"不支持的卷积核大小: {stream2_first_kernel}")

            # 第5层使用卷积核3（符合论文）
            self.s2_conv5 = Tripleconv(512, 512, kernel_size=3)

            self.s2_pool1 = nn.MaxPool1d(3, stride=3)
            self.s2_pool2 = nn.MaxPool1d(3, stride=3)
            self.s2_pool3 = nn.MaxPool1d(2, stride=2)
            self.s2_pool4 = nn.MaxPool1d(2, stride=2)
            self.s2_pool5 = nn.MaxPool1d(2, stride=2)

        # ==================== 自动计算 MLP 输入维度 ====================
        # 1. 创建一个假的输入数据，形状为 [1, 1, 3000]
        # 注意：这里的 3000 必须和你 dataset 中 crop_padding 的长度一致！
        FIXED_LENGTH = 3000
        dummy_input = torch.zeros(1, ch_in, FIXED_LENGTH)

        # 2. 计算 Stream 1 的输出大小
        with torch.no_grad():
            x1 = self.s1_conv1(dummy_input)
            x1 = self.s1_pool1(x1)
            x1 = self.s1_conv2(x1)
            x1 = self.s1_pool2(x1)
            x1 = self.s1_conv3(x1)
            x1 = self.s1_pool3(x1)
            x1 = self.s1_conv4(x1)
            x1 = self.s1_pool4(x1)
            x1 = self.s1_conv5(x1)
            x1 = self.s1_pool5(x1)
            # 获取展平后的维度 (1 * Channels * Length)
            flat_dim = x1.view(1, -1).shape[1]

            # 3. 如果有 Stream 2，计算 Stream 2 的输出大小并累加
            if self.use_stream2:
                x2 = self.s2_conv1(dummy_input)
                x2 = self.s2_pool1(x2)
                x2 = self.s2_conv2(x2)
                x2 = self.s2_pool2(x2)
                x2 = self.s2_conv3(x2)
                x2 = self.s2_pool3(x2)
                x2 = self.s2_conv4(x2)
                x2 = self.s2_pool4(x2)
                x2 = self.s2_conv5(x2)
                x2 = self.s2_pool5(x2)
                flat_dim += x2.view(1, -1).shape[1]

        print(f"自动计算的全连接层输入维度: {flat_dim}")

        # 使用自动计算出的维度初始化 MLP
        self.out = MLP(flat_dim, ch_out)

    def forward(self, x):
        x1 = self.s1_conv1(x)
        x1 = self.s1_pool1(x1)
        x1 = self.s1_conv2(x1)
        x1 = self.s1_pool2(x1)
        x1 = self.s1_conv3(x1)
        x1 = self.s1_pool3(x1)
        x1 = self.s1_conv4(x1)
        x1 = self.s1_pool4(x1)
        x1 = self.s1_conv5(x1)
        x1 = self.s1_pool5(x1)
        x1_flat = x1.view(x1.size(0), -1)

        if self.use_stream2:
            x2 = self.s2_conv1(x)
            x2 = self.s2_pool1(x2)
            x2 = self.s2_conv2(x2)
            x2 = self.s2_pool2(x2)
            x2 = self.s2_conv3(x2)
            x2 = self.s2_pool3(x2)
            x2 = self.s2_conv4(x2)
            x2 = self.s2_pool4(x2)
            x2 = self.s2_conv5(x2)
            x2 = self.s2_pool5(x2)
            x2_flat = x2.view(x2.size(0), -1)

            merge = torch.cat([x1_flat, x2_flat], dim=1)
        else:
            merge = x1_flat

        output = self.out(merge)
        output = torch.sigmoid(output)  # 记得这里要改成 torch.sigmoid

        return output