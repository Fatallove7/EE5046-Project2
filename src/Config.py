# =======================================================
# 项目常量配置
# =======================================================

# 信号长度和处理参数
FIXED_LENGTH = 3000  # 最终裁剪/填充后的ECG信号长度
DOWNSAMPLE_RATE = 3  # 原始数据降采样率 (即 data[::3])
INPUT_CHANNELS = 1   # 输入通道数 (ECG导联数)
OUTPUT_CLASSES = 1   # 输出类别数 (二分类，AF vs Non-AF)

# 训练超参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
# ... 其他参数
AUGMENT_SETTING = True
USE_STREAM2_SETTING = True
EARLY_STOP_PATIENCE = 15
MIN_DELTA = 1e-4

# 实验模式
EXPERIMENT_MODE = "search"  # search/train/full/compare

# 最佳模型文件位置
CNN_WEIGHTS_PATH = "/home/xusi/Logs/FinalTraining/Results_20251212_112634/Best_Models/Mscnn_K37_S2_E39_Aug_LR0p001_BS32_Best_F3.pth"
DATASET_PATH = "/home/xusi/EE5046_Projects/Dataset"
# JSON_METADATA
JSON_PATH = "/home/xusi/EE5046_Projects/Dataset/MMID/multimodal_instruction_data.json"


# ==================== 对比实验配置 ====================
COMPARISON_EXPERIMENT = True  # 是否进行对比实验
COMPARISON_MODE = 'augment'    # 'stream'对比stream配置, 'augment'对比数据增强

# Stream对比配置
STREAM_COMPARISON_CONFIGS = [
    {
        'name': 'Stream1_Only',
        'use_stream2': False,
        'description': '仅使用单尺度卷积流',
        'color': '#FF6B6B',  # 红色
        'line_style': '-'
    },
    {
        'name': 'Stream1+2',
        'use_stream2': True,
        'description': '使用双流多尺度卷积',
        'color': '#4ECDC4',  # 青色
        'line_style': '--'
    }
]

# 数据增强对比配置
AUGMENTATION_COMPARISON_CONFIGS = [
    {
        'name': 'No_Augmentation',
        'augment': False,
        'description': '无数据增强',
        'color': '#FF9F1C',  # 橙色
        'line_style': '-'
    },
    {
        'name': 'With_Augmentation',
        'augment': True,
        'description': '使用数据增强',
        'color': '#2A9D8F',  # 蓝绿色
        'line_style': '--'
    }
]

# 默认卷积核配置（用于对比实验）
DEFAULT_KERNEL_CONFIG = {
    'name': 'MS-CNN(3,7)', 
    'stream1_kernel': 3, 
    'stream2_first_kernel': 7
}