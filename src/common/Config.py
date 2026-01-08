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
EXPERIMENT_MODE = "full"  # search/train/full/compare

# 最佳模型文件位置
CNN_WEIGHTS_PATH = "/home/xusi/Logs/FinalTraining/Results_20251212_112634/Best_Models/Mscnn_K37_S2_E39_Aug_LR0p001_BS32_Best_F3.pth"
DATASET_PATH = "/home/xusi/EE5046_Projects/Dataset"
# JSON_METADATA
JSON_PATH = "/home/xusi/EE5046_Projects/Dataset/MMID/multimodal_instruction_data.json"


# 学习率调度器配置
LR_SCHEDULER_CONFIG = {
    'use_scheduler': True,           # 是否使用学习率调度器
    'scheduler_type': 'plateau',     # 'plateau', 'cosine', 'onecycle', 'step'
    'plateau_config': {
        'mode': 'max',               # 监控指标方向：max/min
        'factor': 0.5,               # 学习率衰减因子
        'patience': 5,               # 容忍epoch数
        'min_lr': 1e-6,              # 最小学习率
        'verbose': True              # 是否打印调整信息
    },
    'cosine_config': {
        'T_max': NUM_EPOCHS,         # 半周期长度
        'eta_min': 1e-6              # 最小学习率
    },
    'onecycle_config': {
        'max_lr': LEARNING_RATE,     # 最大学习率
        'pct_start': 0.3,            # 上升阶段占比
        'div_factor': 25.0,          # 初始学习率 = max_lr/div_factor
        'final_div_factor': 1e4      # 最终学习率 = initial_lr/final_div_factor
    },
    'step_config': {
        'step_size': 10,             # 多少epoch衰减一次
        'gamma': 0.5                 # 衰减因子
    }
}

# ==================== 对比实验配置 ====================
COMPARISON_EXPERIMENT = False  # 是否进行对比实验
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

LOSS_FUNCTION_CONFIG = {
    'type': 'focal',  # 'bce' 或 'focal'
    'focal_alpha': 0.25,  # Focal Loss的alpha参数，None则自动计算
    'focal_gamma': 2.0,   # Focal Loss的gamma参数
    'focal_config': 'focus_positive',  # 预设配置: 'default', 'balanced', 'focus_positive', 'focus_hard'
}

# Focal Loss预设配置
FOCAL_PRESET_CONFIGS = {
    'default': {'alpha': 0.25, 'gamma': 2.0, 'lr_factor': 0.5},
    'balanced': {'alpha': 0.5, 'gamma': 2.0, 'lr_factor': 0.5},
    'focus_positive': {'alpha': 0.85, 'gamma': 3.0, 'lr_factor': 0.1},
    'focus_hard': {'alpha': 0.25, 'gamma': 3.0, 'lr_factor': 0.3},
}

# 是否启用Focal Loss（兼容旧配置）
USE_FOCAL_LOSS = (LOSS_FUNCTION_CONFIG['type'] == 'focal')

def get_loss_config(focal_config_name=None):
    """获取损失函数配置"""
    config = LOSS_FUNCTION_CONFIG.copy()
    
    if focal_config_name and focal_config_name in FOCAL_PRESET_CONFIGS:
        preset = FOCAL_PRESET_CONFIGS[focal_config_name]
        config.update(preset)
    
    # 计算调整后的学习率
    if config['type'] == 'focal':
        config['adjusted_lr'] = LEARNING_RATE * config.get('lr_factor', 0.5)
    else:
        config['adjusted_lr'] = LEARNING_RATE
        
    return config