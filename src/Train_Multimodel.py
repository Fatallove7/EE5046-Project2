from datetime import datetime
import json
import os
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.amp import GradScaler, autocast

# 导入自定义模块
from MultimodelLLM import MultimodalLLM
from MultimodelDataset import MultimodalDataset
from ECGEncoder import ECGEncoder
from Config import CNN_WEIGHTS_PATH, DATASET_PATH, FIXED_LENGTH, JSON_PATH


# ============================================================================
# 配置常量
# ============================================================================
LLM_MODEL_NAME = "Qwen/Qwen-7B"
LLM_PATH = "/home/xusi/EE5046_Projects/LLM_Models/Qwen_Qwen-7B"
JSON_BEST_CONFIG = "/home/xusi/Logs/FinalTraining/Results_20251212_121215/final_results.json"

# 训练超参数
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
GRADIENT_CLIP_NORM = 1.0
LOG_INTERVAL = 100  # 日志记录间隔（步数）

# Qwen 特殊 token ID（硬编码）
QWEN_SPECIAL_TOKENS = {
    '<|endoftext|>': 151643,
    '<|im_start|>': 151644,
    '<|im_end|>': 151645,
    '<|extra_0|>': 151646,
    '<|extra_1|>': 151647,
    '<|extra_2|>': 151648
}


# ============================================================================
# 辅助函数
# ============================================================================
def load_best_cnn_config(json_path):
    """从训练结果 JSON 文件中读取最佳的 CNN 配置"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到训练结果 JSON 文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 提取 JSON 中的 kernel_config 部分
    kernel_config = results.get("best_config", {}).get("kernel_config", {})

    if not kernel_config:
        raise ValueError("JSON 文件中未找到 'best_config' 或 'kernel_config' 结构。")

    # 根据 Mscnn 的构造函数需求，构建 CNN_CONFIG 字典
    cnn_config = {
        'ch_in': 1,               # 默认值：单导联 ECG
        'ch_out': 1,              # 默认值：输出通道数
        'use_stream2': True,      # 默认值：只要 stream2_first_kernel 存在，通常为 True
        'stream1_kernel': kernel_config.get("stream1_kernel"),
        'stream2_first_kernel': kernel_config.get("stream2_first_kernel"),
    }
    
    # 检查关键参数是否成功读取
    if cnn_config['stream1_kernel'] is None or cnn_config['stream2_first_kernel'] is None:
        raise ValueError("无法从 JSON 中提取 stream1_kernel 或 stream2_first_kernel。")

    return cnn_config


def calculate_flat_dim(cnn_config, fixed_length, ECGEncoder_class, cnn_weights_path):
    """通过模拟前向传播来计算 ECGEncoder 的输出维度"""
    model = ECGEncoder_class(cnn_config, cnn_weights_path) 
    
    dummy_input = torch.randn(1, 1, fixed_length) 
    model.eval()
    with torch.no_grad():
        output_tensor = model(dummy_input)
    
    return output_tensor.size(-1)


def create_model_save_dir(base_path='../Trained_Multimodal_Models', experiment_name=None):
    """
    创建有意义的模型保存目录
    
    Args:
        base_path: 基础保存路径
        experiment_name: 实验名称，如果不提供则自动生成
    
    Returns:
        str: 新创建的模型保存目录的绝对路径
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_base_path = os.path.abspath(os.path.join(current_script_dir, base_path))
    
    # 确保基础目录存在
    os.makedirs(absolute_base_path, exist_ok=True)
    
    # 生成有意义的实验名称
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"Qwen7B_ECG_B{BATCH_SIZE}_LR{LEARNING_RATE}_E{EPOCHS}_{timestamp}"
    
    save_dir = os.path.join(absolute_base_path, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"模型将保存到: {save_dir}")
    return save_dir


def monitor_gpu_memory():
    """监控 GPU 内存使用情况"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return gpu_memory, gpu_memory_max
    return None, None


def multimodal_collate_fn(batch, tokenizer):
    """
    自定义的 collate_fn 用于多模态数据集
    处理变长序列的填充
    """
    # ECG 数据是固定长度的，直接堆叠
    ecg_data = torch.stack([item['ecg_data'] for item in batch])
    
    # 文本数据需要填充
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    # 找到批次中最长的序列长度
    max_len = max(len(ids) for ids in input_ids_list)
    
    # 初始化填充后的张量
    batch_size = len(batch)
    padded_input_ids = torch.full(
        (batch_size, max_len), 
        tokenizer.pad_token_id, 
        dtype=torch.long
    )
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    
    # 填充每个序列
    for i, (ids, mask, lbl) in enumerate(zip(input_ids_list, attention_mask_list, labels_list)):
        seq_len = len(ids)
        padded_input_ids[i, :seq_len] = ids
        padded_attention_mask[i, :seq_len] = mask
        padded_labels[i, :seq_len] = lbl
    
    return {
        'ecg_data': ecg_data,
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels
    }


def setup_qwen_tokenizer(llm_path):
    """专门为 Qwen 设置 tokenizer，确保 pad_token 正确配置"""
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, 
        trust_remote_code=True,
        padding_side='right'
    )
    
    print("原始 Qwen tokenizer 配置:")
    print(f"  eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  bos_token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"设置 pad_token 为 eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        else:
            # 使用 <|endoftext|> 作为默认 pad_token
            tokenizer.pad_token = '<|endoftext|>'
            tokenizer.pad_token_id = QWEN_SPECIAL_TOKENS['<|endoftext|>']
            print(f"设置 pad_token 为 '<|endoftext|>': (ID: {tokenizer.pad_token_id})")
    
    # 确保 pad_token_id 不为 None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = QWEN_SPECIAL_TOKENS['<|endoftext|>']
        print(f"强制设置 pad_token 为 '<|endoftext|>': (ID: {tokenizer.pad_token_id})")
    
    print(f"\n最终 tokenizer 配置:")
    print(f"  pad_token: {tokenizer.pad_token}")
    print(f"  pad_token_id: {tokenizer.pad_token_id}")
    
    return tokenizer


def validate_ecg_token(tokenizer, ecg_token="<|extra_0|>"):
    """验证 ECG token 是否存在"""
    ecg_token_id = tokenizer.convert_tokens_to_ids(ecg_token)
    
    print(f"ECG token 验证:")
    print(f"  ECG token: {ecg_token}")
    print(f"  ECG token ID: {ecg_token_id}")
    
    if ecg_token_id == tokenizer.unk_token_id:
        raise ValueError(f"ECG token {ecg_token} 不存在于 Qwen 词表中！")
    
    print(f"✅ ECG token {ecg_token} (ID: {ecg_token_id}) 验证通过。")
    
    return ecg_token_id


def log_model_parameters_to_tensorboard(model, writer, global_step):
    """记录模型参数到TensorBoard"""
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # 记录参数值分布
            writer.add_histogram(f'parameters/{name}', param.data.cpu().numpy(), global_step)
            # 记录梯度分布
            writer.add_histogram(f'gradients/{name}', param.grad.cpu().numpy(), global_step)
            
            # 记录参数范数
            writer.add_scalar(f'norm/parameters/{name}', param.norm().item(), global_step)
            writer.add_scalar(f'norm/gradients/{name}', param.grad.norm().item(), global_step)


def save_model_with_metadata(model, save_dir, epoch, loss, config=None, is_best=False):
    """
    保存模型及相关元数据
    
    Args:
        model: 要保存的模型
        save_dir: 保存目录
        epoch: 当前epoch
        loss: 当前损失
        config: 训练配置
        is_best: 是否是最佳模型
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    model.llm.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    torch.save(model.projector.state_dict(), os.path.join(save_dir, "projector.pth"))
    
    # 创建模型卡信息
    model_card = {
        "model_name": "ECG-Qwen-LLM",
        "model_type": "multimodal_language_model",
        "task": "ecg_classification",
        "framework": "pytorch",
        "base_model": "Qwen-7B",
        "fine_tuning_method": "lora",
        "ecg_encoder": "Mscnn",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_epoch": epoch,
        "loss": loss,
        "is_best_model": is_best,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_length": MAX_LEN,
            "warmup_ratio": WARMUP_RATIO
        }
    }
    
    if config:
        model_card["model_config"] = config
    
    # 保存模型卡
    with open(os.path.join(save_dir, "model_card.json"), "w", encoding='utf-8') as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)
    
    # 保存训练状态
    training_state = {
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(save_dir, "training_state.json"), "w", encoding='utf-8') as f:
        json.dump(training_state, f, indent=2)
    
    print(f"✅ 模型已保存到: {save_dir}")


def save_training_report(save_dir, best_loss, final_loss, total_epochs, training_time=None):
    """生成训练报告"""
    if training_time is None:
        training_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
    ========================================
    ECG-Qwen-LLM 训练报告
    ========================================
    
    训练完成时间: {training_time}
    
    训练统计:
    - 总epoch数: {total_epochs}
    - 最佳训练损失: {best_loss:.4f}
    - 最终训练损失: {final_loss:.4f}
    - 性能提升: {((final_loss - best_loss) / best_loss * 100):+.2f}% (最佳 vs 最终)
    
    模型信息:
    - 基础模型: Qwen-7B
    - 微调方法: LoRA
    - ECG编码器: Mscnn
    - 任务: ECG信号分类
    
    超参数:
    - 学习率: {LEARNING_RATE}
    - 批次大小: {BATCH_SIZE}
    - 最大序列长度: {MAX_LEN}
    - Warmup比例: {WARMUP_RATIO}
    - 梯度裁剪: {GRADIENT_CLIP_NORM}
    
    ========================================
    """
    
    report_path = os.path.join(save_dir, "training_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 训练报告已保存到: {report_path}")
    return report_path


# ============================================================================
# 训练函数（简化版本 - 只保存最佳模型）
# ============================================================================
def train_multimodal_model(cnn_config, json_path, data_dir, model_save_path, experiment_name=None):
    """
    多模态模型训练主函数
    - 每个epoch训练完成后，如果模型表现更好，则保存为最佳模型
    - 训练结束后，保存最终模型
    - 不保存中间检查点，只保留最佳和最终模型
    """
    # ------------------ 1. 初始化 ------------------
    start_time = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    # ------------------ 2. 创建TensorBoard写入器 ------------------
    tb_log_dir = os.path.join(model_save_path, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard日志目录: {tb_log_dir}")
    
    # ------------------ 3. Tokenizer 配置 ------------------
    tokenizer = setup_qwen_tokenizer(LLM_PATH)
    ECG_TOKEN = "<|extra_0|>"
    ecg_token_id = validate_ecg_token(tokenizer, ECG_TOKEN)
    
    # ------------------ 4. 数据准备 ------------------
    print("\n正在创建数据集和数据加载器...")
    
    # 创建数据集
    train_dataset = MultimodalDataset(
        json_path=json_path,
        data_dir=data_dir,
        tokenizer=tokenizer,
        ecg_token=ECG_TOKEN,
        max_len=MAX_LEN,
        is_train=True
    )
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: multimodal_collate_fn(batch, tokenizer)
    )
    
    print(f"数据集大小: {len(train_dataset)}")
    print(f"DataLoader 批次数量: {len(train_loader)}")
    
    # 测试批次数据
    test_batch = next(iter(train_loader))
    print(f"\n测试批次形状:")
    print(f"  ecg_data: {test_batch['ecg_data'].shape}")
    print(f"  input_ids: {test_batch['input_ids'].shape}")
    print(f"  attention_mask: {test_batch['attention_mask'].shape}")
    print(f"  labels: {test_batch['labels'].shape}")
    
    # ------------------ 5. 模型初始化 ------------------
    print("\n正在初始化模型...")
    
    # 计算 FLAT_DIM 和 LLM_EMBEDDING_DIM
    FLAT_DIM = calculate_flat_dim(
        cnn_config, 
        FIXED_LENGTH, 
        ECGEncoder_class=ECGEncoder,
        cnn_weights_path=CNN_WEIGHTS_PATH
    )
    LLM_EMBEDDING_DIM = 4096  # Qwen-7B 的嵌入维度
    
    model = MultimodalLLM(
        llm_path=LLM_PATH,
        cnn_config=cnn_config,
        cnn_weights_path=CNN_WEIGHTS_PATH,
        ecg_token_id=ecg_token_id,
        flat_dim=FLAT_DIM,
        llm_embed_dim=LLM_EMBEDDING_DIM,
        device=device
    )
    
    # 记录超参数到TensorBoard
    hparams = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "max_len": MAX_LEN,
        "warmup_ratio": WARMUP_RATIO,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "flat_dim": FLAT_DIM,
        "llm_embed_dim": LLM_EMBEDDING_DIM,
        "cnn_stream1_kernel": cnn_config.get('stream1_kernel'),
        "cnn_stream2_first_kernel": cnn_config.get('stream2_first_kernel'),
    }
    writer.add_hparams(hparams, {})
    
    # ------------------ 6. 优化器和调度器 ------------------
    print("\n正在配置优化器和学习率调度器...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"总训练步数: {total_steps}")
    print(f"Warmup 步数: {warmup_steps}")
    print(f"初始学习率: {LEARNING_RATE}")
    
    # ------------------ 7. 禁用混合精度训练 ------------------
    # 由于 Qwen 模型有数据类型问题，暂时禁用混合精度
    print("混合精度训练已禁用")
    scaler = None
    
    # 训练统计
    global_step = 0
    best_loss = float('inf')
    best_epoch = 0
    
    # ------------------ 8. 训练循环 ------------------
    print("\n" + "="*50)
    print("开始多模态指令微调...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_steps = 0
        
        # 创建进度条
        progress_bar = tqdm.tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
        
        for step, batch in progress_bar:
            # 准备数据
            ecg_data = batch['ecg_data'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                ecg_data=ecg_data,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新学习率
            scheduler.step()
            
            # 更新统计
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            avg_loss = epoch_loss / epoch_steps
            current_lr = scheduler.get_last_lr()[0]
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{avg_loss:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # 记录到TensorBoard
            if global_step % LOG_INTERVAL == 0:
                # 记录损失和学习率
                writer.add_scalar('train/loss_step', loss.item(), global_step)
                writer.add_scalar('train/loss_avg', avg_loss, global_step)
                writer.add_scalar('train/learning_rate', current_lr, global_step)
                
                # 记录梯度范数
                total_grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_grad_norm = p.grad.data.norm(2).item()
                        total_grad_norm += param_grad_norm ** 2
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar('train/grad_norm', total_grad_norm, global_step)
            
            # 监控 GPU 内存
            if step % 100 == 0 and device.type == 'cuda':
                gpu_memory, gpu_memory_max = monitor_gpu_memory()
                if gpu_memory:
                    # 记录GPU内存使用
                    writer.add_scalar('system/gpu_memory', gpu_memory, global_step)
                    writer.add_scalar('system/gpu_memory_max', gpu_memory_max, global_step)
        
        # ------------------ 9. 每个epoch结束 ------------------
        epoch_avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} 结束, 平均训练损失: {epoch_avg_loss:.4f}")
        
        # 记录epoch级别的指标
        writer.add_scalar('train/loss_epoch', epoch_avg_loss, epoch+1)
        writer.add_scalar('train/learning_rate_epoch', scheduler.get_last_lr()[0], epoch+1)
        
        # 检查是否是最佳模型
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_epoch = epoch + 1
            print(f"🎉 发现新的最佳模型! Epoch: {best_epoch}, 损失: {best_loss:.4f}")
            
            # 保存最佳模型
            best_model_dir = os.path.join(model_save_path, "best_model")
            save_model_with_metadata(
                model=model,
                save_dir=best_model_dir,
                epoch=best_epoch,
                loss=best_loss,
                config={
                    "cnn_config": cnn_config,
                    "flat_dim": FLAT_DIM,
                    "llm_embed_dim": LLM_EMBEDDING_DIM
                },
                is_best=True
            )
    
    # ------------------ 10. 训练完成 ------------------
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "="*50)
    print("训练完成!")
    print(f"训练时长: {training_duration}")
    print(f"最佳模型: Epoch {best_epoch}, 损失: {best_loss:.4f}")
    print(f"最终模型: Epoch {EPOCHS}, 损失: {epoch_avg_loss:.4f}")
    print("="*50)
    
    # 保存最终模型
    final_model_dir = os.path.join(model_save_path, "final_model")
    save_model_with_metadata(
        model=model,
        save_dir=final_model_dir,
        epoch=EPOCHS,
        loss=epoch_avg_loss,
        config={
            "cnn_config": cnn_config,
            "flat_dim": FLAT_DIM,
            "llm_embed_dim": LLM_EMBEDDING_DIM,
            "final_epoch": EPOCHS,
            "final_loss": epoch_avg_loss
        },
        is_best=False
    )
    
    # 生成训练报告
    save_training_report(
        save_dir=model_save_path,
        best_loss=best_loss,
        final_loss=epoch_avg_loss,
        total_epochs=EPOCHS,
        training_time=end_time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f"\n📁 训练结果保存在: {model_save_path}")
    print(f"🏆 最佳模型: {os.path.join(model_save_path, 'best_model')}")
    print(f"✅ 最终模型: {os.path.join(model_save_path, 'final_model')}")
    
    # 提示如何启动TensorBoard
    print(f"\n要启动TensorBoard查看训练曲线，请运行:")
    print(f"tensorboard --logdir={tb_log_dir} --port=6006")
    print("然后在浏览器中打开: http://localhost:6006")
    
    return {
        "best_loss": best_loss,
        "final_loss": epoch_avg_loss,
        "best_epoch": best_epoch,
        "training_duration": str(training_duration),
        "model_save_path": model_save_path,
        "best_model_path": os.path.join(model_save_path, "best_model"),
        "final_model_path": os.path.join(model_save_path, "final_model")
    }


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序入口"""
    print("="*50)
    print("多模态 ECG-LLM 模型训练（优化版）")
    print("="*50)
    
    # 1. 创建模型保存目录（使用有意义的实验名称）
    EXPERIMENT_NAME = f"Qwen7B_ECG_B{BATCH_SIZE}_LR{LEARNING_RATE}_E{EPOCHS}"
    MODEL_SAVE_DIR = create_model_save_dir(
        base_path='../Trained_Multimodal_Models',
        experiment_name=EXPERIMENT_NAME
    )
    
    # 2. 加载最佳 CNN 配置
    print("\n正在加载 CNN 配置...")
    try:
        CNN_CONFIG = load_best_cnn_config(JSON_BEST_CONFIG)
        print(f"✅ 成功从 JSON 文件加载最佳 CNN 配置")
        print(f"  配置内容: {CNN_CONFIG}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return
    
    # 3. 开始训练
    print("\n开始训练过程...")
    training_results = train_multimodal_model(
        cnn_config=CNN_CONFIG,
        json_path=JSON_PATH,
        data_dir=DATASET_PATH,
        model_save_path=MODEL_SAVE_DIR
    )
    
    # 4. 打印最终结果
    print("\n" + "="*50)
    print("训练总结:")
    print(f"  实验名称: {EXPERIMENT_NAME}")
    print(f"  最佳Epoch: {training_results['best_epoch']}")
    print(f"  最佳损失: {training_results['best_loss']:.4f}")
    print(f"  最终损失: {training_results['final_loss']:.4f}")
    print(f"  训练时长: {training_results['training_duration']}")
    print(f"  保存路径: {training_results['model_save_path']}")
    print("="*50)
    
    print("\n✅ 训练脚本执行完成！")


if __name__ == '__main__':
    main()