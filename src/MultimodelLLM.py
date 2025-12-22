# MultimodelLLM.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from ECGEncoder import ECGEncoder

class MultimodalLLM(nn.Module):
    def __init__(self, llm_path, cnn_config, cnn_weights_path, ecg_token_id, flat_dim, llm_embed_dim, device=None):
        super(MultimodalLLM, self).__init__()
        
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"MultimodalLLM 将在设备上运行: {self.device}")
        
        # 1. ECG 编码器 - 使用 GPU 设备
        self.ecg_encoder = ECGEncoder(cnn_config, cnn_weights_path, device=self.device)
        
        # 2. 投影层 (ECG 特征 -> LLM 嵌入空间)
        self.projector = nn.Linear(flat_dim, llm_embed_dim)
        
        # 3. LLM 模型
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False
        )
        
        # 4. ECG Token ID
        self.ecg_token_id = ecg_token_id
        
        # 5. LoRA 配置 - 使用正确的模块名称
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "w1", "w2"],
            bias="none",
            modules_to_save=["embed_tokens", "lm_head"]  # 保存这些层的完整参数
        )
        
        # 应用 LoRA
        self.llm = get_peft_model(self.llm, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.llm.parameters())
        print(f"LoRA 可训练参数: {trainable_params:,} ({(trainable_params/total_params*100):.2f}% of total)")
        
        # 冻结 ECG 编码器（已经在 ECGEncoder 中完成）
        # 这里再次确认 ECG 编码器被冻结
        for param in self.ecg_encoder.parameters():
            param.requires_grad = False
        
        # 将 projector 移动到设备
        self.projector = self.projector.to(self.device)

    
        
    def forward(self, ecg_data, input_ids, attention_mask, labels=None):
        # 确保输入数据在正确的设备上
        ecg_data = ecg_data.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        labels = labels.to(self.device) if labels is not None else None
        
        # ECG 编码
        ecg_features = self.ecg_encoder(ecg_data)  # [batch, flat_dim]
        
        # 投影到 LLM 嵌入空间
        ecg_embeddings = self.projector(ecg_features)  # [batch, llm_embed_dim]
        
        # 获取 LLM 的输入嵌入
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # [batch, seq_len, llm_embed_dim]
        
        # 替换 ECG Token 的嵌入
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # 找到 ECG Token 的位置
            ecg_positions = (input_ids[i] == self.ecg_token_id).nonzero(as_tuple=True)[0]
            for pos in ecg_positions:
                input_embeds[i, pos] = ecg_embeddings[i]
        
        # 前向传播
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs