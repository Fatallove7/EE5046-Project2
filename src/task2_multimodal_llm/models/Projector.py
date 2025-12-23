from torch import nn



# ------------------------------------------------------------------
# II. Projector 投影层 (满足需求 2)
# ------------------------------------------------------------------
class Projector(nn.Module):
    """
    将 CNN 特征向量 (Z_v) 映射到 LLM 嵌入空间 (H_v)。
    使用一个简单的两层 MLP。
    """

    def __init__(self, flat_dim, llm_embed_dim):
        super(Projector, self).__init__()

        # 中间维度可以设置为 LLM 嵌入维度的 2 倍，或与 flat_dim 接近
        mid_dim = llm_embed_dim * 2

        self.fc = nn.Sequential(
            # 第一层：从 CNN 特征维度 映射到 中间维度
            nn.Linear(flat_dim, mid_dim),
            nn.GELU(),  # 使用 GELU 激活函数，比 ReLU 更稳定
            # 第二层：从中维 映射到 LLM 嵌入维度
            nn.Linear(mid_dim, llm_embed_dim)
        )

    def forward(self, x):
        # x 是 Z_v (ECG 特征向量)
        return self.fc(x)  # 返回 H_v (ECG 嵌入向量)