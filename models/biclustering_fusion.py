import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块：用于构建深层分类器"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_out)
        self.norm1 = nn.LayerNorm(dim_out)
        self.fc2 = nn.Linear(dim_out, dim_out)
        self.norm2 = nn.LayerNorm(dim_out)
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        return F.relu(x + residual)


class BiclusteringFusion(nn.Module):
    """
    改进版双聚类融合模块：
    1. 残差连接防止梯度消失
    2. 自注意力机制学习时间-通道特征交互
    3. LayerNorm + Dropout正则化
    """
    def __init__(self, time_feat_dim=128, n_time_clusters=4, n_channel_clusters=4):
        super().__init__()
        self.n_time_clusters = n_time_clusters
        self.n_channel_clusters = n_channel_clusters

        fusion_dim = time_feat_dim + n_channel_clusters

        # 特征投影层
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 残差块 × 2
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        ])

        # 自注意力：学习时间-通道特征的交互关系
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)

        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, h_time, w_channel):
        """
        h_time: (batch, time_hidden_dim) 时间隐特征
        w_channel: (batch, n_channel_clusters) 通道软分配权重
        返回: logits (batch,)
        """
        # 基础融合：拼接时间特征与通道特征
        combined = torch.cat([h_time, w_channel], dim=1)
        x = self.proj(combined)  # (batch, 128)

        # 残差块处理
        for block in self.res_blocks:
            x = block(x)

        # 自注意力（增加序列维度后计算）
        x_attn = x.unsqueeze(1)  # (batch, 1, 128)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x_attn.squeeze(1) + x  # 残差连接

        # 分类输出
        logits = self.classifier(x).squeeze(-1)
        return logits