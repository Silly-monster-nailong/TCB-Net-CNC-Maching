import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入改进后的模块
from models.temporal_clustering import TemporalClustering
from models.channel_clustering import ChannelClustering
from models.biclustering_fusion import BiclusteringFusion

class TCB_Net(nn.Module):
    """
    改进版时间-通道双聚类网络 (TCB-Net)

    改进点：
    1. 时间聚类：1D-CNN替代FC提取时序特征
    2. 通道聚类：可学习参数中心 + 6维特征(相关系数+能量比)
    3. 融合模块：残差连接 + 自注意力 + LayerNorm
    4. 支持FocalLoss和动态阈值策略
    """
    def __init__(self, config):
        super().__init__()
        # 从配置读取参数
        model_cfg = config.model if hasattr(config, 'model') else config

        self.time_hidden_dim = getattr(model_cfg, 'time_hidden_dim', 128)
        self.n_time_clusters = getattr(model_cfg, 'n_time_clusters', 4)
        self.n_channel_clusters = getattr(model_cfg, 'n_channel_clusters', 4)

        print(f"[TCB-Net] 配置: n_time_clusters={self.n_time_clusters}, "
              f"time_hidden_dim={self.time_hidden_dim}, n_channel_clusters={self.n_channel_clusters}")

        self.use_time_clustering = getattr(config.model, 'use_time_clustering', True)
        self.use_channel_clustering = getattr(config.model, 'use_channel_clustering', True)
        self.use_cluster_loss = getattr(config.training, 'use_cluster_loss', True)
        self.time_only = getattr(config.model, 'time_only', False)   # 可选

        # 时间聚类模块（改进：1D-CNN特征提取）
        self.temporal = TemporalClustering(
            channels=getattr(model_cfg, 'channels', 3),
            window_len=getattr(model_cfg, 'window_len', 256),
            time_hidden_dim=self.time_hidden_dim,
            n_clusters=self.n_time_clusters,
            ema_decay=getattr(model_cfg, 'ema_decay', 0.95)
        )

        # 通道聚类模块（改进：可学习中心 + 丰富特征）
        channel_center_init = getattr(model_cfg, 'channel_center_init', 'physical')  # 'physical', 'random', 'kmeans'
        channel_center_learnable = getattr(model_cfg, 'channel_center_learnable', True)   
        self.channel = ChannelClustering(
            n_clusters=self.n_channel_clusters, 
            center_init=channel_center_init, 
            learnable=channel_center_learnable
        )

        # 双聚类融合模块（改进：残差+注意力）
        self.fusion = BiclusteringFusion(
            time_feat_dim=self.time_hidden_dim,
            n_time_clusters=self.n_time_clusters,
            n_channel_clusters=self.n_channel_clusters
        )

        self.config = config
        self.cached_h_time = None
        self.cached_w_time = None

    def forward(self, x):
        """
        x: (batch, time, channels) 或 (batch, channels, time)
        返回: logits (batch,)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")

        # 统一转换为 (batch, channels, time) 供时间聚类使用
        if x.size(-1) == 3:
            x_time = x.permute(0, 2, 1)  # (batch, time, channels) -> (batch, channels, time)
            x_tc = x  # 通道聚类保持 (batch, time, channels)
        elif x.size(1) == 3:
            x_time = x  # 已经是 (batch, channels, time)
            x_tc = x.permute(0, 2, 1)  # -> (batch, time, channels)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # 时间聚模块
        if self.use_time_clustering:
            h_time, w_time, hard_assign = self.temporal(x_time)
        else:
            # 跳过聚类，仅取特征提取器的输出（CNN + FC）
            h = self.temporal.feature_extractor(x_time).squeeze(-1)   # (batch, 64)
            h_time = self.temporal.fc(h)                              # (batch, time_hidden_dim)
            # 软分配权重设为 None 或全零（取决于后续是否需要）
            w_time = None
            hard_assign = None

        self.cached_h_time = h_time
        self.cached_w_time = w_time

        # 通道聚类模块
        if self.use_channel_clustering:
            w_channel, r_vec = self.channel(x_tc)
        else:
            # 不使用通道聚类时，用零向量代替软分配权重
            w_channel = torch.zeros(x_tc.size(0), self.n_channel_clusters, device=x_tc.device)
            r_vec = None

        # 双聚类融合与分类
        if self.time_only:
            w_channel = torch.zeros_like(w_channel)  # 强制置零

        logits = self.fusion(h_time, w_channel)
        return logits

    def compute_loss(self, logits, labels, return_components=False, focal_criterion=None):
        """
        计算总损失 = 分类损失 + 聚类正则项
        """
        # 分类损失（支持FocalLoss或标准BCE）
        if focal_criterion is not None:
            cls_loss = focal_criterion(logits, labels)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        # 聚类正则损失
        cluster_loss = torch.tensor(0.0, device=logits.device)
        lambda_cluster = getattr(self.config.training, 'lambda_cluster', 0.01)
        if self.use_cluster_loss and lambda_cluster > 0 and self.cached_h_time is not None and self.cached_w_time is not None:
            cluster_loss = self.temporal.get_cluster_loss(self.cached_h_time, self.cached_w_time)

        total_loss = cls_loss + lambda_cluster * cluster_loss

        if return_components:
            return total_loss, {'cls': cls_loss.item(), 'cluster': cluster_loss.item()}
        return total_loss

    def get_cluster_statistics(self):
        """获取时间聚类统计信息"""
        return self.temporal.get_cluster_statistics()

    def get_channel_weights(self, x):
        """获取通道聚类软分配权重（用于可解释性分析）"""
        if x.size(-1) == 3:
            x_tc = x
        else:
            x_tc = x.permute(0, 2, 1)
        w_channel, r_vec = self.channel(x_tc)
        return w_channel, r_vec

    def get_time_assignments(self, x):
        """获取时间聚类硬分配标签（用于可解释性分析）"""
        if x.size(-1) == 3:
            x_time = x.permute(0, 2, 1)
        else:
            x_time = x
        _, _, hard_assign = self.temporal(x_time)
        return hard_assign