import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalClustering(nn.Module):
    """
    时间聚类：
    1. centers保持为buffer，不参与梯度优化
    2. 增加死亡簇检测与重生机制
    3. EMA更新增加最小样本保护
    4. 增加簇间分离性正则，防止坍缩
    """
    def __init__(self, channels=3, window_len=256, time_hidden_dim=128, n_clusters=4, ema_decay=0.95):
        super().__init__()
        self.channels = channels
        self.window_len = window_len
        self.n_clusters = n_clusters
        self.time_hidden_dim = time_hidden_dim
        self.ema_decay = ema_decay

        # 1D-CNN特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, time_hidden_dim)

        # 关键：centers必须是buffer，不可学习
        self.register_buffer('centers', torch.randn(n_clusters, time_hidden_dim))
        # 初始化：使用K-means++风格，确保初始分离
        self._init_centers()
        self.update_centers = True   # 控制是否在线更新

        # 记录分配信息
        self.last_hard_assign = None
        self.last_entropy = None
        self.dead_cluster_count = 0  # 死亡簇计数器

    def _init_centers(self):
        """K-means++风格初始化：确保初始中心相互远离"""
        with torch.no_grad():
            # 随机初始化后正交化
            centers = torch.randn(self.n_clusters, self.time_hidden_dim)
            # Gram-Schmidt正交化近似
            for i in range(1, self.n_clusters):
                for j in range(i):
                    proj = (centers[i] @ centers[j]) / (centers[j] @ centers[j] + 1e-8)
                    centers[i] -= proj * centers[j]
                centers[i] = F.normalize(centers[i], dim=0)
            # 缩放至合理范围
            centers *= 0.5
            self.centers.copy_(centers)

    def forward(self, x):
        """
        x: (batch, channels, time)
        返回:
            h_time: (batch, time_hidden_dim) 时间隐特征
            w: (batch, n_clusters) 软分配权重
            hard_assign: (batch,) 硬分配标签
        """
        batch_size = x.size(0)

        # 1D-CNN特征提取
        h = self.feature_extractor(x)  # (batch, 64, 1)
        h = h.squeeze(-1)  # (batch, 64)
        h_time = self.fc(h)  # (batch, time_hidden_dim)

        # 计算到各簇中心的距离
        dist = torch.cdist(h_time, self.centers)  # (batch, n_clusters)
        hard_assign = torch.argmin(dist, dim=1)

        # 软分配
        w = F.softmax(-dist, dim=1)

        # 关键：EMA更新 + 死亡簇重生
        if self.update_centers:
            with torch.no_grad():
                # 统计各簇样本数
                cluster_counts = torch.bincount(hard_assign, minlength=self.n_clusters)
                
                for k in range(self.n_clusters):
                    mask = (hard_assign == k)
                    
                    if mask.sum() >= 5:  # 最小样本保护：至少5个样本才更新
                        center_new = h_time[mask].mean(dim=0)
                        # EMA更新
                        self.centers[k] = self.ema_decay * self.centers[k] + (1 - self.ema_decay) * center_new
                    elif cluster_counts[k] == 0:
                        # 死亡簇重生：选择远离所有中心的样本
                        dist_to_all = torch.cdist(h_time, self.centers).sum(dim=1)
                        farthest_idx = torch.argmax(dist_to_all)
                        self.centers[k] = h_time[farthest_idx].clone()
                        self.dead_cluster_count += 1

        # 记录统计信息
        self.last_hard_assign = hard_assign
        self.last_entropy = -torch.sum(w * torch.log(w + 1e-8), dim=1).mean()

        return h_time, w, hard_assign

    def get_cluster_loss(self, h_time, w):
        """
        修复版聚类正则：
        1. 强簇内紧凑性
        2. 强簇间分离性（防止坍缩）
        3. 空簇惩罚
        """
        hard_assign = torch.argmax(w, dim=1)
        
        # 收集有效中心
        centers = []
        valid_mask = []
        for k in range(self.n_clusters):
            mask = (hard_assign == k)
            if mask.sum() > 1:
                centers.append(h_time[mask].mean(dim=0))
                valid_mask.append(True)
            else:
                centers.append(self.centers[k].clone())
                valid_mask.append(False)
        centers = torch.stack(centers)

        # 簇内紧凑性（仅对有效簇计算）
        intra_loss = 0.0
        valid_count = 0
        for k in range(self.n_clusters):
            if not valid_mask[k]:
                continue
            mask = (hard_assign == k)
            intra_loss += ((h_time[mask] - centers[k].unsqueeze(0)) ** 2).mean()
            valid_count += 1
        intra_loss = intra_loss / max(valid_count, 1)

        # 簇间分离性：强制所有中心对之间距离>阈值
        inter_loss = 0.0
        n_pairs = 0
        min_dist_threshold = 2.0  # 最小距离阈值
        
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                dist = torch.norm(centers[i] - centers[j], p=2)
                # 如果距离太小，施加惩罚
                if dist < min_dist_threshold:
                    inter_loss += (min_dist_threshold - dist) ** 2
                n_pairs += 1
        inter_loss = inter_loss / max(n_pairs, 1)

        # 空簇惩罚
        empty_count = sum(1 for v in valid_mask if not v)
        empty_penalty = empty_count / self.n_clusters

        # 总损失：紧凑性 + 分离性 + 空簇惩罚
        # 分离性权重调高至1.0，强制中心分散
        loss = intra_loss + 1.0 * inter_loss + 0.1 * empty_penalty
        return loss

    def get_cluster_statistics(self):
        """获取聚类统计信息"""
        if self.last_hard_assign is None:
            return None
        unique, counts = torch.unique(self.last_hard_assign, return_counts=True)
        dist = {int(k.item()): int(c.item()) for k, c in zip(unique, counts)}
        entropy = self.last_entropy.item() if self.last_entropy is not None else 0.0
        return {
            'distribution': dist, 
            'entropy': entropy, 
            'num_nonempty': len(unique),
            'dead_cluster_count': self.dead_cluster_count
        }