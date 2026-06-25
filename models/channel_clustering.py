import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelClustering(nn.Module):
    """
    修复版通道聚类：
    1. 分离相关系数中心和能量比中心，避免尺度冲突
    2. 相关系数用tanh约束[-1,1]
    3. 能量比用softmax约束（和为1）
    4. 距离融合时自适应加权
    """
    def __init__(self, n_clusters=4, center_init='physical', learnable=True):
        super().__init__()
        self.n_clusters = n_clusters
        self.learnable = learnable

        # 分离两种特征的中心
        # 相关系数中心（3维）：预设物理模式，tanh约束
        physical_corr = torch.tensor([
            [0.5, 0.4, 0.4],   # 正常耦合
            [0.8, 0.2, 0.2],   # 不对中
            [0.6, 0.6, 0.6],   # 切屑夹持
            [0.2, -0.2, 0.7],  # 卡盘/冲击
        ])

        if center_init == 'physical':
            corr_init = physical_corr.clone()
        elif center_init == 'random':
            corr_init = torch.rand(n_clusters, 3) * 0.1  # 小范围随机初始化
        elif center_init == 'kmeans':
            # 数据驱动：后面会单独用 K‑means 初始化，这里先占位
            corr_init = torch.rand(n_clusters, 3)
        else:
            corr_init = physical_corr.clone()  # 默认使用物理预设
        
        if self.learnable:
            self.corr_centers = nn.Parameter(corr_init)
        else:
            self.register_buffer('corr_centers', corr_init)

        # 能量比中心（3维）：初始化均匀，softmax约束
        energy_init = torch.ones(n_clusters, 3) / 3.0
        self.energy_centers = nn.Parameter(energy_init)

        # 可学习的融合权重（让网络自己决定哪种特征更重要）
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]))

    def extract_channel_features(self, x):
        """
        提取多维通道特征
        x: (batch, time, channels)
        返回: (batch, 6) [r_xy, r_xz, r_yz, e_ratio_x, e_ratio_y, e_ratio_z]
        """
        batch_size, seq_len, channels = x.shape

        features = []

        # 1. 皮尔逊相关系数 (3维)
        corr_matrix = torch.zeros(batch_size, channels, channels, device=x.device)
        for i in range(batch_size):
            # 数值稳定性：减去均值，避免数值溢出
            x_centered = x[i] - x[i].mean(dim=0, keepdim=True)
            cov = x_centered.T @ x_centered / (seq_len - 1 + 1e-8)
            std = torch.sqrt(torch.diag(cov) + 1e-8)
            corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
            corr_matrix[i] = torch.clamp(corr, -1, 1)  # 截断异常值

        r_xy = corr_matrix[:, 0, 1]
        r_xz = corr_matrix[:, 0, 2]
        r_yz = corr_matrix[:, 1, 2]
        features.extend([r_xy, r_xz, r_yz])

        # 2. 能量比 (3维)
        energy = torch.sum(x ** 2, dim=1)  # (batch, channels)
        total_energy = energy.sum(dim=1, keepdim=True) + 1e-8
        energy_ratio = energy / total_energy
        features.extend([energy_ratio[:, 0], energy_ratio[:, 1], energy_ratio[:, 2]])

        return torch.stack(features, dim=1)  # (batch, 6)

    def initialize_with_kmeans(self, features, n_iter=10):
        """features: (num_samples, 6) 通道特征"""
        if not self.learnable:
            return
        from sklearn.cluster import KMeans
        corr_feat = features[:, :3].numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10).fit(corr_feat)
        with torch.no_grad():
            self.corr_centers.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    def forward(self, x):
        """
        x: (batch, time, channels)
        返回:
            w_channel: (batch, n_clusters) 通道软分配权重
            r_vec: (batch, 6) 通道特征向量
        """
        batch_size = x.size(0)

        # 提取多维通道特征
        r_vec = self.extract_channel_features(x)  # (batch, 6)
        corr_feat = r_vec[:, :3]  # 相关系数 (batch, 3)
        energy_feat = r_vec[:, 3:]  # 能量比 (batch, 3)

        # 约束中心
        corr_centers_constrained = torch.tanh(self.corr_centers)  # [-1, 1]
        energy_centers_constrained = F.softmax(self.energy_centers, dim=1)  # [0, 1], sum=1

        # 分别计算距离
        corr_dist = torch.cdist(corr_feat, corr_centers_constrained, p=2)
        energy_dist = torch.cdist(energy_feat, energy_centers_constrained, p=2)

        # 自适应融合权重（softmax归一化确保和为1）
        fusion_w = F.softmax(self.fusion_weight, dim=0)
        dist = fusion_w[0] * corr_dist + fusion_w[1] * energy_dist

        # 软分配权重
        w_channel = F.softmax(-dist, dim=1)

        return w_channel, r_vec