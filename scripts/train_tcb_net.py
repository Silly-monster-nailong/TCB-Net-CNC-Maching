#!/usr/bin/env python
"""
TCB-Net 训练脚本
- 统一使用 FocalLoss
- 统一使用动态阈值（基于训练集正常样本分布，启用验证集微调）
- 早停机制：基于验证集 AUC，耐心值 15
- 训练细节：AdamW 优化器，ReduceLROnPlateau 学习率调度，早停机制
- 评估指标：准确率、精确率、召回率、F1 分数、AUC，以及混淆矩阵
- 结果保存：输出目录下的 results.json 和 tcb_net_model.pth
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import json
import random
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, accuracy_score,confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import Config
from models.tcb_net import TCB_Net


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    FL(pt) = -α_t(1-pt)^γ log(pt)
    """
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        # 标签平滑
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)

        # 正样本损失
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs + 1e-8)
        # 负样本损失
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs + 1e-8)

        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

def get_threshold_from_train_normal(model, train_loader, device, 
                                    val_loader=None, 
                                    percentile=99, 
                                    fine_tune_range=0.2):
        """基于训练集正常样本的分位数获取阈值，可选验证集微调"""
        model.eval()
        normal_probs = []
        with torch.no_grad():
            for X, y in train_loader:
                # 只保留正常样本 (y == 0)
                normal_mask = (y == 0)
                if not normal_mask.any():
                    continue
                X_normal = X[normal_mask].to(device)
                logits = model(X_normal)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                normal_probs.extend(probs)
        base_thr = np.percentile(normal_probs, percentile)
        print(f"[阈值] 训练集正常样本 {percentile}% 分位数 = {base_thr:.4f}")

        if val_loader is not None:
            val_probs, val_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    logits = model(X)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_probs.extend(probs)
                    val_labels.extend(y.numpy())
            val_probs = np.array(val_probs)
            val_labels = np.array(val_labels)
            pos_cnt = (val_labels == 1).sum()
            if pos_cnt >= 10:
                thresholds = np.linspace(base_thr * (1 - fine_tune_range),
                                        base_thr * (1 + fine_tune_range), 21)
                best_f1 = 0
                best_thr = base_thr
                for thr in thresholds:
                    preds = (val_probs >= thr).astype(int)
                    f1 = f1_score(val_labels, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = thr
                print(f"[阈值] 验证集微调后 = {best_thr:.4f} (F1={best_f1:.4f})")
                return best_thr
            else:
                print(f"[阈值] 验证集正样本不足 ({pos_cnt} < 10)，不微调")
        return base_thr

# ========== 数据增强 ==========
class SmallSampleAugmentation:
    @staticmethod
    def time_warp(x, sigma=0.2):
        batch, seq_len, channels = x.shape
        if seq_len < 10:
            return x
        scale = 1.0 + torch.randn(1).item() * sigma
        new_len = int(seq_len * scale)
        new_len = max(min(new_len, seq_len + 20), seq_len - 20)
        x_np = x.cpu().numpy()
        x_warped = np.zeros((batch, new_len, channels))
        for b in range(batch):
            for c in range(channels):
                x_warped[b, :, c] = np.interp(
                    np.linspace(0, seq_len - 1, new_len),
                    np.arange(seq_len),
                    x_np[b, :, c]
                )
        x_resampled = np.zeros((batch, seq_len, channels))
        for b in range(batch):
            for c in range(channels):
                x_resampled[b, :, c] = np.interp(
                    np.arange(seq_len),
                    np.arange(new_len),
                    x_warped[b, :, c]
                )
        return torch.FloatTensor(x_resampled).to(x.device)

    @staticmethod
    def add_noise(x, noise_level=0.03):
        return x + torch.randn_like(x) * noise_level

    @staticmethod
    def freq_mask(x, max_mask=0.2):
        x_fft = torch.fft.rfft(x, dim=1)
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        n_freq = x_fft.shape[1]
        mask_len = int(n_freq * max_mask)
        start = random.randint(0, n_freq - mask_len)
        mask = torch.ones_like(mag)
        mask[:, start:start+mask_len, :] = 0.5
        x_fft_masked = mag * mask * torch.exp(1j * phase)
        return torch.fft.irfft(x_fft_masked, n=x.shape[1], dim=1)

    @classmethod
    def augment_batch(cls, x, y, aug_prob=0.3):
        if np.random.rand() < aug_prob:
            aug_type = random.choice(['warp', 'noise', 'freq'])
            if aug_type == 'warp':
                x = cls.time_warp(x)
            elif aug_type == 'noise':
                x = cls.add_noise(x)
            elif aug_type == 'freq':
                x = cls.freq_mask(x)
        return x, y

class Trainer:
    def __init__(self, config_path, data_dir, **kwargs):
        self.config = Config.from_yaml(config_path)

        # 分离 output_dir 单独处理
        output_dir = kwargs.pop('output_dir', None)
        self.output_dir = output_dir

        for key, value in kwargs.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
            elif hasattr(self.config.training, key):
                setattr(self.config.training, key, value)

        if 'ema_decay' in kwargs:
            setattr(self.config.model, 'ema_decay', kwargs['ema_decay']) 

        # 应用 output_dir
        if output_dir is not None:
            self.config.logging.save_dir = output_dir
            self.config.logging.model_dir = os.path.join(output_dir, 'models')
            os.makedirs(self.config.logging.model_dir, exist_ok=True)
            os.makedirs(self.config.logging.save_dir, exist_ok=True)

        self.config.logging.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TCB_Net(self.config).to(self.device)

        self.experiment_name = os.path.basename(data_dir.rstrip('/\\'))
        print(f"实验名称: {self.experiment_name}")

        # 优化器
        lr = float(self.config.training.learning_rate)
        wd = float(self.config.training.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Focal Loss
        alpha = float(getattr(self.config.training, 'focal_alpha', 0.75))
        gamma = float(getattr(self.config.training, 'focal_gamma', 2.0))
        label_smoothing = float(getattr(self.config.training, 'label_smoothing', 0.05))
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
        print(f"使用 Focal Loss (alpha={alpha}, gamma={gamma}, smoothing={label_smoothing})")

        # 早停配置
        self.early_stop_metric = getattr(self.config.training, 'early_stop_metric', 'auc')
        print(f"早停指标: {self.early_stop_metric}")

        # 训练状态
        self.best_score = 0.0      # 存储最佳 F1 或 AUC
        self.best_epoch = 0
        self.patience = 0
        self.warmup_epochs = getattr(self.config.training, 'warmup_epochs', 5)

        os.makedirs(self.config.logging.model_dir, exist_ok=True)
        os.makedirs(self.config.logging.save_dir, exist_ok=True)

        self.best_model_path = os.path.join(
            self.config.logging.model_dir, f"tcb_net_best_{self.experiment_name}.pth"
        )

        self.small_sample_mode = getattr(self.config.training, 'small_sample_mode', False)
        self.augmentation_prob = getattr(self.config.training, 'augmentation_prob', 0.3)

    def load_data(self):
        data_dir = self.config.logging.data_dir
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val   = np.load(os.path.join(data_dir, 'X_val.npy'))
        X_test  = np.load(os.path.join(data_dir, 'X_test.npy'))      # 新增
        y_train = np.load(os.path.join(data_dir, 'train_labels.npy')).astype(np.float32)
        y_val   = np.load(os.path.join(data_dir, 'val_labels.npy')).astype(np.float32)
        y_test  = np.load(os.path.join(data_dir, 'test_labels.npy')).astype(np.float32)  # 新增

        # 统一形状为 (batch, time, channels)
        for arr in [X_train, X_val, X_test]:
            if arr.shape[1] == 3 and arr.shape[2] != 3:
                arr = arr.transpose(0, 2, 1)   # 注意：这里需要原地修改，建议重新赋值
        # 更安全的写法：
        X_train = X_train.transpose(0,2,1) if X_train.shape[1]==3 else X_train
        X_val   = X_val.transpose(0,2,1)   if X_val.shape[1]==3   else X_val
        X_test  = X_test.transpose(0,2,1)  if X_test.shape[1]==3  else X_test

        # 训练集加权采样
        sample_weights = [1.0 if y==0 else len(y_train)/(2*(y_train==1).sum()) for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=int(self.config.training.batch_size), sampler=sampler
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=int(self.config.training.batch_size), shuffle=False
        )
        self.test_loader = DataLoader(   # 新增
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=int(self.config.training.batch_size), shuffle=False
        )
        print(f"训练集: {len(X_train)} 样本, 异常率 {y_train.mean():.2%}")
        print(f"验证集: {len(X_val)} 样本, 异常率 {y_val.mean():.2%}")
        print(f"测试集: {len(X_test)} 样本, 异常率 {y_test.mean():.2%}") 

    def train_epoch(self, epoch):
        self.model.train()

        # 预热阶段：前5个epoch冻结聚类中心
        if epoch < self.warmup_epochs:
            for param in self.model.temporal.centers:
                param.requires_grad = False
        else:
            for param in self.model.temporal.centers:
                param.requires_grad = True

        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            if self.small_sample_mode and self.augmentation_prob > 0:
                X, y = SmallSampleAugmentation.augment_batch(X, y, self.augmentation_prob)

            self.optimizer.zero_grad()
            logits = self.model(X)
            loss, comp = self.model.compute_loss(logits, y, return_components=True, focal_criterion=self.criterion)
            loss.backward()
            grad_clip = float(self.config.training.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_labels.extend(y.cpu().numpy())
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        if len(np.unique(all_labels)) > 1:
            val_auc = roc_auc_score(all_labels, all_probs)
        else:
            val_auc = 0.5
        return val_auc

    def train(self):
        self.load_data()

        max_patience = int(getattr(self.config.training, 'patience', 15))
        epochs = int(self.config.training.epochs)

        print(f"\n开始训练 (早停耐心值={max_patience}, 早停指标={self.early_stop_metric})")
        print("-" * 60)

        # 如果通道聚类中心需要 KMeans 初始化
        if getattr(self.config.model, 'channel_center_init', 'physical') == 'kmeans' and self.model.use_channel_clustering:
            self.model.eval()
            features_list = []
            num_samples = min(10000, len(self.train_loader.dataset))
            with torch.no_grad():
                for i, (X, _) in enumerate(self.train_loader):
                    if i * self.config.training.batch_size >= num_samples:
                        break
                    _, r_vec = self.model.channel(X.to(self.device))
                    features_list.append(r_vec.cpu())
            features = torch.cat(features_list, dim=0)
            self.model.channel.initialize_with_kmeans(features)
            self.model.train()
            print(f"通道聚类中心已通过 K‑means 初始化，使用 {features.shape[0]} 个样本")

        print("训练前 corr_centers:\n", self.model.channel.corr_centers.data)

        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_auc = self.validate()

            # 学习率调整
            self.scheduler.step(val_auc)

            # 早停判断（基于 AUC）
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [OK] 保存最佳模型 (AUC={val_auc:.4f})")
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= max_patience:
                    print(f"早停触发于 epoch {epoch+1}")
                    break 
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_auc={val_auc:.4f}")

        print(f"\n训练完成")
        print(f"  最佳{self.early_stop_metric}: {self.best_score:.4f} (epoch {self.best_epoch})")
        print(f"  最佳模型保存至: {self.best_model_path}")

        print("训练后 corr_centers:\n", self.model.channel.corr_centers.data)

        # 保存训练结果
        result_file = os.path.join(self.config.logging.model_dir,
                                   f"best_val_{self.early_stop_metric}_{self.experiment_name}.json")
        with open(result_file, 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_epoch': self.best_epoch,
                'early_stop_metric': self.early_stop_metric
            }, f, indent=2)

        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))

        # 动态阈值
        threshold = get_threshold_from_train_normal(
            self.model, self.train_loader, self.device,
            val_loader=self.val_loader,   # 可选微调
            percentile=99,
            fine_tune_range=0.2
        )

        # 测试集评估
        self.model.eval()
        test_probs, test_labels = [], []
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                logits = self.model(X)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                test_probs.extend(probs)
                test_labels.extend(y.numpy())
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        preds = (test_probs >= threshold).astype(int)

        acc = accuracy_score(test_labels, preds)
        prec = precision_score(test_labels, preds, zero_division=0)
        rec = recall_score(test_labels, preds, zero_division=0)
        f1 = f1_score(test_labels, preds, zero_division=0)
        auc = roc_auc_score(test_labels, test_probs)
        cm = confusion_matrix(test_labels, preds).tolist()

        # 转换 numpy 类型
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj

        results = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'threshold': threshold,
            'best_val_auc': self.best_score,
            'confusion_matrix': cm
        }
        results = convert_numpy(results)

        # 保存
        os.makedirs(self.config.logging.save_dir, exist_ok=True)
        with open(os.path.join(self.config.logging.save_dir, f'tcb_net_{self.experiment_name}_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"测试结果: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TCB-Net 训练脚本')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--config', type=str, default='./configs/tcb_net.yaml', help='配置文件路径')
    parser.add_argument('--n_time_clusters', type=int, default=None, help='时间簇数量')
    parser.add_argument('--time_hidden_dim', type=int, default=None, help='时间特征维度')
    parser.add_argument('--n_channel_clusters', type=int, default=None, help='通道簇数量')
    parser.add_argument('--lambda_cluster', type=float, default=None, help='聚类损失权重')
    parser.add_argument('--learning_rate', type=float, default=None, help='学习率')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--ema_decay', type=float, default=None, help='指数移动平均衰减率')
    parser.add_argument('--output_dir', type=str, default=None, help='结果保存目录')
    parser.add_argument('--channel_center_init', type=str, default='physical', 
                    choices=['physical', 'random', 'kmeans'], help='通道聚类中心初始化方式')
    parser.add_argument('--channel_center_learnable', action='store_true', default=True,
                    help='通道聚类中心是否可学习（默认学习）')
    parser.add_argument('--no-channel_center_learnable', action='store_false', dest='channel_center_learnable',
                    help='固定通道聚类中心，不可学习')
    args = parser.parse_args()

    kwargs = {}
    for key in ['n_time_clusters', 'time_hidden_dim', 'n_channel_clusters',
                'lambda_cluster', 'learning_rate', 'batch_size', 'epochs', 'ema_decay', 'output_dir', 'channel_center_init', 'channel_center_learnable']:
        val = getattr(args, key)
        if val is not None:
            kwargs[key] = val

    trainer = Trainer(args.config, args.data_dir, **kwargs)
    trainer.train()


if __name__ == '__main__':
    main()