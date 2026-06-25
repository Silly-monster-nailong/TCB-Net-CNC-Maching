"""
ModernTCN 基线模型
- 统一使用 FocalLoss
- 统一使用动态阈值（基于训练集正常样本分布，启用验证集微调）
- 早停机制：基于验证集 AUC，耐心值 15
- 训练细节：AdamW 优化器，ReduceLROnPlateau 学习率调度，早停机制
- 评估指标：准确率、精确率、召回率、F1 分数、AUC，以及混淆矩阵
- 结果保存：输出目录下的 results.json 和 moderntcn_model.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_curve, recall_score, f1_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix
import os
import argparse
from tqdm import tqdm
import json

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        probs = torch.sigmoid(logits)
        pos_loss = -self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs + 1e-8)
        neg_loss = -(1 - self.alpha) * probs ** self.gamma * (1 - targets) * torch.log(1 - probs + 1e-8)
        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

def get_threshold_from_train_normal(model, train_loader, device, val_loader=None, percentile=99, fine_tune_range=0.2):
    model.eval()
    normal_probs = []
    with torch.no_grad():
        for X, y in train_loader:
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
            thresholds = np.linspace(base_thr * (1 - fine_tune_range), base_thr * (1 + fine_tune_range), 21)
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

class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        skip = x
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return self.norm(x + skip)

class ModernTCN(nn.Module):
    def __init__(self, seq_len=256, n_channels=3, d_model=64, num_blocks=2, kernel_size=7, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.blocks = nn.ModuleList([
            ConvBlock(d_model, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x).squeeze(-1)
        return logits

def train_and_evaluate(data_dir, output_dir, batch_size=256, lr=0.0005, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy')).astype(np.float32)
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy')).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy')).astype(np.float32)

    # 统一形状为 (batch, seq_len, channels)
    if X_train.shape[1] != 256:
        X_train = X_train.transpose(0, 2, 1)
        X_val = X_val.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    seq_len = X_train.shape[1]
    print(f"序列长度: {seq_len}")

    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if y_train.sum() > 0 else 1.0
    sample_weights = [1.0 if y == 0 else pos_weight for y in y_train.numpy()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=2)

    model = ModernTCN(seq_len=seq_len).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ---------- 验证（只计算 AUC）----------
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_labels.extend(y_batch.numpy())
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        scheduler.step(val_auc)

        # 保存最佳模型（基于 AUC）
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            print(f"  [OK] 保存最佳模型 (AUC={best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"早停于 epoch {epoch+1}")
                break

        # 每个 epoch 都打印指标
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_auc={val_auc:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_state)

    # 动态阈值（启用验证集微调）
    threshold = get_threshold_from_train_normal(
        model, train_loader, device,
        val_loader=val_loader,   # 使用验证集微调
        percentile=99,
        fine_tune_range=0.2
    )

    # ---------- 测试集评估 ----------
    model.eval()
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            test_probs.extend(probs)
            test_labels.extend(y_batch.numpy())
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    preds = (test_probs >= threshold).astype(int)

    accuracy = accuracy_score(test_labels, preds)
    precision = precision_score(test_labels, preds, zero_division=0)
    recall = recall_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    cm = confusion_matrix(test_labels, preds).tolist()

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': threshold,
        'best_val_auc': best_auc,
        'confusion_matrix': cm
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    torch.save(model.state_dict(), os.path.join(output_dir, 'moderntcn_model.pth'))
    print(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='moderntcn_baseline')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    train_and_evaluate(args.data_dir, args.output_dir, args.batch_size, args.lr, args.epochs)

if __name__ == '__main__':
    main()