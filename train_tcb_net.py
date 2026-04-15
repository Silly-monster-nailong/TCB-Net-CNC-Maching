#!/usr/bin/env python
"""TCB-Net 训练脚本 - 温和改进版"""

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
from sklearn.metrics import f1_score, recall_score, precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import Config
from models.tcb_net import TCB_Net

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return loss.mean()

class Trainer:
    def __init__(self, config_path, data_dir):
        self.config = Config.from_yaml(config_path)
        self.config.logging.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TCB_Net(self.config).to(self.device)
        self.model.config = self.config
        
        # 优化器
        lr = float(self.config.training.learning_rate)
        wd = float(self.config.training.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.training.epochs)
        
        # 损失函数
        alpha = float(self.config.training.focal_alpha)
        gamma = float(self.config.training.focal_gamma)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        
        self.best_f1 = 0.0
        self.patience = 0
        os.makedirs(self.config.logging.model_dir, exist_ok=True)
        os.makedirs(self.config.logging.save_dir, exist_ok=True)
        
    def load_data(self):
        data_dir = self.config.logging.data_dir
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(data_dir, 'train_labels.npy')).astype(np.float32)
        y_val = np.load(os.path.join(data_dir, 'val_labels.npy')).astype(np.float32)
        
        # 确保形状为 (batch, channels, time)
        if X_train.shape[1] != 3:
            X_train = X_train.transpose(0, 2, 1)
            X_val = X_val.transpose(0, 2, 1)
        
        # 加权采样器（处理不平衡）
        sample_weights = [1.0 if y == 0 else float(len(y_train) / (2 * (y_train == 1).sum())) for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=int(self.config.training.batch_size),
            sampler=sampler,
            num_workers=4
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=int(self.config.training.batch_size),
            shuffle=False,
            num_workers=2
        )
        print(f"训练集: {len(X_train)} 样本, 异常率: {y_train.mean():.2%}")
        print(f"验证集: {len(X_val)} 样本, 异常率: {y_val.mean():.2%}")
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for X, y in tqdm(self.train_loader, desc='训练'):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss, comp = self.model.compute_loss(logits, y, return_components=True, focal_criterion=self.criterion)
            loss.backward()
            grad_clip = float(self.config.training.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y.cpu().numpy())
        # 搜索最佳阈值
        prec, rec, thr = precision_recall_curve(all_labels, all_probs)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1[:-1])
        best_thresh = thr[best_idx]
        preds = (np.array(all_probs) >= best_thresh).astype(int)
        f1_val = f1_score(all_labels, preds)
        rec_val = recall_score(all_labels, preds)
        return f1_val, rec_val, best_thresh
    
    def train(self):
        self.load_data()
        for epoch in range(int(self.config.training.epochs)):
            train_loss = self.train_epoch()
            val_f1, val_rec, best_thresh = self.validate()
            self.scheduler.step()
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_f1={val_f1:.4f}, val_rec={val_rec:.4f}, thresh={best_thresh:.4f}")
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.config.logging.model_dir}/tcb_net_best.pth")
                print(f"  ✅ 保存最佳模型 (F1={val_f1:.4f})")
                self.patience = 0
            else:
                self.patience += 1
                if self.patience >= int(self.config.training.patience):
                    print("早停触发")
                    break
        print(f"训练完成，最佳验证F1: {self.best_f1:.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='./configs/tcb_net.yaml')
    args = parser.parse_args()
    trainer = Trainer(args.config, args.data_dir)
    trainer.train()

if __name__ == '__main__':
    main()