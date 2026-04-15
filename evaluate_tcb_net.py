#!/usr/bin/env python
"""评估 TCB-Net 模型并输出可解释性分析 - 适配温和改进版"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

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

def find_best_threshold(model, val_loader, device):
    """在验证集上搜索最佳阈值（最大化F1）"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            # 确保形状为 (batch, channels, time)
            if X_batch.dim() == 3 and X_batch.size(2) == 3 and X_batch.size(1) != 3:
                X_batch = X_batch.permute(0, 2, 1)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())  # 修复：添加 .cpu()
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"验证集预测概率 - 均值: {all_probs.mean():.4f}, 中位数: {np.median(all_probs):.4f}")
    print(f"最佳阈值: {best_thresh:.4f}, 对应F1: {best_f1:.4f}")
    return best_thresh

def evaluate(data_dir=None):
    config_path = os.path.join(project_root, 'configs', 'tcb_net.yaml')
    config = Config.from_yaml(config_path)
    if data_dir is not None:
        config.logging.data_dir = data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCB_Net(config).to(device)

    # 加载最佳模型
    model_path = os.path.join(config.logging.model_dir, 'tcb_net_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    data_dir = config.logging.data_dir

    # 加载验证集
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))
    print(f"\n验证集样本数: {len(X_val)}")
    print(f"验证集标签分布: 正常={np.sum(y_val==0)}, 异常={np.sum(y_val==1)}")
    print(f"验证集异常率: {y_val.mean():.4f}")
    print(f"X_val 均值: {X_val.mean():.4f}, 标准差: {X_val.std():.4f}")

    # 转置为 (batch, channels, time) 以匹配模型输入
    if X_val.shape[1] != 3:
        X_val = X_val.transpose(0, 2, 1)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 寻找最佳阈值
    best_thresh = find_best_threshold(model, val_loader, device)

    # 加载测试集
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
    print(f"\n测试集样本数: {len(X_test)}")
    print(f"测试集异常率: {y_test.mean():.4f}")
    print(f"X_test 均值: {X_test.mean():.4f}, 标准差: {X_test.std():.4f}")

    # 转置为 (batch, channels, time)
    if X_test.shape[1] != 3:
        X_test = X_test.transpose(0, 2, 1)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 测试集推理
    all_probs = []
    all_preds = []
    all_labels = []
    all_time_weights = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= best_thresh).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy().flatten())
            
            # 获取时间簇权重（用于可解释性）
            # 需要单独调用 temporal 模块，注意输入形状
            X_temporal = X_batch  # 已经是 (batch, channels, time)
            _, w_time, _ = model.temporal(X_temporal)
            all_time_weights.append(w_time.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_time_weights = np.concatenate(all_time_weights, axis=0) if all_time_weights else np.array([])

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("测试集结果 (使用验证集最佳阈值 {:.4f})".format(best_thresh))
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("混淆矩阵:")
    print(cm)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn+fp)>0 else 0.0
        print(f"特异性: {specificity:.4f}")
    print("="*60)

    # 可解释性分析（如果有时间簇权重）
    if len(all_time_weights) > 0:
        print("\n【可解释性分析】")
        n_clusters = config.model.n_time_clusters
        print(f"时间簇数量: {n_clusters}")
        print("\n各时间簇的平均权重（所有样本）:")
        for k in range(n_clusters):
            print(f"  簇{k}: {all_time_weights[:, k].mean():.3f}")

        normal_mask = (all_labels == 0)
        fault_mask = (all_labels == 1)
        print("\n按标签分组的时间簇平均权重:")
        for k in range(n_clusters):
            normal_w = all_time_weights[normal_mask, k].mean() if normal_mask.any() else 0.0
            fault_w = all_time_weights[fault_mask, k].mean() if fault_mask.any() else 0.0
            print(f"  簇{k}: 正常={normal_w:.3f}, 故障={fault_w:.3f}")

        dominant_cluster = np.argmax(all_time_weights, axis=1)
        print("\n主导时间簇分布:")
        unique, counts = np.unique(dominant_cluster, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  簇{u}: {c} 样本 ({c/len(dominant_cluster)*100:.1f}%)")

        print("\n各主导簇的故障率:")
        for u in unique:
            mask = (dominant_cluster == u)
            fault_rate = all_labels[mask].mean()
            print(f"  簇{u}: {fault_rate:.2%}")

    # 保存图表
    save_dir = config.logging.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 混淆矩阵图
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    plt.title('TCB-Net 测试集混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tcb_confusion_matrix.png'))
    plt.close()
    print(f"\n混淆矩阵图已保存: {save_dir}/tcb_confusion_matrix.png")
    
    # 如果有时间簇权重，绘制热力图
    if len(all_time_weights) > 0:
        plt.figure(figsize=(12,5))
        plt.imshow(all_time_weights[:200, :].T, aspect='auto', cmap='viridis')
        plt.colorbar(label='权重')
        plt.xlabel('样本序号')
        plt.ylabel('时间簇')
        plt.title('前200个样本的时间簇分配权重')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tcb_time_cluster_weights.png'))
        plt.close()
        print(f"时间簇权重图已保存: {save_dir}/tcb_time_cluster_weights.png")

    print("\n评估完成。")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()
    evaluate(data_dir=args.data_dir)

if __name__ == '__main__':
    main()