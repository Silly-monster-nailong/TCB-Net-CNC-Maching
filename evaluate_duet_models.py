#!/usr/bin/env python
"""
评估 Duet 模型（SA-FNO 和 Physical-FNO）在测试集上的性能
支持基于验证集的阈值优化，输出召回率、精确率、F1、混淆矩阵等指标
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
import json
import glob

from configs import Config
from models.hybrid_model import FNOSmallSampleDetector
from models.physical_fno_detector_simple import PhysicalFNOAnomalyDetector


def load_model(model_path, config, device):
    """加载模型，根据文件名自动识别模型类型"""
    # 根据文件名判断模型类型
    path_lower = model_path.lower()
    if 'sa_fno' in path_lower:
        from models.hybrid_model import FNOSmallSampleDetector
        model = FNOSmallSampleDetector(config).to(device)
    elif 'physical_fno' in path_lower:
        from models.physical_fno_detector_simple import PhysicalFNOAnomalyDetector
        model = PhysicalFNOAnomalyDetector(config).to(device)
    else:
        raise ValueError(f"无法从路径识别模型类型: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_predictions(model, loader, device):
    """获取模型预测概率，兼容不同输出形状"""
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            
            # 根据输出形状处理
            if logits.dim() == 1:
                # 形状: (batch,)
                probs = torch.sigmoid(logits).cpu().numpy()
            elif logits.shape[-1] == 1:
                # 形状: (batch, 1)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            elif logits.shape[-1] == 2:
                # 形状: (batch, 2) - 二分类
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())
    return np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs, labels):
    """基于验证集精确率-召回率曲线寻找最佳阈值（最大化F1）"""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # 排除最后一个阈值点
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_thresh, best_f1


def evaluate_model(model, test_loader, val_probs=None, val_labels=None, threshold=None, device='cuda'):
    """
    评估模型
    如果提供验证集概率和标签，则自动寻找最佳阈值；否则使用给定阈值
    """
    # 获取测试集预测概率
    test_probs, test_labels = get_predictions(model, test_loader, device)

    # 确定阈值
    if threshold is None:
        if val_probs is not None and val_labels is not None:
            threshold, _ = find_best_threshold(val_probs, val_labels)
            print(f"  基于验证集优化阈值: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"  使用默认阈值: 0.5")

    preds = (test_probs >= threshold).astype(int)

    # 计算指标
    recall = recall_score(test_labels, preds, zero_division=0)
    precision = precision_score(test_labels, preds, zero_division=0)
    f1 = f1_score(test_labels, preds, zero_division=0)
    auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    cm = confusion_matrix(test_labels, preds)

    metrics = {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'threshold': threshold,
        'confusion_matrix': cm.tolist()
    }
    return metrics, test_probs, test_labels


def main():
    parser = argparse.ArgumentParser(description='评估 Duet 模型 (SA-FNO / Physical-FNO)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录，需包含 X_test.npy, y_test.npy 以及可选的 X_val.npy, y_val.npy')
    parser.add_argument('--config', type=str, default='./configs/duet_anomaly.yaml',
                        help='模型配置文件')
    parser.add_argument('--model1', type=str, default=None,
                        help='SA-FNO 模型权重路径，若不指定则自动搜索 models/ 目录')
    parser.add_argument('--model2', type=str, default=None,
                        help='Physical-FNO 模型权重路径，若不指定则自动搜索')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备 (cuda/cpu)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载配置
    config = Config.from_yaml(args.config)
    print(f"配置文件: {args.config}")

    # 加载测试数据
    X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'test_labels.npy'))
    print(f"测试集大小: {X_test.shape}, 异常率: {y_test.mean():.2%}")

    # 转换为 tensor
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 加载验证集（用于阈值优化）
    val_probs = None
    val_labels = None
    val_path_x = os.path.join(args.data_dir, 'X_val.npy')
    val_path_y = os.path.join(args.data_dir, 'val_labels.npy')
    if os.path.exists(val_path_x) and os.path.exists(val_path_y):
        X_val = np.load(val_path_x)
        y_val = np.load(val_path_y)
        print(f"验证集大小: {X_val.shape}, 异常率: {y_val.mean():.2%}")
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    else:
        print("警告: 未找到验证集文件，将使用默认阈值 0.5")
        val_loader = None

    # 自动搜索模型文件
    model_dir = os.path.join(os.path.dirname(args.config), '..', 'models')  # 假设 models 目录在 configs 同级
    if not os.path.exists(model_dir):
        model_dir = './models'

    def find_model(pattern):
        matches = glob.glob(os.path.join(model_dir, f'*{pattern}*.pth'))
        if matches:
            return matches[0]
        return None

    model1_path = args.model1 if args.model1 else find_model('sa_fno')
    model2_path = args.model2 if args.model2 else find_model('physical_fno')

    if model1_path is None:
        print("错误: 未找到 SA-FNO 模型文件，请通过 --model1 指定")
        sys.exit(1)
    if model2_path is None:
        print("错误: 未找到 Physical-FNO 模型文件，请通过 --model2 指定")
        sys.exit(1)

    print(f"\nSA-FNO 模型: {model1_path}")
    print(f"Physical-FNO 模型: {model2_path}")

    # 加载模型
    model1 = load_model(model1_path, config, device)
    model2 = load_model(model2_path, config, device)

    # 如果提供了验证集，先获取两个模型在验证集上的概率（用于阈值优化）
    val_probs1 = val_probs2 = None
    if val_loader is not None:
        val_probs1, val_labels = get_predictions(model1, val_loader, device)
        val_probs2, _ = get_predictions(model2, val_loader, device)
        print(f"\n验证集预测概率统计: SA-FNO mean={val_probs1.mean():.4f}, Physical-FNO mean={val_probs2.mean():.4f}")

    # 评估两个模型
    print("\n" + "="*60)
    print("评估 SA-FNO 模型")
    print("="*60)
    metrics1, _, _ = evaluate_model(model1, test_loader, val_probs1, val_labels, device=device)

    print("\n" + "="*60)
    print("评估 Physical-FNO 模型")
    print("="*60)
    metrics2, _, _ = evaluate_model(model2, test_loader, val_probs2, val_labels, device=device)

    # 打印对比结果
    print("\n" + "="*60)
    print("模型性能对比 (测试集)")
    print("="*60)
    print(f"{'指标':<12} {'SA-FNO':<15} {'Physical-FNO':<15}")
    print("-"*42)
    print(f"{'召回率':<12} {metrics1['recall']:<15.4f} {metrics2['recall']:<15.4f}")
    print(f"{'精确率':<12} {metrics1['precision']:<15.4f} {metrics2['precision']:<15.4f}")
    print(f"{'F1分数':<12} {metrics1['f1']:<15.4f} {metrics2['f1']:<15.4f}")
    print(f"{'AUC':<12} {metrics1['auc']:<15.4f} {metrics2['auc']:<15.4f}")
    print(f"{'阈值':<12} {metrics1['threshold']:<15.4f} {metrics2['threshold']:<15.4f}")

    # 混淆矩阵
    print("\n混淆矩阵 (SA-FNO):")
    cm1 = np.array(metrics1['confusion_matrix'])
    print(f"  TN={cm1[0,0]}  FP={cm1[0,1]}")
    print(f"  FN={cm1[1,0]}  TP={cm1[1,1]}")
    print("\n混淆矩阵 (Physical-FNO):")
    cm2 = np.array(metrics2['confusion_matrix'])
    print(f"  TN={cm2[0,0]}  FP={cm2[0,1]}")
    print(f"  FN={cm2[1,0]}  TP={cm2[1,1]}")

    # 保存结果到 JSON
    output_dir = './evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'sa_fno': metrics1,
        'physical_fno': metrics2,
        'test_info': {
            'num_samples': len(y_test),
            'anomaly_rate': float(y_test.float().mean())
        }
    }
    with open(os.path.join(output_dir, 'duet_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n评估结果已保存至: {output_dir}/duet_evaluation.json")


if __name__ == '__main__':
    main()