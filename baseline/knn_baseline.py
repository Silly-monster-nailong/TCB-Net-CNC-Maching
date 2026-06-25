"""
KNN 基线模型
- 展平特征
- 验证集上全局搜索最佳阈值（最大化 F1）
- 测试集评估：准确率、精确率、召回率、F1 分数、AUC，以及混淆矩阵
- 结果保存：输出目录下的 results.json
- 评估细节：使用 sklearn 的 KNeighborsClassifier 和相关评估函数，支持多线程加速（n_jobs=-1）
- 统一使用全局最佳阈值，保持与其他基线模型一致的评估流程
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)
import os
import argparse
import json


def load_and_flatten(data_dir):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    if X_train.ndim == 3:
        X_train = X_train.reshape(len(X_train), -1)
        X_val = X_val.reshape(len(X_val), -1)
        X_test = X_test.reshape(len(X_test), -1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='knn_baseline')
    parser.add_argument('--n_neighbors', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_flatten(args.data_dir)

    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, weights='distance', n_jobs=-1)
    knn.fit(X_train, y_train)

    # ----- 验证集上全局搜索最佳阈值（最大化 F1）-----
    val_probs = knn.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])          # 忽略最后一个点（阈值=1）
    best_thr = thresholds[best_idx]
    best_f1_val = f1_scores[best_idx]
    print(f"[阈值] 验证集 F1 最大化阈值 = {best_thr:.4f} (F1={best_f1_val:.4f})")

    # ----- 测试集评估 -----
    test_probs = knn.predict_proba(X_test)[:, 1]
    preds = (test_probs >= best_thr).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, preds).tolist()

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': best_thr,
        'best_val_f1': best_f1_val,
        'confusion_matrix': cm
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    print(f"\nTest Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"\n✅ 结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()