"""
RandomForest 基线模型
- 直接使用原始信号（展平）
- 统一使用动态阈值（基于训练集正常样本分布，启用验证集微调）
- 评估指标：准确率、精确率、召回率、F1 分数、AUC，以及混淆矩阵
- 结果保存：输出目录下的 results.json 和 randomforest_model.pkl
- 评估细节：使用 sklearn 的 RandomForestClassifier 和相关评估函数，支持多线程加速（n_jobs=-1）
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
import joblib
import os
import json
import argparse


class RandomForestBaseline:
    def __init__(self, data_dir, output_dir='randomforest_baseline'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = None

    def load_data(self):
        X_train = np.load(os.path.join(self.data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(self.data_dir, 'X_val.npy'))
        X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
        y_train = np.load(os.path.join(self.data_dir, 'train_labels.npy'))
        y_val = np.load(os.path.join(self.data_dir, 'val_labels.npy'))
        y_test = np.load(os.path.join(self.data_dir, 'test_labels.npy'))

        print(f"\n数据加载完成:")
        print(f"  训练集: {X_train.shape}, 异常率: {y_train.mean():.2%}")
        print(f"  验证集: {X_val.shape}, 异常率: {y_val.mean():.2%}")
        print(f"  测试集: {X_test.shape}, 异常率: {y_test.mean():.2%}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def flatten_data(self, X):
        """(n_samples, time, channels) -> (n_samples, time*channels)"""
        return X.reshape(X.shape[0], -1)

    def train(self, X_train, y_train):
        print("\n【模型训练】...")
        X_train_flat = self.flatten_data(X_train)
        print(f"  展平后特征维度: {X_train_flat.shape[1]}")
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_split=50,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_flat, y_train)
        print("  训练完成")

    def get_threshold(self, X_train, y_train, X_val, y_val, percentile=99, fine_tune_range=0.2):
        """
        基于训练集正常样本的分位数获取阈值，并在验证集上微调。
        """
        # 训练集正常样本的概率
        train_normal_mask = (y_train == 0)
        if train_normal_mask.sum() == 0:
            base_thr = 0.5
        else:
            X_train_normal = self.flatten_data(X_train[train_normal_mask])
            train_normal_probs = self.model.predict_proba(X_train_normal)[:, 1]
            base_thr = np.percentile(train_normal_probs, percentile)
        print(f"[阈值] 训练集正常样本 {percentile}% 分位数 = {base_thr:.4f}")

        # 验证集微调（如果正样本足够）
        val_probs = self.model.predict_proba(self.flatten_data(X_val))[:, 1]
        pos_cnt = (y_val == 1).sum()
        if pos_cnt >= 10:
            thresholds = np.linspace(base_thr * (1 - fine_tune_range),
                                     base_thr * (1 + fine_tune_range), 21)
            best_f1 = 0
            best_thr = base_thr
            for thr in thresholds:
                preds = (val_probs >= thr).astype(int)
                f1 = f1_score(y_val, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            print(f"[阈值] 验证集微调后 = {best_thr:.4f} (F1={best_f1:.4f})")
            return best_thr
        else:
            print(f"[阈值] 验证集正样本不足 ({pos_cnt} < 10)，不微调")
            return base_thr

    def evaluate(self, X_test, y_test, threshold):
        X_test_flat = self.flatten_data(X_test)
        y_prob = self.model.predict_proba(X_test_flat)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"\nTest Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'threshold': threshold,
            'confusion_matrix': cm
        }

    def save_results(self, results):
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        joblib.dump(self.model, os.path.join(self.output_dir, 'randomforest_model.pkl'))
        print(f"\n✅ 结果保存至: {self.output_dir}")

    def run(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        self.train(X_train, y_train)
        threshold = self.get_threshold(X_train, y_train, X_val, y_val)
        results = self.evaluate(X_test, y_test, threshold)
        self.save_results(results)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='数据目录')
    parser.add_argument('--output_dir', default='randomforest_baseline', help='输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print("RandomForest 基线模型（动态阈值）")
    print("=" * 60)

    baseline = RandomForestBaseline(args.data_dir, args.output_dir)
    baseline.run()

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)


if __name__ == '__main__':
    main()