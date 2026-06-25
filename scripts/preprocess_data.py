#!/usr/bin/env python
"""CNC振动数据预处理 - 支持跨时段/跨机器/跨工序划分"""

import sys
import os
import argparse
import numpy as np
import h5py
import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import joblib
import json
from collections import defaultdict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'configs'))

from configs import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='cross_time',
                        choices=['cross_time', 'cross_machine', 'cross_process'],
                        help='实验类型: cross_time(跨时段), cross_machine(跨机器), cross_process(跨工序)')
    parser.add_argument('--config', type=str, default='./configs/duet_anomaly.yaml',
                        help='配置文件路径')
    return parser.parse_args()

class CNCDataPreprocessor:
    def __init__(self, config_path, experiment):
        self.config = Config.from_yaml(config_path)
        self.data_cfg = self.config.data
        self.experiment = experiment
        self.scaler = StandardScaler()

        # 采样参数
        self.normal_freq = getattr(self.data_cfg, 'normal_freq', 100)
        self.normal_stride = getattr(self.data_cfg, 'normal_stride', 64)
        self.abnormal_freq = getattr(self.data_cfg, 'abnormal_freq', 100)
        self.abnormal_stride = getattr(self.data_cfg, 'abnormal_stride', 64)

    def load_h5_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = f['vibration_data'][:]  # (n_points, 3)
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"⚠️ 文件 {file_path} 包含 NaN/Inf，跳过")
            return None
        return data

    def downsample(self, data, original_freq=2000, target_freq=100):
        ratio = original_freq // target_freq
        if ratio <= 1:
            return data
        n_samples = len(data) // ratio
        if n_samples == 0:
            return data
        reshaped = data[:n_samples*ratio].reshape(n_samples, ratio, data.shape[1])
        return reshaped.mean(axis=1)

    def create_sequences(self, data, seq_len, stride=None):
        """创建滑动窗口序列"""
        if stride is None:
            stride = max(1, seq_len // 2)
        sequences = []
        n_total = len(data)
        for i in range(0, n_total - seq_len, stride):
            seq = data[i:i+seq_len]
            sequences.append(seq)
        if not sequences:
            return np.array([])
        return np.array(sequences)

    def process_label(self, label):
        """处理指定标签的所有文件，返回数据矩阵和文件信息列表"""
        all_sequences = []
        file_info = []

        if label == 'good':
            target_freq = self.normal_freq
            stride = self.normal_stride
        else:
            target_freq = self.abnormal_freq
            stride = self.abnormal_stride

        for process in self.data_cfg.processes:
            pattern = os.path.join(
                self.data_cfg.base_path,
                self.data_cfg.machine,  # 仅用于跨时段，跨机器时需要扩展
                process,
                label,
                "*.h5"
            )
            files = sorted(glob.glob(pattern))
            if not files and label == 'bad':
                print(f"⚠️  没有找到 {process}/{label} 数据: {pattern}")
                continue

            print(f"📂 处理 {process}/{label} 数据 ({len(files)}个文件)")
            for file_path in tqdm(files, desc=f"{process}文件"):
                try:
                    data = self.load_h5_file(file_path)
                    data_ds = self.downsample(data, 2000, target_freq)
                    seq_len = self.data_cfg.seq_len
                    sequences = self.create_sequences(data_ds, seq_len, stride)
                    if len(sequences) > 0:
                        all_sequences.append(sequences)
                        # 从文件名提取机器、时间段、工序
                        basename = os.path.basename(file_path)
                        parts = basename.split('_')
                        machine = parts[0] if len(parts) > 0 else 'unknown'
                        period = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else 'unknown'
                        proc = parts[3] if len(parts) >= 4 else process
                        for _ in range(len(sequences)):
                            file_info.append({
                                'machine': machine,
                                'period': period,
                                'process': proc,
                                'file': basename,
                                'label': label
                            })
                except Exception as e:
                    print(f"⚠️  处理文件 {os.path.basename(file_path)} 失败: {e}")

        if all_sequences:
            X = np.concatenate(all_sequences, axis=0)
            print(f"  ✅ {label}数据: {len(X)} 序列")
            return X, file_info
        else:
            print(f"  ❌ {label}数据: 没有创建任何序列")
            return None, []

    def split_indices_by_experiment(self, file_info):
        """根据实验类型划分训练/验证/测试索引，确保验证集包含异常"""
        if self.experiment == 'cross_time':
            # ========== 时间顺序划分（验证创新点） ==========
            # 训练集：早期数据
            train_periods = {'Feb_2019', 'Aug_2019'}
            # 验证集：中期数据（用于阈值优化和早停）
            val_period = {'Feb_2021'}
            # 测试集：晚期数据（最终评估）
            test_period = {'Aug_2021'}
            
            print(f"  训练集时间段: {train_periods}")
            print(f"  验证集时间段: {val_period}")
            print(f"  测试集时间段: {test_period}")
            
            all_indices = list(range(len(file_info)))
            
            # 按时间段划分
            train_idx = [i for i in all_indices if file_info[i]['period'] in train_periods]
            val_idx = [i for i in all_indices if file_info[i]['period'] in val_period]
            test_idx = [i for i in all_indices if file_info[i]['period'] in test_period]
            
            # 统计信息
            train_abnormal = sum(1 for i in train_idx if file_info[i]['label'] == 'bad')
            val_abnormal = sum(1 for i in val_idx if file_info[i]['label'] == 'bad')
            test_abnormal = sum(1 for i in test_idx if file_info[i]['label'] == 'bad')
            
            print(f"\n训练集: {len(train_idx)} 样本, 异常率: {train_abnormal/len(train_idx):.2%}")
            print(f"验证集: {len(val_idx)} 样本, 异常率: {val_abnormal/len(val_idx):.2%}")
            print(f"测试集: {len(test_idx)} 样本, 异常率: {test_abnormal/len(test_idx):.2%}")
            
            return train_idx, val_idx, test_idx

        elif self.experiment == 'cross_machine':
            train_machines = ['M01']
            test_machines  = ['M02', 'M03']
            train_idx = [i for i, info in enumerate(file_info) if info['machine'] in train_machines]
            test_idx  = [i for i, info in enumerate(file_info) if info['machine'] in test_machines]
            # 从训练集中分层采样 20% 作为验证集
            train_labels = [1 if file_info[i]['label'] == 'bad' else 0 for i in train_idx]
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                train_idx, test_size=0.2, stratify=train_labels, random_state=42
            )
            return train_idx, val_idx, test_idx

        elif self.experiment == 'cross_process':
            # 定义三组工序（完全不重叠）
            train_processes = {'OP00', 'OP01', 'OP02', 'OP03', 'OP04', 'OP05'}
            val_processes   = {'OP06', 'OP07', 'OP13'}
            test_processes  = {'OP08', 'OP09', 'OP10', 'OP11', 'OP12', 'OP14'}
            
            # 只使用 M01 机器
            all_indices = list(range(len(file_info)))
            train_idx = [i for i in all_indices if file_info[i]['machine'] == 'M01' and file_info[i]['process'] in train_processes]
            val_idx   = [i for i in all_indices if file_info[i]['machine'] == 'M01' and file_info[i]['process'] in val_processes]
            test_idx  = [i for i in all_indices if file_info[i]['machine'] == 'M01' and file_info[i]['process'] in test_processes]
            
            # 可选：如果希望验证集更小（例如从验证工序中采样一部分），可再采样，但一般不需要。
            return train_idx, val_idx, test_idx

        elif self.experiment == 'cross_time_ablation':
            # 消融实验：验证集从训练集采样，不按时间划分
            train_periods = {'Feb_2019', 'Aug_2019'}
            test_period = 'Aug_2021'
            
            all_indices = list(range(len(file_info)))
            
            train_candidates = [i for i in all_indices if file_info[i]['period'] in train_periods]
            test_idx = [i for i in all_indices if file_info[i]['period'] == test_period]
            
            # 从训练候选集中分层采样 20% 作为验证集
            train_labels = [1 if file_info[i]['label'] == 'bad' else 0 for i in train_candidates]
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(
                train_candidates, test_size=0.2, stratify=train_labels, random_state=42
            )
            
            print(f"【消融实验】验证集从训练集采样")
            print(f"训练集: {len(train_idx)} 样本, 异常率: {np.mean([1 if file_info[i]['label']=='bad' else 0 for i in train_idx]):.2%}")
            print(f"验证集: {len(val_idx)} 样本, 异常率: {np.mean([1 if file_info[i]['label']=='bad' else 0 for i in val_idx]):.2%}")
            print(f"测试集: {len(test_idx)} 样本, 异常率: {np.mean([1 if file_info[i]['label']=='bad' else 0 for i in test_idx]):.2%}")
            
            return train_idx, val_idx, test_idx

    def run(self):
        print("="*60)
        print(f"CNC数据预处理 - 实验类型: {self.experiment}")
        print("="*60)

        # 1. 处理正常数据
        print("\n📊 处理正常数据")
        X_normal, normal_info = self.process_label('good')
        if X_normal is None:
            raise ValueError("❌ 没有正常数据")

        # 2. 处理异常数据
        print("\n📊 处理异常数据")
        X_abnormal, abnormal_info = self.process_label('bad')

        # 3. 合并所有文件信息（用于划分）
        all_info = normal_info + abnormal_info
        all_X = np.concatenate([X_normal, X_abnormal], axis=0) if X_abnormal is not None else X_normal

        # 4. 划分训练/验证/测试索引
        train_idx, val_idx, test_idx = self.split_indices_by_experiment(all_info)

        # 5. 分配数据
        X_train = all_X[train_idx]
        X_val   = all_X[val_idx]
        X_test  = all_X[test_idx]

        # 创建标签
        y_train = np.array([1 if all_info[i]['label'] == 'bad' else 0 for i in train_idx])
        y_val   = np.array([1 if all_info[i]['label'] == 'bad' else 0 for i in val_idx])
        y_test  = np.array([1 if all_info[i]['label'] == 'bad' else 0 for i in test_idx])

        # 6. 标准化
        print("\n📊 标准化数据")
        all_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler.fit(all_train_flat)
        X_train = self.scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val   = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test  = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # 裁剪极端值，防止后续计算溢出
        clip_val = 5.0
        X_train = np.clip(X_train, -clip_val, clip_val)
        X_val = np.clip(X_val, -clip_val, clip_val)
        X_test = np.clip(X_test, -clip_val, clip_val)

        # 7. 保存
        output_dir = f"./data/processed/{self.experiment}"
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

        # 保存标签（用于半监督）
        train_labels = y_train.astype(np.float32)
        val_labels   = y_val.astype(np.float32)
        test_labels  = y_test.astype(np.float32)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(output_dir, 'val_labels.npy'), val_labels)
        np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

        # 保存标准化器
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))

        # 保存划分信息
        with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
            json.dump({
                'experiment': self.experiment,
                'train_count': len(train_idx),
                'val_count': len(val_idx),
                'test_count': len(test_idx),
                'train_abnormal': int(y_train.sum()),
                'val_abnormal': int(y_val.sum()),
                'test_abnormal': int(y_test.sum())
            }, f, indent=2)

        print(f"\n✅ 数据保存到 {output_dir}")
        print(f"   训练: {len(X_train)} 样本 (异常率 {y_train.mean():.2%})")
        print(f"   验证: {len(X_val)} 样本 (异常率 {y_val.mean():.2%})")
        print(f"   测试: {len(X_test)} 样本 (异常率 {y_test.mean():.2%})")

def main():
    args = parse_args()
    preprocessor = CNCDataPreprocessor(args.config, args.experiment)
    preprocessor.run()

if __name__ == '__main__':
    main()