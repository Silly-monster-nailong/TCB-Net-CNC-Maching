#!/usr/bin/env python
"""
analyze_data_distribution.py
统计每台机器在各个时间段的异常样本比例（基于滑动窗口序列数）
用于指导 cross_time 实验的划分。
"""

import os
import glob
import h5py
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def count_sequences(data_length, seq_len=256, stride=256):
    """计算给定信号长度能生成的序列数（无重叠）"""
    if data_length < seq_len:
        return 0
    return (data_length - seq_len) // stride + 1

def analyze_machine(machine, base_path, processes, seq_len=256, stride=256):
    """分析一台机器，返回 {(machine, period): {'good': n, 'bad': n}}"""
    results = defaultdict(lambda: {'good': 0, 'bad': 0})
    for process in processes:
        for label in ['good', 'bad']:
            pattern = os.path.join(base_path, machine, process, label, "*.h5")
            files = sorted(glob.glob(pattern))
            if not files:
                continue
            desc = f"{machine} {process} {label}"
            for fpath in tqdm(files, desc=desc, leave=False):
                try:
                    with h5py.File(fpath, 'r') as f:
                        data = f['vibration_data'][:]  # (n_points, 3)
                except Exception as e:
                    print(f"  读取失败 {fpath}: {e}")
                    continue
                n_seq = count_sequences(len(data), seq_len, stride)
                if n_seq == 0:
                    continue
                # 从文件名提取时间段
                basename = os.path.basename(fpath)
                parts = basename.split('_')
                if len(parts) >= 3:
                    period = f"{parts[1]}_{parts[2]}"
                else:
                    period = 'unknown'
                results[(machine, period)][label] += n_seq
    return results

def main():
    # 配置路径（根据您的实际目录调整）
    base_path = "../CNC_Machining/data"   # 原始数据根目录
    machines = ['M01', 'M02', 'M03']
    processes = [f"OP{i:02d}" for i in range(0, 15)]   # OP00 ~ OP14
    seq_len = 256
    stride = 256   # 与预处理中的 abnormal_stride 一致（无重叠）

    all_results = {}
    for machine in machines:
        print(f"\n🔍 分析机器 {machine} ...")
        res = analyze_machine(machine, base_path, processes, seq_len, stride)
        all_results[machine] = res

    # 打印汇总表格
    print("\n" + "="*80)
    print("各机器各时间段的异常比例（基于滑动窗口序列数）")
    print("="*80)
    for machine in machines:
        print(f"\n【机器 {machine}】")
        # 收集该机器涉及的所有时间段
        periods = set([p for (m, p) in all_results[machine].keys()])
        if not periods:
            print("  无数据")
            continue
        # 按时间排序（需要自定义排序）
        month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                     'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
        def period_key(p):
            mon, year = p.split('_')
            return (int(year), month_map[mon])
        periods_sorted = sorted(periods, key=period_key)
        print(f"{'时间段':<12} {'总序列数':>12} {'异常序列数':>12} {'异常比例':>10}")
        print("-"*50)
        for period in periods_sorted:
            good = all_results[machine].get((machine, period), {}).get('good', 0)
            bad  = all_results[machine].get((machine, period), {}).get('bad', 0)
            total = good + bad
            if total == 0:
                continue
            ratio = bad / total
            print(f"{period:<12} {total:>12,} {bad:>12,} {ratio:>9.2%}")

if __name__ == "__main__":
    main()