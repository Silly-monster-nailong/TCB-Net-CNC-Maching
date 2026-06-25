#!/usr/bin/env python
"""
analyze_m01_process_distribution.py
统计 M01 机器各工序的序列数量（正常/故障），用于指导跨工序划分
"""

import os
import glob
import h5py
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def count_sequences(data_len, seq_len=256, stride=256):
    """计算给定信号长度能生成的序列数（无重叠）"""
    if data_len < seq_len:
        return 0
    return (data_len - seq_len) // stride + 1

def main():
    base_path = "../CNC_Machining/data"
    machine = "M01"
    processes = [f"OP{i:02d}" for i in range(0, 15)]
    seq_len = 256
    stride = 256   # 无重叠

    results = defaultdict(lambda: {'good': 0, 'bad': 0})

    for process in processes:
        for label in ['good', 'bad']:
            pattern = os.path.join(base_path, machine, process, label, "*.h5")
            files = sorted(glob.glob(pattern))
            if not files:
                continue
            desc = f"{process} {label}"
            for fpath in tqdm(files, desc=desc, leave=False):
                try:
                    with h5py.File(fpath, 'r') as f:
                        data = f['vibration_data'][:]  # (n_points, 3)
                except Exception as e:
                    print(f"  读取失败 {fpath}: {e}")
                    continue
                n_seq = count_sequences(len(data), seq_len, stride)
                if n_seq > 0:
                    results[process][label] += n_seq

    # 打印结果表格
    print("\n" + "="*70)
    print(f"机器 {machine} 各工序序列统计（窗口{seq_len}，步长{stride}）")
    print("="*70)
    print(f"{'工序':<8} {'总序列数':>12} {'故障序列数':>12} {'故障比例':>10}")
    print("-"*50)

    total_all = 0
    total_bad = 0
    for proc in sorted(results.keys()):
        good = results[proc]['good']
        bad = results[proc]['bad']
        total = good + bad
        total_all += total
        total_bad += bad
        ratio = bad / total if total > 0 else 0.0
        print(f"{proc:<8} {total:>12,} {bad:>12,} {ratio:>9.2%}")

    print("-"*50)
    print(f"{'合计':<8} {total_all:>12,} {total_bad:>12,} {total_bad/total_all:>9.2%}")

if __name__ == "__main__":
    main()