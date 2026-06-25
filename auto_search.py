#!/usr/bin/env python
"""
自动超参数搜索脚本（TCB-Net）
按顺序：n_time_clusters -> time_hidden_dim -> lambda_cluster -> 学习率
每个组合会完整训练，并记录最佳验证 F1。
"""

import subprocess
import json
import os
import time
from itertools import product

# 配置
DATA_DIR = "./data/processed/cross_time"   # 可修改为 cross_process
CONFIG_PATH = "./configs/tcb_net.yaml"
MODEL_DIR = "./models"
RESULT_FILE = "./hyperparam_search_results.json"

# 搜索空间（按顺序）
param_groups = [
    {'name': 'n_time_clusters', 'values': [3]},
    {'name': 'time_hidden_dim', 'values': [32, 64]},
    {'name': 'lambda_cluster', 'values': [0.01, 0.05, 0.1]},
    # 最后微调学习率（可选）
    {'name': 'learning_rate', 'values': [0.0005, 0.001, 0.002]},
]

# 存储所有结果
all_results = []

def run_training(params):
    """运行训练并返回最佳验证 F1（直接显示输出，无编码错误）"""
    cmd = ["python", "scripts/train_tcb_net.py",
           "--data_dir", DATA_DIR,
           "--config", CONFIG_PATH]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"\n{'='*60}")
    print(f"运行参数: {params}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # 删除旧的结果文件
    exp_name = os.path.basename(DATA_DIR.rstrip('/\\'))
    result_file = os.path.join(MODEL_DIR, f"best_val_f1_{exp_name}.json")
    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"已删除旧结果文件: {result_file}")
    
    start = time.time()
    # 直接运行，不捕获输出（输出会显示在终端）
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    
    # 检查训练是否成功
    if result.returncode != 0:
        print(f"训练失败，返回码 {result.returncode}")
        return 0.0
    
    # 等待结果文件生成（最多 60 秒）
    for _ in range(60):
        if os.path.exists(result_file):
            break
        time.sleep(1)
    else:
        print("未找到结果文件，训练可能异常")
        return 0.0
    
    # 读取结果
    with open(result_file, 'r') as f:
        data = json.load(f)
        best_f1 = data['best_val_f1']
    
    print(f"完成，耗时 {elapsed/60:.1f} 分钟，最佳验证 F1 = {best_f1:.4f}")
    return best_f1

def grid_search():
    best_params = {}
    best_f1 = -1
    
    # 第一阶段：调 n_time_clusters
    # print("\n" + "="*60)
    # print("第一阶段：调整 n_time_clusters")
    # print("="*60)
    # for n in param_groups[0]['values']:
    #     params = {'n_time_clusters': n}
    #     f1 = run_training(params)
    #     all_results.append({'stage': 1, 'params': params, 'best_val_f1': f1})
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_params = params.copy()
    # print(f"\n第一阶段完成，最佳 n_time_clusters = {best_params['n_time_clusters']} (F1={best_f1:.4f})")
    
    # 第二阶段：固定 n_time_clusters，调 time_hidden_dim
    print("\n" + "="*60)
    print("第二阶段：调整 time_hidden_dim")
    print("="*60)
    for h in param_groups[1]['values']:
        params = best_params.copy()
        params['time_hidden_dim'] = h
        f1 = run_training(params)
        all_results.append({'stage': 2, 'params': params, 'best_val_f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()
    print(f"\n第二阶段完成，最佳 time_hidden_dim = {best_params.get('time_hidden_dim')} (F1={best_f1:.4f})")
    
    # 第三阶段：固定前两者，调 lambda_cluster
    print("\n" + "="*60)
    print("第三阶段：调整 lambda_cluster")
    print("="*60)
    for l in param_groups[2]['values']:
        params = best_params.copy()
        params['lambda_cluster'] = l
        f1 = run_training(params)
        all_results.append({'stage': 3, 'params': params, 'best_val_f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()
    print(f"\n第三阶段完成，最佳 lambda_cluster = {best_params.get('lambda_cluster')} (F1={best_f1:.4f})")
    
    # 第四阶段：微调学习率（可选）
    print("\n" + "="*60)
    print("第四阶段：微调 learning_rate")
    print("="*60)
    for lr in param_groups[3]['values']:
        params = best_params.copy()
        params['learning_rate'] = lr
        f1 = run_training(params)
        all_results.append({'stage': 4, 'params': params, 'best_val_f1': f1})
        if f1 > best_f1:
            best_f1 = f1
            best_params = params.copy()
    print(f"\n第四阶段完成，最佳 learning_rate = {best_params.get('learning_rate')} (F1={best_f1:.4f})")
    
    # 保存所有结果
    with open(RESULT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"搜索完成！最佳参数组合: {best_params}")
    print(f"最佳验证 F1: {best_f1:.4f}")
    print(f"结果已保存至 {RESULT_FILE}")
    print("="*60)

if __name__ == "__main__":
    grid_search()