#!/usr/bin/env python
"""
参数敏感性分析 - 时间聚类模块
- 变化参数：n_time_clusters, time_hidden_dim, ema_decay
- 每个实验训练10轮（或使用早停），记录测试集 F1 和 AUC
"""

import subprocess
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "./data/processed/cross_time"
BASE_CONFIG = "./configs/tcb_net.yaml"
EPOCHS = 10  # 可适当减少以加快实验，但需保证不同实验对比公平

# 参数范围
PARAMS = {
    'n_time_clusters': [2, 4, 6, 8],
    'time_hidden_dim': [32, 64, 128],
    'ema_decay': [0.8, 0.9, 0.95]   # decay=0.9 相当于新样本权重 0.1
}

def run_experiment(param_name, param_value):
    """运行单次训练并返回 F1 和 AUC"""
    output_dir = f"./ablation/param_{param_name}_{param_value}"
    cmd = [
        "python", "scripts/train_tcb_net.py",
        "--data_dir", DATA_DIR,
        "--config", BASE_CONFIG,
        "--epochs", str(EPOCHS),
        "--output_dir", output_dir
    ]
    if param_name == 'n_time_clusters':
        cmd.extend(["--n_time_clusters", str(param_value)])
    elif param_name == 'time_hidden_dim':
        cmd.extend(["--time_hidden_dim", str(param_value)])
    elif param_name == 'ema_decay':
        cmd.extend(["--ema_decay", str(param_value)])
    
    print(f"Running: {param_name}={param_value}")
    subprocess.run(cmd, check=True)
    
    results_path = os.path.join(output_dir, "tcb_net_cross_time_results.json")
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results['f1'], results['auc']

def main():
    save_dir = "./ablation"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for param_name, values in PARAMS.items():
        for val in values:
            f1, auc = run_experiment(param_name, val)
            results.append({'param': param_name, 'value': val, 'f1': f1, 'auc': auc})
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "param_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(df)
    
    # 绘制折线图
    for param_name in PARAMS:
        sub = df[df['param'] == param_name]
        plt.figure(figsize=(6,4))
        plt.plot(sub['value'], sub['f1'], 'o-', label='F1')
        plt.plot(sub['value'], sub['auc'], 's-', label='AUC')
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.legend()
        plt.title(f'Parameter Sensitivity: {param_name}')
        plt.grid(True)
        png_path = os.path.join(save_dir, f'param_sensitivity_{param_name}.png')
        plt.savefig(png_path, dpi=150)
        plt.close()
    print("结果已保存，图表已生成。")

if __name__ == '__main__':
    main()