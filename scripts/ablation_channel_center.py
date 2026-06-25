import subprocess
import json
import os
import csv

DATA_DIR = "./data/processed/cross_time"
BASE_CONFIG = "./configs/tcb_net.yaml"
EPOCHS = 10

# 定义实验
experiments = [
    ("fixed_physical", {"channel_center_init": "physical", "channel_center_learnable": False}),
    ("random_learnable", {"channel_center_init": "random", "channel_center_learnable": True}),
    ("kmeans_learnable", {"channel_center_init": "kmeans", "channel_center_learnable": True}),
    ("physical_learnable", {"channel_center_init": "physical", "channel_center_learnable": True}),  # 本文方法
]

def run_exp(name, params):
    output_dir = f"./ablation/channel_{name}"
    cmd = ["python", "scripts/train_tcb_net.py",
           "--data_dir", DATA_DIR,
           "--config", BASE_CONFIG,
           "--epochs", str(EPOCHS),
           "--output_dir", output_dir]
    for k, v in params.items():
        if k == 'channel_center_learnable':
            if v:
                cmd.append("--channel_center_learnable")
            else:
                cmd.append("--no-channel_center_learnable")
        else:
            cmd.extend([f"--{k}", str(v)])
    print(f"Running: {name}")
    subprocess.run(cmd, check=True)
    res_path = os.path.join(output_dir, "tcb_net_cross_time_results.json")
    with open(res_path) as f:
        data = json.load(f)
    return data['f1'], data['auc']

def main():
    results = []
    for name, params in experiments:
        f1, auc = run_exp(name, params)
        results.append({'model': name, 'f1': f1, 'auc': auc})
    for r in results:
        print(r)
    
    os.makedirs("ablation", exist_ok=True)
    with open("ablation/channel_center_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'f1', 'auc'])
        writer.writeheader()
        writer.writerows(results)
    print("结果已保存至 ablation/channel_center_results.csv")

if __name__ == '__main__':
    main()