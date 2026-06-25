"""
增强版模型性能对比 - 支持 TCB‑Net 与所有基线
生成顶刊风格图表：F1 排名柱状图、雷达图、召回率-精确率散点图、推理时间图、混淆矩阵热图、多指标横向柱状图
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

# 设置全局绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 定义模型名称映射
MODEL_NAMES = {
    'TCB-Net': 'TCB-Net (Ours)',
    'TimesNet': 'TimesNet',
    'iTransformer': 'iTransformer',
    'RandomForest': 'Random Forest',
    'KNN': 'KNN',
    'PatchTST': 'PatchTST',
    'ModernTCN': 'ModernTCN'
}

MODEL_ORDER = ['TCB-Net', 'PatchTST', 'ModernTCN', 'iTransformer', 'TimesNet', 'RandomForest', 'KNN']


class ModelComparator:
    def __init__(self, exp_type='cross_time', baseline_root='../baseline', tcb_dir='./results', version='', verbose=False):
        self.exp_type = exp_type
        self.baseline_root = baseline_root
        self.tcb_dir = tcb_dir
        self.version = version
        self.verbose = verbose  # 控制是否打印详细信息
        self.results = {}

    def load_tcb_results(self):
        json_path = os.path.join(self.tcb_dir, f'tcb_net_{self.exp_type}_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.results['TCB-Net'] = {
                'accuracy': data.get('accuracy', 0),
                'precision': data.get('precision', 0),
                'recall': data.get('recall', 0),
                'f1': data.get('f1', 0),
                'auc': data.get('auc', 0),
                'best_threshold': data.get('threshold', 0.5),
                'confusion_matrix': data.get('confusion_matrix', [[0,0],[0,0]]),
            }
            if self.verbose:
                print(f"Loaded TCB-Net results from {json_path}")
        else:
            if self.verbose:
                print(f"Warning: TCB-Net metrics file not found at {json_path}")
            self.results['TCB-Net'] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0,
                'best_threshold': 0.5, 'confusion_matrix': [[0,0],[0,0]]
            }

    def load_baseline_results(self):
        model_config = {
            'TimesNet': 'timesnet',
            'iTransformer': 'itransformer',
            'RandomForest': 'randomforest',
            'KNN': 'knn',
            'PatchTST': 'patchtst',
            'ModernTCN': 'moderntcn'
        }
        possible_filenames = ['results.json', 'results_final.json']

        for display_name, dir_prefix in model_config.items():
            found = False
            json_data = None
            used_path = None
            for fname in possible_filenames:
                folder_name = f'{dir_prefix}_{self.exp_type}{self.version}'
                path = os.path.join(self.baseline_root, folder_name, fname)
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        json_data = json.load(f)
                    found = True
                    used_path = path
                    break
            if not found:
                continue

            # 提取字段
            acc = json_data.get('accuracy')
            prec = json_data.get('precision')
            rec = json_data.get('recall')
            f1 = json_data.get('f1')
            auc = json_data.get('auc', 0.5)
            threshold = json_data.get('threshold', 0.5)

            # 从混淆矩阵补全缺失指标
            cm = json_data.get('confusion_matrix')
            if cm is not None and len(cm) == 2:
                tn, fp = cm[0]
                fn, tp = cm[1]
                total = tn + fp + fn + tp
                if acc is None and total > 0:
                    acc = (tn + tp) / total
                if prec is None and (tp + fp) > 0:
                    prec = tp / (tp + fp)
                if rec is None and (tp + fn) > 0:
                    rec = tp / (tp + fn)
                if f1 is None and prec is not None and rec is not None and (prec + rec) > 0:
                    f1 = 2 * prec * rec / (prec + rec)

            # 默认值
            acc = acc if acc is not None else 0.0
            prec = prec if prec is not None else 0.0
            rec = rec if rec is not None else 0.0
            f1 = f1 if f1 is not None else 0.0

            metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'best_threshold': threshold,
                'confusion_matrix': cm if cm is not None else [[0,0],[0,0]],
            }
            self.results[display_name] = metrics
            if self.verbose:
                print(f"Loaded {display_name} from {used_path} (F1={f1:.4f}, Rec={rec:.4f})")

    def load_all(self):
        self.load_tcb_results()
        self.load_baseline_results()
        ordered = []
        for name in MODEL_ORDER:
            if name in self.results:
                ordered.append((name, self.results[name]))
            elif self.verbose:
                print(f"Warning: {name} results missing.")
        return ordered

    # ---------- 绘图函数 ----------
    def plot_f1_ranking(self, save_path=None):
        ordered = self.load_all()
        names = [n for n, _ in ordered]
        f1s = [r['f1'] for _, r in ordered]
        colors = ['#d62728' if n == 'TCB-Net' else '#2c7bb6' for n in names]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, f1s, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
        ax.set_ylabel('F1 分数', fontsize=12)
        ax.set_title(f'{self.exp_type} 实验模型 F1 分数对比', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        plt.tight_layout(pad=1.5)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"F1 ranking plot saved: {save_path}")

    def plot_radar_chart(self, save_path=None):
        ordered = self.load_all()
        top_models = ordered[:6]  # 取前6个
        metrics = ['F1', '召回率', '精确率', '准确率', 'AUC']
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for name, r in top_models:
            values = [r['f1'], r['recall'], r['precision'], r['accuracy'], r.get('auc', 0)]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=MODEL_NAMES.get(name, name), alpha=0.8)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
        ax.set_title(f'{self.exp_type} 实验模型性能雷达图', fontsize=14, fontweight='bold', ha='center')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Radar chart saved: {save_path}")

    def plot_recall_precision_scatter(self, save_path=None):
        ordered = self.load_all()
        names = [n for n, _ in ordered]
        recalls = [r['recall'] for _, r in ordered]
        precisions = [r['precision'] for _, r in ordered]
        f1s = [r['f1'] for _, r in ordered]
        sizes = [f * 500 for f in f1s]

        fig, ax = plt.subplots(figsize=(8, 6))
        norm = plt.Normalize(vmin=min(f1s), vmax=max(f1s))
        scatter = ax.scatter(recalls, precisions, s=sizes, c=f1s, cmap='viridis',
                            norm=norm, alpha=0.7, edgecolors='black', linewidth=0.5)

        # 为每个模型分配编号（从1开始）
        model_numbers = list(range(1, len(names) + 1))
        # 在圆圈中心添加数字（无底色）
        for (x, y), num in zip(zip(recalls, precisions), model_numbers):
            ax.text(x, y, str(num), ha='center', va='center', fontsize=9,
                    color='black', fontweight='bold')

        # 左上角图例（数字->模型），简洁无背景框
        legend_text = '\n'.join([f"{num}: {MODEL_NAMES.get(name, name)}" for num, name in zip(model_numbers, names)])
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlim(0.2, 1.0)
        ax.set_ylim(0.2, 1.0)
        line_x = np.linspace(0.2, 1.0, 100)
        ax.plot(line_x, line_x, 'r--', linewidth=1.5, label='y=x')
        ax.set_xlabel('召回率', fontsize=12)
        ax.set_ylabel('精确率', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        cbar = plt.colorbar(scatter, ax=ax, label='F1 分数')
        ax.set_title(f'{self.exp_type} 实验召回率 vs 精确率', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved: {save_path}")

    def plot_confusion_matrices(self, save_dir=None):
        """每个模型单独保存混淆矩阵图，标注(a)~(g)，使用中文标签"""
        ordered = self.load_all()
        if not ordered:
            return
        if save_dir is None:
            save_dir = '.'
        os.makedirs(save_dir, exist_ok=True)
        # 子图标签字母
        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
        for idx, (name, r) in enumerate(ordered):
            cm = np.array(r['confusion_matrix'])
            if cm.shape != (2,2):
                continue
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                        xticklabels=['预测正常', '预测异常'], yticklabels=['真实正常', '真实异常'])
            ax.set_title(f'{MODEL_NAMES.get(name, name)} 混淆矩阵', fontsize=11)
            # 在正下方添加标注
            plt.figtext(0.5, -0.05, labels[idx], ha='center', fontsize=12, fontweight='bold')
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部文字留空间
            # 构建文件名
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            fname = f'confusion_matrix_{idx+1:02d}_{safe_name}.png'
            save_path = os.path.join(save_dir, fname)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved: {save_path}")

    def plot_metrics_barh(self, save_path=None):
        ordered = self.load_all()
        names = [n for n, _ in ordered]
        metrics = ['f1', 'recall', 'precision', 'accuracy']
        metric_labels = ['F1', '召回率', '精确率', '准确率']
        x = np.arange(len(names))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (m, label) in enumerate(zip(metrics, metric_labels)):
            values = [r[m] for _, r in ordered]
            offset = (i - len(metrics)/2) * width + width/2
            ax.bar(x + offset, values, width, label=label, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('分数', fontsize=12)
        ax.set_ylim(0, 1)
        # 图例移到右下角（柱状图外）
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_title(f'{self.exp_type} 实验多指标对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Multi-metric bar chart saved: {save_path}")

    def generate_all_plots(self, output_dir='./comparison_plots'):
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{self.exp_type}{self.version}"
        # F1 ranking
        self.plot_f1_ranking(os.path.join(output_dir, f'f1_ranking_{base_name}.png'))
        # Radar chart
        self.plot_radar_chart(os.path.join(output_dir, f'radar_chart_{base_name}.png'))
        # Scatter
        self.plot_recall_precision_scatter(os.path.join(output_dir, f'recall_precision_{base_name}.png'))
        # Confusion matrices - separate images
        self.plot_confusion_matrices(save_dir=output_dir)
        # Multi-metric bar chart
        self.plot_metrics_barh(os.path.join(output_dir, f'metrics_barh_{base_name}.png'))
        print(f"All plots saved to {output_dir}")

    def print_table(self):
        ordered = self.load_all()
        print("\n" + "="*70)
        print(f"Model Performance Summary on {self.exp_type}{self.version}")
        print("="*70)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10} ")
        print("-"*70)
        for name, r in ordered:
            display_name = MODEL_NAMES.get(name, name)
            print(f"{display_name:<20} {r.get('accuracy',0):<10.3f} {r['precision']:<10.3f} "
                  f"{r['recall']:<10.3f} {r['f1']:<10.3f} {r.get('auc',0):<10.3f} ")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Model comparison with enhanced plots')
    parser.add_argument('--exp_type', type=str, default='cross_time', 
                        choices=['cross_time', 'cross_machine', 'cross_process'])
    parser.add_argument('--baseline_root', type=str, default='../baseline', 
                        help='Root directory of baseline results')
    parser.add_argument('--output_dir', type=str, default='./comparison_plots', 
                        help='Directory to save plots')
    parser.add_argument('--tcb_dir', type=str, default='./results', 
                        help='Directory of TCB-Net metrics JSON')
    parser.add_argument('--version', type=str, default='', 
                        help='Version suffix for baseline folders (e.g., _v2)')
    parser.add_argument('--verbose', action='store_true', 
                        help='Print detailed loading information')
    args = parser.parse_args()

    comparator = ModelComparator(
        exp_type=args.exp_type, 
        baseline_root=args.baseline_root, 
        tcb_dir=args.tcb_dir,
        version=args.version,
        verbose=args.verbose
    )
    comparator.print_table()
    comparator.generate_all_plots(output_dir=args.output_dir)


if __name__ == '__main__':
    main()