"""
compare_models_enhanced.py
增强版模型性能对比 - 修复TCB-Net评估和混淆矩阵缺失键
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from tqdm import tqdm
import json
import time
import pandas as pd

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from configs import Config
from models.hybrid_model import SAFNOCNCDetector, FNOSmallSampleDetector
from models.tcb_net import TCB_Net

# 导入physical_fno_detector模型
try:
    from models.physical_fno_detector import PhysicalFNOAnomalyDetector
    print("✅ 使用完整版物理FNO检测器")
except ImportError:
    try:
        from models.physical_fno_detector_simple import PhysicalFNOAnomalyDetector
        print("✅ 使用简化版物理FNO检测器")
    except ImportError:
        class PhysicalFNOAnomalyDetector(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3 * 128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.network(x).squeeze(-1)
        print("✅ 使用临时简单模型")


class ModelComparator:
    """增强版模型性能对比器 - 支持每个模型独立阈值优化"""
    
    def __init__(self, config_path, data_dir=None, baseline_dir=None, device='cuda'):
        self.config = Config.from_yaml(config_path)
        if data_dir is not None:
            self.config.logging.data_dir = data_dir
            self.experiment_type = os.path.basename(data_dir.rstrip('/\\'))
        else:
            self.experiment_type = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_dir = self.config.logging.data_dir
        self.baseline_dir = baseline_dir

        # 创建按实验类型命名的结果子目录
        if self.experiment_type:
            self.result_dir = os.path.join('./results', self.experiment_type)
        else:
            self.result_dir = './results'
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"📂 结果保存目录: {self.result_dir}")
        print(f"📱 使用设备: {self.device}")
        
        # 加载数据
        self.load_test_data()
        self.load_val_data()
        
        # 存储所有模型结果
        self.all_results = {}
    
    def load_test_data(self):
        """加载测试数据"""
        print("📂 加载测试数据...")
        X_test = np.load(os.path.join(self.data_dir, 'X_test.npy'))
        test_labels_path = os.path.join(self.data_dir, 'test_labels.npy')
        if os.path.exists(test_labels_path):
            y_test = np.load(test_labels_path)
            print(f"  测试集: {len(X_test)} 序列, 异常率: {y_test.mean():.2%}")
        else:
            y_test = np.zeros(len(X_test))
            print(f"  测试集: {len(X_test)} 序列 (无标签)")
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test)
        self.test_loader = DataLoader(
            TensorDataset(self.X_test, self.y_test),
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
    
    def load_val_data(self):
        """加载验证集用于阈值优化"""
        X_val = np.load(os.path.join(self.data_dir, 'X_val.npy'))
        val_labels_path = os.path.join(self.data_dir, 'val_labels.npy')
        if os.path.exists(val_labels_path):
            y_val = np.load(val_labels_path)
            print(f"  验证集: {len(X_val)} 序列, 异常率: {y_val.mean():.2%}")
        else:
            y_val = np.zeros(len(X_val))
            print(f"  验证集: {len(X_val)} 序列 (无标签)")
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.FloatTensor(y_val)
        self.val_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.config.training.batch_size,
            shuffle=False
        )
    
    def create_model(self, model_type='original'):
        """创建模型"""
        if model_type == 'original':
            print("  创建 SA-FNO 模型")
            small_sample_mode = getattr(self.config.training, 'small_sample_mode', False)
            if small_sample_mode:
                model = FNOSmallSampleDetector(self.config)
            else:
                model = SAFNOCNCDetector(self.config)
        elif model_type == 'new':
            print("  创建 PhysicalFNO 模型")
            model = PhysicalFNOAnomalyDetector(self.config)
        elif model_type == 'tcb_net':
            print("  创建 TCB-Net 模型")
            model = TCB_Net(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    总参数: {total_params:,}, 可训练: {trainable_params:,}")
        return model.to(self.device)
    
    def load_pretrained(self, model, model_type='original'):
        """加载预训练权重"""
        model_dir = self.config.logging.model_dir
        if model_type == 'original':
            candidates = ['sa_fno_best.pth', 'cnc_binary_best.pth', 'duet_best.pth']
        elif model_type == 'new':
            candidates = ['physical_fno_best.pth']
        elif model_type == 'tcb_net':
            candidates = ['tcb_net_best.pth']
        else:
            candidates = []

        for name in candidates:
            model_path = os.path.join(model_dir, name)
            if os.path.exists(model_path):
                print(f"  找到预训练权重: {model_path}")
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                    if model_type == 'tcb_net':
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        model.load_state_dict(state_dict)
                    print(f"  ✅ 成功加载权重")
                    return model
                except Exception as e:
                    print(f"  ⚠️ 加载失败: {e}")
        print(f"  ⚠️ 未找到预训练权重，使用随机初始化")
        return model
    
    def get_probs(self, model, loader):
        """获取模型预测概率（兼容不同输出形状和输入形状）"""
        model.eval()
        all_probs = []
        is_tcb = isinstance(model, TCB_Net)
        with torch.no_grad():
            for X_batch, _ in loader:
                if is_tcb:
                    # TCB-Net 需要 (batch, channels, time)
                    X_batch = X_batch.permute(0, 2, 1).contiguous()
                X_batch = X_batch.to(self.device)
                logits = model(X_batch)
                # 处理输出形状
                if logits.dim() == 1:
                    probs = torch.sigmoid(logits)
                elif logits.shape[-1] == 1:
                    probs = torch.sigmoid(logits).squeeze(-1)
                else:
                    probs = torch.sigmoid(logits)[:, 1] if logits.shape[-1] == 2 else torch.sigmoid(logits).squeeze(-1)
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_probs)
    
    def find_best_threshold(self, probs, labels):
        """基于验证集精确率-召回率曲线寻找最佳阈值（最大化F1）"""
        if len(np.unique(labels)) < 2:
            return 0.5
        precisions, recalls, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])  # 排除最后一个阈值点
        return thresholds[best_idx]
    
    def evaluate_model(self, model, model_name, use_val_threshold=True):
        """
        评估模型，如果 use_val_threshold=True 则使用验证集优化阈值
        """
        # 获取验证集概率并优化阈值
        if use_val_threshold and self.val_loader is not None and hasattr(self, 'y_val'):
            val_probs = self.get_probs(model, self.val_loader)
            val_labels = self.y_val.numpy()
            best_thresh = self.find_best_threshold(val_probs, val_labels)
            print(f"  {model_name} 最佳阈值: {best_thresh:.4f}")
        else:
            best_thresh = 0.5
            print(f"  {model_name} 使用默认阈值: 0.5")
        
        # 测试集推理
        test_probs = self.get_probs(model, self.test_loader)
        preds = (test_probs >= best_thresh).astype(int)
        test_labels = self.y_test.numpy()
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(test_labels, preds),
            'precision': precision_score(test_labels, preds, zero_division=0),
            'recall': recall_score(test_labels, preds, zero_division=0),
            'f1': f1_score(test_labels, preds, zero_division=0),
            'best_threshold': best_thresh,
        }
        if len(np.unique(test_labels)) > 1:
            metrics['auc'] = roc_auc_score(test_labels, test_probs)
        else:
            metrics['auc'] = 0.5
        
        cm = confusion_matrix(test_labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 测量推理时间（取10个batch平均）
        model.eval()
        inference_times = []
        with torch.no_grad():
            for i, (X_batch, _) in enumerate(self.test_loader):
                if i >= 10:
                    break
                X_batch_input = X_batch.permute(0, 2, 1).contiguous() if isinstance(model, TCB_Net) else X_batch
                X_batch_input = X_batch_input.to(self.device)
                start = time.time()
                _ = model(X_batch_input)
                inference_times.append(time.time() - start)
        metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000 if inference_times else 0
        
        return metrics

    def load_baseline_results(self, baseline_dir):
        """从指定目录读取基线模型结果 JSON，适配实验类型"""
        if self.experiment_type is None:
            print("⚠️ 无法确定实验类型，跳过基线加载")
            return
        model_files = {
            'RandomForest': os.path.join(baseline_dir, f'optimized_model_{self.experiment_type}', 'results_final.json'),
            'TimesNet': os.path.join(baseline_dir, f'timesnet_{self.experiment_type}', 'results.json'),
            'iTransformer': os.path.join(baseline_dir, f'itransformer_{self.experiment_type}', 'results.json'),
            'WeightedKNN': os.path.join(baseline_dir, f'weighted_knn_{self.experiment_type}', 'weighted_knn_results.json')
        }
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                results = {
                    'precision': data.get('precision', 0),
                    'recall': data.get('recall', 0),
                    'f1': data.get('f1', 0),
                    'auc': data.get('auc', 0),
                    'avg_inference_time_ms': data.get('avg_inference_time_ms', 0),
                    'best_threshold': data.get('threshold', 0.5)
                }
                # 处理混淆矩阵
                if 'confusion_matrix' in data:
                    results['confusion_matrix'] = data['confusion_matrix']
                else:
                    # 尝试从 accuracy 估算一个默认矩阵（避免绘图崩溃）
                    # 实际最好让模型自己生成，这里提供一个占位符
                    results['confusion_matrix'] = [[0, 0], [0, 0]]
                cm = results['confusion_matrix']
                if cm and len(cm) == 2 and len(cm[0]) == 2:
                    tn, fp = cm[0]
                    fn, tp = cm[1]
                    total = tn + fp + fn + tp
                    results['accuracy'] = (tn + tp) / total if total > 0 else 0
                else:
                    results['accuracy'] = data.get('accuracy', 0)
                self.all_results[model_name] = results
                print(f"✅ 加载基线模型 {model_name} 结果")
            else:
                print(f"⚠️ 未找到基线模型结果: {file_path}")

    def compare_all(self):
        """对比所有模型"""
        print("\n" + "="*60)
        print("完整 SOTA 模型对比")
        print("="*60)
        
        if self.baseline_dir:
            self.load_baseline_results(self.baseline_dir)

        # 评估 SA-FNO
        print(f"\n📊 评估 SA-FNO")
        original_model = self.create_model('original')
        original_model = self.load_pretrained(original_model, 'original')
        self.all_results['SA-FNO'] = self.evaluate_model(original_model, 'SA-FNO', use_val_threshold=True)
        
        # 评估 PhysicalFNO
        print(f"\n📊 评估 PhysicalFNO")
        new_model = self.create_model('new')
        new_model = self.load_pretrained(new_model, 'new')
        self.all_results['PhysicalFNO'] = self.evaluate_model(new_model, 'PhysicalFNO', use_val_threshold=True)
        
        # 评估 TCB-Net
        print("\n📊 评估 TCB-Net")
        tcb_model = self.create_model('tcb_net')
        tcb_model = self.load_pretrained(tcb_model, 'tcb_net')
        # 直接使用验证集优化阈值，不再手动指定
        self.all_results['TCB-Net'] = self.evaluate_model(tcb_model, 'TCB-Net', use_val_threshold=True)

        # 打印结果表格
        self.print_results_table()
        
        # 生成所有独立图表
        self.plot_f1_ranking()
        self.plot_radar_chart()
        self.plot_recall_precision_scatter()
        self.plot_improvement_bar()
        self.plot_inference_time()
        self.plot_confusion_matrices()
        self.plot_metrics_barh()
        
        # 保存结果
        self.save_results()
        
        return self.all_results
    
    def print_results_table(self):
        """打印结果表格"""
        print("\n" + "="*90)
        print("测试集结果汇总")
        print("="*90)
        print(f"{'模型':<20} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'AUC':<8} {'推理时间(ms)':<12}")
        print("-"*90)
        
        sorted_models = sorted(self.all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for name, results in sorted_models:
            print(f"{name:<20} {results.get('accuracy', 0):<10.2%} "
                  f"{results['precision']:<10.2%} {results['recall']:<10.2%} "
                  f"{results['f1']:<10.2%} {results.get('auc', 0):<8.2%} "
                  f"{results.get('avg_inference_time_ms', 0):<12.2f}")
    
    def plot_f1_ranking(self):
        """绘制F1分数排名图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_names = sorted(self.all_results.keys(), key=lambda x: self.all_results[x]['f1'], reverse=True)
        f1_values = [self.all_results[name]['f1'] for name in sorted_names]
        
        colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else '#4CAF50' for i in range(len(sorted_names))]
        bars = ax.bar(sorted_names, f1_values, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylabel('F1分数', fontsize=12)
        ax.set_title(f'{self.experiment_type} 各模型F1分数排名', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='基准线 (0.5)')
        for bar, val in zip(bars, f1_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'f1_ranking.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 F1排名图保存至: {save_path}")
    
    def plot_radar_chart(self):
        """绘制雷达图（前6个模型）"""
        sorted_names = sorted(self.all_results.keys(), key=lambda x: self.all_results[x]['f1'], reverse=True)
        top_models = sorted_names[:min(6, len(sorted_names))]
        metrics_radar = ['F1', '召回率', '精确率', '准确率', 'AUC']
        angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        for name in top_models:
            values = [
                self.all_results[name]['f1'],
                self.all_results[name]['recall'],
                self.all_results[name]['precision'],
                self.all_results[name].get('accuracy', 0),
                self.all_results[name].get('auc', 0)
            ]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=name, alpha=0.7)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_radar, fontsize=9)
        ax.set_ylim([0, 1])
        ax.set_title(f'{self.experiment_type} Top模型雷达图', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'radar_chart.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 雷达图保存至: {save_path}")
    
    def plot_recall_precision_scatter(self):
        """绘制召回率 vs 精确率散点图"""
        fig, ax = plt.subplots(figsize=(8, 6))
        names = list(self.all_results.keys())
        recalls = [self.all_results[n]['recall'] for n in names]
        precisions = [self.all_results[n]['precision'] for n in names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        for i, name in enumerate(names):
            ax.scatter(recalls[i], precisions[i], s=150, c=[colors[i]], alpha=0.6, edgecolors='black', linewidth=1)
            ax.annotate(name, (recalls[i], precisions[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)
        ax.set_xlabel('召回率', fontsize=12)
        ax.set_ylabel('精确率', fontsize=12)
        ax.set_title(f'{self.experiment_type} 召回率 vs 精确率', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'recall_precision_scatter.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 散点图保存至: {save_path}")
    
    def plot_improvement_bar(self):
        """绘制相对于最佳模型的改进幅度图（以最佳模型为基准）"""
        sorted_names = sorted(self.all_results.keys(), key=lambda x: self.all_results[x]['f1'], reverse=True)
        if not sorted_names:
            return
        best_name = sorted_names[0]
        best_f1 = self.all_results[best_name]['f1']
        other_models = sorted_names[1:]
        improvements = [(self.all_results[n]['f1'] - best_f1) * 100 for n in other_models]
        if not improvements:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_imp = ['green' if imp < 0 else 'red' for imp in improvements]
        bars = ax.barh(other_models, improvements, color=colors_imp, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('F1分数差异 (%)', fontsize=12)
        ax.set_title(f'{self.experiment_type} 模型对比 (相对于最佳模型 {best_name})', fontsize=14, fontweight='bold')
        for bar, imp in zip(bars, improvements):
            ax.text(imp + (1 if imp >= 0 else -3), bar.get_y() + bar.get_height()/2, f'{imp:+.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'improvement_bar.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 改进幅度图保存至: {save_path}")
    
    def plot_inference_time(self):
        """绘制推理时间对比图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_names = sorted(self.all_results.keys(), key=lambda x: self.all_results[x].get('avg_inference_time_ms', 0))
        times = [self.all_results[name].get('avg_inference_time_ms', 0) for name in sorted_names]
        times = [t if t > 0 else 0.1 for t in times]
        colors = ['#2E86AB' if 'PhysicalFNO' in n else '#6A994E' for n in sorted_names]
        bars = ax.barh(sorted_names, times, color=colors, alpha=0.8)
        ax.set_xlabel('推理时间 (ms)', fontsize=12)
        ax.set_title(f'{self.experiment_type} 模型推理速度对比', fontsize=14, fontweight='bold')
        for bar, t in zip(bars, times):
            ax.text(t + 0.1, bar.get_y() + bar.get_height()/2, f'{t:.2f} ms', va='center', fontsize=9)
        ax.axvline(x=min(times), color='green', linestyle='--', alpha=0.7, label=f'最快: {min(times):.2f} ms')
        ax.legend()
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'inference_time.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 推理时间图保存至: {save_path}")
    
    def plot_confusion_matrices(self):
        """绘制混淆矩阵对比（多个子图）"""
        model_names = ['SA-FNO', 'PhysicalFNO', 'TCB-Net', 'iTransformer', 'TimesNet', 'RandomForest', 'WeightedKNN']
        valid_names = [n for n in model_names if n in self.all_results]
        if not valid_names:
            return
        n = len(valid_names)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if rows*cols > 1 else [axes]
        for idx, name in enumerate(valid_names):
            cm = np.array(self.all_results[name].get('confusion_matrix', [[0,0],[0,0]]))
            if cm.shape == (2, 2):
                ax = axes[idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['预测正常', '预测异常'],
                           yticklabels=['真实正常', '真实异常'],
                           ax=ax, cbar=False)
                ax.set_title(f'{name}', fontsize=10)
                ax.set_xlabel('预测', fontsize=9)
                ax.set_ylabel('真实', fontsize=9)
        for idx in range(len(valid_names), len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle(f'{self.experiment_type} 混淆矩阵对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 混淆矩阵图保存至: {save_path}")
    
    def plot_metrics_barh(self):
        """绘制横向柱状图展示多个指标"""
        sorted_names = sorted(self.all_results.keys(), key=lambda x: self.all_results[x]['f1'], reverse=True)
        metrics = ['f1', 'recall', 'precision', 'accuracy']
        metric_names = ['F1', '召回率', '精确率', '准确率']
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(sorted_names))
        bar_width = 0.2
        for i, (mname, mkey) in enumerate(zip(metric_names, metrics)):
            values = [self.all_results[name].get(mkey, 0) for name in sorted_names]
            offset = (i - len(metrics)/2) * bar_width + bar_width/2
            ax.barh(y + offset, values, bar_width, label=mname, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('分数', fontsize=12)
        ax.set_title(f'{self.experiment_type} 多指标对比', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        plt.tight_layout()
        save_path = os.path.join(self.result_dir, 'metrics_barh.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 多指标柱状图保存至: {save_path}")
    
    def save_results(self):
        """保存结果到 JSON 和 Markdown 报告"""
        results_path = os.path.join(self.result_dir, 'all_models_comparison.json')
        with open(results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
        print(f"💾 对比结果保存至: {results_path}")
        
        # 生成 Markdown 报告
        self.generate_markdown_report()
    
    def generate_markdown_report(self):
        """生成 Markdown 报告"""
        report = f"""# 模型性能对比报告 ({self.experiment_type})

## 测试数据信息
- 测试集大小: {len(self.X_test)} 序列
- 异常率: {self.y_test.mean():.2%}

## 性能指标对比

| 排名 | 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC | 推理时间(ms) |
|:----:|------|--------|--------|--------|--------|-----|--------------|
"""
        sorted_models = sorted(self.all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        for i, (name, results) in enumerate(sorted_models, 1):
            medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}"
            report += f"| {medal} | {name} | {results.get('accuracy', 0):.2%} | {results['precision']:.2%} | {results['recall']:.2%} | {results['f1']:.2%} | {results.get('auc', 0):.2%} | {results.get('avg_inference_time_ms', 0):.2f} |\n"
        
        # 添加关键发现
        best_model = sorted_models[0]
        report += f"\n## 关键发现\n\n### 🏆 最佳模型\n- **{best_model[0]}** 获得最高F1分数 ({best_model[1]['f1']:.2%})\n"
        
        report_path = os.path.join(self.result_dir, 'comparison_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Markdown报告保存至: {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='增强版模型性能对比工具')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='预处理数据目录（包含 X_test.npy 等）')
    parser.add_argument('--retrain', action='store_true', help='重新训练新模型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--baseline_dir', type=str, default='../baseline',
                        help='存放基线模型结果 JSON 的根目录（如 ./optimized_model_cross_time）')
    args = parser.parse_args()
    
    config_path = './configs/duet_anomaly.yaml'
    comparator = ModelComparator(config_path, data_dir=args.data_dir, baseline_dir=args.baseline_dir)
    comparator.compare_all()


if __name__ == '__main__':
    main()