"""
可视化 CNC 振动数据的正常和异常波形对比 - 完整分析所有机器和工序
修复版本
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

def load_h5_file(file_path):
    """加载 H5 文件并返回振动数据"""
    try:
        with h5py.File(file_path, 'r') as f:
            # 尝试不同的数据集名称
            possible_keys = ['vibration data', 'vibration_data', 'data', 'acceleration', 'vibration']
            
            for key in possible_keys:
                if key in f:
                    data = f[key][:]
                    return data
            
            # 如果找不到标准名称，获取第一个数据集
            first_key = list(f.keys())[0]
            data = f[first_key][:]
            return data
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def analyze_and_visualize(machine, operation, data_root, output_dir):
    """分析单个机器-工序组合的可视化数据"""
    good_dir = os.path.join(data_root, machine, operation, "good")
    bad_dir = os.path.join(data_root, machine, operation, "bad")
    
    # 检查目录是否存在
    if not os.path.exists(good_dir) or not os.path.exists(bad_dir):
        print(f"  Warning: Directory not found for {machine}/{operation}")
        return None, None, None
    
    good_files = [f for f in os.listdir(good_dir) if f.endswith('.h5')]
    bad_files = [f for f in os.listdir(bad_dir) if f.endswith('.h5')]
    
    if not good_files or not bad_files:
        print(f"  Warning: No files found for {machine}/{operation}")
        return None, None, None
    
    print(f"  Found {len(good_files)} good files and {len(bad_files)} bad files")
    
    # 创建该组合的输出目录
    combo_dir = os.path.join(output_dir, f"{machine}_{operation}")
    os.makedirs(combo_dir, exist_ok=True)
    
    # 存储统计信息
    stats = {
        "machine": machine,
        "operation": operation,
        "total_good_files": len(good_files),
        "total_bad_files": len(bad_files),
        "good_samples": [],
        "bad_samples": [],
        "comparisons": []
    }
    
    # 分析前3个匹配对（如果有）
    analyzed_pairs = 0
    max_pairs_to_analyze = min(3, len(good_files), len(bad_files))
    
    for i in range(max_pairs_to_analyze):
        good_file = os.path.join(good_dir, good_files[i])
        bad_file = os.path.join(bad_dir, bad_files[i])
        
        good_data = load_h5_file(good_file)
        bad_data = load_h5_file(bad_file)
        
        if good_data is None or bad_data is None:
            continue
        
        # 计算统计信息
        good_stats = calculate_statistics(good_data, f"good_{i}")
        bad_stats = calculate_statistics(bad_data, f"bad_{i}")
        
        stats["good_samples"].append(good_stats)
        stats["bad_samples"].append(bad_stats)
        
        # 创建对比图
        comparison_id = f"{machine}_{operation}_pair_{i}"
        fig = create_comparison_figure(good_data, bad_data, good_file, bad_file, comparison_id)
        
        # 保存图像
        save_path = os.path.join(combo_dir, f"{comparison_id}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # 计算差异指标
        diff_metrics = calculate_difference_metrics(good_stats, bad_stats)
        comparison_info = {
            "pair_id": i,
            "good_file": good_files[i],
            "bad_file": bad_files[i],
            "good_shape": str(good_data.shape),
            "bad_shape": str(bad_data.shape),
            "difference_metrics": diff_metrics
        }
        stats["comparisons"].append(comparison_info)
        
        analyzed_pairs += 1
    
    # 保存统计信息
    stats_path = os.path.join(combo_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats, good_dir, bad_dir

def calculate_statistics(data, label):
    """计算数据的统计信息"""
    stats = {
        "label": label,
        "shape": str(data.shape),
        "total_samples": data.shape[0],
        "x_axis": {
            "mean": float(np.mean(data[:, 0])),
            "std": float(np.std(data[:, 0])),
            "max": float(np.max(data[:, 0])),
            "min": float(np.min(data[:, 0])),
            "max_abs": float(np.max(np.abs(data[:, 0])))
        },
        "y_axis": {
            "mean": float(np.mean(data[:, 1])),
            "std": float(np.std(data[:, 1])),
            "max": float(np.max(data[:, 1])),
            "min": float(np.min(data[:, 1])),
            "max_abs": float(np.max(np.abs(data[:, 1])))
        },
        "z_axis": {
            "mean": float(np.mean(data[:, 2])),
            "std": float(np.std(data[:, 2])),
            "max": float(np.max(data[:, 2])),
            "min": float(np.min(data[:, 2])),
            "max_abs": float(np.max(np.abs(data[:, 2])))
        }
    }
    return stats

def calculate_difference_metrics(good_stats, bad_stats):
    """计算正常和异常数据之间的差异指标"""
    metrics = {}
    
    for axis in ['x_axis', 'y_axis', 'z_axis']:
        good_mean = good_stats[axis]['mean']
        bad_mean = bad_stats[axis]['mean']
        good_std = good_stats[axis]['std']
        bad_std = bad_stats[axis]['std']
        
        metrics[axis] = {
            "mean_difference": abs(bad_mean - good_mean),
            "mean_difference_percentage": abs((bad_mean - good_mean) / good_mean * 100) if good_mean != 0 else float('inf'),
            "std_difference": abs(bad_std - good_std),
            "std_difference_percentage": abs((bad_std - good_std) / good_std * 100) if good_std != 0 else float('inf'),
            "max_abs_ratio": bad_stats[axis]['max_abs'] / good_stats[axis]['max_abs'] if good_stats[axis]['max_abs'] != 0 else float('inf')
        }
    
    return metrics

def create_comparison_figure(good_data, bad_data, good_file, bad_file, title):
    """创建对比可视化图"""
    fig = plt.figure(figsize=(16, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.5])
    
    # 1. 原始波形对比
    axes_waveform = []
    colors = ['blue', 'orange', 'green']
    
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(good_data[:, i], color=colors[i], alpha=0.6, linewidth=0.5, label='Normal')
        ax.set_ylabel(f'Axis {i} (Normal)')
        ax.grid(True, alpha=0.3)
        axes_waveform.append(ax)
        
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(bad_data[:, i], color='red', alpha=0.6, linewidth=0.5, label='Anomalous')
        ax.set_ylabel(f'Axis {i} (Anomalous)')
        ax.grid(True, alpha=0.3)
        axes_waveform.append(ax)
    
    # 2. 统计信息对比
    ax_stats = fig.add_subplot(gs[3, :])
    ax_stats.axis('off')
    
    # 计算并显示统计信息
    stats_text = generate_stats_text(good_data, bad_data, good_file, bad_file)
    ax_stats.text(0.02, 0.5, stats_text, fontsize=9, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Waveform Comparison: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def generate_stats_text(good_data, bad_data, good_file, bad_file):
    """生成统计信息文本"""
    text_lines = [
        f"File: {Path(good_file).name} vs {Path(bad_file).name}",
        "=" * 70,
        f"{'Statistic':<15} {'Normal':<12} {'Anomalous':<12} {'Diff %':<10} {'Visible Diff':<12}",
        "-" * 70
    ]
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        good_mean = np.mean(good_data[:, i])
        bad_mean = np.mean(bad_data[:, i])
        good_std = np.std(good_data[:, i])
        bad_std = np.std(bad_data[:, i])
        good_max_abs = np.max(np.abs(good_data[:, i]))
        bad_max_abs = np.max(np.abs(bad_data[:, i]))
        
        mean_diff_pct = abs((bad_mean - good_mean) / good_mean * 100) if good_mean != 0 else 999
        std_diff_pct = abs((bad_std - good_std) / good_std * 100) if good_std != 0 else 999
        
        mean_visible = "✓" if mean_diff_pct > 10 else "✗"
        std_visible = "✓" if std_diff_pct > 10 else "✗"
        max_visible = "✓" if bad_max_abs > good_max_abs * 1.2 else "✗"
        
        text_lines.extend([
            f"{f'{axis_name}-Mean':<15} {good_mean:<12.4f} {bad_mean:<12.4f} {mean_diff_pct:<10.1f} {mean_visible:<12}",
            f"{f'{axis_name}-Std':<15} {good_std:<12.4f} {bad_std:<12.4f} {std_diff_pct:<10.1f} {std_visible:<12}",
            f"{f'{axis_name}-MaxAbs':<15} {good_max_abs:<12.4f} {bad_max_abs:<12.4f} {bad_max_abs/good_max_abs:<10.2f} {max_visible:<12}",
            "-" * 70
        ])
    
    # 总体判断
    visible_diffs = sum([
        1 for i in range(3) 
        if abs(np.mean(bad_data[:, i]) - np.mean(good_data[:, i])) / abs(np.mean(good_data[:, i])) > 0.1
    ])
    
    text_lines.append(f"\nSummary: {visible_diffs}/3 axes show >10% difference in mean values")
    text_lines.append(f"         {'Easily distinguishable' if visible_diffs >= 2 else 'Hard to distinguish'}")
    
    return "\n".join(text_lines)

def create_summary_report(all_stats, output_dir):
    """创建所有分析的汇总报告"""
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "total_combinations_analyzed": len([s for s in all_stats if s is not None]),
        "combinations": []
    }
    
    for stats in all_stats:
        if stats is None:
            continue
        
        combo_info = {
            "machine": stats["machine"],
            "operation": stats["operation"],
            "good_files": stats["total_good_files"],
            "bad_files": stats["total_bad_files"],
            "analysis_pairs": len(stats["comparisons"]),
            "average_differences": {}
        }
        
        # 计算平均差异 - 修复：先收集所有值，然后再计算平均值
        all_diff_values = {}
        
        for comp in stats["comparisons"]:
            diff_metrics = comp["difference_metrics"]
            for axis in diff_metrics:
                if axis not in all_diff_values:
                    all_diff_values[axis] = {}
                
                for metric in diff_metrics[axis]:
                    if metric not in all_diff_values[axis]:
                        all_diff_values[axis][metric] = []
                    
                    all_diff_values[axis][metric].append(diff_metrics[axis][metric])
        
        # 计算平均值
        for axis in all_diff_values:
            if axis not in combo_info["average_differences"]:
                combo_info["average_differences"][axis] = {}
            
            for metric in all_diff_values[axis]:
                values = all_diff_values[axis][metric]
                valid_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
                if valid_values:
                    combo_info["average_differences"][axis][f"{metric}_avg"] = float(np.mean(valid_values))
        
        summary["combinations"].append(combo_info)
    
    # 保存汇总报告
    summary_path = os.path.join(output_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 创建可视化摘要
    create_summary_visualization(summary, output_dir)
    
    return summary

def create_summary_visualization(summary, output_dir):
    """创建汇总可视化图表"""
    if not summary["combinations"]:
        print("  Warning: No valid combinations to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 文件数量分布
    machine_ops = []
    good_counts = []
    bad_counts = []
    
    for combo in summary["combinations"]:
        machine_ops.append(f"{combo['machine']}_{combo['operation']}")
        good_counts.append(combo["good_files"])
        bad_counts.append(combo["bad_files"])
    
    x_pos = np.arange(len(machine_ops))
    width = 0.35
    
    if len(machine_ops) > 0:
        axes[0, 0].bar(x_pos - width/2, good_counts, width, label='Good', color='blue', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, bad_counts, width, label='Bad', color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Machine-Operation')
        axes[0, 0].set_ylabel('Number of Files')
        axes[0, 0].set_title('File Distribution by Machine-Operation')
        axes[0, 0].legend()
        
        # 旋转x轴标签以避免重叠
        if len(machine_ops) > 10:
            axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(machine_ops, fontsize=8)
    
    # 2. 平均差异热图
    if len(summary["combinations"]) > 0:
        mean_diff_matrix = np.zeros((3, len(summary["combinations"])))  # 3 axes
        
        for i, combo in enumerate(summary["combinations"]):
            for j, axis in enumerate(['x_axis', 'y_axis', 'z_axis']):
                if axis in combo["average_differences"]:
                    key = 'mean_difference_percentage_avg'
                    if key in combo["average_differences"][axis]:
                        mean_diff_matrix[j, i] = combo["average_differences"][axis][key]
        
        if np.any(mean_diff_matrix > 0):
            im = axes[0, 1].imshow(mean_diff_matrix, aspect='auto', cmap='YlOrRd')
            axes[0, 1].set_xlabel('Machine-Operation Index')
            axes[0, 1].set_ylabel('Axis')
            axes[0, 1].set_title('Mean Difference Percentage (%)')
            axes[0, 1].set_yticks([0, 1, 2])
            axes[0, 1].set_yticklabels(['X', 'Y', 'Z'])
            plt.colorbar(im, ax=axes[0, 1])
    
    # 3. 数据可区分性评分
    if len(summary["combinations"]) > 0:
        distinguishable = []
        for combo in summary["combinations"]:
            # 简单评分：如果有超过2个轴的差异>10%，则认为可区分
            scores = []
            for axis in ['x_axis', 'y_axis', 'z_axis']:
                if axis in combo["average_differences"]:
                    key = 'mean_difference_percentage_avg'
                    if key in combo["average_differences"][axis]:
                        scores.append(1 if combo["average_differences"][axis][key] > 10 else 0)
            
            distinguishable.append(sum(scores) if scores else 0)
        
        axes[1, 0].bar(range(len(distinguishable)), distinguishable, color='green', alpha=0.7)
        axes[1, 0].axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Threshold (2/3 axes)')
        axes[1, 0].set_xlabel('Machine-Operation Index')
        axes[1, 0].set_ylabel('Number of Axes with >10% Diff')
        axes[1, 0].set_title('Distinguishability Score')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 3.5])
    
    # 4. 分类统计
    if len(summary["combinations"]) > 0:
        easy_count = sum(1 for d in distinguishable if d >= 2)
        medium_count = sum(1 for d in distinguishable if d == 1)
        hard_count = sum(1 for d in distinguishable if d == 0)
        
        categories = ['Easy (≥2 axes)', 'Medium (1 axis)', 'Hard (0 axes)']
        counts = [easy_count, medium_count, hard_count]
        colors = ['green', 'orange', 'red']
        
        axes[1, 1].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Distinguishability Distribution')
    
    fig.suptitle(f'CNC Vibration Data Analysis Summary\nTotal Combinations: {summary["total_combinations_analyzed"]}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_plot_path = os.path.join(output_dir, "analysis_summary.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    """主函数：分析所有机器和工序"""
    # 设置路径
    data_root = r"D:\code\PythonProjects\CNC_Machining\data"
    output_dir = "cnc_analysis_all"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义要分析的所有组合
    machines = ["M01", "M02", "M03"]
    operations = [f"OP{i:02d}" for i in range(15)]  # OP00到OP14
    
    print("=" * 70)
    print("CNC Vibration Data - Complete Analysis")
    print("=" * 70)
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Machines to analyze: {machines}")
    print(f"Operations to analyze: {operations}")
    print("=" * 70)
    
    all_stats = []
    total_combinations = len(machines) * len(operations)
    analyzed_count = 0
    
    # 遍历所有组合
    for machine in machines:
        for operation in operations:
            analyzed_count += 1
            print(f"\n[{analyzed_count}/{total_combinations}] Analyzing {machine}/{operation}...")
            
            # 分析当前组合
            stats, good_dir, bad_dir = analyze_and_visualize(machine, operation, data_root, output_dir)
            all_stats.append(stats)
            
            if stats is not None:
                print(f"  ✓ Analysis completed. Results saved to {output_dir}/{machine}_{operation}/")
            else:
                print(f"  ✗ No data found or analysis failed.")
    
    # 创建汇总报告
    print("\n" + "=" * 70)
    print("Creating summary report...")
    
    summary = create_summary_report(all_stats, output_dir)
    
    print(f"✓ Summary report saved to {output_dir}/analysis_summary.json")
    print(f"✓ Summary visualization saved to {output_dir}/analysis_summary.png")
    
    # 生成简单文本报告
    generate_text_report(summary, output_dir)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Total combinations analyzed: {summary['total_combinations_analyzed']}/{total_combinations}")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("=" * 70)

def generate_text_report(summary, output_dir):
    """生成文本格式的简单报告"""
    report_lines = [
        "=" * 80,
        "CNC VIBRATION DATA ANALYSIS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        f"\nTotal Combinations Analyzed: {summary['total_combinations_analyzed']}",
        "\n" + "-" * 80
    ]
    
    if not summary["combinations"]:
        report_lines.append("\nNo valid data combinations found.")
        report_lines.append("=" * 80)
        report_path = os.path.join(output_dir, "analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        return
    
    # 按可区分性排序
    combos_with_diff = []
    for combo in summary["combinations"]:
        # 计算可区分性分数
        scores = []
        for axis in ['x_axis', 'y_axis', 'z_axis']:
            if axis in combo.get("average_differences", {}):
                key = 'mean_difference_percentage_avg'
                if key in combo["average_differences"][axis]:
                    scores.append(1 if combo["average_differences"][axis][key] > 10 else 0)
        
        diff_score = sum(scores)
        combos_with_diff.append((combo, diff_score))
    
    # 按分数排序
    combos_with_diff.sort(key=lambda x: x[1], reverse=True)
    
    report_lines.append("\nTOP 5 MOST DISTINGUISHABLE CASES:")
    report_lines.append("-" * 80)
    
    for i, (combo, score) in enumerate(combos_with_diff[:5]):
        report_lines.append(f"{i+1}. {combo['machine']}/{combo['operation']}:")
        report_lines.append(f"   Files: {combo['good_files']} good, {combo['bad_files']} bad")
        report_lines.append(f"   Distinguishability: {score}/3 axes show >10% difference")
        
        if 'average_differences' in combo:
            for axis in ['x_axis', 'y_axis', 'z_axis']:
                if axis in combo['average_differences']:
                    key = 'mean_difference_percentage_avg'
                    if key in combo['average_differences'][axis]:
                        diff = combo['average_differences'][axis][key]
                        report_lines.append(f"   {axis}: {diff:.1f}% difference")
        
        report_lines.append("")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 80)
    report_lines.append("1. Start analysis with highly distinguishable cases (top of the list)")
    report_lines.append("2. Check waveform patterns for anomalies in visible cases")
    report_lines.append("3. For hard cases, consider time-frequency analysis (FFT)")
    report_lines.append("4. Machine-specific patterns may exist - compare same operation across machines")
    
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"✓ Text report saved to {output_dir}/analysis_report.txt")

if __name__ == "__main__":
    main()