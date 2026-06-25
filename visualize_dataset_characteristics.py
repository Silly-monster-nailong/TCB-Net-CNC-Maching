"""
CNC 数据集特性可视化脚本
1. 类别不平衡：展示三台机器在六个时间段的正常/故障样本比例（堆叠柱状图）
2. 时序分布漂移与轴偏移：以 M01_OP05 为例，自动选择最早和最晚有数据的时间段进行对比
   - 三轴波形对比
   - 概率密度分布直方图
   - 输出统计信息（均值、标准差、最大幅值等）
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据根目录（请根据实际路径修改）
DATA_ROOT = r"D:\code\PythonProjects\CNC_Machining\data"

# 完整时间段列表（按时间顺序）
PERIODS = ['Feb_2019', 'Aug_2019', 'Feb_2020', 'Aug_2020', 'Feb_2021', 'Aug_2021']

# 机器列表
MACHINES = ['M01', 'M02', 'M03']

# 所有工序（用于统计样本数）
ALL_OPS = [f"OP{i:02d}" for i in range(15)]

# ------------------------- 辅助函数 -------------------------
def load_h5_file(file_path):
    """加载 h5 文件中的振动数据，返回 (n_points, 3) 数组"""
    try:
        with h5py.File(file_path, 'r') as f:
            # 常见数据集名称
            for key in ['vibration_data', 'vibration data', 'data', 'acceleration']:
                if key in f:
                    return f[key][:]
            # 否则取第一个
            first_key = list(f.keys())[0]
            return f[first_key][:]
    except Exception as e:
        print(f"  加载失败 {file_path}: {e}")
        return None

def count_samples_by_period(machine):
    """统计某台机器在六个时间段内的正常/故障样本（文件）数量"""
    counts = {period: {'good': 0, 'bad': 0} for period in PERIODS}
    for op in ALL_OPS:
        for period in PERIODS:
            # 正常样本
            good_pattern = os.path.join(DATA_ROOT, machine, op, 'good', f'*{period}*.h5')
            good_files = glob.glob(good_pattern)
            counts[period]['good'] += len(good_files)
            # 故障样本
            bad_pattern = os.path.join(DATA_ROOT, machine, op, 'bad', f'*{period}*.h5')
            bad_files = glob.glob(bad_pattern)
            counts[period]['bad'] += len(bad_files)
    return counts

def compute_statistics(data, axis_names=['X', 'Y', 'Z']):
    """计算三轴的统计量，返回字典"""
    if data is None:
        return None
    stats = {}
    for i, name in enumerate(axis_names):
        col = data[:, i]
        stats[name] = {
            'mean': np.mean(col),
            'std': np.std(col),
            'max_abs': np.max(np.abs(col)),
            'min': np.min(col),
            'max': np.max(col)
        }
    return stats

# ------------------------- 1. 类别不平衡可视化 -------------------------
def plot_class_imbalance(output_dir='./dataset_analysis'):
    """合并为一张图：分组堆叠柱状图，用不同色系区分机器，深浅区分正常/故障"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有时间段、三台机器的正常与故障计数
    data = {machine: count_samples_by_period(machine) for machine in MACHINES}
    
    # 定义色系：每个机器的正常颜色、故障颜色
    color_scheme = {
        'M01': {'good': '#1f77b4', 'bad': '#aec7e8'},   # 深蓝, 浅蓝
        'M02': {'good': '#2ca02c', 'bad': '#98df8a'},   # 深绿, 浅绿
        'M03': {'good': '#ff7f0e', 'bad': '#ffbb78'}    # 橙, 浅橙
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.25
    x = np.arange(len(PERIODS))
    
    for i, machine in enumerate(MACHINES):
        good_counts = [data[machine][p]['good'] for p in PERIODS]
        bad_counts  = [data[machine][p]['bad'] for p in PERIODS]
        offset = (i - 1) * bar_width
        
        # 正常（深色）
        ax.bar(x + offset, good_counts, width=bar_width, 
               label=f'{machine} 正常', color=color_scheme[machine]['good'], 
               alpha=0.9, edgecolor='black', linewidth=0.5)
        # 故障（浅色），堆叠在正常之上
        ax.bar(x + offset, bad_counts, bottom=good_counts, width=bar_width,
               label=f'{machine} 故障', color=color_scheme[machine]['bad'],
               alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # 添加总数标签（可选）
        for j, (g, b) in enumerate(zip(good_counts, bad_counts)):
            total = g + b
            if total > 0:
                ax.text(x[j] + offset, total + 5, str(total), ha='center', va='bottom', fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('样本数量', fontsize=22)
    ax.set_xlabel('时间段', fontsize=22)
    ax.set_title('各机床在不同时间段的样本数量分布（正常 vs 故障）', fontsize=24, fontweight='bold')
    
    # 图例去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=15, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'class_imbalance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 合并后的类别不平衡图已保存: {save_path}")

# ------------------------- 2. 时序分布漂移与轴偏移可视化 -------------------------
def find_available_files(machine, operation, label):
    """查找某个标签（good/bad）在六个时间段中有哪些数据，返回最早和最晚的时间段及数据"""
    available = []
    for period in PERIODS:
        pattern = os.path.join(DATA_ROOT, machine, operation, label, f'*{period}*.h5')
        files = glob.glob(pattern)
        if files:
            data = load_h5_file(files[0])
            if data is not None:
                available.append((period, data))
    if not available:
        return None, None, None, None
    # 最早（按PERIODS顺序的第一个）和最晚（最后一个）
    earliest_period, earliest_data = available[0]
    latest_period, latest_data = available[-1]
    return earliest_period, earliest_data, latest_period, latest_data

def plot_drift_and_bias(output_dir='./dataset_analysis'):
    """以 M01_OP05 为例，自动寻找可用的最早/最晚时间段展示时序漂移与轴偏移（整图统一右上角图例）"""
    os.makedirs(output_dir, exist_ok=True)
    machine = 'M01'
    operation = 'OP05'

    good_early_period, good_early_data, good_late_period, good_late_data = find_available_files(machine, operation, 'good')
    bad_early_period, bad_early_data, bad_late_period, bad_late_data = find_available_files(machine, operation, 'bad')

    if good_early_data is None or good_late_data is None:
        print(f"❌ 无法找到足够的 good 数据（需要至少两个不同时间段的正常样本）")
        return

    n_points = min(2000, good_early_data.shape[0], good_late_data.shape[0])
    good_early = good_early_data[:n_points]
    good_late = good_late_data[:n_points]

    has_bad = (bad_early_data is not None and bad_late_data is not None)
    if has_bad:
        n_points_bad = min(2000, bad_early_data.shape[0], bad_late_data.shape[0])
        bad_early = bad_early_data[:n_points_bad]
        bad_late = bad_late_data[:n_points_bad]

    # 计算统计量
    s_early = compute_statistics(good_early)
    s_late = compute_statistics(good_late)
    if has_bad:
        s_bad_early = compute_statistics(bad_early)
        s_bad_late = compute_statistics(bad_late)

    # 打印统计信息（表格形式，包含均值、标准差、最大绝对值及漂移变化）
    print("\n" + "=" * 90)
    print(f"【{machine}_{operation} 时序漂移分析】")
    print("=" * 90)
    
    # 早期健康 vs 晚期健康
    print("【早期健康 vs 晚期健康】")
    print(f"{'轴':<4} {'状态':<10} {'均值':>12} {'标准差':>12} {'最大绝对值':>12}")
    print("-" * 56)
    for ax, idx in zip(['X','Y','Z'], [0,1,2]):
        early_mean = s_early[ax]['mean']
        early_std  = s_early[ax]['std']
        early_max  = s_early[ax]['max_abs']
        late_mean  = s_late[ax]['mean']
        late_std   = s_late[ax]['std']
        late_max   = s_late[ax]['max_abs']
        print(f"{ax:<4} {'早期':<10} {early_mean:>12.2f} {early_std:>12.2f} {early_max:>12.2f}")
        print(f"{ax:<4} {'晚期':<10} {late_mean:>12.2f} {late_std:>12.2f} {late_max:>12.2f}")
        mean_diff = late_mean - early_mean
        std_diff  = late_std - early_std
        max_ratio = late_max / early_max if early_max != 0 else float('inf')
        print(f"{'':<4} {'漂移':<10} {mean_diff:>+12.2f} {std_diff:>+12.2f} {max_ratio:>12.2f}x")
        print("-" * 56)
    
    if has_bad:
        print("\n【早期故障 vs 晚期故障】")
        print(f"{'轴':<4} {'状态':<10} {'均值':>12} {'标准差':>12} {'最大绝对值':>12}")
        print("-" * 56)
        for ax, idx in zip(['X','Y','Z'], [0,1,2]):
            early_mean = s_bad_early[ax]['mean']
            early_std  = s_bad_early[ax]['std']
            early_max  = s_bad_early[ax]['max_abs']
            late_mean  = s_bad_late[ax]['mean']
            late_std   = s_bad_late[ax]['std']
            late_max   = s_bad_late[ax]['max_abs']
            print(f"{ax:<4} {'早期':<10} {early_mean:>12.2f} {early_std:>12.2f} {early_max:>12.2f}")
            print(f"{ax:<4} {'晚期':<10} {late_mean:>12.2f} {late_std:>12.2f} {late_max:>12.2f}")
            mean_diff = late_mean - early_mean
            std_diff  = late_std - early_std
            max_ratio = late_max / early_max if early_max != 0 else float('inf')
            print(f"{'':<4} {'漂移':<10} {mean_diff:>+12.2f} {std_diff:>+12.2f} {max_ratio:>12.2f}x")
            print("-" * 56)
    print("=" * 90 + "\n")

    # 准备绘图数据
    if has_bad:
        n_rows = 4
        data_list = [good_early, good_late, bad_early, bad_late]
        titles = [f'早期正常 ({good_early_period})', f'晚期正常 ({good_late_period})',
                  f'早期故障 ({bad_early_period})', f'晚期故障 ({bad_late_period})']
    else:
        n_rows = 2
        data_list = [good_early, good_late]
        titles = [f'早期正常 ({good_early_period})', f'晚期正常 ({good_late_period})']

    # 创建子图：n_rows 行, 2 列
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    if n_rows == 2:
        axes = axes.reshape(2, 2)

    axis_colors = {'X': 'blue', 'Y': 'orange', 'Z': 'green'}

    for row, (data, title) in enumerate(zip(data_list, titles)):
        # 左列：三轴波形叠加
        ax_wave = axes[row, 0]
        for axis_name in ['X', 'Y', 'Z']:
            idx = {'X':0, 'Y':1, 'Z':2}[axis_name]
            ax_wave.plot(data[:, idx], label=axis_name, color=axis_colors[axis_name], alpha=0.7, linewidth=0.8)
        ax_wave.set_title(title, fontsize=14)          # 调大标题
        ax_wave.set_xlabel('采样点序号', fontsize=12)   # 添加 x 轴标签
        ax_wave.set_ylabel('加速度值', fontsize=12)     # 添加 y 轴标签
        ax_wave.tick_params(labelsize=11)              # 刻度字体
        ax_wave.grid(True, alpha=0.3)

        # 右列：概率密度分布
        ax_hist = axes[row, 1]
        for axis_name in ['X', 'Y', 'Z']:
            idx = {'X':0, 'Y':1, 'Z':2}[axis_name]
            ax_hist.hist(data[:, idx], bins=80, alpha=0.5, density=True, label=axis_name, color=axis_colors[axis_name])
        ax_hist.set_title('概率密度分布', fontsize=14)   # 调大标题
        ax_hist.set_xlabel('加速度值', fontsize=12)     # 添加 x 轴标签
        ax_hist.set_ylabel('概率密度', fontsize=12)     # 添加 y 轴标签
        ax_hist.tick_params(labelsize=11)              # 刻度字体
    
    # 整图统一图例（从第一个子图获取）
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

    plt.suptitle(f'{machine}_{operation} 时序分布漂移与轴偏移分析', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, 'drift_and_bias.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 漂移与轴偏移图已保存: {save_path}")

# ------------------------- 主函数 -------------------------
def main():
    print("开始生成数据集特性可视化...")
    plot_class_imbalance()
    plot_drift_and_bias()
    print("\n所有图表生成完毕！")

if __name__ == "__main__":
    main()