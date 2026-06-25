"""
check_machine_distribution.py
检查各机器按时间顺序或工序的正常/故障比例分布
用于挑选最合适的机器进行跨时序/跨工序实验
"""

import os
import numpy as np
import h5py
import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


def scan_all_files(base_path):
    """递归扫描所有h5文件，根据文件夹路径判断标签"""
    pattern = os.path.join(base_path, '*', '*', '*', '*.h5')
    files = glob.glob(pattern)
    return files


def extract_info_from_path(file_path, base_path):
    """
    从文件路径提取信息
    路径格式: {base_path}/{machine}/{process}/{label}/{filename}.h5
    例如: ../CNC_Machining/data/M01/OP02/bad/M01_Feb_2019_OP02_000.h5
    """
    # 获取相对于base_path的路径
    rel_path = os.path.relpath(file_path, base_path)
    parts = rel_path.split(os.sep)
    
    # parts 格式: [machine, process, label, filename]
    if len(parts) >= 4:
        machine = parts[0]      # M01, M02, M03
        process = parts[1]      # OP00, OP01, etc.
        label = parts[2]        # good 或 bad
        filename = parts[3]     # M01_Feb_2019_OP02_000.h5
        
        # 从文件名解析时间段
        # 文件名格式: M01_Feb_2019_OP02_000.h5
        name_parts = filename.split('_')
        if len(name_parts) >= 4:
            month = name_parts[1]   # Feb, Aug
            year = name_parts[2]    # 2019, 2020, 2021
            period = f"{month}_{year}"  # Feb_2019, Aug_2019, etc.
        else:
            period = "unknown"
        
        is_abnormal = 1 if label == 'bad' else 0
        
        return {
            'machine': machine,
            'period': period,
            'process': process,
            'label': label,
            'is_abnormal': is_abnormal,
            'file': filename,
            'full_path': file_path
        }
    else:
        return None


def check_h5_structure(file_path):
    """检查h5文件内部结构（调试用）"""
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            return keys
    except Exception as e:
        return None


def analyze_machine(base_path, machine):
    """分析指定机器的数据分布"""
    pattern = os.path.join(base_path, machine, '*', '*', '*.h5')
    files = glob.glob(pattern)
    
    print(f"\n扫描 {machine}: 找到 {len(files)} 个文件")
    
    # 统计每个时间段的样本数
    period_stats = defaultdict(lambda: {'total': 0, 'abnormal': 0, 'normal': 0})
    
    # 统计每个工序的样本数
    process_stats = defaultdict(lambda: {'total': 0, 'abnormal': 0, 'normal': 0})
    
    # 详细记录每个文件的信息
    file_info_list = []
    
    for file_path in tqdm(files, desc=f"处理 {machine}"):
        info = extract_info_from_path(file_path, base_path)
        if info:
            file_info_list.append(info)
            period = info['period']
            process = info['process']
            
            period_stats[period]['total'] += 1
            process_stats[process]['total'] += 1
            
            if info['is_abnormal'] == 1:
                period_stats[period]['abnormal'] += 1
                process_stats[process]['abnormal'] += 1
            else:
                period_stats[period]['normal'] += 1
                process_stats[process]['normal'] += 1
    
    return file_info_list, period_stats, process_stats


def print_machine_summary(machine, period_stats, process_stats):
    """打印单个机器的详细统计"""
    print(f"\n{'='*60}")
    print(f"机器: {machine}")
    print(f"{'='*60}")
    
    # 按时间段统计
    print("\n【按时间段统计】")
    print(f"{'时间段':<15} {'总样本数':<10} {'正常样本':<10} {'故障样本':<10} {'故障率':<10}")
    print("-" * 55)
    
    total_samples = 0
    total_abnormal = 0
    
    for period in sorted(period_stats.keys()):
        stats = period_stats[period]
        total = stats['total']
        abnormal = stats['abnormal']
        rate = abnormal / total if total > 0 else 0
        
        total_samples += total
        total_abnormal += abnormal
        
        print(f"{period:<15} {total:<10} {stats['normal']:<10} {abnormal:<10} {rate:.2%}")
    
    print("-" * 55)
    print(f"{'总计':<15} {total_samples:<10} {total_samples-total_abnormal:<10} {total_abnormal:<10} {total_abnormal/total_samples:.2%}")
    
    # 按工序统计（只显示有故障的工序）
    print("\n【按工序统计（有故障样本的工序）】")
    print(f"{'工序':<10} {'总样本数':<10} {'正常样本':<10} {'故障样本':<10} {'故障率':<10}")
    print("-" * 50)
    
    for process in sorted(process_stats.keys()):
        stats = process_stats[process]
        if stats['abnormal'] > 0 or True:  # 显示所有工序，但高亮故障工序
            total = stats['total']
            abnormal = stats['abnormal']
            rate = abnormal / total if total > 0 else 0
            marker = " ⚠️" if abnormal > 0 else ""
            print(f"{process:<10} {total:<10} {stats['normal']:<10} {abnormal:<10} {rate:.2%}{marker}")
    
    return total_samples, total_abnormal


def recommend_best_machine(machines_data):
    """推荐最适合跨时序实验的机器"""
    print("\n" + "="*80)
    print("跨时序实验机器推荐")
    print("="*80)
    
    recommendations = []
    
    for machine, (period_stats, process_stats, total_samples, total_abnormal) in machines_data.items():
        periods = sorted(period_stats.keys())
        
        # 评估指标
        n_periods = len(periods)
        
        # 检查时间跨度
        years = set()
        for p in periods:
            if '_' in p:
                year = p.split('_')[1]
                years.add(year)
        time_span = len(years)
        
        # 计算各时间段的故障样本数
        period_abnormal = {p: period_stats[p]['abnormal'] for p in periods}
        period_rates = {p: period_stats[p]['abnormal'] / period_stats[p]['total'] if period_stats[p]['total'] > 0 else 0 
                       for p in periods}
        
        # 找出有故障的时间段
        periods_with_fault = [p for p in periods if period_stats[p]['abnormal'] > 0]
        n_periods_with_fault = len(periods_with_fault)
        
        # 检查是否有合适的时间段用于训练/验证/测试
        # 需要：早期有故障数据的2个时间段 + 中期/晚期有故障数据的时间段
        early_periods = [p for p in periods if '2019' in p or '2020' in p]
        late_periods = [p for p in periods if '2021' in p]
        
        # 计算得分
        # 优先级：时间段数量 > 故障时间段数量 > 时间跨度 > 总故障样本数
        score = n_periods * 10 + n_periods_with_fault * 20 + time_span * 5 + total_abnormal / 100
        
        recommendations.append({
            'machine': machine,
            'n_periods': n_periods,
            'periods_with_fault': n_periods_with_fault,
            'time_span_years': time_span,
            'total_abnormal': total_abnormal,
            'periods': periods,
            'periods_with_fault_list': periods_with_fault,
            'period_abnormal': period_abnormal,
            'period_rates': period_rates,
            'score': score
        })
    
    # 按得分排序
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n机器综合评估:")
    print("-"*100)
    print(f"{'机器':<8} {'时间段数':<10} {'故障时间段数':<10} {'时间跨度(年)':<12} {'故障样本总数':<12} {'得分':<8}")
    print("-"*100)
    
    for rec in recommendations:
        print(f"{rec['machine']:<8} {rec['n_periods']:<10} {rec['periods_with_fault']:<10} "
              f"{rec['time_span_years']:<12} {rec['total_abnormal']:<12} {rec['score']:<8.0f}")
    
    print("-"*100)
    
    # 详细分析每个机器的适用性
    print("\n【详细分析】")
    for rec in recommendations:
        print(f"\n{rec['machine']}:")
        print(f"  - 时间段: {rec['periods']}")
        print(f"  - 有故障的时间段: {rec['periods_with_fault_list']}")
        print(f"  - 各时间段故障数: {rec['period_abnormal']}")
        print(f"  - 各时间段故障率: { {k: f'{v:.2%}' for k, v in rec['period_rates'].items()} }")
        
        # 判断是否适合跨时序实验
        if rec['periods_with_fault'] >= 3:
            print(f"  - ✅ 适合跨时序实验 (有 {rec['periods_with_fault']} 个故障时间段)")
            # 推荐划分
            periods = rec['periods']
            # 按年份排序
            def get_year(p):
                if '_' in p:
                    return int(p.split('_')[1])
                return 0
            sorted_periods = sorted(periods, key=get_year)
            
            # 找出前2个有故障的时间段作为训练
            train_candidates = [p for p in sorted_periods if rec['period_abnormal'][p] > 0][:2]
            # 找出最后一个有故障的时间段作为测试
            test_candidates = [p for p in sorted_periods if rec['period_abnormal'][p] > 0][-1:]
            # 剩余作为验证
            val_candidates = [p for p in sorted_periods if p not in train_candidates and p not in test_candidates 
                             and rec['period_abnormal'][p] > 0]
            
            if len(train_candidates) >= 2 and test_candidates:
                print(f"  - 建议划分方案:")
                print(f"    训练集: {train_candidates}")
                print(f"    验证集: {val_candidates[0] if val_candidates else '需从训练集采样'}")
                print(f"    测试集: {test_candidates}")
        else:
            print(f"  - ⚠️ 不适合跨时序实验 (故障时间段不足3个)")
    
    # 返回最佳推荐
    best = recommendations[0]
    print("\n" + "="*80)
    print(f"🏆 推荐机器: {best['machine']}")
    print(f"   理由: 时间段数={best['n_periods']}, 故障时间段数={best['periods_with_fault']}, "
          f"故障样本总数={best['total_abnormal']}")
    print("="*80)
    
    return best['machine']


def main():
    # 配置路径
    base_path = "../CNC_Machining/data"
    
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在 {base_path}")
        print("请确认数据路径是否正确")
        return
    
    # 要分析的机器
    machines = ['M01', 'M02', 'M03']
    
    machines_data = {}
    
    for machine in machines:
        file_info_list, period_stats, process_stats = analyze_machine(base_path, machine)
        total_samples, total_abnormal = print_machine_summary(machine, period_stats, process_stats)
        machines_data[machine] = (period_stats, process_stats, total_samples, total_abnormal)
    
    # 推荐最佳机器
    best_machine = recommend_best_machine(machines_data)
    
    print(f"\n建议: 使用 {best_machine} 进行跨时序实验")
    print("运行预处理命令:")
    print(f"  python scripts/preprocess_data.py --experiment cross_time")
    print(f"  # 如果需要指定机器，请修改 preprocess_data.py 中的 machine 参数")


if __name__ == "__main__":
    main()