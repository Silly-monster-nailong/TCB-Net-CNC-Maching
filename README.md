# TCB‑Net：面向长期服役下数控铣床的故障诊断方法

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**TCB‑Net** 是一种基于时间‑通道双聚类的数控铣床故障诊断方法，由浙江大学本科生王瀚涵在毕业论文中提出（2026）。该方法通过同时聚类振动信号的时间片段和传感器通道，有效应对长期服役带来的时序分布漂移以及跨工序/跨机器的工况异质性。在 Bosch 真实产线 CNC 铣床数据集上，TCB‑Net 取得了领先的诊断性能，并提供了丰富的可解释性信息。

---

## 📁 项目目录结构
TCB‑Net‑CNC‑Machining/
├── configs/ # 配置文件（YAML）
│ ├── duet_anomaly.yaml # 数据预处理参数（路径、窗口、采样率等）
│ ├── tcb_net.yaml # 完整 TCB‑Net 配置（默认）
│ ├── tcb_net_no_channel.yaml # 消融：移除通道聚类
│ ├── tcb_net_no_time.yaml # 消融：移除时间聚类
│ ├── tcb_net_no_loss.yaml # 消融：移除聚类正则损失
│ └── tcb_net_time_only.yaml # 仅使用时间特征（丢弃通道信息）
│
├── models/ # 模型核心模块
│ ├── tcb_net.py # TCB‑Net 主网络（整合三个子模块）
│ ├── temporal_clustering.py # 时间聚类（1D‑CNN + EMA 在线 K‑means）
│ ├── channel_clustering.py # 通道聚类（相关系数+能量比，物理模式软分配）
│ └── biclustering_fusion.py # 融合模块（残差块 + 多头自注意力 + MLP）
│
├── scripts/ # 可执行实验脚本
│ ├── preprocess_data.py # 数据预处理（HDF5 → numpy 窗口，划分数据集）
│ ├── train_tcb_net.py # 训练 TCB‑Net（支持参数覆盖）
│ ├── evaluate_tcb_net.py # 全面评估（指标、混淆矩阵、可解释性图表）
│ ├── param_sensitivity.py # 时间聚类超参数敏感度分析
│ ├── ablation_channel_center.py # 通道中心初始化策略消融
│ └── compare_models_enhanced.py # 与基线模型对比（生成多组图表）
│
├── data/processed/ # 预处理后的数据（.npy 格式，Git 忽略）
│ ├── cross_time/ # 跨时段实验数据集
│ └── cross_process/ # 跨工序实验数据集
│
├── results/ # 实验结果输出（JSON 指标、图表，Git 忽略）
├── ablation/ # 敏感性分析和中心消融结果（Git 忽略）
├── ablations/ # 模块消融实验结果（Git 忽略）
├── appendix/ # 论文附录相关图表（可选）
├── streamlit_app.py # 可视化平台主程序
├── requirements.txt # Python 依赖列表
├── .gitignore # Git 忽略规则
└── README.md # 本文档

> **注意**：`data/processed/`、`results/`、`ablation/`、`ablations/`、`appendix/` 等目录中的大量二进制文件（`.npy`、`.pth`、`.png` 等）已通过 `.gitignore` 忽略，不会上传至仓库，仅保留源码和配置。

---

## 📄 脚本功能详解

| 脚本文件 | 功能描述 | 主要输出 |
|----------|----------|----------|
| `preprocess_data.py` | 读取原始 HDF5 振动数据（采样率 2kHz），降采样（可选），以固定窗口（默认 256 点）滑动切分，并根据 `--experiment` 参数（`cross_time` 或 `cross_process`）划分训练/验证/测试集，标准化后保存为 `.npy`。 | `data/processed/<exp_name>/X_train.npy`, `X_val.npy`, `X_test.npy`, `train_labels.npy`, `val_labels.npy`, `test_labels.npy`, `scaler.pkl` |
| `train_tcb_net.py` | 使用指定配置训练 TCB‑Net。支持通过命令行覆盖超参数（如 `--n_time_clusters`, `--ema_decay`）。训练采用 Focal Loss、动态阈值（基于训练集正常样本分位数，可选验证集微调）、早停、学习率调度。自动保存最佳模型和验证指标。 | `models/tcb_net_best_<exp_name>.pth`，`results/tcb_net_<exp_name>_results.json` |
| `evaluate_tcb_net.py` | 加载训练好的模型，在测试集上计算准确率、精确率、召回率、F1、AUC，绘制混淆矩阵、PR 曲线、ROC 曲线、概率分布直方图、校准曲线；同时输出时间簇/通道簇可解释性分析（各簇故障率、物理模式匹配、双簇交叉热力图等）。 | `results/tcb_net_<exp_name>_metrics.json`，`results/tcb_interpretability_<exp_name>.json`，以及多张 PNG 图表 |
| `param_sensitivity.py` | 对时间聚类模块的三个关键超参数（`n_time_clusters`、`time_hidden_dim`、`ema_decay`）进行网格搜索，每个组合训练一次并记录测试 F1 和 AUC，最后生成三张折线图。 | `ablation/param_sensitivity.csv`，`ablation/param_sensitivity_*.png` |
| `ablation_channel_center.py` | 对比四种通道聚类中心初始化策略（固定物理中心、随机+可学习、K‑means+可学习、物理引导+可学习），输出各策略的 F1 和 AUC 表格。 | `ablation/channel_center_results.csv` |
| `compare_models_enhanced.py` | 将 TCB‑Net 与多个基线模型（PatchTST、ModernTCN、iTransformer、TimesNet、Random Forest、KNN）进行性能对比，自动生成 F1 排名柱状图、雷达图、召回率‑精确率散点图、多指标横向柱状图以及各模型独立的混淆矩阵图。 | `results/<exp_type>/f1_ranking_*.png`，`radar_chart_*.png`，`recall_precision_*.png`，`metrics_barh_*.png`，`confusion_matrix_*.png` |
| `streamlit_app.py` | 启动基于 Streamlit 的交互式可视化 Web 平台，支持上传 `.h5` 或 `.npy` 文件，实时查看三轴波形、聚类权重、故障概率，并自动匹配检修策略。 | 浏览器交互界面（无本地文件输出） |

---

## 🔧 环境配置与依赖安装

### 创建虚拟环境（推荐）
```bash
conda create -n tcbnet python=3.9
conda activate tcbnet