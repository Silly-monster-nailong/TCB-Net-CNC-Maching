# TCB‑Net：面向长期服役下数控铣床的故障诊断方法

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**TCB‑Net**（Time‑Channel Biclustering Network）是一种用于数控铣床长期服役过程中故障诊断的深度学习方法，由浙江大学本科生王瀚涵在毕业论文中提出（2026）。该方法通过同时对三轴振动信号的**时间维度**和**通道维度**进行聚类，有效应对设备老化带来的时序分布漂移以及跨工序/跨机器的工况异质性。在 Bosch 真实产线 CNC 铣床数据集上，TCB‑Net 取得了领先的诊断性能，并提供了丰富的可解释性信息。

---

## 📁 项目目录结构

```
TCB‑Net‑CNC‑Machining/
├── configs/                           # 配置文件（YAML）
│   ├── duet_anomaly.yaml             # 数据预处理参数（路径、窗口、采样率等）
│   ├── tcb_net.yaml                  # 完整 TCB‑Net 配置（默认）
│   ├── tcb_net_no_channel.yaml       # 消融：移除通道聚类
│   ├── tcb_net_no_time.yaml          # 消融：移除时间聚类
│   ├── tcb_net_no_loss.yaml          # 消融：移除聚类正则损失
│   └── tcb_net_time_only.yaml        # 仅使用时间特征（丢弃通道信息）
│
├── models/                            # 模型核心模块
│   ├── tcb_net.py                    # TCB‑Net 主网络（整合三个子模块）
│   ├── temporal_clustering.py        # 时间聚类（1D‑CNN + EMA 在线 K‑means）
│   ├── channel_clustering.py         # 通道聚类（相关系数+能量比，物理模式软分配）
│   └── biclustering_fusion.py        # 融合模块（残差块 + 多头自注意力 + MLP）
│
├── scripts/                           # 可执行实验脚本
│   ├── preprocess_data.py            # 数据预处理（HDF5 → numpy 窗口，划分数据集）
│   ├── train_tcb_net.py              # 训练 TCB‑Net（支持参数覆盖）
│   ├── evaluate_tcb_net.py           # 全面评估（指标、混淆矩阵、可解释性图表）
│   ├── param_sensitivity.py          # 时间聚类超参数敏感度分析
│   ├── ablation_channel_center.py    # 通道中心初始化策略消融
│   └── compare_models_enhanced.py    # 与基线模型对比（生成多组图表）
│
├── data/processed/                    # 预处理后的数据（.npy 格式，Git 忽略）
│   ├── cross_time/                   # 跨时段实验数据集
│   └── cross_process/                # 跨工序实验数据集
│
├── results/                           # 实验结果输出（JSON 指标、图表，Git 忽略）
├── ablation/                          # 敏感性分析和中心消融结果（Git 忽略）
├── ablations/                         # 模块消融实验结果（Git 忽略）
├── appendix/                          # 论文附录相关图表（可选）
├── streamlit_app.py                   # 可视化平台主程序
├── requirements.txt                   # Python 依赖列表
├── .gitignore                         # Git 忽略规则
└── README.md                          # 本文档
```

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
```

### 安装依赖
```bash
pip install -r requirements.txt
```

> **GPU 用户**：如需使用 CUDA，请修改 `requirements.txt` 中的 `torch` 版本为对应 CUDA 版本（例如 `torch==2.5.1+cu121`），并添加 `--index-url https://download.pytorch.org/whl/cu121`。

---

## 📊 数据集准备

本项目使用 **Bosch CNC Machining Dataset**，可从 [官方 GitHub](https://github.com/boschresearch/CNC_Machining) 下载原始 HDF5 文件。

### 放置原始数据
将下载的文件夹（包含 `M01/`、`M02/`、`M03/` 等子目录）放在 `../CNC_Machining/data/` 下（相对路径），或修改 `configs/duet_anomaly.yaml` 中的 `base_path` 指向实际路径。

### 数据预处理
```bash
# 跨时序实验（训练: Feb2019+Aug2019, 验证: Feb2021, 测试: Aug2021）
python scripts/preprocess_data.py --experiment cross_time

# 跨工序实验（训练: OP00~OP05, 验证: OP06+OP07+OP13, 测试: OP08~OP14）
python scripts/preprocess_data.py --experiment cross_process
```

预处理后，数据将保存在 `data/processed/cross_time/` 或 `data/processed/cross_process/` 中，包含 `X_train.npy`、`X_val.npy`、`X_test.npy` 和对应的标签文件。

---

## 🚀 完整实验流程

### 1. 训练 TCB‑Net（跨时序）
```bash
python scripts/train_tcb_net.py \
    --data_dir ./data/processed/cross_time \
    --config ./configs/tcb_net.yaml \
    --epochs 10
```
最佳模型保存在 `./models/tcb_net_best_cross_time.pth`，评估结果（JSON）保存在 `./results/`。

### 2. 详细评估与可视化
```bash
python scripts/evaluate_tcb_net.py --data_dir ./data/processed/cross_time
```
这会在 `./results/` 下生成大量图表（混淆矩阵、PR/ROC 曲线、概率分布、校准曲线、时间簇权重图、双簇交叉热力图等）以及可解释性 JSON 文件。

### 3. 参数敏感性分析
```bash
python scripts/param_sensitivity.py
```
遍历 `n_time_clusters`、`time_hidden_dim`、`ema_decay` 的所有取值，每个组合训练一次，结果汇总在 `./ablation/param_sensitivity.csv`，并输出三张折线图。

### 4. 通道中心初始化消融实验
```bash
python scripts/ablation_channel_center.py
```
对比四种初始化策略，结果保存在 `./ablation/channel_center_results.csv`。

### 5. 与基线模型对比
首先确保你已经训练或获得了以下基线的结果（保存在 `../baseline/` 或指定目录）：
- PatchTST、ModernTCN、iTransformer、TimesNet、Random Forest、KNN

然后执行：
```bash
python scripts/compare_models_enhanced.py \
    --exp_type cross_time \
    --baseline_root ../baseline \
    --tcb_dir ./results
```
生成的对比图表将保存在 `./results/cross_time/` 下，包括 F1 排名、雷达图、召回率‑精确率散点图、多指标横向柱状图以及各模型的混淆矩阵。

---

## 📱 可视化平台使用

启动 Streamlit 应用：
```bash
streamlit run streamlit_app.py
```
浏览器将打开 `http://localhost:8501`。

### 功能说明
- **上传数据**：支持 `.h5` 或 `.npy` 格式的三轴振动信号文件（采样率 2kHz）。
- **自动分析**：应用预训练的 TCB‑Net 模型，对每个滑动窗口（长度 256）输出故障概率、时间簇软分配权重、通道簇软分配权重。
- **交互查看**：点击任意窗口可查看其时域波形、簇权重柱状图、到各时间簇中心的欧氏距离，以及自动匹配的检修建议。
- **策略配置**：在“数据总览”页面中，可为每个（时间簇, 通道簇）组合编辑检修策略，保存后诊断页面将自动匹配显示。

---

## 📈 主要实验结果（复现参考）

| 实验 | 模型 | F1 (%) | AUC (%) |
|------|------|--------|---------|
| 跨时序 | TCB‑Net（Ours） | **79.7** | 93.3 |
| 跨时序 | ModernTCN | 77.3 | 91.2 |
| 跨时序 | TimesNet | 69.4 | 91.7 |
| 跨工序 | TCB‑Net（Ours） | 80.9 | 90.5 |

更多消融和可解释性结果详见论文或运行上述脚本生成。

---

## 📝 引用

如果你在研究中使用了本代码或数据集，请引用：

```bibtex
@thesis{wang2026tcbnet,
  author = {王瀚涵},
  title = {基于时间通道双聚类的长期服役下数控铣床故障诊断方法},
  school = {浙江大学},
  year = {2026},
  type = {本科生毕业论文}
}
```

同时引用 Bosch CNC 数据集：

```bibtex
@article{tnani2022smart,
  title={Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring},
  author={Tnani, Mohamed-Ali and Feil, Michael and Diepold, Klaus},
  journal={Procedia CIRP},
  volume={107},
  pages={131--136},
  year={2022}
}
```

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢 Bosch Research 公开数据集，感谢导师王康教授的悉心指导。

---

**如有问题，欢迎提交 Issue 或联系作者。**