import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import h5py
import os
import sys
import json
import zipfile
import io
import pandas as pd
from itertools import product

st.set_page_config(page_title="CNC 铣床故障诊断平台", layout="wide", initial_sidebar_state="expanded")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1120; color: #e5e9f0; }
    [data-testid="stSidebar"] { background-color: #1a2332; border-right: 1px solid #2a3a4a; }
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextArea label { color: #f0f2f6 !important; font-weight: 500; }
    [data-testid="stSidebar"] input {
        color: white !important;
        background-color: #2d3a4a !important;
    }
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stFileUploader label {
        color: #e5e9f0 !important;
    }
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] {
        background-color: #2d3a4a;
    }
    /* 修复 text_area 文本颜色 */
    .stTextArea textarea {
        color: white !important;
        background-color: #2d3a4a !important;
    }
    .stTextArea label {
        color: #f1f5f9 !important;
    }
    div[data-testid="stMetric"] label {
        color: #f1f5f9 !important;
    }
    div[data-testid="stMetric"] div {
        color: #60a5fa !important;
    }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 6px; border: none; }
    .stButton>button:hover { background-color: #2563eb; }
    .stAlert { background-color: #1e293b; color: #e2e8f0; }
    h1, h2, h3, h4 { color: #f1f5f9; }
    </style>
    """,
    unsafe_allow_html=True,
)

sys.path.append('.')
from configs import Config
from models.tcb_net import TCB_Net

# ------------------- 统一绘图函数 -------------------
def create_unified_figure(figsize=(6, 3.5), left=0.15, right=0.95, bottom=0.15, top=0.85, xlabel_pad=4):
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.set_position([left, bottom, right-left, top-bottom])
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    ax.xaxis.labelpad = xlabel_pad
    ax.yaxis.labelpad = 4
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    return fig, ax

# ------------------- 模型加载 -------------------
@st.cache_resource
def load_model(model_path, config_path):
    config = Config.from_yaml(config_path)
    config.logging.model_dir = os.path.dirname(model_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    n_time_clusters = state_dict['temporal.centers'].shape[0]
    time_hidden_dim = state_dict['temporal.fc.weight'].shape[0] if 'temporal.fc.weight' in state_dict else 128
    config.model.n_time_clusters = n_time_clusters
    config.model.time_hidden_dim = time_hidden_dim
    model = TCB_Net(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.temporal.update_centers = False
    return model, n_time_clusters, time_hidden_dim

MODEL_PATH = "./models/tcb_net_best_cross_time.pth"
CONFIG_PATH = "./configs/tcb_net.yaml"
model, n_time_clusters, time_hidden_dim = load_model(MODEL_PATH, CONFIG_PATH)
time_centers = model.temporal.centers.cpu().numpy()

# ------------------- 数据处理函数 -------------------
def preprocess_data(data, seq_len=256):
    n = data.shape[0]
    sequences = []
    step = seq_len // 2
    for i in range(0, n - seq_len + 1, step):
        sequences.append(data[i:i+seq_len])
    if not sequences:
        pad_len = seq_len - n
        if pad_len > 0:
            padded = np.vstack([data, np.zeros((pad_len, data.shape[1]))])
        else:
            padded = data[:seq_len]
        sequences.append(padded)
    return np.array(sequences)

def load_h5_data(uploaded_file):
    with h5py.File(uploaded_file, 'r') as f:
        possible_keys = ['vibration_data', 'vibration data', 'data', 'acceleration']
        for key in possible_keys:
            if key in f:
                return f[key][:]
        first_key = list(f.keys())[0]
        return f[first_key][:]

def load_custom_data(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, 'r') as zf:
        X_train = np.load(io.BytesIO(zf.read('X_train.npy')))
        X_test = np.load(io.BytesIO(zf.read('X_test.npy')))
        train_labels = np.load(io.BytesIO(zf.read('train_labels.npy')))
        test_labels = np.load(io.BytesIO(zf.read('test_labels.npy')))
    if X_train.shape[1] == 3:
        X_train = X_train.transpose(0,2,1)
        X_test = X_test.transpose(0,2,1)
    return X_train, X_test, train_labels, test_labels

def extract_features(model, X, batch_size=256, device='cpu'):
    feats = []
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
        with torch.no_grad():
            X_time = batch.permute(0,2,1)
            h_time, _, _ = model.temporal(X_time)
        feats.append(h_time.cpu().numpy())
    return np.concatenate(feats, axis=0)

def get_channel_info(model, X, batch_size=256, device='cpu'):
    all_assign = []
    all_r = []
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
        with torch.no_grad():
            w_channel, r_vec = model.channel(batch)
            assign = torch.argmax(w_channel, dim=1).cpu().numpy()
            all_assign.extend(assign)
            all_r.append(r_vec.cpu().numpy())
    return np.array(all_assign), np.concatenate(all_r, axis=0)

def get_time_assign(model, X, batch_size=256, device='cpu'):
    all_weights = []
    for i in range(0, len(X), batch_size):
        batch = torch.FloatTensor(X[i:i+batch_size]).to(device)
        with torch.no_grad():
            X_time = batch.permute(0,2,1)
            _, w_time, _ = model.temporal(X_time)
            all_weights.append(w_time.cpu().numpy())
    all_weights = np.concatenate(all_weights, axis=0)
    return np.argmax(all_weights, axis=1), all_weights

# ------------------- 状态初始化 -------------------
if 'windows' not in st.session_state:
    st.session_state.windows = None
if 'probs' not in st.session_state:
    st.session_state.probs = None
if 'time_weights' not in st.session_state:
    st.session_state.time_weights = None
if 'channel_weights' not in st.session_state:
    st.session_state.channel_weights = None
if 'h_features' not in st.session_state:
    st.session_state.h_features = None
if 'selected_window_idx' not in st.session_state:
    st.session_state.selected_window_idx = 0
if 'show_window_list' not in st.session_state:
    st.session_state.show_window_list = False
if 'custom_strategy' not in st.session_state:
    st.session_state.custom_strategy = {}   # 字典存储策略
if 'view' not in st.session_state:
    st.session_state.view = 'diagnosis'
if 'overview_data_loaded' not in st.session_state:
    st.session_state.overview_data_loaded = False
if 'overview_X_train' not in st.session_state:
    st.session_state.overview_X_train = None
if 'overview_X_test' not in st.session_state:
    st.session_state.overview_X_test = None
if 'overview_y_train' not in st.session_state:
    st.session_state.overview_y_train = None
if 'overview_y_test' not in st.session_state:
    st.session_state.overview_y_test = None

# ------------------- 侧边栏 -------------------
with st.sidebar:
    st.markdown("## 🧭 导航")
    st.success(f"模型状态: {'✅ 已加载' if model else '❌ 未加载'}")
    
    if st.button("🔧 故障诊断", use_container_width=True):
        st.session_state.view = 'diagnosis'
        st.rerun()
    if st.button("📊 数据总览", use_container_width=True):
        st.session_state.view = 'overview'
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.view == 'diagnosis':
        uploaded_file = st.file_uploader("📂 上传振动数据 (.h5 / .npy)", type=['h5','npy'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.h5'):
                data = load_h5_data(uploaded_file)
            else:
                data = np.load(uploaded_file)
            
            if data.ndim == 2 and data.shape[1] != 3:
                st.error("数据格式错误：需要 (n_samples, 3)")
            else:
                windows = preprocess_data(data)
                st.session_state.windows = windows
                st.info(f"原始数据长度: {data.shape[0]} | 生成窗口: {windows.shape[0]}")
                
                with st.spinner("正在批量分析所有窗口..."):
                    X_batch = torch.FloatTensor(windows)
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)
                    with torch.no_grad():
                        logits = model(X_batch.to(device))
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        X_time = X_batch.permute(0,2,1).to(device)
                        h_time, w_time, _ = model.temporal(X_time)
                        w_channel, _ = model.channel(X_batch.to(device))
                    st.session_state.probs = probs
                    st.session_state.time_weights = w_time.cpu().numpy()
                    st.session_state.channel_weights = w_channel.cpu().numpy()
                    st.session_state.h_features = h_time.cpu().numpy()
                st.success("分析完成！")
        
        if st.session_state.windows is not None:
            total_windows = len(st.session_state.windows)
            def jump_to_window():
                new_idx = st.session_state.jump_window_input - 1
                if 0 <= new_idx < total_windows:
                    st.session_state.selected_window_idx = new_idx
            st.number_input(
                f"输入窗口序号 (1~{total_windows}) 并回车",
                min_value=1, max_value=total_windows,
                value=st.session_state.selected_window_idx+1,
                key="jump_window_input",
                on_change=jump_to_window
            )
        
        st.markdown("### 🛠️ 检修策略自定义")
        # 全局策略输入
        global_strat = st.text_area("全局策略（备用）", value=st.session_state.custom_strategy.get('global', ''), height=100)
        if global_strat != st.session_state.custom_strategy.get('global', ''):
            st.session_state.custom_strategy['global'] = global_strat
        
        st.markdown("---")
        st.caption("提示：点击下方按钮可切换窗口列表视图")

# ------------------- 主界面 -------------------
if st.session_state.view == 'diagnosis':
    if st.session_state.windows is not None and st.session_state.probs is not None:
        total_windows = len(st.session_state.windows)
        avg_prob = st.session_state.probs.mean()
        high_risk_ratio = (st.session_state.probs > 0.5).mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔔 平均故障概率", f"{avg_prob:.2%}")
        with col2:
            st.metric("⚠️ 高风险窗口比例", f"{high_risk_ratio:.1%}")
        with col3:
            st.metric("📊 窗口总数", total_windows)
        with col4:
            st.metric("📈 当前窗口", st.session_state.selected_window_idx+1)
        
        if st.button("📋 切换窗口列表", use_container_width=True):
            st.session_state.show_window_list = not st.session_state.show_window_list
        
        if st.session_state.show_window_list:
            st.subheader("所有窗口预览 (🔥=高风险，概率>0.5)")
            if high_risk_ratio == 0:
                st.info("当前数据中无高风险窗口（所有窗口故障概率 ≤ 0.5）")
            else:
                cols_per_row = 6
                for i in range(0, total_windows, cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < total_windows:
                            prob = st.session_state.probs[idx]
                            is_risk = prob > 0.5
                            risk_mark = "🔥" if is_risk else "✅"
                            col.markdown(f"**窗口{idx}** {risk_mark}<br><span style='font-size:0.8rem'>{prob:.1%}</span>", unsafe_allow_html=True)
            st.markdown("---")
        
        idx = st.session_state.selected_window_idx
        window = st.session_state.windows[idx]
        prob = st.session_state.probs[idx]
        time_weights = st.session_state.time_weights[idx]
        channel_weights = st.session_state.channel_weights[idx]
        h_feat = st.session_state.h_features[idx]
        
        time_cluster_id = np.argmax(time_weights)
        channel_cluster_id = np.argmax(channel_weights)
        risk_level = "高危" if prob > 0.7 else ("中危" if prob > 0.4 else "低危")
        
        strat_key = f"{time_cluster_id}_{channel_cluster_id}"
        strategy_text = st.session_state.custom_strategy.get(strat_key, st.session_state.custom_strategy.get('global', '未定义策略'))
        
        st.markdown("---")
        col_left, col_right = st.columns(2, gap="medium")
        
        with col_left:
            st.subheader("📈 三轴振动信号波形")
            fig1, ax1 = create_unified_figure()
            for i, axis in enumerate(['X', 'Y', 'Z']):
                ax1.plot(window[:, i], label=axis, linewidth=1)
            ax1.set_title("当前窗口波形")
            ax1.set_xlabel("采样点")
            ax1.set_ylabel("加速度 (mg)")
            ax1.legend(loc='upper right', fontsize=8, framealpha=0.8)
            st.pyplot(fig1, use_container_width=True)
            
            st.subheader("⏱️ 时间簇软分配权重")
            fig3, ax3 = create_unified_figure()
            bars1 = ax3.bar(range(len(time_weights)), time_weights, color='#4c9aff', alpha=0.8)
            ax3.set_xticks(range(len(time_weights)))
            ax3.set_xticklabels([str(i) for i in range(len(time_weights))])
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("权重")
            ax3.set_title("当前窗口时间簇归属")
            ax3.set_xlabel("时间簇")
            for bar, val in zip(bars1, time_weights):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            st.pyplot(fig3, use_container_width=True)
        
        with col_right:
            st.subheader("📊 当前窗口到各簇中心距离")
            dists = [np.linalg.norm(h_feat - center) for center in time_centers]
            fig_bar, ax_bar = create_unified_figure()
            bars = ax_bar.bar(range(n_time_clusters), dists, color='#4c9aff', alpha=0.8, edgecolor='white')
            ax_bar.set_xticks(range(n_time_clusters))
            ax_bar.set_xticklabels([f'簇{i}' for i in range(n_time_clusters)])
            ax_bar.set_ylabel("欧氏距离")
            ax_bar.set_title("当前窗口到各时间簇中心的距离")
            max_dist = max(dists)
            ax_bar.set_ylim(0, max_dist * 1.1)
            for bar, d in zip(bars, dists):
                ax_bar.text(bar.get_x() + bar.get_width()/2, d + max_dist * 0.02, f'{d:.1f}', ha='center', va='bottom', fontsize=8)
            st.pyplot(fig_bar, use_container_width=True)
            
            st.subheader("🔄 通道簇软分配权重")
            fig4, ax4 = create_unified_figure()
            bars2 = ax4.bar(range(len(channel_weights)), channel_weights, color='#ff9f4c', alpha=0.8)
            ax4.set_xticks(range(len(channel_weights)))
            ax4.set_xticklabels([str(i) for i in range(len(channel_weights))])
            ax4.set_ylim(0, 1)
            ax4.set_ylabel("权重")
            ax4.set_title("当前窗口通道簇归属")
            ax4.set_xlabel("通道簇")   
            for bar, val in zip(bars2, channel_weights):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            st.pyplot(fig4, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚠️ 风险评估与检修建议")
        st.markdown(f"""
        > **该窗口故障概率 {prob:.2%}，属于时间簇{time_cluster_id} + 通道簇{channel_cluster_id}，风险等级：{risk_level}。  
        > 检修策略：{strategy_text}
        """)
    else:
        st.info("👈 请先在左侧导航栏上传振动数据文件（.h5或.npy）以开始诊断")

elif st.session_state.view == 'overview':
    st.title("📊 数据总览 - 聚类分析与策略配置")
    st.markdown("本页面允许研究人员加载自定义数据集，观察模型在训练/测试集上的聚类效果，并为每个(时间簇, 通道簇)组合编写检修策略。")
    
    with st.expander("📂 加载数据", expanded=not st.session_state.overview_data_loaded):
        data_source = st.radio("选择数据源", ["使用默认 cross_time 数据集", "上传自定义 zip 文件"], index=0)
        if data_source == "使用默认 cross_time 数据集":
            data_dir = "./data/processed/cross_time"
            try:
                X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
                y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
                X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
                y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))
                if X_train.shape[1] == 3:
                    X_train = X_train.transpose(0,2,1)
                    X_test = X_test.transpose(0,2,1)
                st.session_state.overview_X_train = X_train
                st.session_state.overview_X_test = X_test
                st.session_state.overview_y_train = y_train
                st.session_state.overview_y_test = y_test
                st.session_state.overview_data_loaded = True
                st.success("成功加载默认数据集")
            except Exception as e:
                st.error(f"加载失败：{e}")
        else:
            uploaded_zip = st.file_uploader("上传数据包 (包含 X_train.npy, X_test.npy, train_labels.npy, test_labels.npy)", type=['zip'])
            if uploaded_zip is not None:
                try:
                    X_train, X_test, y_train, y_test = load_custom_data(uploaded_zip)
                    st.session_state.overview_X_train = X_train
                    st.session_state.overview_X_test = X_test
                    st.session_state.overview_y_train = y_train
                    st.session_state.overview_y_test = y_test
                    st.session_state.overview_data_loaded = True
                    st.success("数据加载成功")
                except Exception as e:
                    st.error(f"解析失败：{e}")
    
    if not st.session_state.overview_data_loaded:
        st.warning("请先加载数据")
        st.stop()
    
    X_train = st.session_state.overview_X_train
    X_test = st.session_state.overview_X_test
    y_train = st.session_state.overview_y_train
    y_test = st.session_state.overview_y_test
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    @st.cache_data
    def get_overview_features(_model, X_train, X_test):
        h_train = extract_features(_model, X_train, device=device)
        h_test = extract_features(_model, X_test, device=device)
        time_assign_test, _ = get_time_assign(_model, X_test, device=device)
        channel_assign_test, r_vec_test = get_channel_info(_model, X_test, device=device)
        return h_train, h_test, time_assign_test, channel_assign_test, r_vec_test
    
    with st.spinner("正在提取特征..."):
        h_train, h_test, time_assign_test, channel_assign_test, r_vec_test = get_overview_features(model, X_train, X_test)
    
    all_h = np.vstack([h_train, h_test])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_h)
    h_train_pca = all_pca[:len(h_train)]
    h_test_pca = all_pca[len(h_train):]
    
    # 四宫格
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("样本类型 PCA 投影")
        fig1, ax1 = plt.subplots(figsize=(6,5))
        y_train_bin = y_train.astype(int)
        y_test_bin = y_test.astype(int)
        mask_train_n = (y_train_bin == 0)
        mask_train_f = (y_train_bin == 1)
        mask_test_n = (y_test_bin == 0)
        mask_test_f = (y_test_bin == 1)
        ax1.scatter(h_train_pca[mask_train_n,0], h_train_pca[mask_train_n,1], c='green', s=10, alpha=0.6, label='早期健康')
        ax1.scatter(h_train_pca[mask_train_f,0], h_train_pca[mask_train_f,1], c='orange', s=10, alpha=0.6, label='早期故障')
        ax1.scatter(h_test_pca[mask_test_n,0], h_test_pca[mask_test_n,1], c='blue', s=10, alpha=0.6, label='晚期健康')
        ax1.scatter(h_test_pca[mask_test_f,0], h_test_pca[mask_test_f,1], c='red', s=10, alpha=0.6, label='晚期故障')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, use_container_width=True)
    
    with col2:
        st.subheader("测试集时间簇着色")
        fig2, ax2 = plt.subplots(figsize=(6,5))
        cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for c in range(n_time_clusters):
            mask = (time_assign_test == c)
            if mask.any():
                ax2.scatter(h_test_pca[mask,0], h_test_pca[mask,1], c=cluster_colors[c], s=10, alpha=0.6, label=f'时间簇{c}')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("通道簇 3D 相关系数")
        fig3 = plt.figure(figsize=(6,5))
        ax3 = fig3.add_subplot(111, projection='3d')
        for c in range(4):
            mask = (channel_assign_test == c)
            if mask.sum() > 0:
                ax3.scatter(r_vec_test[mask,0], r_vec_test[mask,1], r_vec_test[mask,2],
                            c=cluster_colors[c], s=5, alpha=0.6, label=f'通道簇{c}')
        ax3.set_xlabel('r_xy')
        ax3.set_ylabel('r_xz')
        ax3.set_zlabel('r_yz')
        ax3.legend(loc='upper right', fontsize=8)
        st.pyplot(fig3, use_container_width=True)
    
    with col4:
        st.subheader("通道簇平均特征")
        n_chan_clusters = 4
        avg_corr = []
        avg_energy = []
        for c in range(n_chan_clusters):
            mask = (channel_assign_test == c)
            if mask.sum() > 0:
                avg_corr.append(r_vec_test[mask, :3].mean(axis=0))
                avg_energy.append(r_vec_test[mask, 3:].mean(axis=0))
            else:
                avg_corr.append([0,0,0])
                avg_energy.append([0,0,0])
        avg_corr = np.array(avg_corr)
        avg_energy = np.array(avg_energy)
        
        st.write("**平均相关系数**")
        st.dataframe(pd.DataFrame(avg_corr, columns=['r_xy', 'r_xz', 'r_yz'], index=[f'簇{i}' for i in range(4)]))
        
        fig4, ax4 = plt.subplots(figsize=(6,3))
        x = np.arange(4)
        width = 0.25
        for i in range(3):
            ax4.bar(x + i*width, avg_energy[:, i], width, label=f'轴{i}')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([f'簇{i}' for i in range(4)])
        ax4.set_ylabel('能量比')
        ax4.legend()
        st.pyplot(fig4, use_container_width=True)
    
    st.markdown("---")
    st.subheader("✍️ 检修策略配置（按时间簇×通道簇）")
    st.markdown("为每个簇组合编写专属检修策略，保存后将用于故障诊断页面。")
    
    time_clusters = list(range(n_time_clusters))
    chan_clusters = list(range(4))
    combos = list(product(time_clusters, chan_clusters))
    
    for i in range(0, len(combos), 4):
        cols = st.columns(4)
        for j, (tc, cc) in enumerate(combos[i:i+4]):
            key = f"{tc}_{cc}"
            current_val = st.session_state.custom_strategy.get(key, "")
            with cols[j]:
                new_val = st.text_area(f"时间簇{tc} × 通道簇{cc}", value=current_val, key=f"edit_{key}", height=80)
                if new_val != current_val:
                    st.session_state.custom_strategy[key] = new_val
    
    st.markdown("**全局默认策略**（当特定组合未定义时使用）")
    global_strat = st.text_area("全局策略", value=st.session_state.custom_strategy.get('global', ''), height=80)
    if global_strat != st.session_state.custom_strategy.get('global', ''):
        st.session_state.custom_strategy['global'] = global_strat
    
    if st.button("💾 保存所有策略"):
        st.success("策略已保存到会话状态，将在故障诊断界面生效")
    
    st.caption("提示：策略会保存在当前会话中，刷新页面后丢失。如需永久保存，请导出 session_state 数据。")