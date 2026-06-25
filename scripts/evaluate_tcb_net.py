"""评估 TCB-Net 模型 - 完全从 checkpoint 适配参数"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import Config
from models.tcb_net import TCB_Net


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# ================== 新增绘图函数（便于管理）==================
def plot_pr_curve(labels, probs, save_path):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, linewidth=2, color='darkorange')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title(f'PR 曲线 (AUC={auc:.3f})')
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_prob_dist(labels, probs, save_path):
    plt.figure(figsize=(5, 4))
    plt.hist(probs[labels==0], bins=50, alpha=0.5, label='正常', density=True, color='blue')
    plt.hist(probs[labels==1], bins=50, alpha=0.5, label='故障', density=True, color='red')
    plt.xlabel('预测概率')
    plt.ylabel('密度')
    plt.title('预测概率分布对比')
    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_bicluster_heatmap(dominant_time_cluster, channel_cluster, labels, n_time, n_chan, time_labels, chan_labels, save_path):
    """绘制双簇交叉故障率热力图"""
    fault_matrix = np.full((n_time, n_chan), np.nan)
    for i in range(n_time):
        for j in range(n_chan):
            mask = (dominant_time_cluster == i) & (channel_cluster == j)
            if mask.sum() > 0:
                fault_matrix[i, j] = labels[mask].mean()
    plt.figure(figsize=(6, 5))
    sns.heatmap(fault_matrix, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=chan_labels, yticklabels=time_labels)
    plt.title('双簇交叉故障率')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_time_cluster_weights_sampled(weights, save_path, n_sample=1000):
    """随机抽样绘制时间簇软分配权重热力图"""
    n_sample = min(n_sample, weights.shape[0])
    indices = np.random.choice(weights.shape[0], n_sample, replace=False)
    weights_sample = weights[indices].T
    plt.figure(figsize=(12, 5))
    plt.imshow(weights_sample, aspect='auto', cmap='viridis')
    plt.colorbar(label='权重')
    plt.xlabel(f'样本序号（随机抽样 {n_sample} 个）')
    plt.ylabel('时间簇')
    plt.title('时间簇软分配权重（抽样）')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_calibration_curve(labels, probs, save_path):
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)
    plt.figure(figsize=(5, 4))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='TCB-Net')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='完美校准')
    plt.xlabel('平均预测概率')
    plt.ylabel('正样本比例')
    plt.title('可靠性曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC 曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def get_threshold_from_train_normal(model, train_loader, device, 
                                    val_loader=None, 
                                    percentile=99, 
                                    fine_tune_range=0.2):
        """基于训练集正常样本的分位数获取阈值，可选验证集微调"""
        model.eval()
        normal_probs = []
        with torch.no_grad():
            for X, y in train_loader:
                # 只保留正常样本 (y == 0)
                normal_mask = (y == 0)
                if not normal_mask.any():
                    continue
                X_normal = X[normal_mask].to(device)
                logits = model(X_normal)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                normal_probs.extend(probs)
        base_thr = np.percentile(normal_probs, percentile)
        print(f"[阈值] 训练集正常样本 {percentile}% 分位数 = {base_thr:.4f}")

        if val_loader is not None:
            val_probs, val_labels = [], []
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(device)
                    logits = model(X)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    val_probs.extend(probs)
                    val_labels.extend(y.numpy())
            val_probs = np.array(val_probs)
            val_labels = np.array(val_labels)
            pos_cnt = (val_labels == 1).sum()
            if pos_cnt >= 10:
                thresholds = np.linspace(base_thr * (1 - fine_tune_range),
                                        base_thr * (1 + fine_tune_range), 21)
                best_f1 = 0
                best_thr = base_thr
                for thr in thresholds:
                    preds = (val_probs >= thr).astype(int)
                    f1 = f1_score(val_labels, preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = thr
                print(f"[阈值] 验证集微调后 = {best_thr:.4f} (F1={best_f1:.4f})")
                return best_thr
            else:
                print(f"[阈值] 验证集正样本不足 ({pos_cnt} < 10)，不微调")
        return base_thr


def compute_bicluster_quality(time_assign, channel_assign, features, labels, r_vec):
    """计算双簇质量（适配6维特征）"""
    K = len(np.unique(time_assign))
    M = len(np.unique(channel_assign))
    quality = {}
    # 理论模式（仅使用前3维相关系数进行匹配）
    theo_patterns = {
        0: np.array([0.5, 0.4, 0.4]),
        1: np.array([0.8, 0.2, 0.2]),
        2: np.array([0.6, 0.6, 0.6]),
        3: np.array([0.2, -0.2, 0.7])
    }
    for k in range(K):
        for m in range(M):
            mask = (time_assign == k) & (channel_assign == m)
            if mask.sum() < 5:
                quality[f"{k}_{m}"] = None
                continue
            sub_feat = features[mask]
            center = sub_feat.mean(axis=0)
            intra_dist = np.linalg.norm(sub_feat - center, axis=1).mean()
            cons = 1 / (1 + intra_dist)
            min_dist = float('inf')
            for k2 in range(K):
                for m2 in range(M):
                    if k2 == k and m2 == m:
                        continue
                    mask2 = (time_assign == k2) & (channel_assign == m2)
                    if mask2.sum() < 5:
                        continue
                    center2 = features[mask2].mean(axis=0)
                    dist = np.linalg.norm(center - center2)
                    if dist < min_dist:
                        min_dist = dist
            disc = min_dist / (1 + min_dist)
            # 使用前3维相关系数进行物理一致性匹配
            r_mean = r_vec[mask, :3].mean(axis=0)
            phys = 0.0
            if m in theo_patterns:
                phys = np.corrcoef(theo_patterns[m], r_mean)[0, 1]
                phys = max(0, phys)
            alpha, beta, gamma = 0.4, 0.3, 0.3
            Q = alpha * cons + beta * disc + gamma * phys
            quality[f"{k}_{m}"] = {'cons': cons, 'disc': disc, 'phys': phys, 'Q': Q}
    return quality


def compute_mmd(source_feat, target_feat, gamma=0.001):
    """MMD 距离（RBF核）"""
    K_ss = rbf_kernel(source_feat, source_feat, gamma=gamma)
    K_tt = rbf_kernel(target_feat, target_feat, gamma=gamma)
    K_st = rbf_kernel(source_feat, target_feat, gamma=gamma)
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]
    mmd = (K_ss.sum() / (n_s * n_s) + K_tt.sum() / (n_t * n_t) - 2 * K_st.sum() / (n_s * n_t))
    return np.sqrt(mmd) if mmd > 0 else 0


def compute_coral(source_feat, target_feat):
    """CORAL 距离（协方差矩阵差异）"""
    cov_s = np.cov(source_feat, rowvar=False)
    cov_t = np.cov(target_feat, rowvar=False)
    return np.linalg.norm(cov_s - cov_t, ord='fro') / (source_feat.shape[1] ** 0.5)


def compute_a_distance(source_feat, target_feat):
    """A‑distance：训练线性SVM区分源域和目标域"""
    X = np.vstack([source_feat, target_feat])
    y = np.hstack([np.zeros(len(source_feat)), np.ones(len(target_feat))])
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X, y)
    pred = clf.predict(X)
    err = 1 - accuracy_score(y, pred)
    return 2 * (1 - 2 * err)   # 注意：有些定义是 2*(1-2*err)，有些直接用 err。常用 2*(1-2*err) 范围 [0,1]


def plot_cluster_scatter(features, time_assign, channel_assign, save_path):
    """绘制聚类散点图"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=time_assign, cmap='tab10', s=5, alpha=0.6)
    axes[0].set_title('Time Cluster Assignment')
    plt.colorbar(sc1, ax=axes[0], label='Cluster ID')
    sc2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=channel_assign, cmap='tab10', s=5, alpha=0.6)
    axes[1].set_title('Channel Cluster Assignment')
    plt.colorbar(sc2, ax=axes[1], label='Cluster ID')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_channel_corr_3d(r_vec, channel_assign, save_path):
    """绘制3D通道相关系数图"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'orange']
    for i in range(4):
        mask = channel_assign == i
        if mask.sum() == 0:
            continue
        ax.scatter(r_vec[mask, 0], r_vec[mask, 1], r_vec[mask, 2],
                   c=colors[i], label=f'Cluster {i}', s=5, alpha=0.6)
    ax.set_xlabel('r_xy')
    ax.set_ylabel('r_xz')
    ax.set_zlabel('r_yz')
    ax.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(data_dir=None, output_dir=None, custom_threshold=None):
    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tcb_net.yaml')
    config = Config.from_yaml(config_path)
    if data_dir is not None:
        config.logging.data_dir = data_dir
    if output_dir is not None:
        config.logging.save_dir = output_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = os.path.basename(config.logging.data_dir.rstrip('/\\'))

    # 加载模型（同原代码）
    model_path = os.path.join(config.logging.model_dir, f"tcb_net_best_{experiment_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    print(f"使用模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 从 state_dict 推断参数并创建模型（与原代码相同）
    is_v2 = 'temporal.feature_extractor.0.weight' in state_dict
    if is_v2:
        if 'temporal.fc.weight' in state_dict:
            time_hidden_dim = state_dict['temporal.fc.weight'].shape[0]
        else:
            time_hidden_dim = 128
        n_time_clusters = state_dict['temporal.centers'].shape[0]
        if 'channel.corr_centers' in state_dict:
            n_channel_clusters = state_dict['channel.corr_centers'].shape[0]
        elif 'channel.centers' in state_dict:
            n_channel_clusters = state_dict['channel.centers'].shape[0]
        else:
            n_channel_clusters = 4
    else:
        n_time_clusters = state_dict['temporal.centers'].shape[0]
        time_hidden_dim = state_dict['temporal.centers'].shape[1]
        n_channel_clusters = 4

    config.model.n_time_clusters = n_time_clusters
    config.model.time_hidden_dim = time_hidden_dim
    config.model.n_channel_clusters = n_channel_clusters

    model = TCB_Net(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # 加载模型后打印时间簇中心信息
    print("\n【时间簇中心诊断】")
    centers = model.temporal.centers.cpu().numpy()  # shape (n_clusters, time_hidden_dim)
    print(f"中心点数量: {centers.shape[0]}, 维度: {centers.shape[1]}")
    # 计算中心点之间的欧氏距离
    for i in range(centers.shape[0]):
        for j in range(i+1, centers.shape[0]):
            dist = np.linalg.norm(centers[i] - centers[j])
            print(f"  簇{i} <-> 簇{j} 欧氏距离: {dist:.6f}")
    # 打印每个中心的前10维数值（示例）
    print("\n各簇中心前10维数值:")
    for i in range(centers.shape[0]):
        print(f"  簇{i}: {centers[i][:10]}")
    # 打印所有中心点的均值和标准差
    print(f"\n中心点全局均值: {centers.mean():.6f}, 标准差: {centers.std():.6f}")
    print("✅ 模型权重加载完成")

    # ---------- 加载数据 ----------
    data_dir = config.logging.data_dir
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    y_val = np.load(os.path.join(data_dir, 'val_labels.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # 统一形状为 (batch, time, channels)
    if X_train.shape[1] == 3 and X_train.shape[2] != 3:
        X_train = X_train.transpose(0, 2, 1)
        X_val = X_val.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    print(f"\n训练集: {len(X_train)} 样本, 异常率: {y_train.mean():.4f}")
    print(f"验证集: {len(X_val)} 样本, 异常率: {y_val.mean():.4f}")
    print(f"测试集: {len(X_test)} 样本, 异常率: {y_test.mean():.4f}")

    # 创建 DataLoader
    batch_size = 64
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---------- 动态阈值选择 ----------
    if custom_threshold is not None:
        best_thresh = custom_threshold
        print(f"使用手动指定阈值: {best_thresh}")
    else:
        # 调用动态阈值函数（需要提前定义）
        best_thresh = get_threshold_from_train_normal(
            model, train_loader, device,
            val_loader=val_loader,   # 可选微调
            percentile=99,
            fine_tune_range=0.2
        )
        print(f"动态阈值选择结果: {best_thresh:.4f}")

    # ---------- 测试集预测 ----------
    all_probs = []
    all_labels = []
    all_h_time = []
    all_time_weights = []
    all_channel_assign = []
    all_r_vec = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            # 时间聚类需要 (batch, channels, time)
            X_time = X_batch.permute(0, 2, 1)
            # 前向传播
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_batch.cpu().numpy())

            # 获取时间聚类信息
            h_time, w_time, _ = model.temporal(X_time)
            all_h_time.append(h_time.cpu().numpy())
            all_time_weights.append(w_time.cpu().numpy())

            # 获取通道聚类信息
            w_channel, r_vec = model.channel(X_batch)
            channel_assign = torch.argmax(w_channel, dim=1)
            all_channel_assign.extend(channel_assign.cpu().numpy())
            all_r_vec.append(r_vec.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_h_time = np.concatenate(all_h_time, axis=0)
    all_time_weights = np.concatenate(all_time_weights, axis=0)
    all_r_vec = np.concatenate(all_r_vec, axis=0)
    all_channel_assign = np.array(all_channel_assign)
    time_assign = np.argmax(all_time_weights, axis=1)

    # 使用阈值生成预测
    all_preds = (all_probs >= best_thresh).astype(int)

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print(f"测试集结果 (阈值={best_thresh:.4f})")
    print(f"准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print("混淆矩阵:")
    print(cm)
    print("=" * 60)

    # 保存指标 JSON
    save_dir = config.logging.save_dir
    os.makedirs(save_dir, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # 转换键
                if isinstance(k, (np.integer, np.floating)):
                    k = int(k) if isinstance(k, np.integer) else float(k)
                elif isinstance(k, np.bool_):
                    k = bool(k)
                new_dict[k] = convert_numpy(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(i) for i in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    metrics_json = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'threshold': best_thresh,
        'confusion_matrix': cm.tolist(),
    }
    metrics_json = convert_numpy(metrics_json)
    json_path = os.path.join(save_dir, f'tcb_net_{experiment_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"✅ 模型指标保存至: {json_path}")

    # ---------- 可解释性分析 ----------
    print("\n【可解释性分析】")

    # 获取模型配置标志（兼容旧模型，默认 True）
    use_time_clustering = getattr(model, 'use_time_clustering', True)
    use_channel_clustering = getattr(model, 'use_channel_clustering', True)

    # ========== 时间簇分析 ==========
    if use_time_clustering and all_time_weights is not None:
        actual_n_clusters = model.temporal.n_clusters
        normal_mask = (all_labels == 0)
        fault_mask = (all_labels == 1)

        print("\n各时间簇的平均权重（所有样本）:")
        for k in range(actual_n_clusters):
            print(f"  簇{k}: {all_time_weights[:, k].mean():.3f}")

        print("\n按标签分组的时间簇平均权重:")
        for k in range(actual_n_clusters):
            normal_w = all_time_weights[normal_mask, k].mean() if normal_mask.any() else 0.0
            fault_w = all_time_weights[fault_mask, k].mean() if fault_mask.any() else 0.0
            print(f"  簇{k}: 正常={normal_w:.3f}, 故障={fault_w:.3f}")

        dominant_cluster = np.argmax(all_time_weights, axis=1)
        print("\n主导时间簇分布:")
        unique, counts = np.unique(dominant_cluster, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  簇{u}: {c} 样本 ({c / len(dominant_cluster) * 100:.1f}%)")

        print("\n各主导簇的故障率:")
        for u in unique:
            mask = (dominant_cluster == u)
            fault_rate = all_labels[mask].mean()
            print(f"  簇{u}: {fault_rate:.2%}")
    else:
        print("时间聚类未启用或数据为空，跳过时间簇分析")
        actual_n_clusters = 0
        dominant_cluster = None
        unique = np.array([])
        counts = np.array([])

    # ========== 通道簇分析 ==========
    if use_channel_clustering and len(all_channel_assign) > 0:
        unique_chan, counts_chan = np.unique(all_channel_assign, return_counts=True)
        print("\n通道簇分布:")
        for u, c in zip(unique_chan, counts_chan):
            print(f"  簇{u}: {c} 样本 ({c / len(all_channel_assign) * 100:.1f}%)")

        print("\n各通道簇的故障率:")
        for u in unique_chan:
            mask = (all_channel_assign == u)
            fault_rate = all_labels[mask].mean()
            print(f"  簇{u}: {fault_rate:.2%}")

        # 物理模式匹配（仅当有足够样本时）
        preset_patterns = {
            0: np.array([0.5, 0.4, 0.4]),
            1: np.array([0.8, 0.2, 0.2]),
            2: np.array([0.6, 0.6, 0.6]),
            3: np.array([0.2, -0.2, 0.7])
        }
        pattern_names = {0: "正常耦合", 1: "不对中", 2: "切屑夹持", 3: "卡盘/冲击"}
        print("\n【通道簇物理模式匹配】")
        cluster_matching = {}
        for cluster_id in unique_chan:
            mask = (all_channel_assign == cluster_id)
            if mask.sum() == 0:
                continue
            mean_corr = all_r_vec[mask, :3].mean(axis=0)
            best_sim = -1
            best_pid = None
            for pid, pattern in preset_patterns.items():
                sim = np.dot(mean_corr, pattern) / (np.linalg.norm(mean_corr) * np.linalg.norm(pattern) + 1e-8)
                if sim > best_sim:
                    best_sim = sim
                    best_pid = pid
            cluster_matching[str(cluster_id)] = {
                "matched_pattern_id": int(best_pid) if best_pid is not None else None,
                "matched_pattern_name": pattern_names[best_pid] if best_pid is not None else None,
                "similarity": float(best_sim),
                "mean_correlation": mean_corr.tolist()
            }
            print(f"  通道簇{cluster_id}: 平均相关系数 = {mean_corr}, 最匹配模式 = {pattern_names[best_pid]} (相似度 = {best_sim:.3f})")
    else:
        print("通道聚类未启用或数据为空，跳过通道簇分析")
        unique_chan = np.array([])
        counts_chan = np.array([])
        cluster_matching = {}

    # ========== 双簇交叉分析 ==========
    if use_time_clustering and use_channel_clustering and dominant_cluster is not None and len(unique_chan) > 0:
        print("\n【双簇交叉分析】")
        for k in range(actual_n_clusters):
            for m in unique_chan:
                mask = (dominant_cluster == k) & (all_channel_assign == m)
                count = mask.sum()
                if count > 0:
                    fault_rate = all_labels[mask].mean()
                    print(f"  时间簇{k}+通道簇{m}: {count}样本, 故障率={fault_rate:.2%}")
    else:
        print("\n双簇交叉分析需要时间聚类和通道聚类同时启用，已跳过")

    # ========== 保存可解释性 JSON ==========
    interpretability = {}

    if use_time_clustering and all_time_weights is not None:
        interpretability.update({
            'n_time_clusters': actual_n_clusters,
            'time_cluster_weights': {f'cluster_{k}': float(all_time_weights[:, k].mean()) for k in range(actual_n_clusters)},
            'time_cluster_weights_by_label': {
                f'cluster_{k}': {
                    'normal': float(all_time_weights[normal_mask, k].mean()) if normal_mask.any() else 0.0,
                    'fault': float(all_time_weights[fault_mask, k].mean()) if fault_mask.any() else 0.0
                } for k in range(actual_n_clusters)
            },
            'dominant_cluster_distribution': {f'cluster_{u}': int(c) for u, c in zip(unique, counts)},
            'dominant_cluster_fault_rate': {f'cluster_{u}': float(all_labels[dominant_cluster == u].mean()) for u in unique}
        })
    else:
        interpretability['n_time_clusters'] = 0
        interpretability['time_cluster_weights'] = {}
        interpretability['time_cluster_weights_by_label'] = {}
        interpretability['dominant_cluster_distribution'] = {}
        interpretability['dominant_cluster_fault_rate'] = {}

    if use_channel_clustering and len(all_channel_assign) > 0:
        interpretability.update({
            'channel_cluster_distribution': {f'cluster_{u}': int(c) for u, c in zip(unique_chan, counts_chan)},
            'channel_cluster_fault_rate': {f'cluster_{u}': float(all_labels[all_channel_assign == u].mean()) for u in unique_chan},
            'channel_cluster_physical_matching': cluster_matching
        })
    else:
        interpretability['channel_cluster_distribution'] = {}
        interpretability['channel_cluster_fault_rate'] = {}
        interpretability['channel_cluster_physical_matching'] = {}

    # 递归转换 numpy 类型
    interpretability = convert_numpy(interpretability)

    # 保存 JSON 文件
    interp_path = os.path.join(save_dir, f'tcb_interpretability_{experiment_name}.json')
    with open(interp_path, 'w') as f:
        json.dump(interpretability, f, indent=2)
    print(f"\n✅ 可解释性分析保存至: {interp_path}")

    # ========== 域适应定量分析 ==========
    print("\n【域适应定量分析】")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 由于训练集样本量很大，随机采样 5000 个样本用于计算（可根据内存调整）
    num_samples_src = 5000
    num_samples_tgt = 5000

    # 提取训练集（源域）原始特征和隐特征（采样）
    train_flat_list = []
    train_h_list = []
    with torch.no_grad():
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            # 原始展平特征
            train_flat_list.append(X_batch.cpu().numpy().reshape(len(X_batch), -1))
            # 隐特征
            X_time = X_batch.permute(0,2,1)
            h_time, _, _ = model.temporal(X_time)
            train_h_list.append(h_time.cpu().numpy())
    train_flat_all = np.concatenate(train_flat_list, axis=0)
    train_h_all = np.concatenate(train_h_list, axis=0)
    if len(train_flat_all) > num_samples_src:
        idx_src = np.random.choice(len(train_flat_all), num_samples_src, replace=False)
        train_flat = train_flat_all[idx_src]
        train_h = train_h_all[idx_src]
    else:
        train_flat = train_flat_all
        train_h = train_h_all

    # 提取测试集（目标域）原始特征和隐特征（采样）
    test_flat_list = []
    test_h_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            test_flat_list.append(X_batch.cpu().numpy().reshape(len(X_batch), -1))
            X_time = X_batch.permute(0,2,1)
            h_time, _, _ = model.temporal(X_time)
            test_h_list.append(h_time.cpu().numpy())
    test_flat_all = np.concatenate(test_flat_list, axis=0)
    test_h_all = np.concatenate(test_h_list, axis=0)
    if len(test_flat_all) > num_samples_tgt:
        idx_tgt = np.random.choice(len(test_flat_all), num_samples_tgt, replace=False)
        test_flat = test_flat_all[idx_tgt]
        test_h = test_h_all[idx_tgt]
    else:
        test_flat = test_flat_all
        test_h = test_h_all

    # 定义 MMD 函数（使用随机采样和分块计算，避免内存爆炸）
    def compute_mmd_safe(source, target, gamma=0.001, batch_size=2000):
        """安全计算 RBF MMD，不构建全尺寸核矩阵"""
        n_s = len(source)
        n_t = len(target)
        # 为避免过大的矩阵，随机采样到 batch_size 以内？这里采用分块累加
        # 计算 K_ss
        k_ss = 0.0
        for i in range(0, n_s, batch_size):
            X_batch = source[i:i+batch_size]
            for j in range(0, n_s, batch_size):
                Y_batch = source[j:j+batch_size]
                K = rbf_kernel(X_batch, Y_batch, gamma=gamma)
                k_ss += K.sum()
        k_ss_avg = k_ss / (n_s * n_s)
        # 计算 K_tt
        k_tt = 0.0
        for i in range(0, n_t, batch_size):
            X_batch = target[i:i+batch_size]
            for j in range(0, n_t, batch_size):
                Y_batch = target[j:j+batch_size]
                K = rbf_kernel(X_batch, Y_batch, gamma=gamma)
                k_tt += K.sum()
        k_tt_avg = k_tt / (n_t * n_t)
        # 计算 K_st
        k_st = 0.0
        for i in range(0, n_s, batch_size):
            X_batch = source[i:i+batch_size]
            for j in range(0, n_t, batch_size):
                Y_batch = target[j:j+batch_size]
                K = rbf_kernel(X_batch, Y_batch, gamma=gamma)
                k_st += K.sum()
        k_st_avg = k_st / (n_s * n_t)
        mmd = np.sqrt(k_ss_avg + k_tt_avg - 2 * k_st_avg)
        return mmd

    def compute_coral(source, target):
        """CORAL 距离（协方差矩阵差异）"""
        cov_s = np.cov(source, rowvar=False)
        cov_t = np.cov(target, rowvar=False)
        return np.linalg.norm(cov_s - cov_t, ord='fro') / (source.shape[1] ** 0.5)

    def compute_a_distance(source, target, max_samples=5000):
        """A‑distance：训练线性SVM区分源域和目标域"""
        # 采样防止过慢
        if len(source) > max_samples:
            idx_s = np.random.choice(len(source), max_samples, replace=False)
            source = source[idx_s]
        if len(target) > max_samples:
            idx_t = np.random.choice(len(target), max_samples, replace=False)
            target = target[idx_t]
        X = np.vstack([source, target])
        y = np.hstack([np.zeros(len(source)), np.ones(len(target))])
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, y)
        pred = clf.predict(X)
        err = 1 - accuracy_score(y, pred)
        # 返回 A-distance (值越大表示域差异越小，常见取值范围 0~1)
        return 1 - 2 * err

    print("计算 MMD...")
    mmd_flat = compute_mmd_safe(train_flat, test_flat, gamma=0.001)
    mmd_h = compute_mmd_safe(train_h, test_h, gamma=0.001)

    print("计算 CORAL...")
    coral_flat = compute_coral(train_flat, test_flat)
    coral_h = compute_coral(train_h, test_h)

    print("计算 A-distance...")
    a_flat = compute_a_distance(train_flat, test_flat)
    a_h = compute_a_distance(train_h, test_h)

    print(f"原始空间: MMD={mmd_flat:.4f}, CORAL={coral_flat:.4f}, A-distance={a_flat:.4f}")
    print(f"隐空间:   MMD={mmd_h:.4f}, CORAL={coral_h:.4f}, A-distance={a_h:.4f}")

    # 保存结果
    domain_metrics = {
        'original': {'mmd': float(mmd_flat), 'coral': float(coral_flat), 'a_distance': float(a_flat)},
        'latent': {'mmd': float(mmd_h), 'coral': float(coral_h), 'a_distance': float(a_h)}
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'domain_adaptation_metrics_{experiment_name}.json'), 'w') as f:
        json.dump(domain_metrics, f, indent=2)
    print("域适应指标已保存。")

    # 保存图表（与原代码相同）
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    plt.title(f'TCB-Net 混淆矩阵 ({experiment_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tcb_confusion_matrix_{experiment_name}.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.imshow(all_time_weights[:200, :].T, aspect='auto', cmap='viridis')
    plt.colorbar(label='权重')
    plt.xlabel('样本序号')
    plt.ylabel('时间簇')
    plt.title(f'时间簇分配权重 ({experiment_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'tcb_time_cluster_weights_{experiment_name}.png'))
    plt.close()

    # 双簇质量分析（复用原函数，但需要 convert_numpy 已在上面定义）
    # from evaluate_tcb_net import compute_bicluster_quality, plot_cluster_scatter, plot_channel_corr_3d
    quality = compute_bicluster_quality(time_assign, all_channel_assign, all_h_time, all_labels, all_r_vec)
    quality = convert_numpy(quality)
    with open(os.path.join(save_dir, f'bicluster_quality_{experiment_name}.json'), 'w') as f:
        json.dump(quality, f, indent=2)

    plot_cluster_scatter(all_h_time, time_assign, all_channel_assign,
                         os.path.join(save_dir, f'cluster_scatter_{experiment_name}.png'))
    plot_channel_corr_3d(all_r_vec[:, :3], all_channel_assign,
                         os.path.join(save_dir, f'channel_corr_3d_{experiment_name}.png'))
    
    # ========== 新增图表（函数调用）==========
    save_dir = config.logging.save_dir  # 确保已经定义

    # 1. PR 曲线
    plot_pr_curve(all_labels, all_probs,
                os.path.join(save_dir, f'pr_curve_{experiment_name}.png'))

    # 2. 概率分布直方图
    plot_prob_dist(all_labels, all_probs,
                os.path.join(save_dir, f'prob_dist_{experiment_name}.png'))

    # 3. 双簇交叉故障率热力图
    n_time = actual_n_clusters
    n_chan = len(unique_chan)
    time_labels = [f'时间簇{i}' for i in range(n_time)]
    chan_labels = [f'通道簇{i}' for i in unique_chan]
    plot_bicluster_heatmap(dominant_cluster, all_channel_assign, all_labels,
                        n_time, n_chan, time_labels, chan_labels,
                        os.path.join(save_dir, f'bicluster_heatmap_{experiment_name}.png'))

    # 4. 时间簇权重抽样热力图
    plot_time_cluster_weights_sampled(all_time_weights,
                                    os.path.join(save_dir, f'time_cluster_weights_sampled_{experiment_name}.png'),
                                    n_sample=1000)

    # 5. 校准曲线
    plot_calibration_curve(all_labels, all_probs,
                        os.path.join(save_dir, f'calibration_curve_{experiment_name}.png'))

    # 6. ROC 曲线
    plot_roc_curve(all_labels, all_probs,
                os.path.join(save_dir, f'roc_curve_{experiment_name}.png'))

    print(f"\n图表已保存至: {save_dir}")


def visualize_domain_adaptation(data_dir, output_dir=None, num_train_samples=500):
    """
    补充实验：左图展示训练集+测试集全貌，右图只展示测试集并按时间簇着色。
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    import torch
    import os
    from torch.utils.data import DataLoader, TensorDataset

    # 设置全局字体大小
    plt.rcParams.update({
        'font.size': 14,           # 全局字体大小
        'axes.titlesize': 16,      # 子图标题字体大小
        'axes.labelsize': 14,      # 坐标轴标签字体大小
        'xtick.labelsize': 12,     # X轴刻度字体大小
        'ytick.labelsize': 12,     # Y轴刻度字体大小
        'legend.fontsize': 12,     # 图例字体大小
    })

    # 加载配置和模型（与 evaluate 中一致）
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'tcb_net.yaml')
    config = Config.from_yaml(config_path)
    config.logging.data_dir = data_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_name = os.path.basename(data_dir.rstrip('/\\'))
    model_path = os.path.join(config.logging.model_dir, f"tcb_net_best_{experiment_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 推断模型参数
    n_time_clusters = state_dict['temporal.centers'].shape[0]
    time_hidden_dim = state_dict['temporal.fc.weight'].shape[0] if 'temporal.fc.weight' in state_dict else 128
    config.model.n_time_clusters = n_time_clusters
    config.model.time_hidden_dim = time_hidden_dim
    model = TCB_Net(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("模型加载完成，开始提取隐特征...")

    # 加载数据
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # 统一形状为 (batch, time, channels)
    if X_train.shape[1] == 3:
        X_train = X_train.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)

    # ---------- 采样 ----------
    # 训练集：早期健康、早期故障各采样 num_train_samples 个
    train_normal_idx = np.where(y_train == 0)[0]
    train_fault_idx  = np.where(y_train == 1)[0]
    num_train_normal = min(num_train_samples, len(train_normal_idx))
    num_train_fault  = min(num_train_samples, len(train_fault_idx))
    train_normal_sample = np.random.choice(train_normal_idx, num_train_normal, replace=False)
    train_fault_sample  = np.random.choice(train_fault_idx,  num_train_fault,  replace=False)

    # 测试集全量（晚期健康、晚期故障）
    test_normal_idx = np.where(y_test == 0)[0]
    test_fault_idx  = np.where(y_test == 1)[0]

    print(f"采样数量: 早期健康={len(train_normal_sample)}, 早期故障={len(train_fault_sample)}, "
          f"晚期健康={len(test_normal_idx)}, 晚期故障={len(test_fault_idx)}")

    # ----- 左图：训练集+测试集全量（用于观察漂移）-----
    def extract_features(X_batch, y_batch, label_type):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_batch).to(device)
            X_time = X_tensor.permute(0, 2, 1)
            h_time, _, _ = model.temporal(X_time)
            return h_time.detach().cpu().numpy(), [label_type] * len(X_batch)

    h_left = []
    labels_left = []
    # 早期健康
    h, lbl = extract_features(X_train[train_normal_sample], y_train[train_normal_sample], 0)
    h_left.extend(h); labels_left.extend(lbl)
    # 早期故障
    h, lbl = extract_features(X_train[train_fault_sample], y_train[train_fault_sample], 3)
    h_left.extend(h); labels_left.extend(lbl)
    # 晚期健康
    h, lbl = extract_features(X_test[test_normal_idx], y_test[test_normal_idx], 1)
    h_left.extend(h); labels_left.extend(lbl)
    # 晚期故障
    h, lbl = extract_features(X_test[test_fault_idx], y_test[test_fault_idx], 2)
    h_left.extend(h); labels_left.extend(lbl)
    h_left = np.array(h_left)
    labels_left = np.array(labels_left)

    # 左图 PCA
    pca_left = PCA(n_components=2)
    h_left_pca = pca_left.fit_transform(h_left)
    print(f"左图 PCA 解释方差比: {pca_left.explained_variance_ratio_}")

    # ----- 右图：仅测试集（用于观察时间簇在测试集上的分布）-----
    # 提取测试集所有样本的 h_time 和簇分配
    X_test_all = X_test  # shape (N, time, channels)
    N_test = len(X_test_all)
    h_test_list = []
    cluster_test_list = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, N_test, batch_size):
            batch = torch.FloatTensor(X_test_all[i:i+batch_size]).to(device)
            X_time = batch.permute(0, 2, 1)
            h_time, _, hard_assign = model.temporal(X_time)
            h_test_list.append(h_time.detach().cpu().numpy())
            cluster_test_list.append(hard_assign.detach().cpu().numpy())
    h_test = np.concatenate(h_test_list, axis=0)
    cluster_test = np.concatenate(cluster_test_list, axis=0)

    # 右图 PCA (单独对测试集计算)
    pca_right = PCA(n_components=2)
    h_test_pca = pca_right.fit_transform(h_test)
    print(f"右图 PCA 解释方差比: {pca_right.explained_variance_ratio_}")

    # ---------- 绘图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：按样本类型着色
    colors_left = {0: 'green', 1: 'blue', 2: 'red', 3: 'orange'}
    labels_left_map = {0: '早期健康', 1: '晚期健康', 2: '晚期故障', 3: '早期故障'}
    for lt in [0,1,2,3]:
        mask = labels_left == lt
        if mask.sum() == 0:
            continue
        axes[0].scatter(h_left_pca[mask,0], h_left_pca[mask,1],
                        c=colors_left[lt], label=labels_left_map[lt],
                        alpha=0.3, s=10)
    axes[0].set_title('PCA 隐空间分布（训练集采样+测试集全量）')
    axes[0].legend()
    axes[0].grid(True)

    # 右图：按时间簇着色（仅测试集）
    cmap = plt.cm.tab10
    sc = axes[1].scatter(h_test_pca[:,0], h_test_pca[:,1], c=cluster_test,
                         cmap=cmap, alpha=0.3, s=10, vmin=0, vmax=3)
    # 创建离散图例（方框）
    import matplotlib.patches as mpatches
    legend_handles = []
    for cluster_id in range(4):   # 假设有4个时间簇
        color = cmap(cluster_id / 3)   # 归一化到 [0,1]
        patch = mpatches.Patch(color=color, label=f'簇 {cluster_id}')
        legend_handles.append(patch)
    axes[1].legend(handles=legend_handles, title='时间簇 ID', loc='upper right')
    axes[1].set_title('PCA 隐空间分布（仅测试集，按时间簇着色）')
    axes[1].grid(True)

    if output_dir is None:
        output_dir = config.logging.save_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'domain_adaptation_pca_{experiment_name}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"可视化图保存至: {save_path}")

    # 打印仅测试集中各时间簇的故障率（供参考）
    print("\n【仅测试集各时间簇的故障率】")
    for c in range(n_time_clusters):
        mask = cluster_test == c
        if mask.sum() > 0:
            fault_rate = np.mean(y_test[mask])
            print(f"  簇{c}: 样本数 {mask.sum()}, 故障率 {fault_rate:.2%}")
        else:
            print(f"  簇{c}: 样本数 0")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--threshold', type=float, default=None, help='手动指定阈值（不指定则自动搜索）')
    parser.add_argument('--visualize_drift', action='store_true', help='执行跨时段可视化补充实验')
    args = parser.parse_args()
    if args.visualize_drift:
        visualize_domain_adaptation(args.data_dir, args.output_dir)
    else:
        evaluate(data_dir=args.data_dir, output_dir=args.output_dir, custom_threshold=args.threshold)

if __name__ == '__main__':
    main()