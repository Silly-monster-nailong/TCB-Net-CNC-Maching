#!/usr/bin/env python
"""训练CNC异常检测模型 - 支持SA-FNO"""
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== FocalLoss类 ==========
class FocalLoss(nn.Module):
    """Focal Loss for binary classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        logits: 模型输出(logits), shape: [batch_size]
        targets: 真实标签, shape: [batch_size], 值在0-1之间
        """
        # 使用带logits的BCE，避免数值问题
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算概率
        pt = torch.exp(-BCE_loss)  # pt = p if y=1 else 1-p
        
        # Focal Loss公式
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
    
    def __repr__(self):
        return f'FocalLoss(alpha={self.alpha}, gamma={self.gamma})'

# ========== 数据增强类（小样本专用） ==========
class SmallSampleAugmentation:
    """小样本数据增强策略"""
    
    @staticmethod
    def time_warp(x, sigma=0.2):
        """时间扭曲"""
        batch, seq_len, channels = x.shape
        if seq_len < 10:
            return x
        
        # 随机缩放
        scale = 1.0 + torch.randn(1).item() * sigma
        new_len = int(seq_len * scale)
        new_len = max(min(new_len, seq_len + 20), seq_len - 20)
        
        # 插值
        x_np = x.cpu().numpy()
        x_warped = np.zeros((batch, new_len, channels))
        
        for b in range(batch):
            for c in range(channels):
                x_warped[b, :, c] = np.interp(
                    np.linspace(0, seq_len-1, new_len),
                    np.arange(seq_len),
                    x_np[b, :, c]
                )
        
        return torch.FloatTensor(x_warped).to(x.device)
    
    @staticmethod
    def add_noise(x, noise_level=0.03):
        """添加高斯噪声"""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    @staticmethod
    def freq_mask(x, max_mask=0.2):
        """频域掩码（增强鲁棒性）"""
        # 转换到频域
        x_fft = torch.fft.rfft(x, dim=1)
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # 随机掩码
        n_freq = x_fft.shape[1]
        mask_len = int(n_freq * max_mask)
        start = random.randint(0, n_freq - mask_len)
        
        mask = torch.ones_like(mag)
        mask[:, start:start+mask_len, :] = 0.5  # 衰减而不是完全掩码
        
        # 重建
        x_fft_masked = mag * mask * torch.exp(1j * phase)
        x_recon = torch.fft.irfft(x_fft_masked, n=x.shape[1], dim=1)
        
        return x_recon
    
    @classmethod
    def augment_batch(cls, x, y, aug_prob=0.5):
        """对批次应用增强"""
        if np.random.rand() < aug_prob:
            # 随机选择增强方式
            aug_type = random.choice(['warp', 'noise', 'freq'])
            
            if aug_type == 'warp':
                x = cls.time_warp(x)
            elif aug_type == 'noise':
                x = cls.add_noise(x)
            elif aug_type == 'freq':
                x = cls.freq_mask(x)
        
        return x, y

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from configs import Config

class CNCTrainer:
    def __init__(self, config_path):
        self.config = Config.from_yaml(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_sa_fno = False
        self.is_physical_fno = False
        self.is_binary_detector = False
        self.is_tcb_net = False
        # 获取小样本模式设置
        self.small_sample_mode = getattr(self.config.training, 'small_sample_mode', False)
        self.augmentation_prob = getattr(self.config.training, 'augmentation_prob', 0.3)
        
        # 根据配置选择模型类型
        model_type = getattr(self.config.model, 'type', 'duet_core')
        print(f"📋 配置的模型类型: {model_type}")
        
        if model_type == 'sa_fno_detector':
            print("🔧 尝试加载SA-FNO增强检测器...")
            try:
                # 添加调试信息
                import sys
                print(f"Python路径: {sys.path}")
                
                from models.hybrid_model import SAFNOCNCDetector, FNOSmallSampleDetector
                print("✅ 成功导入hybrid_model")
                
                if self.small_sample_mode:
                    print("  使用小样本优化版本")
                    self.model = FNOSmallSampleDetector(self.config).to(self.device)
                else:
                    print("  使用标准SA-FNO版本")
                    self.model = SAFNOCNCDetector(self.config).to(self.device)
                    
                self.is_binary_detector = True
                self.is_sa_fno = True
                print(f"  SA-FNO配置: hidden={getattr(self.config.model, 'hidden_dim', 32)}, "
                    f"modes={getattr(self.config.model, 'modes', 16)}")
                    
            except ImportError as e:
                print(f"⚠️  SA-FNO模块导入失败: {e}")
                print("  尝试直接导入cnc_detector...")
                from models.cnc_detector import CNCBinaryAnomalyDetector
                self.model = CNCBinaryAnomalyDetector(self.config).to(self.device)
                self.is_binary_detector = True
                self.is_sa_fno = False
                
        elif model_type == 'cnc_binary_detector':
            print("🔧 训练CNC频域二元异常检测器")
            from models.cnc_detector import CNCBinaryAnomalyDetector
            self.model = CNCBinaryAnomalyDetector(self.config).to(self.device)
            self.is_binary_detector = True
            self.is_sa_fno = False

        elif model_type == 'physical_fno_detector':
            print("🔧 训练物理约束FNO检测器")
            from models.physical_fno_detector_simple import PhysicalFNOAnomalyDetector
            self.model = PhysicalFNOAnomalyDetector(self.config).to(self.device)
            self.is_binary_detector = True
            self.is_sa_fno = False
            self.is_physical_fno = True   # 新增标志

        elif model_type == 'tcb_net':
            print("🔧 训练 TCB-Net")
            from models.tcb_net import TCB_Net
            self.model = TCB_Net(self.config).to(self.device)
            self.is_binary_detector = True
            self.is_sa_fno = False
            self.is_tcb_net = True   # 明确设为 True

        elif getattr(self.config.model, 'use_frequency_features', False):
            print("🔧 训练时频域DUET模型")
            from models.duet_frequency import DUETFrequency
            self.model = DUETFrequency(self.config).to(self.device)
            self.is_binary_detector = False
            self.is_sa_fno = False
    
        else:
            print("🔧 训练时域DUET模型")
            from models.duet_core import DUETCore
            self.model = DUETCore(self.config).to(self.device)
            self.is_binary_detector = False
            self.is_sa_fno = False
        
        # 打印模型类型确认
        print(f"✅ 实际加载的模型类型: {type(self.model).__name__}")
        
        # 创建输出目录
        os.makedirs(self.config.logging.model_dir, exist_ok=True)
        os.makedirs(self.config.logging.save_dir, exist_ok=True)
        
        # 可视化设置
        self.visualize = getattr(self.config.training, 'visualize', True)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"📊 模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
        
    def train(self):
        """训练模型"""
        print("开始训练CNC异常检测模型...")
        
        # 加载数据
        data_dir = self.config.logging.data_dir
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))

        # 创建训练标签
        if self.is_binary_detector:
            # 检查是否有半监督训练标签
            train_labels_path = os.path.join(data_dir, 'train_labels.npy')
            
            if os.path.exists(train_labels_path):
                # 加载预生成的半监督标签
                train_labels_np = np.load(train_labels_path)
                train_labels = torch.FloatTensor(train_labels_np).to(self.device)
                
                # 统计
                normal_count = np.sum(train_labels_np == 0)
                abnormal_count = np.sum(train_labels_np == 1)
                print(f"📊 使用半监督训练:")
                print(f"  训练集: {len(X_train)} 序列（正常:{normal_count}, 异常:{abnormal_count}）")
                
                # 小样本模式提示
                if self.small_sample_mode:
                    print(f"  🔄 小样本模式激活: 增强概率={self.augmentation_prob}")
                    
            else:
                # 纯无监督（仅正常数据）
                print("📊 使用无监督训练（仅正常数据）")
                train_labels = torch.zeros(len(X_train), device=self.device)
                print(f"  训练集: {len(X_train)} 序列（全正常）")
        else:
            # 重建模型：使用重建目标
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
            train_labels = torch.FloatTensor(y_train).to(self.device)
        
        # 创建数据加载器
        batch_size = self.config.training.batch_size
        
        if self.is_binary_detector:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                train_labels
            )
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                train_labels
            )
        
        train_loader = DataLoader(train_dataset, 
                                 batch_size=batch_size,
                                 shuffle=True)
        
        # 验证集
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))

        if self.is_binary_detector:
            # 检查是否有验证标签
            val_labels_path = os.path.join(data_dir, 'val_labels.npy')
            
            if os.path.exists(val_labels_path):
                # 加载预生成的验证标签
                val_labels_np = np.load(val_labels_path)
                val_labels = torch.FloatTensor(val_labels_np).to(self.device)
                
                # 统计
                normal_val = np.sum(val_labels_np == 0)
                abnormal_val = np.sum(val_labels_np == 1)
                print(f"  验证集: {len(X_val)} 序列（正常:{normal_val}, 异常:{abnormal_val}）")
            else:
                # 纯正常验证集
                val_labels = torch.zeros(len(X_val), device=self.device)
                print(f"  验证集: {len(X_val)} 序列（全正常）")
            
            # 创建验证数据集
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                val_labels
            )
        else:
            y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
            val_labels = torch.FloatTensor(y_val).to(self.device)
            
            # 创建验证数据集
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                val_labels
            )

        # 创建验证数据加载器
        val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
        
        # 训练模型
        history = self._train_epochs(train_loader, val_loader)
        
        # 保存模型和训练历史
        self._save_model(history)
        
        return history
    
    def _get_loss_function(self):
        """获取损失函数 - 动态调整"""
        if self.is_binary_detector:
            # 加载训练标签计算类别比例
            data_dir = self.config.logging.data_dir
            train_labels_path = os.path.join(data_dir, 'train_labels.npy')
            
            if os.path.exists(train_labels_path):
                train_labels = np.load(train_labels_path)
                abnormal_count = np.sum(train_labels == 1)
                total_count = len(train_labels)
                
                # 计算异常样本比例
                abnormal_ratio = abnormal_count / total_count
                
                # 动态调整Focal Loss参数
                if self.small_sample_mode:
                    # 小样本模式：更关注异常样本
                    alpha = min(0.9, 0.75 + (0.5 - abnormal_ratio) * 2)
                    gamma = 2.5  # 增加对困难样本的关注
                else:
                    alpha = 0.75
                    gamma = 2.0
                
                print(f"📊 Focal Loss配置:")
                print(f"  异常样本: {abnormal_count}/{total_count} ({abnormal_ratio:.2%})")
                print(f"  alpha={alpha:.3f}, gamma={gamma}")
                
                return FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
            else:
                print("⚠️  找不到训练标签，使用Focal Loss默认配置")
                return FocalLoss(alpha=0.75, gamma=2.0, reduction='mean')
        else:
            # 重建模型使用MSE或其他损失
            loss_type = getattr(self.config.training, 'loss_type', 'mse')
            
            if loss_type == 'mse':
                return nn.MSELoss()
            elif loss_type == 'mae':
                return nn.L1Loss()
            elif loss_type == 'huber':
                return nn.SmoothL1Loss()
            else:
                print(f"⚠️  未知损失类型: {loss_type}，使用MSE")
                return nn.MSELoss()
    
    def _train_epochs(self, train_loader, val_loader):
        """训练循环 - 支持小样本增强"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 小样本学习率预热
        warmup_epochs = getattr(self.config.training, 'warmup_epochs', 5) if self.small_sample_mode else 0
        
        criterion = self._get_loss_function()
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'train_acc': [] if self.is_binary_detector else [],
            'val_acc': [] if self.is_binary_detector else []
        }
        
        # 创建实时可视化
        if self.visualize:
            plt.ion()
            fig, axes = plt.subplots(2 if self.is_binary_detector else 1, 2, 
                                    figsize=(12, 6) if self.is_binary_detector else (12, 4))
            if self.is_binary_detector:
                axes = axes.reshape(2, 2)
        
        for epoch in range(self.config.training.epochs):
            # 学习率预热
            if epoch < warmup_epochs:
                lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scale * self.config.training.learning_rate
            
            # 训练
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                X_batch, labels_batch = batch
                X_batch = X_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                # 小样本数据增强
                if self.small_sample_mode and self.is_binary_detector:
                    # 只对异常样本进行增强？或者随机增强
                    X_batch, labels_batch = SmallSampleAugmentation.augment_batch(
                        X_batch, labels_batch, self.augmentation_prob
                    )
                
                optimizer.zero_grad()
                
                if self.is_binary_detector:
                        # 二元检测器前向传播
                        logits = self.model(X_batch)
                        
                        # 计算损失
                        if self.is_tcb_net:
                            # TCB-Net 专用：计算总损失（含辅助损失）
                            loss = self.model.compute_loss(logits, labels_batch)
                        else:
                            # 其他二元检测器（SA-FNO等）直接计算分类损失
                            loss = criterion(logits, labels_batch)
                        
                        # 计算准确率
                        probabilities = torch.sigmoid(logits)
                        pred_labels = (probabilities > 0.5).float()
                        train_correct += (pred_labels == labels_batch).sum().item()
                        train_total += len(labels_batch)
                    
                else:
                    # 重建模型前向传播
                    output = self.model(X_batch)
                    if isinstance(output, tuple):
                        reconstructed = output[0]
                    else:
                        reconstructed = output
                    
                    if reconstructed.shape[1] > X_batch.shape[1]:
                        reconstructed = reconstructed[:, :X_batch.shape[1], :]
                    
                    loss = criterion(reconstructed, X_batch)
                

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config.training.grad_clip)
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            if self.is_binary_detector:
                train_acc = train_correct / train_total if train_total > 0 else 0
                history['train_acc'].append(train_acc)
            
            # 验证
            val_metrics = self._validate(val_loader, criterion)
            avg_val_loss = val_metrics['loss']
            history['val_loss'].append(avg_val_loss)
            
            if self.is_binary_detector:
                history['val_acc'].append(val_metrics.get('accuracy', 0))
            
            # 学习率
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 打印进度
            log_msg = f'Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}'
            if self.is_binary_detector:
                log_msg += f', train_acc={train_acc:.4f}, val_acc={val_metrics.get("accuracy", 0):.4f}'
            log_msg += f', lr={history["lr"][-1]:.6f}'
            
            # 小样本模式特殊提示
            if self.small_sample_mode and epoch < warmup_epochs:
                log_msg += f' (warmup)'
            
            print(log_msg)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 实时可视化（每5轮更新一次）
            if self.visualize and (epoch + 1) % 5 == 0:
                self._plot_training_progress(history, axes, epoch + 1)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                if self.is_sa_fno:
                    model_prefix = 'sa_fno'
                elif getattr(self, 'is_physical_fno', False):
                    model_prefix = 'physical_fno'
                elif self.is_binary_detector:
                    model_prefix = 'cnc_binary'
                elif getattr(self.config.model, 'use_frequency_features', False):
                    model_prefix = 'duet_freq'
                else:
                    model_prefix = 'duet'
                
                torch.save(self.model.state_dict(), 
                          f'{self.config.logging.model_dir}/{model_prefix}_best.pth')
                print(f'  ✅ 保存最佳模型 (val_loss={best_val_loss:.6f})')
            else:
                patience_counter += 1
                if patience_counter >= self.config.training.patience:
                    print(f'  ⏹️  早停触发，第{epoch+1}轮')
                    break
        
        # 关闭交互模式，保存最终图像
        if self.visualize:
            plt.ioff()
            self._save_training_plots(history)
            plt.close()
        
        return history
    
    def _validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X_batch, labels_batch = batch
                X_batch = X_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                if self.is_binary_detector:
                    # 输出logits
                    logits = self.model(X_batch)
                    
                    # 计算损失
                    if self.is_tcb_net:
                        # 验证时仅使用分类损失（辅助损失可设为0）
                        loss = F.binary_cross_entropy_with_logits(logits, labels_batch)
                    else:
                        loss = criterion(logits, labels_batch)
                    
                    # 计算准确率
                    probabilities = torch.sigmoid(logits)
                    pred_labels = (probabilities > 0.5).float()
                    val_correct += (pred_labels == labels_batch).sum().item()
                    val_total += len(labels_batch)
                    
                else:
                    output = self.model(X_batch)
                    if isinstance(output, tuple):
                        reconstructed = output[0]
                    else:
                        reconstructed = output
                    
                    if reconstructed.shape[1] > X_batch.shape[1]:
                        reconstructed = reconstructed[:, :X_batch.shape[1], :]
                    
                    loss = criterion(reconstructed, X_batch)
                
                val_loss += loss.item()
        
        metrics = {
            'loss': val_loss / len(val_loader)
        }
        
        if self.is_binary_detector and val_total > 0:
            metrics['accuracy'] = val_correct / val_total
        
        return metrics
    
    def _plot_training_progress(self, history, axes, epoch):
        """绘制训练进度"""
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # 损失曲线
        ax_idx = 0
        axes[ax_idx, 0].clear()
        axes[ax_idx, 0].plot(epochs_range, history['train_loss'], 'b-', label='训练损失')
        axes[ax_idx, 0].plot(epochs_range, history['val_loss'], 'r-', label='验证损失')
        axes[ax_idx, 0].set_xlabel('Epoch')
        axes[ax_idx, 0].set_ylabel('损失')
        axes[ax_idx, 0].set_title(f'训练过程 (Epoch {epoch})')
        axes[ax_idx, 0].legend()
        axes[ax_idx, 0].grid(True, alpha=0.3)
        
        # 损失比例
        axes[ax_idx, 1].clear()
        if len(history['val_loss']) > 0:
            loss_ratio = [t/v if v > 0 else 1.0 
                         for t, v in zip(history['train_loss'], history['val_loss'])]
            axes[ax_idx, 1].plot(epochs_range, loss_ratio, 'g-')
            axes[ax_idx, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想线')
            axes[ax_idx, 1].set_xlabel('Epoch')
            axes[ax_idx, 1].set_ylabel('训练/验证损失比')
            axes[ax_idx, 1].set_title('过拟合监控')
            axes[ax_idx, 1].legend()
            axes[ax_idx, 1].grid(True, alpha=0.3)
        
        # 如果是二元检测器，添加准确率曲线
        if self.is_binary_detector and 'train_acc' in history:
            ax_idx = 1
            axes[ax_idx, 0].clear()
            if len(history['train_acc']) > 0:
                axes[ax_idx, 0].plot(epochs_range, history['train_acc'], 'b-', label='训练准确率')
                axes[ax_idx, 0].plot(epochs_range, history['val_acc'], 'r-', label='验证准确率')
                axes[ax_idx, 0].set_xlabel('Epoch')
                axes[ax_idx, 0].set_ylabel('准确率')
                axes[ax_idx, 0].set_title('分类准确率')
                axes[ax_idx, 0].legend()
                axes[ax_idx, 0].grid(True, alpha=0.3)
                axes[ax_idx, 0].set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.pause(0.1)
    
    def _save_training_plots(self, history):
        """保存训练图像"""
        if self.is_binary_detector:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes = axes.reshape(1, 2)
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # 1. 损失曲线
        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', linewidth=2, label='训练损失')
        axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', linewidth=2, label='验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练验证损失曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 损失比例
        loss_ratio = [t/v if v > 0 else 1.0 
                    for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[0, 1].plot(epochs_range, loss_ratio, 'g-', linewidth=2)
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='理想线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('训练/验证损失比')
        axes[0, 1].set_title('过拟合监控')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 准确率曲线（如果是二元检测器）
        if self.is_binary_detector and 'train_acc' in history:
            axes[1, 0].plot(epochs_range, history['train_acc'], 'b-', linewidth=2, label='训练准确率')
            axes[1, 0].plot(epochs_range, history['val_acc'], 'r-', linewidth=2, label='验证准确率')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('准确率')
            axes[1, 0].set_title('分类准确率')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1.1])
        
        # 4. 学习率变化
        if self.is_binary_detector:
            ax = axes[1, 1]
        else:
            ax = axes[0, 1].twinx()
        
        if 'lr' in history:
            ax.plot(epochs_range, history['lr'], color='purple', linestyle='-', linewidth=2)
            ax.set_ylabel('学习率', color='purple')
            if not self.is_binary_detector:
                ax.tick_params(axis='y', labelcolor='purple')
        
        plt.tight_layout()
        save_path = f'{self.config.logging.save_dir}/training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'📈 训练图像保存到: {save_path}')
    
    def _save_model(self, history):
        """保存模型和训练历史"""
        # 确定模型前缀
        if self.is_sa_fno:
            model_prefix = 'sa_fno'
        elif getattr(self, 'is_physical_fno', False):
            model_prefix = 'physical_fno'
        elif self.is_binary_detector:
            model_prefix = 'cnc_binary'
        elif getattr(self.config.model, 'use_frequency_features', False):
            model_prefix = 'duet_freq'
        else:
            model_prefix = 'duet'
        
        # 保存最终模型
        torch.save(self.model.state_dict(), 
                  f'{self.config.logging.model_dir}/{model_prefix}_final.pth')
        
        # 保存训练历史
        with open(f'{self.config.logging.save_dir}/training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        
        print(f'💾 模型保存到 {self.config.logging.model_dir}/{model_prefix}_*.pth')
        print(f'💾 训练历史保存到 {self.config.logging.save_dir}/training_history.json')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None,
                        help='预处理数据目录（包含 X_train.npy 等）')
    args = parser.parse_args()
    config_path = './configs/duet_anomaly.yaml'
    trainer = CNCTrainer(config_path)
    if args.data_dir is not None:
        trainer.config.logging.data_dir = args.data_dir
    trainer.train()
    print('✅ 训练完成')

if __name__ == '__main__':
    main()