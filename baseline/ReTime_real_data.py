import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class SimpleScaler:
    """
    简单的数据标准化器
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # 避免除以0
        self.std[self.std == 0] = 1
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        return data * self.std + self.mean


class EarlyStopping:
    """
    早停机制
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def save_checkpoint(self, model):
        # 这里可以添加保存模型的逻辑
        pass


class RelationalRetrieval(nn.Module):
    """
    关系检索模块
    """
    def __init__(self, damping=0.9):
        super(RelationalRetrieval, self).__init__()
        self.damping = damping

    def construct_extended_adjacency(self, A_prime, R):
        """
        构造扩展邻接矩阵
        Args:
            A_prime: 数据库的邻接矩阵 (N, N)
            R: 目标与数据库的关系向量 (N,)

        Returns:
            A: 扩展邻接矩阵 (N+1, N+1)
        """
        N = A_prime.shape[0]
        A = torch.zeros(N+1, N+1)
        A[:N, :N] = A_prime
        A[N, :N] = R  # 目标节点到数据库节点的连接
        A[:N, N] = R  # 数据库节点到目标节点的连接
        return A

    def normalize_adjacency(self, A):
        """
        归一化邻接矩阵
        Args:
            A: 邻接矩阵

        Returns:
            A_norm: 归一化后的邻接矩阵
        """
        # 添加自环
        A = A + torch.eye(A.shape[0])
        # 计算度矩阵
        D = torch.diag(torch.sum(A, dim=1))
        # 归一化
        D_inv_sqrt = torch.inverse(torch.sqrt(D))
        A_norm = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        return A_norm

    def random_walk_with_restart(self, A_norm, target_idx, max_iter=100, tol=1e-6):
        """
        带重启的随机游走
        Args:
            A_norm: 归一化邻接矩阵
            target_idx: 目标节点索引
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            p: 稳态概率分布
        """
        N = A_norm.shape[0]
        # 初始化概率分布
        p = torch.zeros(N)
        p[target_idx] = 1.0
        
        # 重启向量
        r = torch.zeros(N)
        r[target_idx] = 1.0
        
        # 迭代直到收敛
        for _ in range(max_iter):
            p_new = self.damping * torch.mv(A_norm.t(), p) + (1 - self.damping) * r
            if torch.norm(p_new - p) < tol:
                break
            p = p_new
        
        return p

    def select_top_k(self, p, database, K, T):
        """
        选出 Top-K 参考序列
        Args:
            p: 稳态概率分布
            database: 数据库中的时间序列集合 (N, T', v)
            K: 选出的参考序列数量
            T: 目标序列长度

        Returns:
            Y_k: K 个参考序列 (K, T, v)
        """
        # 排除目标节点（最后一个节点）
        p_without_target = p[:-1]
        
        # 获取 Top-K 索引
        _, top_k_indices = torch.topk(p_without_target, K)
        
        # 提取对应的序列
        Y_k = database[top_k_indices, :T, :]
        
        return Y_k

    def forward(self, X, database, A_prime, R, K):
        """
        前向传播
        Args:
            X: 目标时间序列 (T, v)
            database: 数据库中的时间序列集合 (N, T', v)
            A_prime: 数据库的邻接矩阵 (N, N)
            R: 目标与数据库的关系向量 (N,)
            K: 选出的参考序列数量

        Returns:
            Y_k: K 个参考序列 (K, T, v)
        """
        T = X.shape[0]

        # 1. 构造扩展邻接矩阵
        A = self.construct_extended_adjacency(A_prime, R)

        # 2. 归一化邻接矩阵
        A_norm = self.normalize_adjacency(A)

        # 3. Random Walk With Restart
        target_idx = A.shape[0] - 1  # 目标节点是最后一个节点
        p = self.random_walk_with_restart(A_norm, target_idx)

        # 4. 选出 Top-K 参考序列
        Y_k = self.select_top_k(p, database, K, T)

        return Y_k


class InputModule(nn.Module):
    """
    输入嵌入模块
    """
    def __init__(self, input_dim, d_model):
        super(InputModule, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def positional_encoding(self, seq_len, d_model):
        """
        位置编码
        """
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, sequences):
        """
        Args:
            sequences: (K+1, T, v) - K个参考序列 + 1个目标序列

        Returns:
            H_hat: (K+1, T, d) - 嵌入后的序列
        """
        K_plus_1, T, v = sequences.shape
        d_model = self.linear.out_features

        # 线性变换
        embedded = self.linear(sequences)

        # 添加位置编码
        pe = self.positional_encoding(T, d_model).to(sequences.device)
        embedded = embedded + pe.unsqueeze(0)  # 广播到所有序列

        # LayerNorm
        H_hat = self.layer_norm(embedded)

        return H_hat


class ContentAttention(nn.Module):
    """
    内容注意力（跨序列注意力）
    在每个时间步上，让目标序列与参考序列交互
    """
    def __init__(self, d_model, n_heads):
        super(ContentAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, H_hat):
        """
        Args:
            H_hat: (K+1, T, d) - K个参考序列 + 1个目标序列

        Returns:
            H_hat: (K+1, T, d) - 注意力后的序列
        """
        K_plus_1, T, d = H_hat.shape

        # 目标序列是最后一个序列
        target_seq = H_hat[-1].unsqueeze(0)  # (1, T, d)
        reference_seqs = H_hat[:-1]  # (K, T, d)

        # 合并所有序列用于计算注意力
        all_seqs = torch.cat([target_seq, reference_seqs], dim=0)  # (K+1, T, d)

        # 计算注意力
        # Reshape for multi-head attention
        Q = self.w_q(all_seqs).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)
        K = self.w_k(all_seqs).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)
        V = self.w_v(all_seqs).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (K+1, n_heads, T, T)

        # 在序列维度上计算注意力
        attn_weights = F.softmax(scores, dim=-1)  # (K+1, n_heads, T, T)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # (K+1, n_heads, T, d_k)

        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(K_plus_1, T, d)  # (K+1, T, d)

        # 输出投影
        output = self.w_o(attn_output)  # (K+1, T, d)

        # 残差连接和层归一化
        output = output + all_seqs

        return output


class TemporalAttention(nn.Module):
    """
    时间注意力（跨时间注意力）
    每条序列内部的 self-attention
    """
    def __init__(self, d_model, n_heads):
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, H_hat):
        """
        Args:
            H_hat: (K+1, T, d) - K个参考序列 + 1个目标序列

        Returns:
            H_hat: (K+1, T, d) - 注意力后的序列
        """
        K_plus_1, T, d = H_hat.shape

        # 计算注意力
        # Reshape for multi-head attention
        Q = self.w_q(H_hat).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)
        K = self.w_k(H_hat).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)
        V = self.w_v(H_hat).view(K_plus_1, T, self.n_heads, self.d_k).transpose(1, 2)  # (K+1, n_heads, T, d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (K+1, n_heads, T, T)

        # 在时间维度上计算注意力
        attn_weights = F.softmax(scores, dim=-1)  # (K+1, n_heads, T, T)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)  # (K+1, n_heads, T, d_k)

        # 拼接多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(K_plus_1, T, d)  # (K+1, T, d)

        # 输出投影
        output = self.w_o(attn_output)  # (K+1, T, d)

        # 残差连接和层归一化
        output = output + H_hat

        return output


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AggregationBlock(nn.Module):
    """
    聚合模块的一个块
    包含 Content Attention、Temporal Attention 和 FFN
    """
    def __init__(self, d_model, n_heads, d_ff):
        super(AggregationBlock, self).__init__()
        self.content_attn = ContentAttention(d_model, n_heads)
        self.temporal_attn = TemporalAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, H_hat):
        """
        Args:
            H_hat: (K+1, T, d) - K个参考序列 + 1个目标序列

        Returns:
            H_hat: (K+1, T, d) - 处理后的序列
        """
        # Content Attention
        H_hat = self.content_attn(H_hat)
        H_hat = self.norm1(H_hat)

        # Temporal Attention
        H_hat = self.temporal_attn(H_hat)
        H_hat = self.norm2(H_hat)

        # FFN
        H_hat = self.ffn(H_hat)
        H_hat = self.norm3(H_hat)

        return H_hat


class AggregationModule(nn.Module):
    """
    聚合模块
    堆叠多个 AggregationBlock
    """
    def __init__(self, d_model, n_heads, d_ff, n_layers):
        super(AggregationModule, self).__init__()
        self.layers = nn.ModuleList([
            AggregationBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

    def forward(self, H_hat):
        """
        Args:
            H_hat: (K+1, T, d) - K个参考序列 + 1个目标序列

        Returns:
            H_hat: (K+1, T, d) - 处理后的序列
        """
        for layer in self.layers:
            H_hat = layer(H_hat)

        return H_hat


class OutputModule(nn.Module):
    """
    输出模块
    """
    def __init__(self, d_model, output_dim):
        super(OutputModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, output_dim)
        )

    def forward(self, H_hat, output_length=None):
        """
        Args:
            H_hat: (K+1, T, d) - K个参考序列 + 1个目标序列
            output_length: 输出序列长度，如果为None则输出与输入相同长度

        Returns:
            X_hat: (output_length or T, v) - 预测序列
        """
        # 提取目标序列的 embedding（最后一个序列）
        X_embed = H_hat[-1]  # (T, d)

        if output_length is not None and output_length != X_embed.shape[0]:
            # 需要输出不同长度的序列
            # 使用最后一个时间步的隐状态，通过一个线性层生成目标长度的序列
            last_hidden = X_embed[-1:]  # (1, d)

            # 创建位置编码
            pos_encoding = torch.arange(output_length, device=X_embed.device).float().unsqueeze(-1)
            pos_encoding = pos_encoding / (output_length ** 0.5)

            # 扩展隐状态到目标长度
            repeated_hidden = last_hidden.repeat(output_length, 1)  # (output_length, d)

            # 添加位置信息
            X_embed_out = repeated_hidden + 0.1 * pos_encoding

            # 通过 MLP 输出预测序列
            X_hat = self.mlp(X_embed_out)  # (output_length, v)
        else:
            # 通过 MLP 输出预测序列（与输入相同长度）
            X_hat = self.mlp(X_embed)  # (T, v)

        return X_hat


class ContentSynthesis(nn.Module):
    """
    内容合成模块（Content Synthesis）
    """
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers):
        super(ContentSynthesis, self).__init__()
        self.input_module = InputModule(input_dim, d_model)
        self.aggregation_module = AggregationModule(d_model, n_heads, d_ff, n_layers)
        self.output_module = OutputModule(d_model, input_dim)

    def forward(self, X, Y_k, horizon=None):
        """
        Args:
            X: 目标序列 (T, v)
            Y_k: K 个参考序列 (K, T, v)
            horizon: 预测窗口长度，如果提供则输出horizon长度的序列

        Returns:
            X_hat: 预测序列 (horizon or T, v)
        """
        # 合并目标序列和参考序列
        sequences = torch.cat([Y_k, X.unsqueeze(0)], dim=0)  # (K+1, T, v)

        # 输入嵌入
        H_hat = self.input_module(sequences)  # (K+1, T, d)

        # 聚合模块
        H_hat = self.aggregation_module(H_hat)  # (K+1, T, d)

        # 输出模块
        X_hat = self.output_module(H_hat, output_length=horizon)  # (horizon or T, v)

        return X_hat


class ReTimeModel(nn.Module):
    """
    ReTime 模型
    整合检索与合成两部分
    """
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, K, damping=0.9):
        super(ReTimeModel, self).__init__()
        self.retrieval = RelationalRetrieval(damping)
        self.synthesis = ContentSynthesis(input_dim, d_model, n_heads, d_ff, n_layers)
        self.K = K

    def apply_mask(self, X, mask_ratio=0.5):
        """
        应用 mask
        Args:
            X: 完整序列 (T, v)
            mask_ratio: mask 的比例

        Returns:
            X_masked: mask 后的序列 (T, v)
            mask: mask 矩阵 (T, v)
        """
        T, v = X.shape
        mask = torch.rand(T, v) < mask_ratio
        X_masked = X.clone()
        X_masked[mask] = 0
        return X_masked, mask

    def forward(self, X, database, A_prime, R, horizon=None):
        """
        前向传播
        Args:
            X: 目标时间序列 (T, v) - 可能是 mask 后的序列
            database: 数据库中的时间序列集合 (N, T', v)
            A_prime: 数据库的邻接矩阵 (N, N)
            R: 目标与数据库的关系向量 (N,)
            horizon: 预测窗口长度，如果提供则输出horizon长度的序列

        Returns:
            X_hat: 预测序列 (horizon or T, v)
        """
        # 关系检索模块
        Y_k = self.retrieval(X, database, A_prime, R, self.K)  # (K, T, v)

        # 内容合成模块
        X_hat = self.synthesis(X, Y_k, horizon=horizon)  # (horizon or T, v)

        return X_hat


def load_dataset(dataset_path, look_back, horizon):
    """
    加载并预处理数据集
    Args:
        dataset_path: 数据集路径
        look_back: 回看窗口长度
        horizon: 预测窗口长度

    Returns:
        data: 预处理后的数据 (N, T, v)
        scaler: 标准化器
    """
    # 读取数据
    df = pd.read_parquet(dataset_path)

    # 提取数值列（排除日期列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[numeric_cols].values

    # 数据标准化
    scaler = SimpleScaler()
    data = scaler.fit_transform(data)

    # 处理缺失值
    data = np.nan_to_num(data)

    # 滑动窗口切分 - 结合回看窗口和预测窗口
    total_length = look_back + horizon
    sequences = []
    for i in range(len(data) - total_length + 1):
        sequences.append(data[i:i+total_length])

    sequences = np.array(sequences)

    return sequences, scaler


def train_on_dataset(dataset_path, dataset_name, look_back, horizon, device='cuda'):
    """
    在单个数据集上训练模型
    Args:
        dataset_path: 数据集路径
        dataset_name: 数据集名称
        look_back: 回看窗口长度
        horizon: 预测窗口长度
        device: 设备
    """
    print(f"\n{'='*50}")
    print(f"开始在数据集 {dataset_name} 上训练...")
    print(f"回看窗口: {look_back}, 预测窗口: {horizon}")
    print(f"{'='*50}")

    # 设置参数
    K = 5  # 选出的参考序列数量
    d_model = 64  # 模型维度
    n_heads = 4  # 注意力头数
    d_ff = 128  # FFN 隐藏层维度
    n_layers = 2  # 聚合模块层数
    batch_size = 16  # 批大小
    epochs = 100  # 最大训练轮数
    lr = 0.001  # 学习率
    patience = 10  # early stopping patience

    # 加载数据
    sequences, scaler = load_dataset(dataset_path, look_back, horizon)
    print(f"数据形状: {sequences.shape}")

    # 数据集划分 - 7:1:2 比例
    train_size = int(0.7 * len(sequences))
    val_size = int(0.1 * len(sequences))
    test_size = len(sequences) - train_size - val_size

    train_data = sequences[:train_size]
    val_data = sequences[train_size:train_size+val_size]
    test_data = sequences[train_size+val_size:]

    print(f"数据集划分 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 获取特征维度
    v = train_data.shape[2]

    # 创建模型
    model = ReTimeModel(v, d_model, n_heads, d_ff, n_layers, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)

    # 训练记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        # 随机采样批次
        for batch_idx in range(0, len(train_data), batch_size):
            batch_end = min(batch_idx + batch_size, len(train_data))
            batch_size_actual = batch_end - batch_idx

            # 创建数据库（从其他序列中随机选择）
            database_indices = np.random.choice(len(train_data), size=100, replace=False)
            database = torch.tensor(train_data[database_indices], dtype=torch.float32).to(device)
            N = database.shape[0]

            # 创建邻接矩阵（随机图）
            A_prime = torch.randn(N, N) > 0
            A_prime = A_prime.float()
            A_prime = A_prime / (A_prime.sum(dim=1, keepdim=True) + 1e-8)

            # 遍历批次中的每个序列
            for i in range(batch_idx, batch_end):
                # 目标序列（包含回看窗口和预测窗口）
                full_sequence = torch.tensor(train_data[i], dtype=torch.float32).to(device)

                # 分离回看窗口和预测窗口
                X_input = full_sequence[:look_back]  # 回看窗口作为输入
                X_target = full_sequence[look_back:]  # 预测窗口作为目标

                # 应用 mask（只在回看窗口上应用）
                X_masked, mask = model.apply_mask(X_input, mask_ratio=0.5)

                # 创建关系向量（随机）
                R = torch.randn(N).to(device)

                # 前向传播
                X_pred = model(X_masked, database, A_prime, R, horizon=horizon)

                # 计算损失
                loss = criterion(X_pred, X_target)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        train_loss /= len(train_data)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_end = min(i + batch_size, len(val_data))

                # 创建数据库
                database_indices = np.random.choice(len(train_data), size=100, replace=False)
                database = torch.tensor(train_data[database_indices], dtype=torch.float32).to(device)
                N = database.shape[0]
                A_prime = torch.randn(N, N) > 0
                A_prime = A_prime.float()
                A_prime = A_prime / (A_prime.sum(dim=1, keepdim=True) + 1e-8)

                for j in range(i, batch_end):
                    full_sequence = torch.tensor(val_data[j], dtype=torch.float32).to(device)

                    # 分离回看窗口和预测窗口
                    X_input = full_sequence[:look_back]
                    X_target = full_sequence[look_back:]

                    X_masked, mask = model.apply_mask(X_input, mask_ratio=0.5)
                    R = torch.randn(N).to(device)

                    X_pred = model(X_masked, database, A_prime, R, horizon=horizon)
                    loss = criterion(X_pred, X_target)
                    val_loss += loss.item()

        val_loss /= len(val_data)
        val_losses.append(val_loss)

        # 打印进度
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./checkpoint/ReTime_best_model_{dataset_name}.pth')

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    print(f"\n训练完成！用时: {training_time:.2f}秒")
    print(f"最佳验证损失: {best_val_loss:.6f}")

    # 测试阶段
    print("\n开始测试...")
    model.load_state_dict(torch.load(f'./checkpoint/ReTime_best_model_{dataset_name}.pth'))
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_end = min(i + batch_size, len(test_data))

            database_indices = np.random.choice(len(train_data), size=100, replace=False)
            database = torch.tensor(train_data[database_indices], dtype=torch.float32).to(device)
            N = database.shape[0]
            A_prime = torch.randn(N, N) > 0
            A_prime = A_prime.float()
            A_prime = A_prime / (A_prime.sum(dim=1, keepdim=True) + 1e-8)

            for j in range(i, batch_end):
                full_sequence = torch.tensor(test_data[j], dtype=torch.float32).to(device)

                # 分离回看窗口和预测窗口
                X_input = full_sequence[:look_back]
                X_target = full_sequence[look_back:]

                X_masked, mask = model.apply_mask(X_input, mask_ratio=0.8)  # 更高的mask比例
                R = torch.randn(N).to(device)

                X_pred = model(X_masked, database, A_prime, R, horizon=horizon)
                loss = criterion(X_pred, X_target)
                test_loss += loss.item()

    test_loss /= len(test_data)
    print(f"测试损失: {test_loss:.6f}")

    return {
        'dataset': dataset_name,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'training_time': training_time
    }


def main():
    """
    主函数：在所有数据集上训练模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据集配置 - 每个数据集有不同的参数
    # 数据集配置 - 每个数据集有不同的参数
    dataset_configs = {
        'Energy': {
            'path': '../dataset/Time-MMD/numerical/Energy/Energy.parquet',
            'look_back': 36,
            'horizons': [12, 24, 36, 48]
        },
        'Climate': {
            'path': '../dataset/Time-MMD/numerical/Climate/Climate.parquet',
            'look_back': 8,
            'horizons': [6, 8, 10, 12]
        },
        'Health_US': {
            'path': '../dataset/Time-MMD/numerical/Health_US/Health_US.parquet',
            'look_back': 36,
            'horizons': [12, 24, 36, 48]
        },
        'Environment': {
            'path': '../dataset/Time-MMD/numerical/Environment/Environment.parquet',
            'look_back': 96,
            'horizons': [48, 96, 192, 336]
        }
    }

    # 存储所有结果
    all_results = []

    # 在每个数据集的每个配置上训练
    for dataset_name, config in dataset_configs.items():
        # 添加调试信息
        print(f"当前工作目录: {os.getcwd()}")
        print(f"检查文件路径: {config['path']}")
        print(f"文件是否存在: {os.path.exists(config['path'])}")
        
        if os.path.exists(config['path']):
            print(f"\n{'='*80}")
            print(f"处理数据集: {dataset_name}")
            print(f"{'='*80}")

            for horizon in config['horizons']:
                result = train_on_dataset(
                    config['path'],
                    f"{dataset_name}_lb{config['look_back']}_h{horizon}",
                    config['look_back'],
                    horizon,
                    device
                )
                all_results.append(result)
        else:
            print(f"警告: 数据集 {config['path']} 不存在，跳过...")

    # 打印总结
    print("\n" + "="*100)
    print("所有数据集训练完成！")
    print("="*100)

    print("\n各数据集测试结果:")
    print("-"*80)
    for result in all_results:
        print(f"{result['dataset']}:")
        print(f"  - 最佳验证损失: {result['best_val_loss']:.6f}")
        print(f"  - 测试损失: {result['test_loss']:.6f}")
        print(f"  - 训练时间: {result['training_time']:.2f}秒")
        print()


if __name__ == "__main__":
    main()
