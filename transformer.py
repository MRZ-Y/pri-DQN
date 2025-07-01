import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedPositionalEncoding(nn.Module):
    """固定位置编码（正弦余弦编码）"""

    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.pe[:x.size(1), :]


class CausalSelfAttention(nn.Module):
    """带因果掩码的自注意力层"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV投影
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 因果掩码（下三角矩阵）
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(100, 100), diagonal=0)
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # 计算QKV
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [tensor.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
                   for tensor in qkv]

        # 计算注意力得分
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用因果掩码（确保只关注过去的位置）
        mask = self.mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算输出
        output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(output)


class TransformerLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力和残差连接
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络和残差连接
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x


class TransformerPredictor(nn.Module):
    """基于Transformer的时间序列预测模型"""

    def __init__(
            self,
            input_dim,  # 输入特征维度
            output_dim,  # 输出特征维度
            d_model=256,  # 模型基础维度
            n_heads=8,  # 注意力头数量
            n_layers=3,  # Transformer层数
            d_ff=1024,  # 前馈网络中间层维度
            max_seq_len=20,  # 最大序列长度
            dropout=0.1,  # Dropout概率
            activation='gelu',  # 激活函数
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = FixedPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # 创建多层Transformer
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU() if activation == 'gelu' else nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x = self.input_proj(x)

        # 添加位置编码
        pos_emb = self.pos_embedding(x)
        x = self.dropout(x + pos_emb)

        # 通过多层Transformer
        for layer in self.layers:
            x = layer(x)

        # 只使用最后一个时间步的输出进行预测
        output = self.output_proj(x[:, -1, :])
        return output

# 初始化模型
model = TransformerPredictor(
    input_dim=5,        # 输入特征维度（如风速、温度等）
    output_dim=1,       # 输出维度（如预测的发电量）
    d_model=128,        # 模型维度
    n_heads=4,          # 注意力头数
    n_layers=2,         # Transformer层数
    max_seq_len=20,     # 最大序列长度
    dropout=0.1         # Dropout率
)

# 生成随机输入数据
x = torch.randn(32, 20, 5)  # [batch_size, seq_len, input_dim]

# 前向传播
output = model(x)  # [batch_size, output_dim]
print(f"输出形状: {output.shape}")  # 应输出: torch.Size([32, 1])