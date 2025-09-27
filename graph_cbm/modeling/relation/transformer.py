import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Type, Any, Optional

from torch.nn.utils.rnn import pad_sequence


def pad_packed_sequence(x, num_objs):
    recovered_chunks = []
    for i in range(len(num_objs)):
        seq_len = num_objs[i]
        recovered_chunks.append(x[i, :seq_len, :])
    recovered_x = torch.cat(recovered_chunks, dim=0)
    return recovered_x


class TransformerContext(nn.Module):
    def __init__(
            self,
            # transformer
            embedding_dim,
            num_heads,
            depth,
            # linear
            hidden_dim,
            use_cls: False
    ):
        super(TransformerContext, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.use_cls = use_cls
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = Block(embedding_dim=embedding_dim, num_heads=num_heads)
            self.blocks.append(block)
        self.mlp = MLP(embedding_dim, embedding_dim // 2, hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)
        if use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x, num_objs):
        x = pad_sequence(x.split(num_objs, dim=0), batch_first=True)
        if self.use_cls:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.ln(self.mlp(x))
        if self.use_cls:
            x = x[:, 0, :]
        else:
            x = pad_packed_sequence(x, num_objs)
        return x


class MultiLayerCrossAttentionFusion(nn.Module):
    def __init__(self, representation_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionFusion(representation_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, local_features, global_features):
        for layer in self.layers:
            local_features = layer(local_features, global_features)
        return local_features


class CrossAttentionFusion(nn.Module):
    def __init__(self, representation_dim, hidden_dim, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        # 线性变换层
        self.local_query = nn.Linear(representation_dim, hidden_dim)
        self.global_key = nn.Linear(hidden_dim, hidden_dim)
        self.global_value = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.output_proj = nn.Linear(hidden_dim, representation_dim)
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(representation_dim)
        self.norm2 = nn.LayerNorm(representation_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(representation_dim, representation_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(representation_dim * 4, representation_dim)
        )

    def forward(self, local_features, global_features):
        """
        local_features: [batch_size, representation_dim] 作为Query
        global_features: [batch_size, hidden_dim] 作为Key和Value
        """
        batch_size = local_features.size(0)

        # 保存残差连接
        residual = local_features

        # 线性变换
        Q = self.local_query(local_features)  # [batch_size, hidden_dim]
        K = self.global_key(global_features)  # [batch_size, hidden_dim]
        V = self.global_value(global_features)  # [batch_size, hidden_dim]

        # 多头注意力 reshape
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # 输出投影
        attn_output = self.output_proj(attn_output.squeeze(1))
        attn_output = self.dropout(attn_output)

        # 残差连接和层归一化
        attn_output = self.norm1(residual + attn_output)

        # FFN
        ffn_output = self.ffn(attn_output)
        output = self.norm2(attn_output + ffn_output)

        return output


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super(Block, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, mlp_dim=2)
        self.norm = nn.LayerNorm(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        q = self.norm1(x)
        x = x + self.dropout1(self.self_attn(q, q, q))
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x
