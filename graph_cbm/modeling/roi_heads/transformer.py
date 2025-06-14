import math
import numpy as np
import torch
from torch import nn, Tensor


class TransformerContext(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_heads
    ):
        super(TransformerContext, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        print(obj_feats.shape)

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        print(inp_feats.shape)

    def forward(self, x, proposals, rel_pair_idxs, logger=None, all_average=False, ctx_average=False):
        print(x.shape)


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


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 10.0
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor):
        coords = 2 * coords - 1  # 将输入坐标从 [0,1] 线性映射到 [-1,1]
        coords = coords @ self.positional_encoding_gaussian_matrix  # 2×128
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        h_coord = torch.linspace(1, h, h, dtype=torch.float32, device=device)
        w_coord = torch.linspace(1, h, w, dtype=torch.float32, device=device)
        h_coord = (h_coord - 0.5) / h
        w_coord = (w_coord - 0.5) / w

        pe = self._pe_encoding(torch.stack(torch.meshgrid([h_coord, w_coord], indexing='ij'), dim=-1))

        return pe.permute(2, 0, 1)  # C x H x W
