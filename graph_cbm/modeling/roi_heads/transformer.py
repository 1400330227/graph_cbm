import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TransformerContext(nn.Module):
    def __init__(
            self,
            embedding_dim,
            weight_dim,
            hidden_dim,
            num_heads,
            feature_extractor,
            obj_classes,
            representation_size,
            depth=2,

    ):
        super(TransformerContext, self).__init__()
        self.obj_classes = obj_classes
        self.feature_extractor = feature_extractor
        self.obj_embed1 = nn.Embedding(self.obj_classes - 1, weight_dim)
        self.obj_embed2 = nn.Embedding(self.obj_classes - 1, weight_dim)
        self.position_encoder = PositionEmbeddingRandom(num_pos_feats=embedding_dim // 2)
        self.obj_ctx_attns = nn.ModuleList()
        self.edge_ctx_attns = nn.ModuleList()
        self.obj_mlp = MLP((representation_size + weight_dim), hidden_dim, hidden_dim, 2)
        self.edge_mlp = MLP((representation_size + weight_dim + hidden_dim), hidden_dim, hidden_dim, 2)
        self.embedding_dim = embedding_dim
        self.weight_dim = weight_dim
        self.hidden_dim = hidden_dim

        for _ in range(depth):
            block = Block(
                embedding_dim=(representation_size + weight_dim),
                num_heads=num_heads,
            )
            self.obj_ctx_attns.append(block)

        for _ in range(depth):
            block = Block(
                embedding_dim=(representation_size + weight_dim + hidden_dim),
                num_heads=num_heads,
            )
            self.edge_ctx_attns.append(block)
        self.decoder = nn.Linear((representation_size + weight_dim + hidden_dim), obj_classes - 1)

    def obj_ctx(self, obj_feats, proposals, obj_labels=None, boxes_per_cls=None, ctx_average=False):
        per_image = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]
        obj_feats = torch.stack(obj_feats.split(per_image, dim=0), dim=0)
        for obj_ctx_attn in self.obj_ctx_attns:
            obj_feats = obj_ctx_attn(obj_feats)
        encoder_reps = self.obj_mlp(obj_feats)
        decoder_reps = torch.concat([encoder_reps, obj_feats], dim=-1)
        obj_dists = self.decoder(decoder_reps)
        obj_dists = obj_dists.reshape(-1, (self.obj_classes - 1))
        nonzero_pred = obj_dists.max(dim=1)[1] + 1
        obj_preds = obj_labels.clone()
        is_bg = (obj_preds == 0).nonzero()
        if is_bg.dim() > 0:
            obj_preds[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
        obj_ctx = encoder_reps.reshape(-1, self.hidden_dim)
        return obj_dists, obj_preds, obj_ctx

    def edge_ctx(self, edge_feats, proposals):
        per_image = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]
        edge_feats = torch.stack(edge_feats.split(per_image, dim=0), dim=0)
        for edge_ctx_attn in self.edge_ctx_attns:
            edge_feats = edge_ctx_attn(edge_feats)
        encoder_reps = self.edge_mlp(edge_feats)
        edge_ctx = encoder_reps.reshape(-1, self.hidden_dim)
        return edge_ctx

    def forward(self, roi_features, proposals):
        _, _, h, w = roi_features.shape
        pos_embed = self.position_encoder((h, w)).unsqueeze(0)
        pos_embed = torch.repeat_interleave(pos_embed, roi_features.shape[0], dim=0)
        roi_features = roi_features + pos_embed
        roi_features = self.feature_extractor(roi_features)

        obj_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
        obj_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0).detach()
        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        obj_pre_rep = torch.concat((roi_features, obj_embed), -1)
        obj_dists, obj_preds, obj_ctx = self.obj_ctx(obj_pre_rep, proposals, obj_labels)
        obj_embed2 = self.obj_embed2(obj_preds.long())
        obj_rel_rep = torch.concat((obj_embed2, roi_features, obj_ctx), -1)
        edge_ctx = self.edge_ctx(obj_rel_rep, proposals)
        return obj_dists, obj_preds, edge_ctx


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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


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
    def __init__(self, embedding_dim, num_heads):
        super(Block, self).__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim, 2048)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.self_attn(q=x, k=x, v=x)
        x = shortcut + x
        x = x + self.feed_forward(self.norm2(x))
        return x

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs*self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:,:,0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:,:,1].contiguous().view(batch_size, 1, num_obj)

        return joint_prob.view(batch_size, num_obj*num_obj)  @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)
