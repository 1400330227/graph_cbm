import math
import torch
from torch import nn, Tensor

from graph_cbm.modeling.roi_heads.transformer import TransformerContext, PositionEmbeddingRandom, FrequencyBias


class RoIRelationPredictor(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_heads,
            feature_extractor,
            obj_classes,
            num_rel_cls,
            representation_size
    ):
        super(RoIRelationPredictor, self).__init__()
        self.weight_dim = 256
        self.hidden_dim = 512
        self.representation_size = representation_size
        self.obj_classes = obj_classes
        self.num_rel_cls = num_rel_cls
        self.context_layer = TransformerContext(
            embedding_dim=embedding_dim,
            weight_dim=self.weight_dim,
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            feature_extractor=feature_extractor,
            obj_classes=obj_classes,
            representation_size=representation_size,
        )
        # self.freq_bias = FrequencyBias(config, statistics)
        self.position_encoder = PositionEmbeddingRandom(num_pos_feats=embedding_dim // 2)
        self.feature_extractor = feature_extractor
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.representation_size)
        self.rel_compress = nn.Linear(self.representation_size, num_rel_cls)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals)
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [b["boxes"].shape[0] for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)
        prod_rep = prod_rep * union_features
        rel_dists = self.rel_compress(prod_rep)
        # rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)
        return obj_dists, rel_dists, {}


def build_roi_relation_predictor(
        embedding_dim,
        num_heads,
        feature_extractor,
        obj_classes,
        num_rel_cls,
        representation_size
):
    roi_relation_predictor = RoIRelationPredictor(
        embedding_dim,
        num_heads,
        feature_extractor,
        obj_classes,
        num_rel_cls,
        representation_size
    )
    return roi_relation_predictor
