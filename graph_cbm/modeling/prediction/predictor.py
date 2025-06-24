import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Type, Any, Optional

from graph_cbm.modeling.prediction.transformer import PositionEmbeddingRandom, Block, TransformerEncoder


def relation_loss(proposals, rel_labels, relation_logits, refine_obj_logits):
    criterion_loss = nn.CrossEntropyLoss()
    relation_logits = torch.concat(relation_logits, dim=0)
    refine_obj_logits = torch.concat(refine_obj_logits, dim=0)

    fg_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
    rel_labels = torch.concat(rel_labels, dim=0)

    loss_relation = criterion_loss(relation_logits, rel_labels.long())
    loss_refine_obj = criterion_loss(refine_obj_logits, fg_labels.long())
    return loss_relation, loss_refine_obj


class Predictor(nn.Module):
    def __init__(
            self,
            obj_classes,
            relation_classes,
            feature_extractor,
            representation_dim=1024,
            embedding_dim=256,
            num_heads=8,
            hidden_dim=512,
    ):
        super(Predictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.obj_classes = obj_classes
        self.representation_dim = representation_dim
        self.feature_extractor = feature_extractor

        self.position_encoder = PositionEmbeddingRandom(num_pos_feats=embedding_dim // 2)
        self.lin_obj = nn.Linear(representation_dim + embedding_dim, embedding_dim)
        self.lin_edge = nn.Linear(representation_dim + embedding_dim, embedding_dim)

        self.obj_embed1 = nn.Embedding(self.obj_classes, embedding_dim)
        self.obj_embed2 = nn.Embedding(self.obj_classes, embedding_dim)

        self.obj_encoder = TransformerEncoder(representation_dim + embedding_dim, num_heads, 2, hidden_dim)
        self.edge_encoder = TransformerEncoder(representation_dim + embedding_dim + hidden_dim, num_heads, 2, hidden_dim)

        self.obj_classifier = nn.Linear(hidden_dim, obj_classes)
        self.edge_classifier = nn.Linear(self.representation_dim, relation_classes)

        self.postprocessor_embedding = nn.Linear(hidden_dim, hidden_dim * 2)
        self.post_cat = nn.Linear(hidden_dim * 2, representation_dim)
        self.ctx_linear = nn.Linear(self.representation_dim, relation_classes)

    def forward(self, features, proposals, rel_pair_idxs, union_features, rel_labels):
        _, _, h, w = features.shape
        pos_embed = self.position_encoder((h, w)).unsqueeze(0)
        pos_embed = torch.repeat_interleave(pos_embed, features.shape[0], dim=0)
        features = features + pos_embed
        features = self.feature_extractor(features)

        obj_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
        obj_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0).detach()
        obj_embedding = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        num_objs = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]
        obj_representation = torch.concat((features, obj_embedding), -1)

        obj_features = self.obj_encoder(obj_representation, num_objs)
        obj_logits = self.obj_classifier(obj_features)
        obj_preds = obj_logits[:, 1:].max(1)[1] + 1

        obj_embed2 = self.obj_embed2(obj_preds.long())
        edge_representation = torch.concat((features, obj_features, obj_embed2), dim=-1)

        edge_ctx = self.edge_encoder(edge_representation, num_objs)
        edge_representation = self.postprocessor_embedding(edge_ctx)

        edge_representation = edge_representation.view(edge_representation.size(0), 2, self.hidden_dim)
        head_representation = edge_representation[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_representation = edge_representation[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [b["boxes"].shape[0] for b in proposals]

        head_reps = head_representation.split(num_objs, dim=0)
        tail_reps = tail_representation.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)
        ctx_gate = ctx_gate * self.feature_extractor(union_features)
        rel_logits = self.edge_classifier(ctx_gate) + self.ctx_linear(prod_rep)
        obj_logits = obj_logits.split(num_objs, dim=0)
        rel_logits = rel_logits.split(num_rels, dim=0)

        if self.training:
            loss_relation, loss_refine = relation_loss(proposals, rel_labels, rel_logits, obj_logits)
            predictor_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
            return rel_logits, obj_logits, predictor_losses
        return rel_logits, obj_logits, {}
