import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Tuple, Type, Any, Optional

from graph_cbm.modeling.relation.transformer import PositionEmbeddingRandom, Block, TransformerEncoder
from graph_cbm.utils.boxes import obj_prediction_nms


def relation_loss(proposals, rel_labels, relation_logits, refine_obj_logits):
    criterion_loss = nn.CrossEntropyLoss()
    relation_logits = torch.concat(relation_logits, dim=0)
    # refine_obj_logits = torch.concat(refine_obj_logits, dim=0)

    # fg_labels = torch.concat([proposal["gt_labels"] for proposal in proposals], dim=0)
    rel_labels = torch.concat(rel_labels, dim=0)

    loss_relation = criterion_loss(relation_logits, rel_labels.long())
    # loss_refine_obj = criterion_loss(refine_obj_logits, fg_labels.long())
    return loss_relation, 0


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
            later_nms_pred_thres=0.5,
    ):
        super(Predictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.obj_classes = obj_classes
        self.representation_dim = representation_dim
        self.feature_extractor = feature_extractor
        self.later_nms_pred_thres = later_nms_pred_thres

        self.position_encoder = PositionEmbeddingRandom(num_pos_feats=embedding_dim // 2)
        self.lin_obj = nn.Linear(3712, hidden_dim)
        self.lin_edge = nn.Linear(representation_dim + embedding_dim, embedding_dim)

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.obj_embed1 = nn.Embedding(self.obj_classes, embedding_dim)
        self.obj_embed2 = nn.Embedding(self.obj_classes, embedding_dim)

        self.obj_encoder = TransformerEncoder(representation_dim + embedding_dim + 128, num_heads, 2, hidden_dim)
        self.edge_encoder = TransformerEncoder(hidden_dim, num_heads, 4,
                                               hidden_dim)

        self.obj_classifier = nn.Linear(hidden_dim, obj_classes)
        self.edge_classifier = nn.Linear(self.representation_dim, relation_classes)

        self.postprocessor_embedding = nn.Linear(hidden_dim, hidden_dim * 2)
        self.post_cat = nn.Linear(hidden_dim * 2, representation_dim)
        self.ctx_linear = nn.Linear(self.representation_dim, relation_classes)

    def post_processor(self, obj_logits, relation_logits, rel_pair_idxs, proposals):
        device = obj_logits[0].device
        result = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(
                zip(relation_logits, obj_logits, rel_pair_idxs, proposals)):
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
            obj_pred = obj_pred + 1

            # obj_pred = obj_prediction_nms(box["logits_boxes"], obj_logit, self.later_nms_pred_thres)
            # obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
            # obj_scores = obj_class_prob.view(-1)[obj_score_ind]

            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            batch_size = obj_class.shape[0]
            regressed_box_idxs = obj_class

            bbox = box["logits_boxes"][torch.arange(batch_size, device=device), regressed_box_idxs]

            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1

            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)

            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            result.append({
                "boxes": bbox,
                "obj_logit": obj_logit,
                "labels": obj_class,
                "scores": obj_scores,
                "rel_pair_idxs": rel_pair_idx,
                "pred_rel_scores": rel_class_prob,
                "pred_rel_labels": rel_labels,
            })
        return result

    def encode_box_info(self, proposals):
        boxes_info = []
        for proposal in proposals:
            boxes = proposal['boxes']
            img_size = proposal["image_size"]
            wid = img_size[0]
            hei = img_size[1]
            wh = boxes[:, 2:] - boxes[:, :2] + 1.0
            xy = boxes[:, :2] + 0.5 * wh
            w, h = wh.split([1, 1], dim=-1)
            x, y = xy.split([1, 1], dim=-1)
            x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
            info = torch.concat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                                 w * h / (wid * hei)], dim=-1).view(-1, 9)
            boxes_info.append(info)
        return torch.concat(boxes_info, dim=0)

    def forward(self, box_features, proposals, rel_pair_idxs, union_features, rel_labels):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [b["boxes"].shape[0] for b in proposals]
        proposal_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0)
        box_features = self.feature_extractor(box_features)
        union_features = self.feature_extractor(union_features)
        box_features = box_features.split(num_objs, dim=0)
        union_features = union_features.split(num_rels, dim=0)
        pos_embeds = self.bbox_embed(self.encode_box_info(proposals))
        pos_embeds = pos_embeds.split(num_objs, dim=0)
        rel_features = []
        for i, (box_feature, union_feature, rel_pair_idx, proposal, pos_embed) in enumerate(
                zip(box_features, union_features, rel_pair_idxs, proposals, pos_embeds)):
            head_representation = box_feature[rel_pair_idx[:, 0]].contiguous()
            tail_representation = box_feature[rel_pair_idx[:, 1]].contiguous()
            pos_embed1 = pos_embed[rel_pair_idx[:, 0]].contiguous()
            pos_embed2 = pos_embed[rel_pair_idx[:, 1]].contiguous()
            pos_feature = pos_embed1 + pos_embed2
            obj_feature = torch.cat([head_representation, tail_representation, union_feature, pos_feature], dim=-1)
            labels = proposal["labels"]
            labels_representation = self.obj_embed1(labels)
            head_labels_representation = labels_representation[rel_pair_idx[:, 0]].contiguous()
            tail_labels_representation = labels_representation[rel_pair_idx[:, 1]].contiguous()
            labels_feature = torch.cat([head_labels_representation, tail_labels_representation], dim=-1)
            rel_feature = torch.cat([obj_feature, labels_feature], dim=-1)
            rel_features.append(rel_feature)

        rel_features = self.lin_obj(torch.concat(rel_features, dim=0))
        prod_rep = self.edge_encoder(rel_features, num_rels)
        rel_logits = self.ctx_linear(self.postprocessor_embedding(prod_rep))
        obj_logits = proposal_logits

        obj_logits = obj_logits.split(num_objs, dim=0)
        rel_logits = rel_logits.split(num_rels, dim=0)

        result = self.post_processor(obj_logits, rel_logits, rel_pair_idxs, proposals)

        losses = {}
        if self.training:
            loss_relation, loss_refine = relation_loss(proposals, rel_labels, rel_logits, obj_logits)
            losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        return box_features, result, losses


    # def forward(self, feature, proposals, rel_pair_idxs, union_features, rel_labels):
    #     _, _, h, w = feature.shape
    #     pe_layer = self.position_encoder((h, w)).unsqueeze(0)
    #     pe_layer = torch.repeat_interleave(pe_layer, feature.shape[0], dim=0)
    #     box_feature = feature + pe_layer
    #     box_feature = self.feature_extractor(box_feature)
    #     pos_embed = self.bbox_embed(self.encode_box_info(proposals))
    #
    #     proposal_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
    #     proposal_embedding = self.obj_embed1(proposal_labels)
    #     proposal_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0)
    #     # proposal_embedding = F.softmax(proposal_logits, dim=1) @ self.obj_embed1.weight
    #
    #     num_objs = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]
    #     obj_representation = torch.concat((box_feature, proposal_embedding, pos_embed), -1)
    #
    #     obj_embedding = self.obj_encoder(obj_representation, num_objs)
    #     obj_preds = proposal_labels
    #     obj_logits = proposal_logits
    #
    #     obj_embed2 = self.obj_embed2(obj_preds.long())
    #     obj_features = torch.concat((box_feature, obj_embedding, obj_embed2), dim=-1)
    #
    #     edge_ctx = self.edge_encoder(obj_features, num_objs)
    #     edge_representation = self.postprocessor_embedding(edge_ctx)
    #
    #     edge_representation = edge_representation.view(edge_representation.size(0), 2, self.hidden_dim)
    #     head_representation = edge_representation[:, 0].contiguous().view(-1, self.hidden_dim)
    #     tail_representation = edge_representation[:, 1].contiguous().view(-1, self.hidden_dim)
    #
    #     num_rels = [r.shape[0] for r in rel_pair_idxs]
    #     num_objs = [b["boxes"].shape[0] for b in proposals]
    #
    #     head_reps = head_representation.split(num_objs, dim=0)
    #     tail_reps = tail_representation.split(num_objs, dim=0)
    #     obj_preds = obj_preds.split(num_objs, dim=0)
    #
    #     prod_reps = []
    #     pair_preds = []
    #     for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
    #         prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
    #         pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
    #     prod_rep = torch.cat(prod_reps, dim=0)
    #     pair_pred = torch.cat(pair_preds, dim=0)
    #
    #     ctx_gate = self.post_cat(prod_rep)
    #     ctx_gate = ctx_gate * self.feature_extractor(union_features)
    #     rel_logits = self.edge_classifier(ctx_gate) + self.ctx_linear(prod_rep)
    #
    #     obj_logits = obj_logits.split(num_objs, dim=0)
    #     rel_logits = rel_logits.split(num_rels, dim=0)
    #
    #     result = self.post_processor(obj_logits, rel_logits, rel_pair_idxs, proposals)
    #
    #     losses = {}
    #     if self.training:
    #         loss_relation, loss_refine = relation_loss(proposals, rel_labels, rel_logits, obj_logits)
    #         losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
    #     return box_feature, result, losses
