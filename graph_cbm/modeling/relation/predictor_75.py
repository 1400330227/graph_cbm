import torch
import torch.nn.functional as F
from torch import nn

from graph_cbm.modeling.relation.transformer import PositionEmbeddingRandom, TransformerEncoder, TransformerEdgeEncoder


def relation_loss(rel_labels, relation_logits):
    criterion_loss = nn.CrossEntropyLoss()
    relation_logits = torch.concat(relation_logits, dim=0)
    rel_labels = torch.concat(rel_labels, dim=0)
    loss_relation = criterion_loss(relation_logits, rel_labels.long())

    # object_labels = [proposal['gt_labels'] for proposal in proposals]
    # object_labels = torch.concat(object_labels, dim=0)
    # object_logits = torch.concat(object_logits, dim=0)
    # loss_refine_obj = criterion_loss(object_logits, object_labels.long())
    return loss_relation


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Predictor(nn.Module):
    def __init__(
            self,
            obj_classes,
            relation_classes,
            feature_extractor_dim,
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
        self.feature_extractor = FeatureExtractor(feature_extractor_dim, representation_dim)
        self.later_nms_pred_thres = later_nms_pred_thres

        self.position_encoder = PositionEmbeddingRandom(num_pos_feats=embedding_dim // 2)
        self.lin_obj = nn.Linear(representation_dim + embedding_dim + 128, hidden_dim)
        self.lin_edge = nn.Linear(embedding_dim + hidden_dim, hidden_dim)

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.obj_embed1 = nn.Embedding(self.obj_classes, embedding_dim)
        self.obj_embed2 = nn.Embedding(self.obj_classes, embedding_dim)

        self.obj_encoder = TransformerEncoder(hidden_dim, num_heads, 4, hidden_dim)
        self.edge_encoder = TransformerEdgeEncoder(hidden_dim, num_heads, 4, hidden_dim)

        self.obj_classifier = nn.Linear(hidden_dim, obj_classes)
        self.edge_classifier = nn.Linear(self.representation_dim, relation_classes)

        self.post_cat = nn.Linear(hidden_dim * 2, representation_dim)
        self.semantic_fusion_layer = nn.Linear(representation_dim + hidden_dim, representation_dim)
        self.relation_classifier = nn.Linear(self.representation_dim * 2, relation_classes)

        self.complex_path_classifier = nn.Linear(self.representation_dim * 2, relation_classes)
        self.direct_path_classifier = nn.Linear(self.representation_dim, relation_classes)


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

    def forward(self, features, proposals, rel_pair_idxs, union_features, rel_labels):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]

        box_features = features
        box_features = self.feature_extractor(box_features)

        proposal_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
        proposal_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0)

        obj_embed1 = self.obj_embed1(proposal_labels)
        pos_embed = self.bbox_embed(self.encode_box_info(proposals))

        obj_representation = torch.concat((box_features, obj_embed1, pos_embed), -1)
        obj_representation = self.lin_obj(obj_representation)

        # 1. 获取每个对象的精炼特征
        obj_embedding = self.obj_encoder(obj_representation, num_objs)

        obj_preds = proposal_labels
        obj_logits = proposal_logits
        obj_embed2 = self.obj_embed2(obj_preds.long())

        # 2. 准备 edge_encoder 的输入，并获取全局上下文
        features_for_edge_encoder = torch.concat((obj_embedding, obj_embed2), dim=-1)
        features_for_edge_encoder = self.lin_edge(features_for_edge_encoder)
        global_context = self.edge_encoder(features_for_edge_encoder, num_objs)  # Shape: (batch_size, hidden_dim)

        # 3. 提取每个关系对的局部特征（head & tail）
        head_obj_reps = obj_embedding.split(num_objs, dim=0)
        tail_obj_reps = obj_embedding.split(num_objs, dim=0)

        pair_reps_list = []
        for pair_idx, head_rep, tail_rep in zip(rel_pair_idxs, head_obj_reps, tail_obj_reps):
            head = head_rep[pair_idx[:, 0]]
            tail = tail_rep[pair_idx[:, 1]]
            pair_reps_list.append(torch.cat((head, tail), dim=-1))

        pair_rep = torch.cat(pair_reps_list, dim=0)  # Shape: (total_relations, hidden_dim * 2)

        # 4. 投射局部特征
        local_semantic_rep = self.post_cat(pair_rep)  # Shape: (total_relations, representation_dim)

        # 5. 扩展并投射全局上下文，使其与每个关系对应
        num_rels_per_image = torch.tensor(num_rels, device=global_context.device)
        global_context_expanded = global_context.repeat_interleave(num_rels_per_image, dim=0)

        # 6. 使用拼接 + 线性层的方式融合局部和全局语义信息
        fused_semantic_rep = torch.cat([local_semantic_rep, global_context_expanded], dim=-1)
        semantic_rep = F.relu(self.semantic_fusion_layer(fused_semantic_rep))  # 使用ReLU增加非线性

        # 7. 提取并融合视觉联合特征
        visual_union_features = self.feature_extractor(union_features)

        # 通过逐元素相乘融合语义和视觉信息
        final_rep = torch.cat([semantic_rep, visual_union_features], dim=-1)

        # 8. 预测 rel_logits
        rel_logits = self.complex_path_classifier(final_rep) + self.direct_path_classifier(local_semantic_rep)

        # 9. 后续处理和损失计算
        obj_logits = obj_logits.split(num_objs, dim=0)
        rel_logits = rel_logits.split(num_rels, dim=0)

        result = self.post_processor(obj_logits, rel_logits, rel_pair_idxs, proposals)

        losses = {}
        if self.training:
            loss_relation = relation_loss(rel_labels, rel_logits)
            losses = dict(loss_rel=loss_relation)
        return box_features, result, losses
