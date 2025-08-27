import torch
import torch.nn.functional as F
import numpy.random as npr
from torch import nn
from torchvision.ops import boxes as box_ops, MultiScaleRoIAlign

from graph_cbm.modeling.detection.transform import resize_boxes
from graph_cbm.modeling.relation.mamba import MambaRelationEncoder
from graph_cbm.modeling.relation.transformer import CrossAttentionFusion
from graph_cbm.utils.boxes import box_union


# from graph_cbm.modeling.relation.transformer import TransformerEncoder, TransformerEdgeEncoder


def relation_loss(rel_labels, relation_logits, rel_weights=None):
    criterion_loss = nn.CrossEntropyLoss(weight=rel_weights)
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
            in_channels=256,
            use_c2ymodel=False,
            embedding_dim=256,
            hidden_dim=512,
            later_nms_pred_thres=0.5,
            batch_size_per_image=1024,
            positive_fraction=0.25,
            num_sample_per_gt_rel=4,
            fg_thres=0.5,
    ):
        super(Predictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj_classes = obj_classes
        self.representation_dim = representation_dim
        self.feature_extractor = FeatureExtractor(feature_extractor_dim, representation_dim)
        self.later_nms_pred_thres = later_nms_pred_thres
        self.use_c2ymodel = use_c2ymodel
        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.rect_size = self.box_roi_pool.output_size[0] * 4 - 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_pos_per_img = int(batch_size_per_image * positive_fraction)
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.fg_thres = fg_thres
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

        self.lin_obj = nn.Linear(representation_dim + embedding_dim + 128, hidden_dim)
        self.lin_edge = nn.Linear(embedding_dim + hidden_dim, hidden_dim)

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.obj_embed1 = nn.Embedding(self.obj_classes, embedding_dim)
        self.obj_embed2 = nn.Embedding(self.obj_classes, embedding_dim)

        self.obj_encoder = MambaRelationEncoder(embed_dim=hidden_dim, depth=4, if_cls_token=False)
        self.edge_encoder = MambaRelationEncoder(embed_dim=hidden_dim, depth=4, if_cls_token=True)

        self.obj_classifier = nn.Linear(hidden_dim, obj_classes)
        self.edge_classifier = nn.Linear(self.representation_dim, relation_classes)

        self.post_cat = nn.Linear(hidden_dim * 2, representation_dim)
        self.semantic_fusion_layer = nn.Linear(representation_dim + hidden_dim, representation_dim)
        self.relation_classifier = nn.Linear(self.representation_dim * 2, relation_classes)

        self.complex_path_classifier = nn.Linear(self.representation_dim * 2, relation_classes)
        self.direct_path_classifier = nn.Linear(self.representation_dim, relation_classes)

        self.cross_attention_fusion = CrossAttentionFusion(representation_dim, hidden_dim, 8, 0.1)

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

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,
                                                                                                                    1)
            fg_rel_i = torch.concat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(
                torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = torch.concat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = torch.concat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return torch.concat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel

    def select_training_samples(self, proposals, targets):
        device = proposals[0]["boxes"].device
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            prp_box = proposal["boxes"]
            prp_lab = proposal["labels"].long()
            tgt_box = target["boxes"]
            tgt_lab = target["labels"].long()
            tgt_rel_matrix = target["relation"]  # [tgt, tgt]
            # IoU matching
            # ious = boxlist_iou(target, proposal)  # [tgt, prp]
            ious = box_ops.box_iou(tgt_box, prp_box)
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)  # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = box_ops.box_iou(prp_box, prp_box)  # [prp, prp]

            num_prp = prp_box.shape[0]
            rel_possibility = (torch.ones((num_prp, num_prp), device=device).long() -
                               torch.eye(num_prp, device=device).long())
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(
                device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def select_test_pairs(self, proposals):
        device = proposals[0]["boxes"].device
        rel_pair_idxs = []
        for p in proposals:
            n = p["boxes"].shape[0]
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    def union_feature_extractor(self, images, features, proposals, rel_pair_idxs):
        device = features["0"].device
        image_sizes = images.image_sizes
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx, sizes in zip(proposals, rel_pair_idxs, image_sizes):
            head_proposal = proposal["boxes"][rel_pair_idx[:, 0]]
            tail_proposal = proposal["boxes"][rel_pair_idx[:, 1]]
            union_proposal = box_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            num_rel = len(rel_pair_idx)

            dummy_x_range = (torch.arange(self.rect_size, device=device).view(1, 1, -1)
                             .expand(num_rel, self.rect_size, self.rect_size))
            dummy_y_range = (torch.arange(self.rect_size, device=device).view(1, -1, 1)
                             .expand(num_rel, self.rect_size, self.rect_size))

            head_proposal = resize_boxes(head_proposal, sizes, [self.rect_size, self.rect_size])
            tail_proposal = resize_boxes(tail_proposal, sizes, [self.rect_size, self.rect_size])

            head_rect = ((dummy_x_range >= head_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= head_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= head_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= head_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= tail_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= tail_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= tail_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1)
            rect_inputs.append(rect_input)

        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        union_vis_features = self.box_roi_pool(features, union_proposals, images.image_sizes)
        union_features = union_vis_features + rect_features

        return union_features

    def forward(self, features, proposals, targets, images, rel_weights=None):
        if self.training:
            with torch.no_grad():
                proposals, rel_labels, rel_pair_idxs, rel_binarys = self.select_training_samples(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.select_test_pairs(proposals)

        box_features = self.box_roi_pool(features, [t["boxes"] for t in proposals], images.image_sizes)
        union_features = self.union_feature_extractor(images, features, proposals, rel_pair_idxs)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]
        # edge_index_list = [edge_pairs.t().contiguous() for edge_pairs in rel_pair_idxs]

        # box_features = features
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
        global_context = self.edge_encoder(features_for_edge_encoder, num_objs)

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
        semantic_rep = self.cross_attention_fusion(local_semantic_rep, global_context_expanded)
        # fused_semantic_rep = torch.cat([local_semantic_rep, global_context_expanded], dim=-1)
        # semantic_rep = F.relu(self.semantic_fusion_layer(fused_semantic_rep))  # 使用ReLU增加非线性

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
        rel_features = box_features
        # if self.use_c2ymodel:
        #     rel_features = box_features
        #     return rel_features, result, losses
        # else:
        if self.training:
            loss_relation = relation_loss(rel_labels, rel_logits, rel_weights)
            losses = dict(loss_rel=loss_relation)
        return rel_features, result, losses
