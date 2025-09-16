import torch
import numpy.random as npr
import torch.nn.functional as F
from torch import nn
from graph_cbm.modeling.detection.backbone import (
    build_vgg_backbone, build_resnet50_backbone, build_mobilenet_backbone, build_efficientnet_backbone,
    build_swin_transformer_backbone)
from graph_cbm.modeling.detection.detector import FasterRCNN
from graph_cbm.modeling.relation.predictor import Predictor
from torchvision.ops import boxes as box_ops, MultiScaleRoIAlign
from graph_cbm.modeling.detection.transform import resize_boxes
from graph_cbm.utils.boxes import box_union


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


class SceneGraph(nn.Module):
    def __init__(
            self,
            detector: FasterRCNN,
            predictor: Predictor,
            feature_extractor_dim,
            representation_dim=1024,
            in_channels=256,
            batch_size_per_image=50,
            positive_fraction=0.25,
            num_sample_per_gt_rel=4,
            fg_thres=0.5,
            rel_score_thresh=0.,
            use_cbm=False,
    ):
        super().__init__()
        self.detector = detector
        self.predictor = predictor

        self.feature_extractor = FeatureExtractor(feature_extractor_dim, representation_dim)

        self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        self.rect_size = self.box_roi_pool.output_size[0] * 4 - 1
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_pos_per_img = int(batch_size_per_image * positive_fraction)
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.fg_thres = fg_thres
        self.rel_score_thresh = rel_score_thresh
        self.use_cbm = use_cbm

        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

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

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
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

    def post_processor(self, obj_logits, relation_logits, rel_pair_idxs, proposals):
        device = obj_logits[0].device
        result = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(
                zip(relation_logits, obj_logits, rel_pair_idxs, proposals)):
            obj_class_prob = F.softmax(obj_logit, -1)
            obj_class_prob[:, 0] = 0
            num_obj_bbox = obj_class_prob.shape[0]

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

            valid_mask = rel_scores > self.rel_score_thresh
            if valid_mask.any():
                rel_pair_idx = rel_pair_idx[valid_mask]
                rel_class_prob = rel_class_prob[valid_mask]
                rel_scores = rel_scores[valid_mask]
                rel_class = rel_class[valid_mask]
                triple_scores = triple_scores[valid_mask]
                _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)

                rel_pair_idx = rel_pair_idx[sorting_idx]
                rel_class_prob = rel_class_prob[sorting_idx]
                rel_scores = rel_scores[sorting_idx]
                rel_labels = rel_class[sorting_idx]
            else:
                rel_pair_idx = torch.zeros((0, 2), dtype=torch.int64, device=device)
                rel_class_prob = torch.zeros((0, rel_logit.shape[1]), dtype=torch.float32, device=device)
                rel_scores = torch.zeros((0,), dtype=torch.float32, device=device)
                rel_labels = torch.zeros((0,), dtype=torch.int64, device=device)

            result.append({
                "boxes": bbox,
                "obj_logit": obj_logit,
                "labels": obj_class,
                "scores": obj_scores,
                "rel_pair_idxs": rel_pair_idx,
                "pred_rel_scores": rel_class_prob,
                "pred_rel_labels": rel_labels,
                "rel_scores": rel_scores
            })
        return result

    def forward(self, images, targets=None, rel_weights=None):
        proposals, features, images, targets = self.detector(images, targets)
        if self.training:
            with torch.no_grad():
                proposals, rel_labels, rel_pair_idxs, rel_binarys = self.select_training_samples(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.select_test_pairs(proposals)

        roi_features = self.box_roi_pool(features, [t["boxes"] for t in proposals], images.image_sizes)
        union_features = self.union_feature_extractor(images, features, proposals, rel_pair_idxs)
        box_features = self.feature_extractor(roi_features)
        overlap_features = self.feature_extractor(union_features)

        rel_logits, obj_logits, loss_relation, = self.predictor(
            box_features,
            overlap_features,
            proposals,
            rel_pair_idxs,
            rel_labels,
            rel_weights
        )
        result = self.post_processor(obj_logits, rel_logits, rel_pair_idxs, proposals)
        losses = {}
        if self.use_cbm:
            return proposals, result
        losses.update(loss_relation)
        if self.training:
            return losses
        return result


def build_scene_graph(
        backbone_name,
        num_classes,
        relation_classes,
        detector_weights_path="",
        weights_path="",
        rel_score_thresh=0.,
        use_cbm=False,
):
    if backbone_name == 'resnet50':
        backbone = build_resnet50_backbone(pretrained=False)
    elif backbone_name == 'mobilenet':
        backbone = build_mobilenet_backbone(pretrained=False)
    elif backbone_name == 'efficientnet':
        backbone = build_efficientnet_backbone(pretrained=False)
    elif backbone_name == 'squeezenet':
        backbone = build_vgg_backbone(pretrained=False)
    elif backbone_name == 'swin_transformer':
        backbone = build_swin_transformer_backbone(pretrained=False)
    else:
        backbone = build_resnet50_backbone(pretrained=False)

    detector = FasterRCNN(backbone=backbone, num_classes=num_classes, use_relation=True)

    out_channels = backbone.out_channels
    resolution = detector.roi_heads.box_roi_pool.output_size[0]
    feature_extractor_dim = out_channels * resolution ** 2

    if detector_weights_path != "":
        detector_weights = torch.load(detector_weights_path, map_location='cpu', weights_only=True)
        detector_weights = detector_weights['model'] if 'model' in detector_weights else detector_weights
        detector.load_state_dict(detector_weights)

    representation_dim = detector.roi_heads.box_predictor.cls_score.in_features
    predictor = Predictor(num_classes, relation_classes, representation_dim)

    model = SceneGraph(
        detector,
        predictor,
        feature_extractor_dim,
        representation_dim,
        out_channels,
        rel_score_thresh=rel_score_thresh,
        use_cbm=use_cbm,
    )

    if weights_path != "":
        weights_dict = torch.load(weights_path, map_location='cpu')
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict, strict=False)

    return model
