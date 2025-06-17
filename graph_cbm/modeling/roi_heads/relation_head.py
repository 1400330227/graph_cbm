import torch
import numpy.random as npr
from torch import nn

from graph_cbm.modeling.roi_heads.roi_relation_predictors import build_roi_relation_predictor
from graph_cbm.modeling.structures import boxes as box_ops, det_utils
from graph_cbm.modeling.structures.det_utils import roi_relation_loss


class RelationHead(nn.Module):
    def __init__(
            self,
            relation_roi_pool,
            feature_extractor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            representation_size,
            fg_thres,
            use_union_box,
            num_sample_per_gt_rel,
            embedding_dim,
            num_heads,
            obj_classes,
            num_rel_cls,
    ):
        super(RelationHead, self).__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.fg_thres = fg_thres
        self.relation_roi_pool = relation_roi_pool
        self.feature_extractor = feature_extractor
        self.use_union_box = use_union_box
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.rect_size = 27
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False
        )
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        ])
        self.predictor = build_roi_relation_predictor(
            embedding_dim,
            num_heads,
            feature_extractor,
            obj_classes,
            num_rel_cls,
            representation_size
        )

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
            fg_labels = (torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device)
                         .view(-1, 1))
            fg_rel_i = (torch.concat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1)
                        .to(torch.int64))
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = ((ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs])
                              .view(-1).detach().cpu().numpy())
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

    def detect_training_samples(self, proposals, targets):
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_idx, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal["boxes"].device
            prp_box = proposal["boxes"]
            prp_lab = proposal["labels"].long()
            tgt_box = target["boxes"]
            tgt_lab = target["labels"].long()
            tgt_rel_matrix = target["relation"]
            ious = box_ops.box_iou(tgt_box, prp_box)
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)
            num_prp = prp_box.shape[0]
            rel_possibility = (torch.ones((num_prp, num_prp), device=device).long() -
                               torch.eye(num_prp, device=device).long())
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0
            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(
                device,
                tgt_rel_matrix,
                ious,
                is_match,
                rel_possibility
            )
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)
        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def resize_boxe(self, boxes, original_size, new_size):
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratios_height, ratios_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratios_width
        xmax = xmax * ratios_width
        ymin = ymin * ratios_height
        ymax = ymax * ratios_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def union_feature_extractor(self, features, proposals, rel_pair_idxs, image_shapes):
        device = features["0"].device
        prp_boxes = [t["boxes"].to(proposals[0]["boxes"].dtype) for t in proposals]
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx, original_size in zip(prp_boxes, rel_pair_idxs, image_shapes):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = box_ops.box_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = (torch.arange(self.rect_size, device=device).view(1, 1, -1)
                             .expand(num_rel, self.rect_size, self.rect_size))
            dummy_y_range = (torch.arange(self.rect_size, device=device).view(1, -1, 1)
                             .expand(num_rel, self.rect_size, self.rect_size))
            # resize bbox to the scale rect_size
            head_proposal = self.resize_boxe(head_proposal, original_size, (self.rect_size, self.rect_size))
            tail_proposal = self.resize_boxe(tail_proposal, original_size, (self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= head_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= head_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= head_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= tail_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= tail_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= tail_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1)  # (num_rel, 4, rect_size, rect_size)
            rect_inputs.append(rect_input)

        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        union_vis_features = self.relation_roi_pool(features, union_proposals, image_shapes)
        union_features = union_vis_features + rect_features
        # union_features = self.feature_extractor(union_features)

        return union_features

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, rel_labels, rel_pair_idxs, rel_binarys = self.detect_training_samples(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.select_test_pairs(features[0].device, proposals)
        roi_features = self.relation_roi_pool(features, [t["boxes"] for t in proposals], image_shapes)
        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs, image_shapes)
        else:
            union_features = None
        refine_logits, relation_logits = self.predictor(
            proposals=proposals,
            rel_pair_idxs=rel_pair_idxs,
            roi_features=roi_features,
            union_features=union_features,
        )
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}
        loss_relation, loss_refine = roi_relation_loss(proposals, rel_labels, relation_logits, refine_logits)
        output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        return roi_features, proposals, output_losses



