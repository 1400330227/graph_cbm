import torch
from torch import nn
from . import boxes as box_ops


class RelationHead(nn.Module):
    def __init__(self, cfg):
        super(RelationHead, self).__init__()
        self.cfg = cfg.clone()

    def forward(self, features, proposals, targets=None, logger=None):
        if self.training:
            proposals, rel_labels, rel_pair_idxs, rel_binarys = self.detect_relsample(proposals, targets)

        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.prepare_test_pairs(features[0].device, proposals)
        roi_features = self.box_feature_extractor(features, proposals)
        union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        refine_logits, relation_logits, add_losses = self.predictor(
            proposals,
            rel_pair_idxs,
            rel_labels,
            rel_binarys,
            roi_features,
            union_features,
            logger
        )
        if not self.training:
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}
        loss_relation, loss_refine, = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
        output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
        output_losses.update(add_losses)
        return roi_features, proposals, output_losses

    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]
            # IoU matching
            ious = box_ops.box_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)  # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = box_ops.box_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box):
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                                   device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix, ious, is_match,
                                                                         rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys
