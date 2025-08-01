from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import boxes as box_ops, roi_align
import torchvision.models.detection._utils as det_utils


def fastrcnn_loss(
        class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        labels: list[torch.Tensor],
        regression_targets: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):

    def __init__(
            self,
            box_roi_pool,
            box_head,
            box_predictor,
            # Faster R-CNN training
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            score_thresh,
            nms_thresh,
            detections_per_img,
            # Relation
            use_relation=False,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.use_relation = use_relation

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (list[Tensor], list[Tensor], list[Tensor]) -> tuple[list[Tensor], list[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (list[Tensor]) -> list[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (list[Tensor], list[Tensor]) -> list[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[list[dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(
            self,
            proposals,  # type: list[Tensor]
            targets,  # type: Optional[list[dict[str, Tensor]]]
    ):
        # type: (...) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
            self,
            class_logits,  # type: Tensor
            box_regression,  # type: Tensor
            proposals,  # type: list[Tensor]
            image_shapes,  # type: list[tuple[int, int]]
    ):
        # type: (...) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        num_classes_no_bg = num_classes - 1

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        class_logits_list = class_logits.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_logits = []
        all_logits_boxes = []
        for boxes, logits, scores, image_shape in zip(
                pred_boxes_list,
                class_logits_list,
                pred_scores_list,
                image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            logits_boxes = boxes[:, :]
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            logits = logits[:, :]

            n_proposals = scores.shape[0]
            image_inds = torch.arange(n_proposals, device=device)
            image_inds = image_inds.view(-1, 1).expand(n_proposals, num_classes_no_bg).reshape(-1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, logits, scores, labels = boxes[inds], logits[image_inds[inds]], scores[inds], labels[inds]
            logits_boxes = logits_boxes[image_inds[inds]]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, logits, scores, labels = boxes[keep], logits[keep], scores[keep], labels[keep]
            logits_boxes = logits_boxes[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, logits, scores, labels = boxes[keep], logits[keep], scores[keep], labels[keep]
            logits_boxes = logits_boxes[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_logits.append(logits)
            all_logits_boxes.append(logits_boxes)

        return all_boxes, all_scores, all_labels, all_logits, all_logits_boxes

    def assign_label_to_proposals(self, targets, proposals):
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        prp_boxes = [t["boxes"] for t in proposals]
        for img_idx, (prp_boxe, gt_boxe, gt_label) in enumerate(zip(prp_boxes, gt_boxes, gt_labels)):
            match_quality_matrix = box_ops.box_iou(gt_boxe, prp_boxe)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            clamped_matched_idxs = matched_idxs.clamp(min=0)
            labels_in_image = gt_label[clamped_matched_idxs]
            labels_in_image = labels_in_image.to(dtype=torch.int64)
            labels_in_image[matched_idxs < 0] = 0
            proposals[img_idx]['gt_labels'] = labels_in_image
        return proposals

    def forward(
            self,
            features: dict[str, torch.Tensor],
            proposals: list[torch.Tensor],
            image_shapes: list[tuple[int, int]],
            targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.use_relation:
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)
            boxes, scores, labels, logits, logits_boxes = self.postprocess_detections(class_logits, box_regression,
                                                                                      proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s, "logits": i, "logits_boxes": j}
                      for b, l, s, i, j in zip(boxes, labels, scores, logits, logits_boxes)]
            if self.training:
                result = self.assign_label_to_proposals(targets, result)
            return result, {}

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result: list[dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels, _, _ = self.postprocess_detections(class_logits, box_regression,
                                                                      proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s} for b, l, s in zip(boxes, labels, scores)]
        return result, losses
