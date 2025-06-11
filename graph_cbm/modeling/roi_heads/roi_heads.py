from collections import OrderedDict

from torch import nn

from graph_cbm.modeling.roi_heads.box_head import BoxHead
from graph_cbm.modeling.roi_heads.relation_head import RelationHead


class RoIHeads(nn.ModuleDict):
    def __init__(self, heads, relation_on=False):
        super(RoIHeads, self).__init__(heads)
        self.relation_on = relation_on

    def forward(self, features, proposals, image_shapes, targets=None):
        losses = OrderedDict()
        x, detections, loss_box = self.box(features, proposals, image_shapes, targets)
        if not self.relation_on:
            losses.update(loss_box)
        if self.relation_on:
            x, detections, loss_relation = self.relation(features, detections, image_shapes, targets)
            losses.update(loss_relation)
        return x, detections, losses


def build_roi_heads(
        box_roi_pool,
        feature_extractor,
        box_predictor,
        box_fg_iou_thresh,
        box_bg_iou_thresh,  # 0.5  0.5
        box_batch_size_per_image,
        box_positive_fraction,  # 512  0.25
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
        relation_on=False,
):
    roi_heads = []
    box_head = BoxHead(
        box_roi_pool,
        feature_extractor,
        box_predictor,
        box_fg_iou_thresh,
        box_bg_iou_thresh,  # 0.5  0.5
        box_batch_size_per_image,
        box_positive_fraction,  # 512  0.25
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
        relation_on
    )
    roi_heads.append(("box", box_head))
    if relation_on:
        relation_head = RelationHead(box_roi_pool)
        roi_heads.append(("rel", relation_head))
    roi_heads = RoIHeads(roi_heads, relation_on=relation_on)
    return roi_heads
