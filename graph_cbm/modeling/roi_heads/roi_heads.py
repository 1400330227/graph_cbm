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
        box_fg_iou_thresh, # 0.5
        box_bg_iou_thresh,  # 0.5
        box_batch_size_per_image,
        box_positive_fraction,  # 512  0.25
        bbox_reg_weights,
        box_score_thresh,
        box_nms_thresh,
        box_detections_per_img,
        representation_size,
        relation_on=False,
        obj_classes=150,
        num_rel_cls=51,

):
    roi_heads = []
    box_head = BoxHead(
        box_roi_pool=box_roi_pool,
        feature_extractor=feature_extractor,
        box_predictor=box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.3,  # 0.5  0.5
        batch_size_per_image=box_batch_size_per_image,
        positive_fraction=box_positive_fraction,  # 512  0.25
        bbox_reg_weights=bbox_reg_weights,
        score_thresh=box_score_thresh,
        nms_thresh=box_nms_thresh,
        detection_per_img=box_detections_per_img,
        relation_on=relation_on,
    )
    roi_heads.append(("box", box_head))
    if relation_on:
        relation_head = RelationHead(
            relation_roi_pool=box_roi_pool,
            feature_extractor=feature_extractor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.3,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            representation_size=representation_size,
            fg_thres=0.5,
            use_union_box=True,
            num_sample_per_gt_rel=4,
            embedding_dim=256,
            num_heads=8,
            obj_classes=obj_classes,
            num_rel_cls=num_rel_cls,
        )
        roi_heads.append(("relation", relation_head))
    roi_heads = RoIHeads(roi_heads, relation_on=relation_on)
    return roi_heads
