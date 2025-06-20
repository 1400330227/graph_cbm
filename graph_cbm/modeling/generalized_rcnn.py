import torch
import warnings
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
from collections import OrderedDict
from torch import nn, Tensor
from torchvision.ops import MultiScaleRoIAlign
from graph_cbm.modeling.roi_heads.box_head import BoxHead
from graph_cbm.modeling.roi_heads.roi_heads import RoIHeads, build_roi_heads
from graph_cbm.modeling.structures.transform import GeneralizedRCNNTransform
from graph_cbm.modeling.rpn.rpn import AnchorsGenerator, RPNHead, RegionProposalNetwork


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform, relation_on):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.relation_on = relation_on

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        x, detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.relation_on:
                losses.update(proposal_losses)
            return losses
        else:
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def build_detection_model(
        backbone,
        num_classes,
        num_rel_cls,
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        box_roi_pool=None,
        feature_extractor=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        relation_on=True,
        representation_size=4096,
):
    if not hasattr(backbone, "out_channels"):
        raise ValueError(
            "backbone should contain an attribute out_channels"
            "specifying the number of output channels  (assumed to be the"
            "same for all the levels"
        )

    assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
    assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

    if num_classes is not None:
        if box_predictor is not None:
            raise ValueError("num_classes should be None when box_predictor "
                             "is specified")
    else:
        if box_predictor is None:
            raise ValueError("num_classes should not be None when box_predictor "
                             "is not specified")
    out_channels = backbone.out_channels
    if rpn_anchor_generator is None:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorsGenerator(
            anchor_sizes, aspect_ratios
        )
    if rpn_head is None:
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )
    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
        score_thresh=rpn_score_thresh)
    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=[7, 7],
            sampling_ratio=2
        )
    if feature_extractor is None:
        resolution = box_roi_pool.output_size[0]  # 默认等于7
        # representation_size = 4096
        feature_extractor = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
        )
    if box_predictor is None:
        # representation_size = 4096
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes
        )
    roi_heads = build_roi_heads(
        box_roi_pool=box_roi_pool,
        feature_extractor=feature_extractor,
        box_predictor=box_predictor,
        box_fg_iou_thresh=box_fg_iou_thresh,
        box_bg_iou_thresh=box_bg_iou_thresh,  # 0.5  0.5
        box_batch_size_per_image=box_batch_size_per_image,
        box_positive_fraction=box_positive_fraction,  # 512  0.25
        bbox_reg_weights=bbox_reg_weights,
        box_score_thresh=box_score_thresh,  # 0.05
        box_nms_thresh=box_nms_thresh,  # 0.5
        box_detections_per_img=box_detections_per_img,  # 100
        representation_size=representation_size,
        relation_on=relation_on,
        obj_classes=num_classes,
        num_rel_cls=num_rel_cls,
    )

    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    generalized_rcnn = GeneralizedRCNN(backbone, rpn, roi_heads, transform, relation_on)
    return generalized_rcnn
