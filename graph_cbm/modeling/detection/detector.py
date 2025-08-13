import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign

from graph_cbm.modeling.detection.backbone import build_resnet50_backbone, build_mobilenet_backbone, \
    build_efficientnet_backbone, build_vgg_backbone
from graph_cbm.modeling.detection.generalized_rcnn import GeneralizedRCNN
from graph_cbm.modeling.detection.roi_heads import RoIHeads
from graph_cbm.modeling.detection.transform import GeneralizedRCNNTransform


class FasterRCNN(GeneralizedRCNN):
    def __init__(
            self,
            backbone,
            num_classes=None,
            # transform parameters
            min_size=800,
            max_size=1333,
            image_mean=None,
            image_std=None,
            # RPN parameters
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
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.05,
            # box_score_thresh=0.01,
            box_nms_thresh=0.5,
            # box_nms_thresh=0.3,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # box_predictor
            representation_dim=1024,
            # Relation
            use_relation=False,
            **kwargs,
    ):
        out_channels = backbone.out_channels
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            # representation_dim = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_dim)

        if box_predictor is None:
            # representation_dim = 1024
            box_predictor = FastRCNNPredictor(representation_dim, num_classes)
        roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            use_relation,
        )
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        super().__init__(backbone, rpn, roi_heads, transform)


def build_detector(backbone_name='', num_classes=91, weights_path="", is_train=True):
    if backbone_name == 'resnet50':
        backbone = build_resnet50_backbone(pretrained=False)
    elif backbone_name == 'mobilenet':
        backbone = build_mobilenet_backbone(pretrained=False)
    elif backbone_name == 'efficientnet':
        backbone = build_efficientnet_backbone(pretrained=False)
    elif backbone_name == 'squeezenet':
        backbone = build_vgg_backbone(pretrained=False)
    else:
        backbone = build_resnet50_backbone(pretrained=False)

    model = FasterRCNN(backbone=backbone, num_classes=91 if is_train else num_classes)
    if is_train:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if weights_path != "":
        weights_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict)

    return model
