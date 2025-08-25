import torch
from torch import nn
from graph_cbm.modeling.c2y_model import C2yModel
from graph_cbm.modeling.detection.backbone import (
    build_vgg_backbone, build_resnet50_backbone, build_mobilenet_backbone, build_efficientnet_backbone,
    build_swin_transformer_backbone)
from graph_cbm.modeling.detection.detector import FasterRCNN
from graph_cbm.modeling.relation.predictor import Predictor


class GraphCBM(nn.Module):
    def __init__(self, detector: FasterRCNN, predictor: Predictor, c2y_model: C2yModel, use_c2ymodel=False):
        super().__init__()
        self.detector = detector
        self.predictor = predictor
        self.use_c2ymodel = use_c2ymodel
        self.c2y_model = c2y_model

    def forward(self, images, targets=None, rel_weights=None):
        proposals, features, images, targets = self.detector(images, targets)
        rel_features, rel_graphs, loss_relation, = self.predictor(features, proposals, targets, images, rel_weights)
        result = rel_graphs
        losses = {}
        losses.update(loss_relation)
        if self.use_c2ymodel:
            cbm_graphs, loss_task = self.c2y_model(rel_features, rel_graphs, targets)
            losses.update(loss_task)
            result = cbm_graphs
        if self.training:
            return losses
        return result


def build_Graph_CBM(
        backbone_name,
        num_classes,  # 目标检测的数量
        relation_classes,  # 关系的数量
        n_tasks,  # 分类的数量
        detector_weights_path="",
        weights_path="",
        use_c2ymodel=False
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
    predictor = Predictor(num_classes, relation_classes, feature_extractor_dim, representation_dim, out_channels,
                          use_c2ymodel)
    c2y_model = None
    if use_c2ymodel:
        c2y_model = C2yModel(num_classes, relation_classes, n_tasks, representation_dim)

    model = GraphCBM(detector, predictor, c2y_model, use_c2ymodel)
    if weights_path != "":
        weights_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict, strict=False)

    return model
