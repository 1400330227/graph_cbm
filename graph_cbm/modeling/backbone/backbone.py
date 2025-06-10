import os
import torch
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor

from graph_cbm.modeling.backbone.fpn import LastLevelMaxPool, BackboneWithFPN
from graph_cbm.modeling.backbone.resnet import ResNet, Bottleneck
from torchvision.ops.misc import FrozenBatchNorm2d

from graph_cbm.modeling.structures.cfg_node import CfgNode


def overwrite_eps(model, eps):
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def build_resnet50_backbone(cfg, norm_layer=FrozenBatchNorm2d):
    pretrain_path = cfg.pretrain_path
    trainable_layers = cfg.trainable_layers
    extra_blocks = cfg.extra_blocks
    returned_layers = cfg.returned_layers
    resnet_backbone = ResNet(
        block=Bottleneck,
        blocks_num=[3, 4, 6, 3],
        include_top=False,
        norm_layer=norm_layer
    )
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)
    if pretrain_path != "":
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in resnet_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = resnet_backbone.in_channel // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    backbone = BackboneWithFPN(
        backbone=resnet_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks
    )
    img = torch.randn(1, 3, 224, 224)
    outputs = backbone(img)
    [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    return backbone


def build_mobilenet_backbone(cfg):
    models = torchvision.models.mobilenet_v3_large(pretrained=cfg.pretrained)
    return_nodes = {
        "features.6": "0",
        "features.12": "1",
        "features.16": "2"
    }
    in_channels_list = [40, 112, 960]
    backbone = create_feature_extractor(models, return_nodes)
    backbone = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_nodes,
        in_channels_list=in_channels_list,
        out_channels=256,
        re_getter=False,
        extra_blocks=LastLevelMaxPool(),
    )
    return backbone


def build_efficientnet_backbone(cfg):
    models = torchvision.models.efficientnet_b0(pretrained=cfg.pretrained)
    return_nodes = {
        "features.3": "0",
        "features.4": "1",
        "features.8": "2"
    }
    in_channels_list = [40, 80, 1280]
    backbone = create_feature_extractor(models, return_nodes)
    backbone = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_nodes,
        in_channels_list=in_channels_list,
        out_channels=256,
        re_getter=False,
        extra_blocks=LastLevelMaxPool(),
    )
    return backbone

def build_vgg_backbone(cfg):
    models = torchvision.models.vgg16(pretrained=cfg.pretrained)
    return_nodes = {
        "features.5": "0",
        "features.14": "1",
        "features.28": "2"
    }
    in_channels_list = [128, 256, 512]
    backbone = create_feature_extractor(models, return_nodes)
    # img = torch.randn(1, 3, 224, 224)
    # outputs = backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    backbone = BackboneWithFPN(
        backbone=backbone,
        return_layers=return_nodes,
        in_channels_list=in_channels_list,
        out_channels=256,
        re_getter=False,
        extra_blocks=LastLevelMaxPool(),
    )
    # img = torch.randn(1, 3, 224, 224)
    # outputs = backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    return backbone

if __name__ == '__main__':
    cfg = CfgNode({
        'pretrained': True,
    })
    build_vgg_backbone(cfg)
    # cfg = CfgNode({
    #     'pretrain_path': "checkpoints/backbone/resnet50.pth",
    #     'norm_layer': torch.nn.BatchNorm2d,
    #     'trainable_layers': 3,
    #     'extra_blocks': None,
    #     'returned_layers': None,
    # })
    # backbone = build_resnet50_backbone(cfg)
