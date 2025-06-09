import os
import torch
from graph_cbm.modeling.backbone.fpn import LastLevelMaxPool, BackboneWithFPN
from graph_cbm.modeling.backbone.resnet import ResNet, Bottleneck
from torchvision.ops.misc import FrozenBatchNorm2d


def overwrite_eps(model, eps):
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def build_resnet_backbone(cfg, norm_layer=FrozenBatchNorm2d):
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
    model = BackboneWithFPN(
        backbone=resnet_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks
    )
    return model
