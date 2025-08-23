import torch
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from graph_cbm.modeling.detection.fpn import LastLevelMaxPool, BackboneWithFPN, SwinFPNAdapter


def build_resnet50_backbone(pretrained=False):
    models = torchvision.models.resnet50(pretrained=pretrained)
    return_nodes = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
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


def build_mobilenet_backbone(pretrained=False):
    models = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
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


def build_efficientnet_backbone(pretrained=False):
    models = torchvision.models.efficientnet_b0(pretrained=pretrained)
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


def build_vgg_backbone(pretrained=False):
    models = torchvision.models.vgg16(pretrained=pretrained)
    return_nodes = {
        "features.5": "0",
        "features.14": "1",
        "features.28": "2"
    }
    in_channels_list = [128, 256, 512]
    backbone = create_feature_extractor(models, return_nodes)
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


def build_swin_transformer_backbone(pretrained=False):
    try:
        if pretrained:
            weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
        else:
            weights = None
        models = torchvision.models.swin_t(weights=weights)
    except AttributeError:
        models = torchvision.models.swin_t(pretrained=pretrained)
    return_nodes = {
        "features.0": "0",
        "features.3": "1",
        "features.5": "2",
        "features.7": "3",
    }
    in_channels_list = [96, 192, 384, 768]
    backbone = create_feature_extractor(models, return_nodes)
    adapted_backbone = SwinFPNAdapter(backbone=backbone)
    fpn_backbone = BackboneWithFPN(
        backbone=adapted_backbone,
        return_layers=return_nodes,
        in_channels_list=in_channels_list,
        out_channels=256,
        re_getter=False,
        extra_blocks=LastLevelMaxPool(),
    )
    # img = torch.randn(1, 3, 224, 224)
    # outputs = fpn_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    return fpn_backbone


if __name__ == '__main__':
    build_swin_transformer_backbone()
    # cfg = CfgNode({
    #     'pretrain_path': "checkpoints/backbone/resnet50.pth",
    #     'norm_layer': torch.nn.BatchNorm2d,
    #     'trainable_layers': 3,
    #     'extra_blocks': None,
    #     'returned_layers': None,
    # })
    # backbone = build_resnet50_backbone(cfg)
