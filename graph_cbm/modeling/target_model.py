import torch
from torch import Tensor, nn
from torchvision import models
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchvision.models.feature_extraction import create_feature_extractor

from datasets import transforms


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.Normalize(mean=target_mean, std=target_std)]
    )
    return preprocess


def get_target_model(target_name):
    out_channels = 256
    if target_name == 'resnet18_places':
        target_model = models.resnet18(pretrained=True)
        return_node = {'layer4': '0'}
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True)
        return_node = {'features.stage3': '0'}
    elif target_name == 'resnet50':
        target_model = models.resnet50(pretrained=True)
        out_channels = 2048
        return_node = {'layer4': '0'}
    else:
        target_model = models.resnet50(pretrained=True)
        return_node = {'layer4': '0'}
    # print(target_model)
    backbone = create_feature_extractor(target_model, return_node)
    backbone.out_channels = out_channels
    preprocess = get_resnet_imagenet_preprocess()
    return backbone, preprocess


if __name__ == '__main__':
    backbone, preprocess = get_target_model("resnet18_cub")
    output = backbone(torch.rand(1, 3, 224, 224))
    print(output)
