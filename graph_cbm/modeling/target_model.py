from torchvision import models, transforms
from pytorchcv.model_provider import get_model as ptcv_get_model


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std)]
    )
    return preprocess


def get_target_model(target_name):
    if target_name == 'resnet18_places':
        target_model = models.resnet18(pretrained=True)
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True)
    elif target_name == 'resnet50':
        target_model = models.resnet50(pretrained=True)
    else:
        target_model = models.resnet50(pretrained=True)
    target_model.eval()
    preprocess = get_resnet_imagenet_preprocess()
    return target_model, preprocess
