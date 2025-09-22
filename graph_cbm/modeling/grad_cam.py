import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """
    一个可复用的 Grad-CAM 实现类，用于生成类别激活热力图。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        for (name, module) in self.model.named_modules():
            if name == self.target_layer:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handler in self.handlers:
            handler.remove()

    def _get_features_hook(self, module, input, output):
        self.features = output

    def _get_grads_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmaps(self, input_image_list: list, target_class_idx: int = None):
        self.model.zero_grad()
        output = self.model(input_image_list)  # CBMModel 需要一个 list 作为输入

        if target_class_idx is None:
            target_class_idx = torch.argmax(output[0]["y_prob"]).item()  # 获取第一个结果的预测类别

        target_logit = output[0]["y_logit"][target_class_idx]

        target_logit.backward()

        gradients = self.gradients.detach()
        features = self.features.detach()

        n, C, _, _ = gradients.shape

        weights = torch.mean(gradients, dim=[2, 3])
        weights = weights.view(n, C, 1, 1)

        weighted_activations = features * weights
        heatmaps = torch.sum(weighted_activations, dim=1)
        heatmaps = F.relu(heatmaps)

        output_heatmaps = []
        for heatmap in heatmaps:
            # if torch.max(heatmap) > 0:
            heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))
            output_heatmaps.append(heatmap.cpu().numpy())

        return output_heatmaps, output
