from typing import Optional, Union, Any

import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            rpn: nn.Module,
            roi_heads: nn.Module,
            transform: nn.Module,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(
            self,
            images: list[torch.Tensor],
            targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ):

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if self.roi_heads.use_relation:
            return detections, features, images, targets

        losses = {}
        if self.training:
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return detections
