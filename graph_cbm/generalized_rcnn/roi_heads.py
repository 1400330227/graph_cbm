import torch


class ROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(ROIHeads, self).__init__(heads)

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}
        x, detections, loss_box_head = self.box_head(features, proposals, targets)
        losses.update(loss_box_head)
        x, detections, loss_relation = self.relation(features, detections, targets, logger)
        losses.update(loss_relation)
        return x, detections, losses
