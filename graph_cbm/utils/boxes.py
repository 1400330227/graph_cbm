import torch


def box_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    union_box = torch.cat((lt, rb), dim=1)
    return union_box
