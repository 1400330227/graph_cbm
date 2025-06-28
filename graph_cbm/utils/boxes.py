import numpy as np
import torch
import torch.nn.functional as F

def box_union(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    union_box = torch.cat((lt, rb), dim=1)
    return union_box


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]
    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0),
                                                  boxes_per_cls.size(1)).cpu().detach().numpy() >= nms_thresh
    prob_sampled = F.softmax(pred_logits, 1).cpu().detach().numpy()
    prob_sampled[:, 0] = 0

    pred_label = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0 # This way we won't re-sample

    return pred_label

def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    inters = inter[:, :, :, 0] * inter[:, :, :, 1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
            boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union
