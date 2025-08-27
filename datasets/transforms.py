import random

import torch
from torchvision.transforms import functional as F
import torchvision.transforms as T


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            if len(bbox) > 0:
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
                target["boxes"] = bbox
            else:
                print(f"No boxes found for sample {target.get('image_id')} in batch")
        return image, target


class RandomResizedCrop(T.RandomResizedCrop):
    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        boxes = target["boxes"]
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] - j
            boxes[:, [1, 3]] = boxes[:, [1, 3]] - i
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)

            scale_x = self.size[1] / w
            scale_y = self.size[0] / h

            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

            cropped_boxes_w = boxes[:, 2] - boxes[:, 0]
            cropped_boxes_h = boxes[:, 3] - boxes[:, 1]
            keep = (cropped_boxes_w > 0) & (cropped_boxes_h > 0)

            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]
        else:
            print(f"No boxes found for sample {target.get('image_id')} in batch")

        return image, target


class Resize(T.Resize):
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
        else:
            w, h = image.size
        resized_image = super().forward(image)
        if isinstance(resized_image, torch.Tensor):
            _, new_h, new_w = resized_image.shape
        else:
            new_w, new_h = resized_image.size
        if w > 0 and h > 0:
            scale_x = new_w / w
            scale_y = new_h / h
            boxes = target.get("boxes")
            if boxes is not None and len(boxes) > 0:
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                target["boxes"] = boxes
            else:
                print(f"No boxes found for sample {target.get('image_id')} in batch")
        return resized_image, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
