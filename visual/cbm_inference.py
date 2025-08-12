import cv2
import numpy as np
import torch

from graph_cbm.modeling.detection.backbone import build_resnet50_backbone
from graph_cbm.modeling.detection.detector import build_detector
from graph_cbm.modeling.graph_cbm import GraphCBM
from graph_cbm.modeling.relation.predictor import Predictor


def create_model(num_classes, relation_classes):
    backbone = build_resnet50_backbone(pretrained=False)
    weights_path = "../save_weights/detector/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    detector = build_detector(backbone, num_classes, weights_path, use_relation=True)

    predictor = Predictor(
        obj_classes=num_classes,
        relation_classes=relation_classes,
        feature_extractor=detector.roi_heads.box_head,
    )
    model = GraphCBM(detector, predictor)
    return model


device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

model = create_model(21, 51)
model = model.to(device)


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


img = cv2.imread("../graph_cbm/finetuning/2007_002293.jpg")
img = img_transform(img)


def inference(img, model):
    model.eval()

    with torch.no_grad():
        img = img.to(device)
        outputs = model([img, img])

        return outputs


if __name__ == "__main__":
    outputs = inference(img, model)
