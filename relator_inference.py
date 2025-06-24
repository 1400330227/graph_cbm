import cv2
import numpy as np
import torch

from detector_inference import create_detector_model
from graph_cbm.modeling.graph import Graph
from graph_cbm.modeling.prediction.predictor import Predictor


def create_model():
    detector = create_detector_model(load_pretrain_weights=True)
    predictor = Predictor(
        obj_classes=91,
        relation_classes=51,
        feature_extractor=detector.roi_heads.box_head,
    )
    model = Graph(detector, predictor)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_model()
model = model.to(device)


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


img = cv2.imread("./graph_cbm/finetuning/2007_002293.jpg")
img = img_transform(img)


def inference(img, model):
    model.eval()

    img = img.to(device)
    outputs = model([img, img])

    return outputs


if __name__ == "__main__":
    outputs = inference(img, model)
