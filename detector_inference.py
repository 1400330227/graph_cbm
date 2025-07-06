import cv2
import numpy as np
import torch
import json
from matplotlib import patches, pyplot as plt

from graph_cbm.modeling.detection.backbone import build_resnet50_backbone
from graph_cbm.modeling.detection.detector import FasterRCNN


def create_detector_model(num_classes=91, load_pretrain_weights=False):
    backbone = build_resnet50_backbone(pretrained=False)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    if load_pretrain_weights:
        weights_dict = torch.load("./save_weights/resnet-fpn-model-1299.pth",
                                  map_location='cpu', weights_only=True)
        model.load_state_dict(weights_dict, strict=False)
    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_detector_model(load_pretrain_weights=True, num_classes=18)
model = model.to(device)


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def inference(img, model, detection_threshold=0.50):
    '''
    Infernece of a single input image

    inputs:
      img: input-image as torch.tensor (shape: [C, H, W])
      model: model for infernce (torch.nn.Module)
      detection_threshold: Confidence-threshold for NMS (default=0.7)

    returns:
      boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
      labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
      scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
    '''
    model.eval()

    with torch.no_grad():
        img = img.to(device)
        outputs = model([img])
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()
        labels = outputs[0]['labels'].data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        labels = labels[scores >= detection_threshold]
        scores = scores[scores >= detection_threshold]

        return boxes, scores, labels


def plot_image(img, boxes, scores, labels, dataset, save_path=None):
    '''
    Function that draws the BBoxes, scores, and labels on the image.

    inputs:
      img: input-image as numpy.array (shape: [H, W, C])
      boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
      scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
      labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
      dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
    '''

    cmap = plt.get_cmap("tab20b")
    class_labels = np.array(dataset)
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = img.shape
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img)
    for i, box in enumerate(boxes):
        class_pred = labels[i]
        conf = scores[i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            box[0], box[1],
            s=class_labels[int(class_pred)] + " " + str(int(100 * conf)) + "%",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Used to save inference phase results
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


if __name__ == "__main__":
    img = cv2.imread("./data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0014_89.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = img_transform(img)
    boxes, scores, labels = inference(img, model)

    with open("./data/CUB_200_2011/cub_attibutes.json", "r") as f:
        json_data = json.load(f)

    COCO_LABELS = []
    for i, key in enumerate(json_data.keys()):
        COCO_LABELS.append(key)

    img = img.cpu().permute(1, 2, 0).numpy()
    plot_image(img, boxes, scores, labels, COCO_LABELS)

    # with open("./graph_cbm/finetuning/coco_labels.txt", "r") as coco:
    #     COCO_LABELS = coco.readlines()
    #
    # for i, _ in enumerate(COCO_LABELS):
    #     COCO_LABELS[i] = COCO_LABELS[i].replace("\n", "")
    #
    # img = img.cpu().permute(1, 2, 0).numpy()
    # plot_image(img, boxes, scores, labels, COCO_LABELS)
