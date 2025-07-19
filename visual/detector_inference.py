import os
from os.path import join, isdir

import cv2
import numpy as np
import torch
import json

from PIL import Image
from matplotlib import patches, pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from graph_cbm.modeling.detection.backbone import build_resnet50_backbone
from graph_cbm.modeling.detection.detector import FasterRCNN


def create_detector_model(num_classes, load_pretrain_weights=False):
    backbone = build_resnet50_backbone(pretrained=False)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    if load_pretrain_weights:
        weight_path = "../save_weights/detector/resnet-fpn-model-best.pth"
        weights_dict = torch.load(weight_path, map_location='cpu')
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict)
    return model


class CubDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([transforms.ToTensor()])
        folder_list = [f for f in os.listdir(self.root) if isdir(join(self.root, f))]
        folder_list.sort()

        self.data = []
        for i, folder in enumerate(folder_list):
            folder_path = join(self.root, folder)
            classfile_list = [cf for cf in os.listdir(folder_path)
                              if (os.path.isfile(os.path.join(folder_path, cf)) and cf[0] != '.')
                              and cf.lower().endswith('.jpg')]

            for image_path in classfile_list:
                image_path = os.path.join(folder_path, image_path)
                self.data.append(image_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image, img_path

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_detector_model(load_pretrain_weights=True, num_classes=25)
model = model.to(device)


def img_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


def inference(img, model, detection_threshold=0.5):
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
        # plt.text(
        #     box[0], box[1],
        #     s=class_labels[int(class_pred)] + " " + str(int(100 * conf)) + "%",
        #     color="white",
        #     verticalalignment="top",
        #     bbox={"color": colors[int(class_pred)], "pad": 0},
        # )

    # Used to save inference phase results
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def save_images_json(folder_path, dataset):
    cub_dataset = CubDataset(folder_path)
    data_loader = torch.utils.data.DataLoader(cub_dataset, batch_size=10, collate_fn=cub_dataset.collate_fn)
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (images, paths) in pbar:
            images = list(image.to(device) for image in images)
            save_paths = [path for path in paths]
            outputs = model(images)
            save_x_anylabeling_json_list(images, outputs, dataset, save_paths)


def save_x_anylabeling_json_list(images, outputs, dataset, save_paths, detection_threshold=0.5):
    for i, (img, output, save_path) in enumerate(zip(images, outputs, save_paths)):
        boxes, scores, labels = output["boxes"], output["scores"], output["labels"]
        boxes = boxes.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        boxes = boxes[scores >= detection_threshold].astype(np.float32)
        labels = labels[scores >= detection_threshold]
        scores = scores[scores >= detection_threshold]

        save_x_anylabeling_json(img, boxes, scores, labels, dataset, save_path)


def save_x_anylabeling_json(img, boxes, scores, labels, dataset, file_path):
    result = {
        "version": "3.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": "image.jpg",  # 占位文件名，实际使用时应替换为真实文件名
        "imageData": None,  # 设置为 null 符合 X-AnyLabeling 规范
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1]
    }

    file_name = os.path.basename(file_path)
    result["imagePath"] = file_name

    # Process each detection
    for i, box in enumerate(boxes):
        class_id = int(labels[i])
        class_name = dataset[class_id]
        score = float(scores[i])

        # 提取坐标值
        xmin, ymin, xmax, ymax = box
        # 创建四个点（顺时针顺序：左上 → 右上 → 右下 → 左下）
        points = [
            [float(xmin), float(ymin)],  # 左上角
            [float(xmax), float(ymin)],  # 右上角
            [float(xmax), float(ymax)],  # 右下角
            [float(xmin), float(ymax)]  # 左下角
        ]

        shape = {
            "kie_linking": [],
            "label": class_name,
            "score": score,
            "points": points,
            "group_id": None,
            "description": f"{class_name} ({score:.4f})",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {}
        }
        result["shapes"].append(shape)
    # Save JSON file
    save_path = file_path.replace(".jpg", ".json")
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)


def show_img():
    img = cv2.imread("../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = img_transform(img)
    boxes, scores, labels = inference(img, model)

    with open("../data/CUB_200_2011/cub_attributes.json", "r") as f:
        json_data = json.load(f)
    #
    COCO_LABELS = []
    for i, key in enumerate(json_data.keys()):
        COCO_LABELS.append(key)
    COCO_LABELS.insert(0, 'background')
    img = img.cpu().permute(1, 2, 0).numpy()
    plot_image(img, boxes, scores, labels, COCO_LABELS)


def save_json():
    with open("../data/CUB_200_2011/cub_attributes.json", "r") as f:
        json_data = json.load(f)
    COCO_LABELS = []
    for i, key in enumerate(json_data.keys()):
        COCO_LABELS.append(key)
    COCO_LABELS.insert(0, 'background')
    folder_path = "/home/txw/pycharm_project/efficient_cbm/data/CUB_200_2011/images"
    save_images_json(folder_path, COCO_LABELS)


if __name__ == "__main__":
    show_img()
