import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt, patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets import transforms

from datasets.cub_dataset1 import CubDataset
from graph_cbm.finetuning import utils
from graph_cbm.finetuning.engine import train_one_epoch, evaluate

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


def get_model_object_detection(num_classes):
    weight_path = 'checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train():
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    dataset = CubDataset("data/CUB_200_2011", data_transform["train"], True)
    dataset_test = CubDataset("data/CUB_200_2011", data_transform["val"], False)

    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=10,
        shuffle=False,
        collate_fn=utils.collate_fn
    )

    num_classes = 2
    model = get_model_object_detection(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.33
    )

    num_epochs = 100

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}

        torch.save(save_files, "save_weights/mm-resnet-fpn-model-{}.pth".format(epoch))



def evaluate():
    num_classes = 2
    model = get_model_object_detection(num_classes)
    model.eval()



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


if __name__ == '__main__':
    train()
