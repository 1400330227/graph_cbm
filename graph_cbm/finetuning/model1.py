import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets import transforms

from datasets.cub_dataset1 import CubDataset
from graph_cbm.finetuning import utils
from graph_cbm.finetuning.engine import train_one_epoch, evaluate

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


def get_model_object_detection(num_classes):
    weight_path = 'graph_cbm/finetuning/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
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


if __name__ == '__main__':
    train()
