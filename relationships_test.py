import os
import datetime
import torch
from datasets import transforms
from datasets.cub_dataset import CubDataset
from graph_cbm.modeling.detection.backbone import build_resnet50_backbone
from graph_cbm.modeling.detection.detector import build_detector
from graph_cbm.modeling.graph_cbm import GraphCBM, build_Graph_CBM
from graph_cbm.modeling.relation.predictor import Predictor
from graph_cbm.utils.eval_utils import sg_evaluate
from graph_cbm.utils.plot_curve import plot_map


def create_model(num_classes, relation_classes, n_tasks=200):
    backbone = build_resnet50_backbone(pretrained=False)
    detector = build_detector(backbone, num_classes, use_relation=True, is_train=False)
    predictor = Predictor(obj_classes=num_classes, relation_classes=relation_classes,
                          feature_extractor=detector.roi_heads.box_head)
    weights_path = "save_weights/relations/relations-model-best.pth"
    model = build_Graph_CBM(detector, predictor, num_classes, relation_classes, n_tasks, weights_path)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)
    val_dataset = CubDataset("data/CUB_200_2011", data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    model = create_model(num_classes=args.num_classes + 1, relation_classes=args.relation_classes + 1)
    model.to(device)

    val_map = []
    for epoch in range(args.start_epoch, args.epochs):
        coco_info, sgg_info = sg_evaluate(model, val_data_set_loader, device=device, mode=args.mode)
        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--data-path', default='data', help='dataset')
    parser.add_argument('--num-classes', default=24, type=int, help='num_classes')
    parser.add_argument('--relation-classes', default=42, type=int, help='relation_classes')
    parser.add_argument('--output-dir', default='save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--mode", default='sgdet', choices=['predcls', 'sgcls', 'sgdet', 'preddet'],
                        help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    print(args)
    main(args)
