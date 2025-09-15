import os
import datetime

import numpy as np
import torch
from datasets import transforms
from datasets.cub_dataset import CubDataset
from datasets.voc_dataset import VOCDataSet
from graph_cbm.modeling.detection.backbone import build_resnet50_backbone
from graph_cbm.modeling.detection.detector import build_detector
from graph_cbm.modeling.scene_graph import SceneGraph, build_scene_graph
from graph_cbm.utils.eval_utils import train_one_epoch, sg_evaluate
from graph_cbm.utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from graph_cbm.utils.plot_curve import plot_loss_and_lr, plot_map


def create_model(num_classes, relation_classes, n_tasks, args):
    backbone_name = args.backbone
    detector_weights_path = f"save_weights/detector/{args.backbone}-fpn-model-best.pth"
    weights_path = ""
    model = build_scene_graph(
        backbone_name=backbone_name,
        num_classes=num_classes,
        relation_classes=relation_classes,
        detector_weights_path=detector_weights_path,
        weights_path=weights_path,
        use_cbm=False,
    )

    detector_params = model.detector.parameters()
    predictor_params = model.predictor.parameters()

    params = [
        {"params": detector_params, "lr": args.lr * 0.01},  # 例如，主学习率的1% (1e-6)
        {"params": predictor_params, "lr": args.lr}  # 例如, 1e-4
    ]

    return model, params


def calculate_class_weights(dataset, num_classes, device):
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    total_possible_rels = 0
    for _, target in dataset:
        rel_matrix = target['relation']
        num_objs = len(target['labels'])
        total_possible_rels += num_objs * (num_objs - 1)
        unique_rels = rel_matrix[rel_matrix > 0].unique()
        for rel_idx in unique_rels:
            class_counts[rel_idx] += (rel_matrix == rel_idx).sum()
    total_fg_rels = class_counts.sum()
    class_counts[0] = total_possible_rels - total_fg_rels
    zero_counts_mask = class_counts == 0
    class_counts_smoothed = class_counts.clone()
    class_counts_smoothed[zero_counts_mask] = 1
    log_counts = torch.log(class_counts_smoothed)
    epsilon = 1e-6
    weights = torch.max(log_counts) / (log_counts + epsilon)
    background_weight_factor = 0.1
    weights[0] = weights[0] * background_weight_factor
    weights = weights / torch.sum(weights) * num_classes
    return weights.to(device)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }
    # VOC_root = args.data_path
    # if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
    #     raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))
    # train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt")
    train_dataset = CubDataset("data/CUB_200_2011", data_transform["train"], True)
    relation_weights = calculate_class_weights(train_dataset, args.relation_classes + 1, device)
    train_sampler = None
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            collate_fn=train_dataset.collate_fn
        )
    # val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt")
    val_dataset = CubDataset("data/CUB_200_2011", data_transform["val"], False)
    val_data_set_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    model, params = create_model(args.num_classes + 1, args.relation_classes + 1, args.n_tasks, args)
    model.to(device)
    # params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        # lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
        last_epoch=-1
    )

    best_acc = 0.
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        best_acc = checkpoint.get('best_acc', 0.)
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            device=device,
            epoch=epoch,
            print_freq=50,
            warmup=True,
            scaler=scaler,
            weights=relation_weights,
            sgg_weight=args.sgg_weight,
            cls_weight=args.cls_weight,
        )
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)
        lr_scheduler.step()
        coco_info, sgg_info = sg_evaluate(model, val_data_set_loader, device=device, mode=args.mode)
        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch
        }
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        test_acc = sum(sgg_info.values()) / len(sgg_info)
        if test_acc > best_acc:
            best_acc = test_acc
            save_files['best_acc'] = test_acc
            torch.save(save_files, f"save_weights/relations/{args.backbone}-model-best.pth")
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('--data-path', default='data', help='dataset')
    parser.add_argument('--backbone', default='resnet50', help='backbone')
    parser.add_argument('--num-classes', default=24, type=int, help='num_classes')
    parser.add_argument('--relation-classes', default=18, type=int, help='relation_classes')
    parser.add_argument('--n_tasks', default=20, type=int, help='n_tasks')
    parser.add_argument('--output-dir', default='save_weights', help='path where to save')
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--batch_size', default=5, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--mode", default='predcls', choices=['predcls', 'sgcls', 'sgdet', 'preddet'],
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--sgg_weight', default=1, type=int, help='sgg_weight')
    parser.add_argument('--cls_weight', default=1, type=int, help='cls_weight')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(args)
