import math
import sys
import time

import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from .cbm_eval import CBMEvaluator
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import graph_cbm.utils.distributed_utils as utils
from .sg_eval import BasicSceneGraphEvaluator
from sklearn.metrics import confusion_matrix

# CLASS_LABELS = {
#     0: "Black_footed_Albatross", 1: "Laysan_Albatross", 2: "Sooty_Albatross", 3: "Groove_billed_Ani",
#     4: "Crested_Auklet", 5: "Least_Auklet", 6: "Parakeet_Auklet", 7: "Rhinoceros_Auklet",
#     8: "Brewer_Blackbird", 9: "Red_winged_Blackbird", 10: "Rusty_Blackbird", 11: "Yellow_headed_Blackbird",
#     12: "Bobolink", 13: "Indigo_Bunting", 14: "Lazuli_Bunting", 15: "Painted_Bunting",
#     16: "Cardinal", 17: "Spotted_Catbird", 18: "Gray_Catbird", 19: "Yellow_breasted_Chat"
# }


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, warmup=False, scaler=None,
                    weights=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            loss_dict = model(images, targets, weights)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


@torch.no_grad()
def sg_evaluate(model, data_loader, device, mode):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    sg_evaluator = BasicSceneGraphEvaluator.all_modes()

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        sg_val_batch(targets, outputs, sg_evaluator[mode])

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    sg_evaluator[mode].print_stats()
    sgg_info = sg_evaluator[mode].recall_means

    return coco_info, sgg_info


@torch.no_grad()
def cbm_evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    cbm_evaluator = CBMEvaluator()
    # all_preds = []
    # all_labels = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        # cbm_val_batch(targets, outputs, cbm_evaluator)
        y_probs = torch.stack([output["y_logit"] for output in outputs], dim=0).detach().cpu().numpy()
        y_pred = np.argmax(y_probs, axis=1)
        y_true = torch.concat([target["class_label"] for target in targets], dim=0).detach().cpu().numpy()
        cbm_evaluator.compute_bin_accuracy(y_pred, y_true)
        # all_preds.extend(y_pred)
        # all_labels.extend(y_true)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # cm = confusion_matrix(all_labels, all_preds)
    # fig, ax = plt.subplots(figsize=(max(12, 20 // 2), max(10, 20 // 2.5)))
    # sns.heatmap(
    #     cm,
    #     annot=True,  # 在每个单元格中显示数值
    #     fmt='d',  # 将数值格式化为整数
    #     cmap='Blues',  # 选择一个颜色映射
    #     ax=ax,
    #     xticklabels=CLASS_LABELS.values(),
    #     yticklabels=CLASS_LABELS.values()
    # )
    # ax.set_xlabel('Predicted Label', fontsize=14)
    # ax.set_ylabel('True Label', fontsize=14)
    # ax.set_title('Confusion Matrix', fontsize=18)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.tight_layout()
    # plt.show()
    # plt.close(fig)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    cbm_evaluator.print_results()
    cbm_info = cbm_evaluator.result_means
    return cbm_info


def sg_val_batch(targets, outputs, evaluator):
    for i, (target, output) in enumerate(zip(targets, outputs)):
        gt_entry = {
            'gt_classes': target["labels"].detach().cpu().numpy(),
            'gt_relations': target["relation_tuple"].detach().cpu().numpy(),
            'gt_boxes': target["boxes"].detach().cpu().numpy(),
            'gt_image_id': target["image_id"].detach().cpu().numpy(),
        }
        pred_entry = {
            # about objects
            'pred_boxes': output["boxes"].detach().cpu().numpy(),
            'pred_classes': output["labels"].detach().cpu().numpy(),
            'obj_scores': output["scores"].detach().cpu().numpy(),
            # about relations
            'pred_rel_inds': output["rel_pair_idxs"].detach().cpu().numpy(),
            'rel_scores': output["pred_rel_scores"].detach().cpu().numpy(),
            'pred_rel_labels': output["pred_rel_labels"].detach().cpu().numpy(),
        }
        evaluator.evaluate_scene_graph_entry(gt_entry, pred_entry)


def cbm_val_batch(targets, outputs, cbm_evaluator):
    y_probs = torch.stack([output["y_logit"] for output in outputs], dim=0)
    y_true = torch.concat([target["class_label"] for target in targets], dim=0).cpu().detach().numpy()
    cbm_evaluator.compute_bin_accuracy(y_probs, y_true)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


def compute_base_value(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()

    all_scene_features = []
    for batch_idx, (image, targets) in enumerate(data_loader):
        image = list(img.to(device) for img in image)
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        scene_feat = outputs['scene_features']
        all_scene_features.append(outputs)