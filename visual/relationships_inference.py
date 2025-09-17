# 文件: relationships_inference.py

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import argparse

from matplotlib import pyplot as plt

# 导入您项目中的相关模块
# 请确保这些路径是正确的
from graph_cbm.modeling.scene_graph import build_scene_graph

CUB_CLASSES = [
    "background",
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "left eye",
    "left leg",
    "left wing",
    "nape",
    "right eye",
    "right leg",
    "right wing",
    "tail",
    "throat",
    "seawater",
    "nostril",
    "sky",
    "snow",
    "grass",
    "tree",
    "stone",
    "beach",
    "foreground",
]

CUB_RELATIONS = [
    "__background__",
    "above",
    "against",
    "along",
    "and",
    "at",
    "attached to",
    "behind",
    "belonging to",
    "covering",
    "flying in",
    "for",
    "from",
    "has",
    "in",
    "in front of",
    "laying on",
    "looking at",
    "lying on",
    "near",
    "of",
    "on",
    "on back of",
    "over",
    "parked on",
    "part of",
    "playing",
    "riding",
    "sitting on",
    "standing on",
    "to",
    "under",
    "using",
    "walking in",
    "walking on",
    "watching",
    "wearing",
    "wears",
    "with",
    "on the left of",
    "on the right of",
    "on the side of",
    "below",
]


# --- 可视化辅助函数 ---

def get_random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_relationships_on_image(image, boxes, rel_pairs, rel_labels, obj_colors, score_threshold=0.3):
    draw = ImageDraw.Draw(image)
    try:
        font_large = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font_large = ImageFont.load_default()

    rel_scores, rel_pred_ids = rel_labels[:, 1:].max(dim=-1)  # 忽略背景类 (索引0)
    rel_pred_ids += 1  # 恢复正确的类别ID

    drawn_relations = 0
    for i in range(len(rel_pairs)):
        score = rel_scores[i].item()
        if score < score_threshold:
            continue
        drawn_relations += 1
        subj_idx = rel_pairs[i, 0].item()
        obj_idx = rel_pairs[i, 1].item()
        subj_box = boxes[subj_idx]
        obj_box = boxes[obj_idx]
        subj_center = ((subj_box[0] + subj_box[2]) / 2, (subj_box[1] + subj_box[3]) / 2)
        obj_center = ((obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2)
        draw.line([subj_center, obj_center], fill=obj_colors[subj_idx], width=2)
        draw.ellipse([c - 3 for c in obj_center] + [c + 3 for c in obj_center], fill=obj_colors[obj_idx])
        rel_id = rel_pred_ids[i].item()
        rel_text = CUB_RELATIONS[rel_id]
        mid_point = ((subj_center[0] + obj_center[0]) / 2, (subj_center[1] + obj_center[1]) / 2)
        rel_text_bbox = draw.textbbox(mid_point, rel_text, font=font_large)
        padding = 2
        padded_bbox = (
            rel_text_bbox[0] - padding,
            rel_text_bbox[1] - padding,
            rel_text_bbox[2] + padding,
            rel_text_bbox[3] + padding
        )
        draw.rectangle(padded_bbox, fill="white", outline="black", width=1)  # outline="black" 可以增加一个黑色边框
        draw.text(mid_point, rel_text, fill="black", font=font_large)
    return image


def draw_object_detection_only(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    obj_colors = [get_random_color() for _ in range(len(boxes))]
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        label_id = labels[i].item()
        label_text = CUB_CLASSES[label_id]
        color = obj_colors[i]
        draw.rectangle(box, outline=color, width=3)
        text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), label_text, fill="black", font=font)
    return image, obj_colors


def draw_graph_on_blank_canvas(
        image_size, boxes, labels, rel_pairs, rel_labels, score_threshold=0.3
):
    width, height = image_size
    num_relations_to_draw = (rel_labels[:, 1:].max(dim=-1)[0] > score_threshold).sum().item()
    estimated_height = (num_relations_to_draw + 2) * 20  # 每行20像素
    canvas_height = max(height, estimated_height)  # 取原图高度和预估高度中较大的一个

    canvas = Image.new('RGB', (width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)

    try:
        font_text = ImageFont.truetype("cour.ttf", 15)  # Courier New 是一种常见的等宽字体
    except IOError:
        print("警告：找不到等宽字体 'cour.ttf'，将使用默认字体，对齐可能不完美。")
        font_text = ImageFont.load_default()

    rel_scores, rel_pred_ids = rel_labels[:, 1:].max(dim=-1)
    rel_pred_ids += 1
    relations_text = []
    header = f"{'Subject':<20} {'Predicate':<15} {'Object':<20} {'Score':<5}"
    relations_text.append(header)
    relations_text.append("-" * len(header) * 2)
    for i in range(len(rel_pairs)):
        score = rel_scores[i].item()
        if score < score_threshold:
            continue
        subj_idx = rel_pairs[i, 0].item()
        obj_idx = rel_pairs[i, 1].item()
        subj_label_id = labels[subj_idx].item()
        obj_label_id = labels[obj_idx].item()
        subj_text = CUB_CLASSES[subj_label_id]
        obj_text = CUB_CLASSES[obj_label_id]
        rel_id = rel_pred_ids[i].item()
        rel_text = CUB_RELATIONS[rel_id]
        relations_text.append(f"{subj_text:<20} {rel_text:<15} {obj_text:<20} {score:.3f}")
    for text in relations_text:
        print(text)
    text_y = 15
    line_height = 20
    for text in relations_text:
        draw.text((15, text_y), text, fill="black", font=font_text)
        text_y += line_height
    return canvas


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    num_obj_classes = len(CUB_CLASSES)
    num_rel_classes = len(CUB_RELATIONS)
    num_task_classes = 20

    model = build_scene_graph(
        backbone_name=args.backbone,
        num_classes=num_obj_classes,
        relation_classes=num_rel_classes,
        detector_weights_path="",
        weights_path=args.model_path,
    )
    model.to(device)
    model.eval()

    image_path = args.image_path
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return

    transform = T.Compose([
        T.ToTensor(),
    ])
    image_tensor = transform(raw_image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]

    obj_scores = pred['scores']
    keep_indices = obj_scores > args.obj_threshold

    final_boxes = pred['boxes'][keep_indices]
    final_labels = pred['labels'][keep_indices]
    final_rel_pairs = pred['rel_pair_idxs']
    final_rel_labels_and_scores = pred['pred_rel_scores']

    keep_indices_map = torch.where(keep_indices)[0]

    idx_map = -torch.ones(len(obj_scores), dtype=torch.long, device=device)
    idx_map[keep_indices_map] = torch.arange(len(keep_indices_map), device=device)

    subj_mapped = idx_map[final_rel_pairs[:, 0]]
    obj_mapped = idx_map[final_rel_pairs[:, 1]]

    keep_rel = (subj_mapped != -1) & (obj_mapped != -1)

    filtered_rel_pairs = torch.stack((subj_mapped[keep_rel], obj_mapped[keep_rel]), dim=1)
    filtered_rel_scores = final_rel_labels_and_scores[keep_rel]

    if 'scene_prob' in pred:
        scene_probs = pred['scene_prob']
        top_prob, top_class_id = torch.max(scene_probs, dim=0)
        top_class_name = CUB_CLASSES[top_class_id.item() + 1]

    detection_image, obj_colors = draw_object_detection_only(
        raw_image.copy(),
        final_boxes.cpu(),
        final_labels.cpu()
    )

    scene_graph_image_on_real = draw_relationships_on_image(
        detection_image.copy(),
        final_boxes.cpu(),
        filtered_rel_pairs.cpu(),
        filtered_rel_scores.cpu(),
        obj_colors,
        score_threshold=args.rel_threshold
    )

    pure_graph_image = draw_graph_on_blank_canvas(
        image_size=raw_image.size,
        boxes=final_boxes.cpu(),
        labels=final_labels.cpu(),
        rel_pairs=filtered_rel_pairs.cpu(),
        rel_labels=filtered_rel_scores.cpu(),
        score_threshold=args.rel_threshold
    )

    scene_graph_image_on_real.save(args.output_path)
    pure_graph_image.save("pure_" + args.output_path)
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 创建一个1行3列的画布
    axes[0].imshow(detection_image)
    axes[0].set_title('Original Image', fontsize=20)
    axes[0].axis('off')
    axes[1].imshow(scene_graph_image_on_real)
    axes[1].set_title('Scene Graph on Image', fontsize=20)
    axes[1].axis('off')
    axes[2].imshow(pure_graph_image)
    axes[2].set_title('Pure Graph Structure', fontsize=20)
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCBM Scene Graph Inference and Visualization")
    parser.add_argument("--image_path", type=str, required=True, help="待推理的单张图片路径。")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重文件 (.pth) 的路径。")
    parser.add_argument("--backbone", type=str, default="resnet50", help="模型使用的骨干网络，必须与训练时一致。")
    parser.add_argument("--output_path", type=str, default="inference_result.jpg", help="可视化结果的保存路径。")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行推理的设备，例如 'cuda:0' 或 'cpu'。")
    parser.add_argument("--obj_threshold", type=float, default=0.5, help="只显示分数高于此阈值的物体。")
    parser.add_argument("--rel_threshold", type=float, default=0.01, help="只显示分数高于此阈值的关系。")

    args = parser.parse_args()

    main(args)
