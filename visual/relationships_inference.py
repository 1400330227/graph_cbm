import json
import os
from os.path import isdir, join, isfile

import torch
import random
import argparse
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from graph_cbm.modeling.scene_graph import build_scene_graph

with open("data/CUB_200_2011/attributes.json", "r") as f:
    attributes_json_data = json.load(f)

with open("data/CUB_200_2011/relations.json", "r") as f:
    relations_json_data = json.load(f)

CUB_CLASSES = ['background'] + list(attributes_json_data.keys())
CUB_RELATIONS = ['background'] + list(relations_json_data.keys())


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
        draw.rectangle(padded_bbox, fill="white", outline="black", width=1)
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


def save_graph_to_json(boxes, labels, rel_pairs, rel_labels, score_threshold=0.3, output_path="graph_output.json"):
    """
    将场景图保存为JSON格式
    """
    rel_scores, rel_pred_ids = rel_labels[:, 1:].max(dim=-1)
    rel_pred_ids += 1

    # 构建rel_pairs列表（文本格式）
    rel_pairs_text = []
    # 构建relationships列表（索引格式）
    relationships = []
    # 构建predicates列表（关系ID）
    predicates = []

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

        # 添加到rel_pairs（文本格式）
        rel_pairs_text.append([subj_text, rel_text, obj_text])

        # 添加到relationships（索引格式）
        relationships.append([int(subj_idx), int(obj_idx)])

        # 添加到predicates（关系ID）
        predicates.append(int(rel_id))

    # 构建最终的JSON数据结构
    graph_data = {
        "rel_pairs": rel_pairs_text,
        "relationships": relationships,
        "predicates": predicates
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)

    print(f"场景图已保存到: {output_path}")
    return graph_data


def build_model(args):
    num_obj_classes = len(CUB_CLASSES)
    num_rel_classes = len(CUB_RELATIONS)
    model = build_scene_graph(
        backbone_name=args.backbone,
        num_classes=num_obj_classes,
        relation_classes=num_rel_classes,
        detector_weights_path="",
        weights_path=args.model_path,
    )
    return model


def process_single_image(image_path, model, device, args):
    """
    处理单张图片并生成场景图JSON
    """
    image_path = Path(image_path)
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"无法打开图片 {image_path}: {e}")
        return None

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(raw_image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]

    obj_scores = pred['scores']
    keep_indices = obj_scores > args.obj_threshold

    if not keep_indices.any():
        print(f"在 {image_path} 中未检测到任何物体")
        return None

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

    if not keep_rel.any():
        print(f"在 {image_path} 中未检测到任何有效关系")
        return None

    filtered_rel_pairs = torch.stack((subj_mapped[keep_rel], obj_mapped[keep_rel]), dim=1)
    filtered_rel_scores = final_rel_labels_and_scores[keep_rel]

    # 生成JSON文件
    json_output_path = Path(str(image_path).replace("images", "relations")).with_suffix(".json")
    graph_data = save_graph_to_json(
        boxes=final_boxes.cpu(),
        labels=final_labels.cpu(),
        rel_pairs=filtered_rel_pairs.cpu(),
        rel_labels=filtered_rel_scores.cpu(),
        score_threshold=args.rel_threshold,
        output_path=str(json_output_path)
    )

    return graph_data


def batch_process_images(args):
    """
    批量处理images文件夹中的所有图片
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    model.to(device)
    model.eval()

    # 构建images文件夹路径
    images_path = Path(args.images_dir)
    if not images_path.exists():
        print(f"错误: 找不到images文件夹: {images_path}")
        return

    # 统计信息
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'no_objects': 0,
        'no_relations': 0
    }

    print(f"开始处理文件夹: {images_path}")
    print(f"设备: {device}")
    print(f"物体阈值: {args.obj_threshold}, 关系阈值: {args.rel_threshold}")
    print("-" * 50)

    folder_list = [f for f in os.listdir(images_path) if isdir(join(images_path, f))]
    folder_list.sort()
    folder_list = folder_list[20:]
    for i, folder in enumerate(folder_list):
        folder_path = join(images_path, folder)
        classfile_list = [cf for cf in os.listdir(folder_path)
                          if (isfile(join(folder_path, cf)) and cf[0] != '.')
                          and cf.lower().endswith('.jpg')]

        for cf in classfile_list:
            image_path = join(images_path, folder, cf)
            stats['total_processed'] += 1
            try:
                result = process_single_image(image_path, model, device, args)
                if result is None:
                    stats['failed'] += 1
                    # 检查具体失败原因
                    if stats['failed'] == stats['no_objects'] + stats['no_relations']:
                        # 如果失败数等于已知原因的总和，说明是新类型的失败
                        print(f"  处理失败")
                    else:
                        print(f"  处理失败")
                else:
                    stats['successful'] += 1
                    print(f"  成功生成JSON")

            except Exception as e:
                stats['failed'] += 1
                print(f"  处理异常: {e}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("处理完成统计:")
    print(f"总处理图片数: {stats['total_processed']}")
    print(f"成功生成JSON: {stats['successful']}")
    print(f"失败数: {stats['failed']}")
    if stats['no_objects'] > 0:
        print(f"  - 无检测物体: {stats['no_objects']}")
    if stats['no_relations'] > 0:
        print(f"  - 无检测关系: {stats['no_relations']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCBM Scene Graph Inference and Visualization")
    parser.add_argument("--image_path", type=str, required=False, help="待推理的单张图片路径。")
    parser.add_argument("--images_dir", type=str, required=False, help="数据集images文件夹路径")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重文件 (.pth) 的路径。")
    parser.add_argument("--backbone", type=str, default="resnet50", help="模型使用的骨干网络，必须与训练时一致。")
    parser.add_argument("--output_path", type=str, default="inference_result.jpg", help="可视化结果的保存路径。")
    parser.add_argument("--device", type=str, default="cuda:3", help="运行推理的设备，例如 'cuda:0' 或 'cpu'。")
    parser.add_argument("--obj_threshold", type=float, default=0.5, help="只显示分数高于此阈值的物体。")
    parser.add_argument("--rel_threshold", type=float, default=0.1, help="只显示分数高于此阈值的关系。")

    args = parser.parse_args()

    batch_process_images(args)
