# 文件: relationships_inference.py

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import argparse

# 导入您项目中的相关模块
# 请确保这些路径是正确的
from graph_cbm.modeling.graph_cbm import build_Graph_CBM
from datasets.cub_dataset import CUB_CLASSES, CUB_RELATIONS  # 假设您的类别名称列表在这里


# --- 可视化辅助函数 ---

def get_random_color():
    """生成一个随机的 RGB 颜色"""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_scene_graph(image, boxes, labels, rel_pairs, rel_labels, score_threshold=0.3):
    """
    在图片上绘制场景图。

    Args:
        image (PIL.Image): 原始图片。
        boxes (Tensor): 物体检测框，形状 (N, 4)。
        labels (Tensor): 物体标签ID，形状 (N,)。
        rel_pairs (Tensor): 关系对索引，形状 (K, 2)。
        rel_labels (Tensor): 关系标签ID和分数，形状 (K, num_rel_classes)。
        score_threshold (float): 只显示分数高于此阈值的关系。
    """
    draw = ImageDraw.Draw(image)

    # 使用一种常见的字体，如果找不到，Pillow 会使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        font_large = ImageFont.truetype("arial.ttf", 25)
    except IOError:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()

    # 为每个物体分配一种颜色
    obj_colors = [get_random_color() for _ in range(len(boxes))]

    # 1. 绘制物体框和标签
    for i in range(len(boxes)):
        box = boxes[i].tolist()
        label_id = labels[i].item()
        label_text = CUB_CLASSES[label_id]
        color = obj_colors[i]

        draw.rectangle(box, outline=color, width=3)

        text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), label_text, fill="black", font=font)

    # 2. 绘制关系
    rel_scores, rel_pred_ids = rel_labels[:, 1:].max(dim=-1)  # 忽略背景类 (索引0)
    rel_pred_ids += 1  # 恢复正确的类别ID

    drawn_relations = 0
    for i in range(len(rel_pairs)):
        score = rel_scores[i].item()
        if score < score_threshold:
            continue

        drawn_relations += 1

        # 获取主语 (subject) 和宾语 (object) 的信息
        subj_idx = rel_pairs[i, 0].item()
        obj_idx = rel_pairs[i, 1].item()

        subj_box = boxes[subj_idx]
        obj_box = boxes[obj_idx]

        # 计算框的中心点
        subj_center = ((subj_box[0] + subj_box[2]) / 2, (subj_box[1] + subj_box[3]) / 2)
        obj_center = ((obj_box[0] + obj_box[2]) / 2, (obj_box[1] + obj_box[3]) / 2)

        # 绘制从主语指向宾语的箭头
        draw.line([subj_center, obj_center], fill=obj_colors[subj_idx], width=2)
        # 在宾语一端画一个小圆点
        draw.ellipse([c - 3 for c in obj_center] + [c + 3 for c in obj_center], fill=obj_colors[obj_idx])

        # 在连线的中点写上关系标签
        rel_id = rel_pred_ids[i].item()
        rel_text = CUB_RELATIONS[rel_id]
        mid_point = ((subj_center[0] + obj_center[0]) / 2, (subj_center[1] + obj_center[1]) / 2)

        rel_text_bbox = draw.textbbox(mid_point, rel_text, font=font_large)
        draw.rectangle(rel_text_bbox, fill="white")
        draw.text(mid_point, rel_text, fill="black", font=font_large)

    print(f"绘制了 {len(boxes)} 个物体和 {drawn_relations} 条关系 (阈值 > {score_threshold})。")
    return image


def main(args):
    # --- 1. 设置设备 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 2. 加载模型 ---
    # 注意：这里的 num_classes, relation_classes, n_tasks 需要与您训练时使用的完全一致
    num_obj_classes = len(CUB_CLASSES)
    num_rel_classes = len(CUB_RELATIONS)
    num_task_classes = 200  # 假设是CUB-200的分类任务

    model = build_Graph_CBM(
        backbone_name=args.backbone,
        num_classes=num_obj_classes,
        relation_classes=num_rel_classes,
        n_tasks=num_task_classes,
        detector_weights_path="",  # 在推理时，detector的权重已经包含在SGG模型中了
        weights_path=args.model_path,
        use_c2ymodel=False  # 如果您的模型不包含分类头，可以设为False
    )
    model.to(device)
    model.eval()  # 切换到评估模式
    print(f"模型已从 '{args.model_path}' 加载。")

    # --- 3. 准备图像和预处理 ---
    image_path = args.image_path
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图像文件 '{image_path}'")
        return

    # 定义与训练时兼容的预处理
    transform = T.Compose([
        T.ToTensor(),
    ])
    image_tensor = transform(raw_image).to(device)

    # --- 4. 模型推理 ---
    with torch.no_grad():
        # 模型期望的输入是一个批次 (batch)，所以我们需要给图像增加一个批次维度
        # 同时，模型内部可能需要图像的原始尺寸，所以我们传递一个列表
        predictions = model([image_tensor])

    # --- 5. 解析并可视化结果 ---
    # 由于我们只输入了一张图片，所以只取第一个预测结果
    pred = predictions[0]

    # 从预测结果中提取场景图的关键信息
    # 我们只保留分数高于阈值的物体
    obj_scores = pred['scores']
    keep_indices = obj_scores > args.obj_threshold

    final_boxes = pred['boxes'][keep_indices]
    final_labels = pred['labels'][keep_indices]
    final_rel_pairs = pred['rel_pair_idxs']
    final_rel_labels_and_scores = pred['pred_rel_scores']

    # 过滤关系对，只保留那些主语和宾语都在我们保留的物体列表中的关系
    # 这是一个比较复杂的索引操作
    keep_indices_map = torch.where(keep_indices)[0]

    # 创建一个映射，从旧索引到新索引
    idx_map = -torch.ones(len(obj_scores), dtype=torch.long, device=device)
    idx_map[keep_indices_map] = torch.arange(len(keep_indices_map), device=device)

    # 过滤关系对
    subj_mapped = idx_map[final_rel_pairs[:, 0]]
    obj_mapped = idx_map[final_rel_pairs[:, 1]]

    keep_rel = (subj_mapped != -1) & (obj_mapped != -1)

    filtered_rel_pairs = torch.stack((subj_mapped[keep_rel], obj_mapped[keep_rel]), dim=1)
    filtered_rel_scores = final_rel_labels_and_scores[keep_rel]

    print("\n--- 推理结果 ---")
    if 'scene_prob' in pred:
        scene_probs = pred['scene_prob']
        top_prob, top_class_id = torch.max(scene_probs, dim=0)
        top_class_name = CUB_CLASSES[top_class_id.item() + 1]  # 假设分类任务的类别与物体类别一致
        print(f"场景分类结果: '{top_class_name}' (置信度: {top_prob.item():.4f})")

    # 绘制场景图
    output_image = draw_scene_graph(
        raw_image.copy(),
        final_boxes.cpu(),
        final_labels.cpu(),
        filtered_rel_pairs.cpu(),
        filtered_rel_scores.cpu(),
        score_threshold=args.rel_threshold
    )

    # --- 6. 保存或显示结果 ---
    output_path = args.output_path
    output_image.save(output_path)
    print(f"\n可视化结果已保存到: {output_path}")

    # 如果您想直接显示图片（在支持GUI的环境中），可以取消以下注释
    # output_image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCBM Scene Graph Inference and Visualization")

    parser.add_argument("--image_path", type=str, required=True, help="待推理的单张图片路径。")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型权重文件 (.pth) 的路径。")
    parser.add_argument("--backbone", type=str, default="resnet50", help="模型使用的骨干网络，必须与训练时一致。")
    parser.add_argument("--output_path", type=str, default="inference_result.jpg", help="可视化结果的保存路径。")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行推理的设备，例如 'cuda:0' 或 'cpu'。")
    parser.add_argument("--obj_threshold", type=float, default=0.5, help="只显示分数高于此阈值的物体。")
    parser.add_argument("--rel_threshold", type=float, default=0.3, help="只显示分数高于此阈值的关系。")

    args = parser.parse_args()

    main(args)