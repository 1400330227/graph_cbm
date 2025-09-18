import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch import Tensor, nn
from torchvision import transforms
from PIL import Image
from graph_cbm.modeling.cbm import build_model, CBMModel
from torchvision.transforms import functional as F
from matplotlib.patches import Rectangle

from graph_cbm.modeling.grad_cam import GradCAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NODE_LABELS = {
    0: "background", 1: "back", 2: "beak", 3: "belly", 4: "breast", 5: "crown",
    6: "forehead", 7: "left eye", 8: "left leg", 9: "left wing", 10: "nape",
    11: "right eye", 12: "right leg", 13: "right wing", 14: "tail", 15: "throat",
    16: "seawater", 17: "nostril", 18: "sky", 19: "snow", 20: "grass",
    21: "tree", 22: "stone", 23: "beach", 24: "foreground"
}

EDGE_LABELS = {
    0: "background", 1: "has", 2: "below", 3: "above", 4: "part of", 5: "right of",
    6: "left of", 7: "behind", 8: "in front of", 9: "near", 10: "covering",
    11: "on", 12: "attached to", 13: "side of", 14: "under",
    15: "standing on", 16: "over", 17: "flying in", 18: "looking at"
}

CLASS_LABELS = {
    0: "Black_footed_Albatross", 1: "Laysan_Albatross", 2: "Sooty_Albatross", 3: "Groove_billed_Ani",
    4: "Crested_Auklet", 5: "Least_Auklet", 6: "Parakeet_Auklet", 7: "Rhinoceros_Auklet",
    8: "Brewer_Blackbird", 9: "Red_winged_Blackbird", 10: "Rusty_Blackbird", 11: "Yellow_headed_Blackbird",
    12: "Bobolink", 13: "Indigo_Bunting", 14: "Lazuli_Bunting", 15: "Painted_Bunting",
    16: "Cardinal", 17: "Spotted_Catbird", 18: "Gray_Catbird", 19: "Yellow_breasted_Chat"
}


def get_resnet_imagenet_preprocess(resize_to=(448, 448)):
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(resize_to),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std)
    ])
    return preprocess


def create_model(num_classes, relation_classes, n_tasks):
    backbone_name = 'resnet50'
    weights_path = f"save_weights/classification/{backbone_name}-model-best.pth"
    model = build_model(
        target_name=backbone_name,
        num_classes=num_classes,
        relation_classes=relation_classes,
        n_tasks=n_tasks,
        weights_path=weights_path,
    )
    return model


def visualize_subgraph(graph, edge_importance, target_node_idx, filename, node_map, edge_map, class_map, top_k=None):
    edge_index = graph['rel_pair_idxs'].t().cpu().numpy()
    edge_type = graph['pred_rel_labels'].cpu().numpy()
    edge_weights = edge_importance.cpu().numpy()
    num_nodes = graph['boxes'].shape[0]
    node_class_key = 'pred_box_labels' if 'pred_box_labels' in graph else 'labels'
    if node_class_key not in graph:
        raise ValueError(f"graph_data 字典中缺少节点类别信息 (需要 '{node_class_key}')")
    node_classes = graph[node_class_key].cpu().numpy()
    assert edge_index.shape[1] == len(edge_weights)

    if top_k is not None and 0 < top_k < len(edge_weights):
        important_indices = np.argsort(edge_weights)[-top_k:]
        edge_index = edge_index[:, important_indices]
        edge_type = edge_type[important_indices]
        edge_weights = edge_weights[important_indices]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        G.add_edge(u, v, weight=edge_weights[i], rel_type=edge_type[i])

    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=100)

    node_display_labels = {i: f"{node_map.get(node_classes[i])}" for i in range(num_nodes)}
    node_colors = ['skyblue'] * num_nodes
    node_colors[target_node_idx] = 'tomato'
    node_sizes = [3500] * num_nodes
    node_sizes[target_node_idx] = 4500
    weights_for_viz = [G.edges[u, v]['weight'] for u, v in G.edges()]
    cmap = plt.cm.viridis

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrowstyle="->", arrowsize=25,
        edge_color=weights_for_viz, edge_cmap=cmap,
        width=[1 + w * 5 for w in weights_for_viz], node_size=node_sizes,
        connectionstyle="arc3,rad=0.1"
    )
    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_display_labels, font_size=10, font_color='black')

    processed_pairs = set()
    for u, v in G.edges():
        pair = tuple(sorted((u, v)))
        pos_u, pos_v = pos[u], pos[v]
        mid_point = (pos_u + pos_v) / 2.0
        diff = pos_v - pos_u
        angle = np.degrees(np.arctan2(diff[1], diff[0]))

        if -90 < angle < 90:
            pass
        else:
            angle += 180
        offset = np.array([0.0, 0.0])
        if G.has_edge(v, u) and pair not in processed_pairs:
            perp_vec = np.array([-diff[1], diff[0]])
            perp_vec /= np.linalg.norm(perp_vec)
            offset = perp_vec * 0.1
            G.edges[v, u]['offset'] = -offset
            processed_pairs.add(pair)

        if 'offset' in G.edges[u, v]:
            offset = G.edges[u, v]['offset']

        rel_type = G.edges[u, v]['rel_type']
        weight = G.edges[u, v]['weight']
        label_text = f"{edge_map.get(rel_type, f'未知')}\n({weight:.2f})"

        ax.text(mid_point[0] + offset[0], mid_point[1] + offset[1], label_text,
                ha='center', va='center', rotation=angle, color='navy', size=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

    pred_class_idx = graph['y_prob'].argmax(dim=-1).item()
    title = f"Prediction category: {class_map.get(pred_class_idx, f'Unknown_{pred_class_idx}')}"
    ax.set_title(title, fontsize=22)

    norm = mcolors.Normalize(vmin=min(weights_for_viz, default=0), vmax=max(weights_for_viz, default=1))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Importance score (weight) of relationship', size=14, rotation=270, labelpad=25)

    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def explain_with_draw_bbox(output_graphs, vis_image_rgb):
    output_graph = output_graphs[0]
    attention_weights = output_graph['object_attention_weights'].cpu().numpy()
    boxes = output_graph['boxes'].cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the base image
    ax.imshow(vis_image_rgb)
    ax.axis('off')  # Hide the axes ticks

    # 5. Draw transparent rectangles on top of the image
    if boxes.shape[0] > 0:
        # Normalize weights to [0, 1] for alpha (transparency)
        min_w, max_w = attention_weights.min(), attention_weights.max()
        norm_weights = (attention_weights - min_w) / (max_w - min_w + 1e-6)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1

            # Get the attention weight for this box
            alpha = norm_weights[i][0]

            # Create a Rectangle patch
            rect = Rectangle(
                (x1, y1),  # (x,y) bottom-left corner
                box_width,  # Width
                box_height,  # Height
                facecolor='red',  # Fill color
                alpha=alpha,  # Transparency based on attention
                edgecolor='none'  # No border
            )

            # Add the patch to the axes
            ax.add_patch(rect)
    else:
        print("模型没有检测到任何对象。")

    ax.set_title("Attention Heatmap (Matplotlib)")

    # 6. Save the plot to a file (This is the server-safe method)
    output_filename = "attention_heatmap_matplotlib.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print(f"解释性结果已成功保存到文件: {output_filename}")

    # 7. Try to display the plot (This may fail on a server without a display)
    try:
        print("正在尝试显示图像窗口...")
        plt.show()
    except Exception as e:
        print(f"\n无法显示图像窗口 (这是在服务器上的预期行为)。错误: {e}")
        print("请查看已保存的文件。")


def explain_with_gat_attention(model, pil_image:Image, node_labels_map, edge_labels_map, class_labels_map):
    image_tensor = torch.tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])
    output_graph = output_graphs[0]
    if "gat_relation_attention" not in output_graph:
        print("未在模型输出中找到GAT注意力权重。请检查模型修改是否正确。")
        return
    gat_attention = output_graph['gat_relation_attention']
    edge_importance = gat_attention.mean(dim=1)
    target_node_idx = np.argmax(output_graph['object_attention_weights'].cpu().numpy())
    visualize_subgraph(
        output_graph,
        edge_importance,
        target_node_idx,
        "relational_gat_explanation.png",
        node_map=node_labels_map,
        edge_map=edge_labels_map,
        class_map=class_labels_map,
    )


def explain_with_heatmap(model: CBMModel, pil_image: Image, class_labels_map: dict):
    try:
        target_layer = get_last_conv_name(model)
    except AttributeError:
        return

    grad_cam = GradCAM(model, target_layer)

    image_tensor = F.to_tensor(pil_image).to(device)
    heatmaps, output = grad_cam.generate_heatmap([image_tensor])

    boxes = output[0]['boxes'].cpu().numpy()
    object_attention_weights = output[0]['object_attention_weights'].detach().cpu().numpy()

    original_np_image = np.array(pil_image)
    H, W, _ = original_np_image.shape
    unified_heatmap = np.zeros((H, W), dtype=np.float32)

    for box, heatmap, attn_weight in zip(boxes, heatmaps, object_attention_weights):
        x1, y1, x2, y2 = map(int, box)
        box_w, box_h = x2 - x1, y2 - y1
        heatmap_resized = cv2.resize(heatmap, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
        scaled_heatmap = heatmap_resized * attn_weight.item()
        # scaled_heatmap = heatmap_resized
        roi_on_canvas = unified_heatmap[y1:y2, x1:x2]
        unified_heatmap[y1:y2, x1:x2] = np.maximum(roi_on_canvas, scaled_heatmap)

    if np.max(unified_heatmap) > 0:
        unified_heatmap /= np.max(unified_heatmap)

    unified_heatmap = cv2.GaussianBlur(unified_heatmap, (15, 15), 0)

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(original_np_image)
    ax.imshow(unified_heatmap, cmap='jet', alpha=0.5)

    ax.axis('off')

    pred_idx = output[0]['y_prob'].argmax().item()
    title = f"Prediction: {class_labels_map.get(pred_idx, 'Unknown')}"
    ax.set_title(title, fontsize=20)

    output_filename = "unified_heatmap_explanation.jpg"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()
    plt.close(fig)


def explain_with_object_attention(model: CBMModel, pil_image: Image, class_labels_map: dict):
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])
    output_graph = output_graphs[0]
    boxes = output_graph['boxes'].cpu().numpy()
    attn_weights = output_graph['object_attention_weights'].cpu().numpy().flatten()
    if len(attn_weights) > 0:
        min_w, max_w = attn_weights.min(), attn_weights.max()
        norm_weights = (attn_weights - min_w) / (max_w - min_w + 1e-6)
    else:
        norm_weights = np.array([])

    cmap = plt.cm.plasma

    original_np_image = np.array(pil_image)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(original_np_image)
    for i in np.argsort(norm_weights):
        box = boxes[i]
        norm_w = norm_weights[i]
        original_w = attn_weights[i]
        x1, y1, x2, y2 = map(int, box)
        color = cmap(norm_w)
        linewidth = 1.0 + norm_w * 4.0  # 线宽范围: 1.0 -> 5.0
        rect = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            facecolor='none',
            edgecolor=color,
            linewidth=linewidth
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{original_w:.2f}", color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax.axis('off')
    pred_idx = output_graph['y_prob'].argmax().item()
    title = f"Object Attention Weights for Prediction: {class_labels_map.get(pred_idx, 'Unknown')}"
    ax.set_title(title, fontsize=20)
    norm = mcolors.Normalize(vmin=attn_weights.min() if len(attn_weights) > 0 else 0,
                             vmax=attn_weights.max() if len(attn_weights) > 0 else 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Object Attention Weight', size=14, rotation=270, labelpad=25)
    output_filename = "object_attention_explanation.jpg"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)

    try:
        plt.show()
    except Exception as e:
        print(f"无法显示图像窗口: {e}")

    plt.close(fig)


def explain_attribution_heatmap(model: CBMModel, pil_image: Image, class_labels_map):
    """
    通过前向传播归因（非梯度），生成一个统一的、融合了所有重要性分数的场景热力图。
    """
    print("正在通过前向归因生成统一热力图...")

    # --- 1. 获取模型输出 ---
    # 关键：传入原始 PIL 图像，让模型自己处理预处理，确保数据正确
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])

    output_graph = output_graphs[0]

    # --- 2. 提取所有需要的基础数据 ---
    pred_idx = output_graph['y_prob'].argmax().item()

    # a. 空间特征图 (RoIAlign -> Proj Layer 的输出)
    proj_maps = output_graph['proj_map'].detach()  # Shape: [N, C, H, W]

    # b. 对象级注意力权重 (Attention Layer 的输出)
    object_attention_weights = output_graph['object_attention_weights'].detach()  # Shape: [N, 1]

    # d. 其他元数据
    boxes = output_graph['boxes'].cpu().numpy()

    spatial_heatmaps = torch.mean(proj_maps, dim=1)
    spatial_heatmaps = nn.functional.relu(spatial_heatmaps)

    # c. 对象加权: 将每个空间热力图乘以其对象的注意力权重
    #    广播: [N, H, W] * [N, 1, 1] -> [N, H, W]
    final_heatmaps = spatial_heatmaps * object_attention_weights.view(-1, 1, 1)

    # --- 4. 投影到统一画布并可视化 ---
    original_np_image = np.array(pil_image)
    H, W, _ = original_np_image.shape
    unified_heatmap = np.zeros((H, W), dtype=np.float32)

    for box, heatmap_tensor in zip(boxes, final_heatmaps):
        x1, y1, x2, y2 = map(int, box)
        box_w, box_h = x2 - x1, y2 - y1

        if box_w <= 0 or box_h <= 0: continue

        heatmap = heatmap_tensor.cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (box_w, box_h), interpolation=cv2.INTER_CUBIC)

        # 使用 np.maximum 进行正确的融合
        roi_on_canvas = unified_heatmap[y1:y2, x1:x2]
        unified_heatmap[y1:y2, x1:x2] = np.maximum(roi_on_canvas, heatmap_resized)

    if np.max(unified_heatmap) > 0:
        unified_heatmap /= np.max(unified_heatmap)

    unified_heatmap = cv2.GaussianBlur(unified_heatmap, (15, 15), 0)

    # 可视化
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(original_np_image)
    ax.imshow(unified_heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Prediction: {class_labels_map.get(pred_idx, 'Unknown')}", fontsize=20)

    output_filename = "attribution_heatmap_explanation.jpg"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"前向归因热力图已保存到: {output_filename}")
    plt.show()
    plt.close(fig)

def interpretable():
    model = create_model(25, 19, 20)
    model.eval()
    model.to(device)
    # image_path = 'data/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0003_8337.jpg'
    image_path = 'data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"错误: 找不到图像文件 '{image_path}'。请创建一个虚拟图像或提供正确路径。")
        pil_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    # explain_with_heatmap(model, pil_image, CLASS_LABELS)
    # explain_with_object_attention(model, pil_image, CLASS_LABELS)
    # explain_with_gat_attention(model, pil_image, NODE_LABELS, EDGE_LABELS, CLASS_LABELS)
    explain_attribution_heatmap(model, pil_image, CLASS_LABELS)


if __name__ == '__main__':
    interpretable()
