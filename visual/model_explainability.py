import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import shap
import pandas as pd
from torch import Tensor, nn
from torchvision import transforms
from PIL import Image
from graph_cbm.modeling.cbm import build_model, CBMModel
from torchvision.transforms import functional as F
from matplotlib.patches import Rectangle

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


def explain_with_graph(model, pil_image: Image, node_labels_map, edge_labels_map, class_labels_map):
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])
    output_graph = output_graphs[0]
    gat_attention = output_graph['relation_attention']
    edge_importance = gat_attention.mean(dim=1)
    target_node_idx = np.argmax(output_graph['object_attention'].cpu().numpy())
    visualize_subgraph(
        output_graph,
        edge_importance,
        target_node_idx,
        "relational_gat_explanation.png",
        node_map=node_labels_map,
        edge_map=edge_labels_map,
        class_map=class_labels_map,
    )


def explain_with_object_detection(model: CBMModel, pil_image: Image, node_labels_map, class_labels_map: dict):
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])
    output_graph = output_graphs[0]
    boxes = output_graph['boxes'].cpu().numpy()
    obj_labels = output_graph['labels'].cpu().numpy()
    attn_weights = output_graph['object_attention'].cpu().numpy().flatten()
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
        label_id = obj_labels[i]
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
        label_name = node_labels_map.get(label_id)
        display_text = f"{label_name}: {original_w:.2f}"
        ax.text(x1, y1 - 5, display_text, color='white', fontsize=10, fontweight='bold',
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


def explain_with_heatmap(model: CBMModel, pil_image: Image, class_labels_map):
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graphs = model([image_tensor])
    output_graph = output_graphs[0]
    pred_idx = output_graph['y_prob'].argmax().item()
    proj_maps = output_graph['proj_map'].detach()  # Shape: [N, C, H, W]
    object_attention_weights = output_graph['object_attention'].detach()  # Shape: [N, 1]
    boxes = output_graph['boxes'].cpu().numpy()
    spatial_heatmaps = torch.mean(proj_maps, dim=1)
    final_heatmaps = spatial_heatmaps * object_attention_weights.view(-1, 1, 1)
    original_np_image = np.array(pil_image)
    H, W, _ = original_np_image.shape
    unified_heatmap = np.zeros((H, W), dtype=np.float32)
    for box, heatmap_tensor in zip(boxes, final_heatmaps):
        x1, y1, x2, y2 = map(int, box)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0: continue
        heatmap = heatmap_tensor.cpu().numpy()
        heatmap = (heatmap - np.mean(heatmap)) / np.std(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
        unified_heatmap[y1:y2, x1:x2] = heatmap
    unified_heatmap = cv2.GaussianBlur(unified_heatmap, (29, 29), 0)
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(original_np_image)
    ax.imshow(unified_heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title(f"Prediction: {class_labels_map.get(pred_idx, 'Unknown')}", fontsize=20)
    plt.show()
    plt.close(fig)


def explain_with_waterfall(model: CBMModel, pil_image: Image,
                           node_labels_map: dict, edge_labels_map: dict,
                           class_labels_map: dict, top_k: int = 15):
    shap.initjs()  # 初始化JS，以便在Jupyter等环境中显示

    # --- 1. 获取模型输出 ---
    image_tensor = F.to_tensor(pil_image).to(device)
    model.eval()
    with torch.no_grad():
        output_graph = model([image_tensor])[0]

    pred_idx = output_graph['y_prob'].argmax().item()
    pred_class_name = class_labels_map.get(pred_idx, f"Class_{pred_idx}")
    contributions = output_graph['triplet_contributions'].cpu().numpy()
    edge_index = output_graph['rel_pair_idxs'].t().cpu().numpy()
    obj_labels = output_graph['labels'].cpu().numpy()
    rel_labels = output_graph['pred_rel_labels'].cpu().numpy()
    gat_attention = output_graph['relation_attention'].mean(dim=1).cpu().numpy()

    if len(contributions) == 0:
        return

    feature_names = [
        f"{node_labels_map.get(obj_labels[s], 'Unk')}-[{edge_labels_map.get(r, 'Unk')}]-{node_labels_map.get(obj_labels[t], 'Unk')}"
        for s, t, r in zip(edge_index[0], edge_index[1], rel_labels)
    ]

    df = pd.DataFrame({
        'feature_name': feature_names,
        'contribution': contributions,
        'gat_attention': gat_attention
    })

    df_sorted = df[df['contribution'] > 0].sort_values(by="contribution", ascending=False).head(top_k)
    df_display = df_sorted.iloc[::-1]
    base_value = model.classifier.bias[pred_idx].item()

    explanation = shap.Explanation(
        values=df_display['contribution'].values,
        base_values=base_value,
        data=df_display['gat_attention'].values,
        feature_names=df_display['feature_name'].tolist()
    )
    fig, ax = plt.subplots(figsize=(24, 12))
    shap.plots.waterfall(explanation, max_display=top_k, show=False)
    current_fig = plt.gcf()
    current_fig.subplots_adjust(left=0.4)
    ax.tick_params(axis='y', which='major', labelsize=10)
    final_logit = base_value + df_display['contribution'].sum()
    plt.title(f"Triplet Contributions for '{pred_class_name}'\nFinal Logit ≈ {final_logit:.2f}", fontsize=14)
    output_filename = "triplet_contribution_waterfall_final_fixed.jpg"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"最终修复版三元组贡献瀑布图已保存到: {output_filename}")
    plt.show()


def interpretable():
    model = create_model(25, 19, 20)
    model.eval()
    model.to(device)
    image_path = 'data/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0003_8337.jpg'
    # image_path = 'data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
    # image_path = 'data/CUB_200_2011/images/003.Sooty_Albatross/Sooty_Albatross_0001_1071.jpg'
    pil_image = Image.open(image_path).convert('RGB')
    explain_with_object_detection(model, pil_image, NODE_LABELS, CLASS_LABELS)
    # explain_with_graph(model, pil_image, NODE_LABELS, EDGE_LABELS, CLASS_LABELS)
    explain_with_heatmap(model, pil_image, CLASS_LABELS)
    explain_with_waterfall(model, pil_image, NODE_LABELS, EDGE_LABELS, CLASS_LABELS)


if __name__ == '__main__':
    interpretable()
