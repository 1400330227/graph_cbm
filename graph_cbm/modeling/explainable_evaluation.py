import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import Tensor
from typing import List
from data_utils import transforms
from graph_cbm.modeling.cbm import CBMModel
from graph_cbm.modeling.cbm import build_model
from data_utils.cub_dataset import CubDataset


@torch.no_grad()
def evaluate_object_faithfulness(
        model: CBMModel,
        images: List[Tensor],
        k: float,  # (0.0 to 1.0)
        test_type: str  # 'comprehensiveness' 或 'sufficiency'
):
    model.eval()
    result_orig = model(images)
    probs_orig_list = [res['y_prob'] for res in result_orig]
    attentions_list = [res['object_attention'] for res in result_orig]

    batched_probs_orig = torch.stack(probs_orig_list)
    batched_attentions = torch.concat(attentions_list, dim=0)
    pred_probs_orig, pred_classes_orig = batched_probs_orig.max(dim=-1)

    node_features = torch.concat([res['object_feature'] for res in result_orig], dim=0)
    num_total_nodes = node_features.shape[0]
    num_nodes_to_perturb = int(k * num_total_nodes)

    sorted_indices = torch.argsort(batched_attentions, descending=True)

    if test_type == 'comprehensiveness':
        # 删除测试: 屏蔽掉注意力最高的 top-k% 节点
        indices_to_mask = sorted_indices[:num_nodes_to_perturb]
        perturbed_node_features = node_features.clone()
        perturbed_node_features[indices_to_mask] = 0.0  # 置零
    elif test_type == 'sufficiency':
        # 保留测试: 只保留注意力最高的 top-k% 节点
        indices_to_keep = sorted_indices[:num_nodes_to_perturb]
        perturbed_node_features = torch.zeros_like(node_features)
        if num_nodes_to_perturb > 0:
            perturbed_node_features[indices_to_keep] = node_features[indices_to_keep]
    else:
        raise ValueError("test_type must be 'comprehensiveness' or 'sufficiency'")

    result_perturbed = model(images, targets=None, override_object_features=perturbed_node_features)
    probs_perturbed_list = [res['y_prob'] for res in result_perturbed]
    batched_probs_perturbed = torch.stack(probs_perturbed_list)

    prob_of_orig_class_after_perturb = torch.gather(
        batched_probs_perturbed, 1, pred_classes_orig.unsqueeze(-1)
    ).squeeze()

    if test_type == 'comprehensiveness':
        score = pred_probs_orig - prob_of_orig_class_after_perturb
    else:  # sufficiency
        score = prob_of_orig_class_after_perturb
    return score.mean().item()


@torch.no_grad()
def evaluate_relation_faithfulness(
        model: CBMModel,
        images: List[Tensor],
        k: float,  # (0.0 to 1.0)
        test_type: str  # 'comprehensiveness' 或 'sufficiency'
):
    model.eval()

    if k == 0 and test_type == 'comprehensiveness': return 0.0

    result_orig = model(images, None)

    probs_orig_list = [res['y_prob'] for res in result_orig]
    batched_probs_orig = torch.stack(probs_orig_list)
    pred_probs_orig, pred_classes_orig = batched_probs_orig.max(dim=-1)

    original_relation_list = [res['rel_pair_idxs'].t() for res in result_orig]
    original_rel_type_list = [res['pred_rel_labels'] for res in result_orig]
    attention_matrices = [res['edge_attention'] for res in result_orig]

    perturbed_relation = []
    perturbed_rel_type = []

    for i in range(len(result_orig)):
        edge_index = original_relation_list[i]
        edge_type = original_rel_type_list[i]
        attn_matrix = attention_matrices[i]

        num_edges = edge_index.shape[1]
        if num_edges == 0:
            perturbed_relation.append(edge_index)
            perturbed_rel_type.append(edge_type)
            continue

        source_nodes, target_nodes = edge_index[0], edge_index[1]
        edge_attentions = attn_matrix[target_nodes, source_nodes]

        sorted_indices = torch.argsort(edge_attentions, descending=True)
        num_edges_to_perturb = int(k * num_edges)

        if test_type == 'comprehensiveness':
            # 删除测试: 保留注意力最低的边
            indices_to_keep = sorted_indices[num_edges_to_perturb:]
        else:  # sufficiency
            # 保留测试: 保留注意力最高的边
            indices_to_keep = sorted_indices[:num_edges_to_perturb]

        # 根据保留的索引创建新的图结构
        perturbed_edge_index = edge_index[:, indices_to_keep]
        perturbed_edge_type = edge_type[indices_to_keep]

        perturbed_relation.append(perturbed_edge_index)
        perturbed_rel_type.append(perturbed_edge_type)

    # --- 3. 使用被扰动的图列表进行前向传播 ---
    result_perturbed = model(images, None, override_relation_structure=(perturbed_relation, perturbed_rel_type))

    # --- 4. 计算得分 (与之前完全相同) ---
    probs_perturbed_list = [res['y_prob'] for res in result_perturbed]
    batched_probs_perturbed = torch.stack(probs_perturbed_list)
    prob_of_orig_class_after_perturb = torch.gather(
        batched_probs_perturbed, 1, pred_classes_orig.unsqueeze(-1)
    ).squeeze()

    if test_type == 'comprehensiveness':
        score = pred_probs_orig - prob_of_orig_class_after_perturb
    else:
        score = prob_of_orig_class_after_perturb

    # 处理只有一个样本的批次的特殊情况
    if score.dim() == 0:
        return score.item()
    else:
        return score.mean().item()


def run_perturbation_evaluation(model, dataloader, device):
    perturbation_ratios = np.linspace(0, 1.0, 11)  # 0%, 10%, ..., 100%

    all_obj_comp_scores = []
    all_obj_suff_scores = []
    all_rel_comp_scores = []
    all_rel_suff_scores = []

    for k in tqdm(perturbation_ratios, desc="Perturbation Ratios"):
        obj_comp_scores = []
        obj_suff_scores = []
        rel_comp_scores = []
        rel_suff_scores = []
        for batch in dataloader:
            images, targets = batch
            # 将数据移动到正确的设备
            images = [img.to(device) for img in images]
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            obj_comp_score = evaluate_object_faithfulness(model, images, k, 'comprehensiveness')
            obj_suff_score = evaluate_object_faithfulness(model, images, k, 'sufficiency')
            obj_comp_scores.append(obj_comp_score)
            obj_suff_scores.append(obj_suff_score)

            rel_comp_score = evaluate_relation_faithfulness(model, images, k, 'comprehensiveness')
            rel_suff_score = evaluate_relation_faithfulness(model, images, k, 'sufficiency')
            rel_comp_scores.append(rel_comp_score)
            rel_suff_scores.append(rel_suff_score)
        all_obj_comp_scores.append(np.mean(obj_comp_scores))
        all_obj_suff_scores.append(np.mean(obj_suff_scores))
        all_rel_comp_scores.append(np.mean(rel_comp_scores))
        all_rel_suff_scores.append(np.mean(rel_suff_scores))
    result = (perturbation_ratios, all_obj_comp_scores, all_obj_suff_scores, all_rel_comp_scores, all_rel_suff_scores)
    return result


def plot_faithfulness_curves(ratios, comp_scores, suff_scores, entity_type="objects"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 2. 计算 AOPC (Area Over the Perturbation Curve) ---
    # 使用梯形法则进行数值积分，这是计算曲线下面积的标准方法。
    # AOPC 越高，通常意味着解释的忠实性越好。
    aopc_comp = np.trapezoid(comp_scores, ratios)
    aopc_suff = np.trapezoid(suff_scores, ratios)

    # --- 3. 绘制左侧的 Comprehensiveness (删除) 图 ---
    ax1 = axes[0]
    ax1.plot(ratios, comp_scores, marker='o', linestyle='-', label=f'AOPC = {aopc_comp:.3f}')

    # 设置标题和坐标轴标签
    ax1.set_title("Comprehensiveness (Deletion) Test", fontsize=14)
    ax1.set_xlabel(f"Fraction of {entity_type} Removed (k)", fontsize=12)
    ax1.set_ylabel("Drop in Original Prediction Probability", fontsize=12)

    # 设置 Y 轴范围，使其更具可读性（可选）
    # ax1.set_ylim(min(comp_scores) - 0.01, max(comp_scores) + 0.01)

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    # --- 4. 绘制右侧的 Sufficiency (保留) 图 ---
    ax2 = axes[1]
    ax2.plot(ratios, suff_scores, marker='o', linestyle='-', label=f'AOPC = {aopc_suff:.3f}')

    # 设置标题和坐标轴标签
    ax2.set_title("Sufficiency (Insertion) Test", fontsize=14)
    ax2.set_xlabel(f"Fraction of {entity_type} Kept (k)", fontsize=12)
    ax2.set_ylabel("Recovered Probability of Original Prediction", fontsize=12)

    # 设置 Y 轴范围（可选）
    # ax2.set_ylim(min(suff_scores) - 0.05, max(suff_scores) + 0.05)

    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)

    # --- 5. 整体美化和显示 ---
    # 添加一个总标题
    fig.suptitle(f"Faithfulness Evaluation for {entity_type} Attention", fontsize=16, y=1.03)

    # 自动调整子图布局，防止标签重叠
    plt.tight_layout()

    # 显示图像
    plt.show()


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


if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = CubDataset("data/CUB_200_2011", transform, False)
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    model = create_model(25, 20, 20)
    model.to(device)

    ratios, object_comprehensiveness, object_sufficiency, relation_comprehensiveness, relation_sufficiency = \
        run_perturbation_evaluation(model, val_dataloader, device)
    plot_faithfulness_curves(ratios, object_comprehensiveness, object_sufficiency, 'objects')
    plot_faithfulness_curves(ratios, relation_comprehensiveness, relation_sufficiency, 'relations')
