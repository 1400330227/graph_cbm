import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch import nn, Tensor
from torchvision.ops import MultiScaleRoIAlign
from graph_cbm.modeling.detection.transform import resize_boxes
from graph_cbm.modeling.relation.relation_aggregation import Aggregation
from graph_cbm.modeling.scene_graph import SceneGraph, build_scene_graph
from graph_cbm.modeling.target_model import get_target_model
from torch_scatter import scatter_add


def task_loss(y_logits, y, n_tasks, task_class_weights=None):
    criterion_loss = (torch.nn.CrossEntropyLoss(weight=task_class_weights)
                      if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights))
    loss_task = criterion_loss(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
    with  torch.no_grad():
        preds = y_logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
    return loss_task


def explanation_loss(num_channels, object_features, num_objs, y_true, object_attentions, relation_attentions,
                     relation_indexes, factor, importance_offset=0.1):
    batch_idx_node = torch.repeat_interleave(
        torch.arange(len(num_objs), device=object_features.device),
        torch.tensor(num_objs, device=object_features.device)
    )
    (node_expl_factor, edge_expl_factor) = factor

    pooled_importance_node = torch.zeros(len(num_objs), num_channels, device=object_features.device)
    if node_expl_factor > 0 and object_attentions is not None:
        if object_attentions.dim() == 3:  # (L, V, num_channels)
            node_importance = object_attentions.mean(dim=0)  # (V, K)
        else:
            node_importance = object_attentions
        node_transformed = importance_offset * node_importance
        pooled_importance_node = scatter_add(node_transformed, batch_idx_node, dim=0, dim_size=len(num_objs))  # (B, K)

    pooled_importance_edge = torch.zeros(len(num_objs), num_channels, device=object_features.device)
    if edge_expl_factor > 0 and relation_attentions is not None and relation_indexes is not None:
        if relation_attentions.dim() == 3:  # (L, E, num_channels)
            edge_importance = relation_attentions.mean(dim=0)  # (E, K)
        else:
            edge_importance = relation_attentions
        source_batch = batch_idx_node[relation_indexes[0]]  # (E,)
        target_batch = batch_idx_node[relation_indexes[1]]
        if not torch.all(source_batch == target_batch):
            print("Warning: Some edges connect nodes from different graphs!")
        batch_idx_edge = source_batch
        edge_transformed = importance_offset * edge_importance
        # len(num_objs) denotes the number of graphs in the batch.
        pooled_importance_edge = scatter_add(edge_transformed, batch_idx_edge, dim=0, dim_size=len(num_objs))

    pooled_importance = (node_expl_factor * pooled_importance_node + edge_expl_factor * pooled_importance_edge)  # (B,K)

    if y_true.dim() == 1:
        values_true = F.one_hot(y_true, num_classes=num_channels).float()  # (B, C)[0,0,0,1,0,...,0,0]
    else:
        values_true = y_true.float()
    # torch.sigmoid(0.1 * pooled_importance) ≈ smoothed_labels
    loss_expl = F.binary_cross_entropy_with_logits(pooled_importance, values_true, reduction='mean')
    # values_pred = torch.sigmoid(0.1 * pooled_importance)  # [0, 1]
    # values_true = values_true * 0.9 + (1 - 0.9) / num_channels # [0, 0.9]
    # loss_expl = F.binary_cross_entropy(values_pred, values_true)

    # pooled_importance = pooled_importance.sum(dim=-1)  # (B, )
    # if y_true.dim() == 1:
    #     values_true = F.one_hot(y_true, num_classes=num_channels).float()
    # else:
    #     values_true = y_true.float()
    # loss_expl = F.binary_cross_entropy_with_logits(pooled_importance, values_true, reduction='mean')
    return loss_expl


class SoftmaxPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftmaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        N, C, H, W = x.shape
        h, w = self.kernel_size

        # Unfold the input tensor to extract sliding patches
        patches = F.unfold(x, kernel_size=(h, w), stride=self.stride, padding=self.padding)

        # Reshape to (N, C, h*w, num_patches)
        patches = patches.view(N, C, h * w, -1)

        # Apply softmax over the spatial dimensions (h*w)
        softmax_weights = F.softmax(patches, dim=2)

        # Compute weighted average
        weighted_patches = patches * softmax_weights
        pooled = weighted_patches.sum(dim=2)

        # Reshape back to (N, C, H', W')
        H_out = (H + 2 * self.padding - h) // self.stride[0] + 1
        W_out = (W + 2 * self.padding - w) // self.stride[1] + 1
        pooled = pooled.view(N, C, H_out, W_out)

        return pooled


class CBMModel(nn.Module):
    def __init__(
            self,
            target_model: nn.Module,
            graph: SceneGraph,
            transform: nn.Module,
            n_tasks,
            relation_classes,
            roi_pooling=MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2),
            out_channels=2048,
            representation_dim=256,
            kernel_size=(7, 7),
            share_weights=True,
            use_channel_bias=True,
            num_layers=2,
            concat=True,
            dropout_prob=0.3,
            relation_embedding_dim=64,
            importance_factor=1.0

    ):
        super(CBMModel, self).__init__()
        self.target_model = target_model
        self.graph = graph
        self.transform = transform
        self.n_tasks = n_tasks
        self.num_relations = relation_classes
        self.out_channels = out_channels
        self.representation_dim = representation_dim
        self.num_channels = n_tasks
        self.share_weights = share_weights
        self.use_channel_bias = use_channel_bias
        self.roi_align = roi_pooling
        self.importance_factor = importance_factor

        self.proj_layer = nn.Conv2d(out_channels, representation_dim, 1)
        self.aggregation = Aggregation(
            node_feature_dim=representation_dim,
            num_relations=relation_classes,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            num_channels=n_tasks,
            relation_dim=relation_embedding_dim,
            concat=concat,
            share_weights=share_weights,
            use_channel_bias=use_channel_bias
        )
        self.pool_layer = SoftmaxPooling2D(kernel_size=kernel_size)
        aggregation_output_dim = (representation_dim * n_tasks) if concat else representation_dim
        self.classifier = nn.Linear(aggregation_output_dim, n_tasks)

    def preprocess(self, images, rel_graphs):
        images_list = []
        rel_graph_list = []
        for i in range(len(images)):
            image = images[i]
            target = rel_graphs[i] if rel_graphs[i] is not None else None
            image, target = self.transform(image, target)
            images_list.append(image)
            rel_graph_list.append(target)
        images_tensor = torch.stack(images_list, dim=0)
        return images_tensor, rel_graph_list

    def forward(
            self,
            images: List[Tensor],
            targets: Optional[list[dict[str, Tensor]]] = None,
            weights=None,
            override_object_features: Optional[Tensor] = None,
            override_relation_structure: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        original_image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]
        with torch.no_grad():
            rel_graphs, _ = self.graph(images, targets)
        x, rel_graphs = self.preprocess(images, rel_graphs)
        image_sizes = [(x.shape[-2], x.shape[-1])] * x.shape[0]
        boxes = [rel['boxes'] for rel in rel_graphs]

        x = self.target_model(x)
        x = self.roi_align(x, boxes, image_sizes)
        x = self.proj_layer(x)

        proj_maps = x

        relations = [rel['rel_pair_idxs'].t() for rel in rel_graphs]
        rel_labels = [rel['pred_rel_labels'] for rel in rel_graphs]
        # num_rels = [e.shape[1] for e in relations]
        num_objs = [rel['boxes'].shape[0] for rel in rel_graphs]
        object_features = torch.squeeze(self.pool_layer(x))
        x = object_features

        if override_object_features is not None:
            assert override_object_features.shape == x.shape
            x = override_object_features
        if override_relation_structure is not None:
            relations, rel_labels = override_relation_structure

        scene_features, attentions = self.aggregation(x, relations, rel_labels, num_objs)
        object_attentions, relation_attentions, relation_indexes = attentions
        if scene_features.dim() > 2:
            batch_size = scene_features.size(0)
            scene_features = scene_features.view(batch_size, -1)

        y_logits = self.classifier(scene_features)  # [N,20]

        loss = {}
        if self.training:
            y = torch.as_tensor([t['class_label'] for t in targets], dtype=torch.int64, device=y_logits.device)
            loss_task = task_loss(y_logits, y, self.n_tasks)
            loss['loss_task'] = loss_task

            if self.importance_factor > 0:
                loss_expl = explanation_loss(self.n_tasks, object_features, num_objs, y, object_attentions,
                                             relation_attentions, relation_indexes, (0.8, 0.2))
                loss['loss_expl'] = loss_expl
            return loss

        result = self.post_processor(y_logits, rel_graphs, object_attentions, relation_attentions, image_sizes,
                                     original_image_sizes, num_objs, relation_indexes, proj_maps, object_features)
        return result

    def post_processor(self, y_logits, relation_graphs, object_attentions, relation_attentions, image_sizes,
                       original_image_sizes, num_objs, relation_indexes, proj_maps, object_features):
        if relation_attentions is not None:
            if relation_attentions.dim() == 3:  # (L, E, K)
                relation_attentions = relation_attentions.mean(dim=0)  # (E, K)

        attention_matrix = torch.zeros((sum(num_objs), sum(num_objs), self.n_tasks), device=y_logits.device)
        if relation_indexes is not None and relation_attentions is not None:
            source_nodes = relation_indexes[0]
            target_nodes = relation_indexes[1]
            attention_matrix[target_nodes, source_nodes] = relation_attentions

        if object_attentions is not None:
            if object_attentions.dim() == 3:  # (L, V, K)
                object_attentions = object_attentions.mean(dim=0)  # (V, K)
            object_attentions = object_attentions.split(num_objs, dim=0)  # (V, K)
        else:
            object_attentions = [None] * len(num_objs)
        proj_maps = proj_maps.split(num_objs, dim=0)
        object_features = object_features.split(num_objs, dim=0)
        offsets = [0] + list(np.cumsum(num_objs))

        result = []
        for i, graph in enumerate(relation_graphs):
            start_idx, end_idx = offsets[i], offsets[i + 1]
            relation_attention = attention_matrix[start_idx:end_idx, start_idx:end_idx]  # [V, V, K]
            boxes = graph["boxes"]
            boxes = resize_boxes(boxes, image_sizes[i], original_image_sizes[i])
            graph["boxes"] = boxes
            graph["y_logit"] = y_logits[i]
            graph["y_prob"] = F.softmax(y_logits[i], dim=-1)
            graph["object_attention"] = object_attentions[i]  # shape = (V, K)
            graph["proj_map"] = proj_maps[i]
            graph["object_feature"] = object_features[i]
            graph["relation_attention"] = relation_attention  # shape = [V, V, K]
            result.append(graph)
        return result


def build_model(
        target_name,
        num_classes,
        relation_classes,
        n_tasks,
        weights_path="",
        rel_score_thresh=0.4
):
    scene_graph_weights_path = f"save_weights/relations/{target_name}-model-best.pth"
    scene_graph = build_scene_graph(
        backbone_name=target_name,
        num_classes=num_classes,
        relation_classes=relation_classes,
        detector_weights_path='',
        weights_path=scene_graph_weights_path,
        rel_score_thresh=rel_score_thresh,
        use_cbm=True,
    )
    target_model, transform = get_target_model(target_name)
    roi_pooling = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    representation_dim = target_model.out_channels
    model = CBMModel(target_model, scene_graph, transform, n_tasks, relation_classes, roi_pooling, representation_dim)
    if weights_path != "":
        weights_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict, strict=False)

    return model
