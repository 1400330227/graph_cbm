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


def task_loss(y_logits, y, n_tasks, task_class_weights=None):
    criterion_loss = (torch.nn.CrossEntropyLoss(weight=task_class_weights)
                      if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights))
    loss_task = criterion_loss(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)
    with  torch.no_grad():
        preds = y_logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
    return loss_task


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
            kernel_size=(7, 7)
    ):
        super(CBMModel, self).__init__()
        self.target_model = target_model
        self.graph = graph
        self.transform = transform
        self.n_tasks = n_tasks
        self.num_relations = relation_classes
        self.out_channels = out_channels
        self.representation_dim = representation_dim
        self.roi_align = roi_pooling
        self.proj_layer = nn.Conv2d(out_channels, representation_dim, 1)
        self.aggregation = Aggregation(representation_dim, self.num_relations)
        self.pool_layer = SoftmaxPooling2D(kernel_size=kernel_size)
        self.classifier = nn.Linear(representation_dim, n_tasks)

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
        scene_features, x_aggregation, attentions = self.aggregation(x, relations, rel_labels, num_objs)
        object_attentions, relation_attentions, relation_indexes = attentions
        y_logits = self.classifier(scene_features)  # [N,20]

        loss = {}
        if self.training:
            y = torch.as_tensor([t['class_label'] for t in targets], dtype=torch.int64, device=y_logits.device)
            loss_task = task_loss(y_logits, y, self.n_tasks)
            loss['loss_task'] = loss_task
            return loss

        result = self.post_processor(y_logits, rel_graphs, object_attentions, relation_attentions, image_sizes,
                                     original_image_sizes, num_objs, relation_indexes, proj_maps, object_features)
        return result

    def post_processor(self, y_logits, relation_graphs, object_attentions, relation_attentions, image_sizes,
                       original_image_sizes, num_objs, relation_indexes, proj_maps, object_features):
        attention_matrix = torch.zeros((sum(num_objs), sum(num_objs)), device=y_logits.device)
        if relation_indexes is not None:
            source_nodes = relation_indexes[0]
            target_nodes = relation_indexes[1]
            attention_matrix[target_nodes, source_nodes] = relation_attentions
        if object_attentions is not None:
            object_attentions = object_attentions.split(num_objs, dim=0)
        else:
            object_attentions = [None] * len(num_objs)
        proj_maps = proj_maps.split(num_objs, dim=0)
        object_features = object_features.split(num_objs, dim=0)
        offsets = [0] + list(np.cumsum(num_objs))

        result = []
        for i, graph in enumerate(relation_graphs):
            start_idx, end_idx = offsets[i], offsets[i + 1]
            edge_attention = attention_matrix[start_idx:end_idx, start_idx:end_idx]
            boxes = graph["boxes"]
            boxes = resize_boxes(boxes, image_sizes[i], original_image_sizes[i])
            graph["boxes"] = boxes
            graph["y_logit"] = y_logits[i]
            graph["y_prob"] = F.softmax(y_logits[i], dim=-1)
            graph["object_attention"] = object_attentions[i]
            graph["proj_map"] = proj_maps[i]
            graph["object_feature"] = object_features[i]
            graph["edge_attention"] = edge_attention
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
