import torch
import torch.nn.functional as F
from typing import List, Optional
from torch import nn, Tensor
from torch.nn import Parameter
from torchvision.ops import MultiScaleRoIAlign
from graph_cbm.modeling.detection.transform import resize_boxes
from graph_cbm.modeling.relation.relation_aggregation import Aggregation
from graph_cbm.modeling.scene_graph import SceneGraph, build_scene_graph
from graph_cbm.modeling.target_model import get_target_model
from torch_scatter import scatter_add, scatter_softmax, scatter_mean


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

        # self.attention_layer = nn.Sequential(
        #     nn.Linear(representation_dim, representation_dim),
        #     nn.ReLU(),
        #     nn.Linear(representation_dim, 1)
        # )
        self.att = Parameter(torch.Tensor(1, representation_dim))
        self.proj_layer = nn.Conv2d(out_channels, representation_dim, 1)
        self.aggregation = Aggregation(representation_dim, self.num_relations)
        self.pool_layer = SoftmaxPooling2D(kernel_size=kernel_size)
        self.classifier = nn.Linear(representation_dim, n_tasks)

    def preprocess(self, images, targets):
        images_list = []
        targets_list = []
        for i in range(len(images)):
            image = images[i]
            target = targets[i]
            image, target = self.transform(image, target)
            images_list.append(image)
            targets_list.append(target)
        images_tensor = torch.stack(images_list, dim=0)
        return images_tensor, targets_list


    def forward(self, images: List[Tensor], targets: Optional[list[dict[str, Tensor]]] = None, weights=None):
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

        edge_index_list = [rel['rel_pair_idxs'].t() for rel in rel_graphs]
        edge_type_list = [rel['pred_rel_labels'] for rel in rel_graphs]
        num_rels = [e.shape[1] for e in edge_index_list]
        num_objs = [rel['boxes'].shape[0] for rel in rel_graphs]
        x = torch.squeeze(self.pool_layer(x))
        # x_aggregation, relation_attentions = self.aggregation(x, edge_index_list, edge_type_list, num_objs)  # [N_total_objs,256]
        # batch_size = len(num_objs)
        # batch_idx = torch.repeat_interleave(
        #     torch.arange(batch_size, device=x_aggregation.device),
        #     torch.tensor(num_objs, device=x_aggregation.device)
        # )
        # scene_context_features = scatter_mean(x_aggregation, batch_idx, dim=0)
        # attention_input = x_aggregation + scene_context_features[batch_idx]
        # attention_input = F.leaky_relu(attention_input, negative_slope=0.2)
        # attn_logits = (attention_input * self.att).sum(dim=-1)  # [N_total_objs, 1]
        # d_k = torch.tensor(x_aggregation.shape[-1], dtype=torch.float32)
        # attn_logits = attn_logits / torch.sqrt(d_k)
        # attn_weights = scatter_softmax(attn_logits, batch_idx, dim=0)  # [N_total_objs, 1]
        # weighted_features = x_aggregation * attn_weights.unsqueeze(-1)  # [N_total_objs,256]
        # scene_features_batch = scatter_add(weighted_features, batch_idx, dim=0)  # [N_total_objs,256]->[N,256]
        scene_features_batch, x_aggregation, attentions = self.aggregation(x, edge_index_list, edge_type_list, num_objs)
        object_attentions, relation_attentions = attentions
        y_logits = self.classifier(scene_features_batch)  # [N,20]

        loss = {}
        if self.training:
            y = torch.as_tensor([t['class_label'] for t in targets], dtype=torch.int64, device=y_logits.device)
            loss_task = task_loss(y_logits, y, self.n_tasks)
            loss['loss_task'] = loss_task
            return loss

        result = self.post_processor(y_logits, rel_graphs, object_attentions, relation_attentions, image_sizes,
                                     original_image_sizes, num_objs, num_rels, proj_maps, x_aggregation)
        return result

    def post_processor(self, y_logits, relation_graphs, object_attentions, gat_attention_info, image_sizes,
                       original_image_sizes, num_objs, num_rels, proj_maps, x_aggregation):
        relation_attentions = gat_attention_info[1].split(num_rels, dim=0)
        proj_maps = proj_maps.split(num_objs, dim=0)
        object_attentions = object_attentions.split(num_objs, dim=0)
        x_aggregation = x_aggregation.split(num_objs, dim=0)
        result = []
        for i, graph in enumerate(relation_graphs):
            boxes = graph["boxes"]
            boxes = resize_boxes(boxes, image_sizes[i], original_image_sizes[i])
            graph["boxes"] = boxes
            graph["y_logit"] = y_logits[i]
            graph["y_prob"] = y_prob = F.softmax(y_logits[i], dim=-1)
            graph["object_attention"] = object_attentions[i]
            graph["relation_attention"] = relation_attentions[i]
            graph["proj_map"] = proj_maps[i]
            # Contribution(i -> j on class c) ≈ ObjectAttn(j) * GATAttn(i->j) * FeatureContribution(i to c)
            # rel_pairs = graph["rel_pair_idxs"]
            # source_nodes, target_nodes = rel_pairs[:, 0], rel_pairs[:, 1]
            # pred_class_idx = torch.argmax(y_prob)
            # classifier_weights = self.classifier.weight[pred_class_idx]
            # feat_contrib_of_sources = torch.matmul(x_gnns[i][source_nodes], classifier_weights)
            # obj_attn_of_targets = object_attentions[i][target_nodes].squeeze(-1)
            # rel_attn_scores = relation_attentions[i].squeeze(-1)
            # contributions = (obj_attn_of_targets + rel_attn_scores + feat_contrib_of_sources) / 2
            # graph['triplet_contributions'] = F.relu(contributions)
            result.append(graph)
        return result


def build_model(
        target_name,
        num_classes,
        relation_classes,
        n_tasks,
        weights_path="",
        rel_score_thresh=0.1
):
    graph_backbone = 'resnet50'
    scene_graph_weights_path = f"save_weights/relations/{graph_backbone}-model-best.pth"
    scene_graph = build_scene_graph(
        backbone_name=graph_backbone,
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
