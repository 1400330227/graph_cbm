import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool

def task_loss(y_logits, y, n_tasks, task_class_weights=None):
    loss_task = (torch.nn.CrossEntropyLoss(weight=task_class_weights)
                 if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights))
    return loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)


class C2yModel(nn.Module):
    def __init__(self, representation_dim, num_classes, relation_classes, n_tasks, hidden_dim=512):
        super(C2yModel, self).__init__()
        self.num_classes = num_classes
        self.relation_classes = relation_classes
        self.n_tasks = n_tasks
        self.representation_dim = representation_dim
        self.hidden_dim = hidden_dim
        # self.dropout = nn.Dropout(p=0.5)
        # self.conv1 = RGCNConv(representation_dim, self.hidden_dim, num_relations=relation_classes)
        # self.conv2 = RGCNConv(self.hidden_dim, self.hidden_dim, num_relations=relation_classes)

        self.classifier = nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_dim * 2, n_tasks)
        ])

    def forward(self, relation_features, relation_graphs, targets):
        # edge_index_list = [rg['rel_pair_idxs'].t().contiguous() for rg in relation_graphs]
        # edge_type = torch.cat([rg['pred_rel_labels'] for rg in relation_graphs], dim=0)
        # num_nodes_list = [rg['labels'].size(0) for rg in relation_graphs]
        #
        # edge_index_batch = []
        # offset = 0
        # for i, edge_index in enumerate(edge_index_list):
        #     edge_index_batch.append(edge_index + offset)
        #     offset += num_nodes_list[i]
        # edge_index_batch = torch.cat(edge_index_batch, dim=1)

        # batch_vector = torch.cat([
        #     torch.full((n,), i, dtype=torch.long, device=relation_features.device)
        #     for i, n in enumerate(edge_index_list)
        # ], dim=0)

        # batch_vector = torch.cat([
        #     torch.full((len(rg['rel_pair_idxs']),), i, dtype=torch.long, device=relation_features.device)
        #     for i, rg in enumerate(relation_graphs)
        # ], dim=0)

        x = relation_features
        # x = F.relu(self.conv1(x, edge_index_batch, edge_type))
        # x = self.dropout(x)
        # x = F.relu(self.conv2(x, edge_index_batch, edge_type))
        # x = self.dropout(x)

        # avg_pool = global_mean_pool(x, batch_vector)
        # max_pool = global_max_pool(x, batch_vector)
        # graph_features = torch.cat([avg_pool, max_pool], dim=1)
        y_logits = self.classifier(x)

        result = self.post_processor(y_logits, relation_graphs)
        loss = {}
        if self.training:
            y = torch.as_tensor([t['class_label'] for t in targets], dtype=torch.int64, device=y_logits.device)
            loss_task = task_loss(y_logits, y, self.n_tasks)
            loss['loss_task'] = loss_task
        return result, loss

    def post_processor(self, y_logits, relation_graphs):
        result = []
        for i, relation_graph in enumerate(relation_graphs):
            relation_graph["y_logit"] = y_logits[i]
            relation_graph["y_prob"] = F.softmax(y_logits[i], dim=-1)
            result.append(relation_graph)
        return result
