import torch
import torch.nn.functional as F
from torch import nn


def task_loss(y_logits, y, n_tasks, task_class_weights=None):
    loss_task = (torch.nn.CrossEntropyLoss(weight=task_class_weights)
                 if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights))
    return loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)


class C2yModel(nn.Module):
    def __init__(self, num_classes, relation_classes, n_tasks, representation_dim=1024, hidden_dim=512):
        super(C2yModel, self).__init__()
        self.num_classes = num_classes
        self.relation_classes = relation_classes
        self.n_tasks = n_tasks
        self.representation_dim = representation_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)

        self.classifier = nn.Sequential(*[
            nn.Linear(256, n_tasks)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, relation_features, relation_graphs, proposals, features, images, targets):
        if 'pool' in features:
            x_cls = features['pool']
        else:
            x_cls = features['3']
        x_cls = self.global_avg_pool(x_cls)
        x_cls = torch.flatten(x_cls, 1)
        y_logits = self.classifier(x_cls)

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
