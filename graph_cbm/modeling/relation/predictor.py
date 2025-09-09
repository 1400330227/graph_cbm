import torch
from torch import nn
from graph_cbm.modeling.relation.mamba import MambaRelationEncoder


def relation_loss(rel_labels, relation_logits, rel_weights=None):
    criterion_loss = nn.CrossEntropyLoss(weight=rel_weights)
    relation_logits = torch.concat(relation_logits, dim=0)
    rel_labels = torch.concat(rel_labels, dim=0)
    with torch.no_grad():
        preds = relation_logits.argmax(dim=1)
        accuracy = (preds == rel_labels).float().mean()
    loss_relation = criterion_loss(relation_logits, rel_labels.long())
    return loss_relation


class Predictor(nn.Module):
    def __init__(
            self,
            obj_classes,
            relation_classes,
            representation_dim=1024,
            embedding_dim=256,
            hidden_dim=512,
    ):
        super(Predictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.obj_classes = obj_classes
        self.representation_dim = representation_dim

        self.lin_obj = nn.Linear(representation_dim + embedding_dim + 128, hidden_dim)
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        self.obj_embed = nn.Embedding(self.obj_classes, embedding_dim)

        self.local_encoder = MambaRelationEncoder(embed_dim=hidden_dim, depth=4, if_cls_token=False)

        self.relation_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + representation_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, relation_classes)
        )

    def encode_box_info(self, proposals):
        boxes_info = []
        for proposal in proposals:
            boxes = proposal['boxes']
            img_size = proposal["image_size"]
            wid = img_size[0]
            hei = img_size[1]
            wh = boxes[:, 2:] - boxes[:, :2] + 1.0
            xy = boxes[:, :2] + 0.5 * wh
            w, h = wh.split([1, 1], dim=-1)
            x, y = xy.split([1, 1], dim=-1)
            x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
            info = torch.concat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                                 w * h / (wid * hei)], dim=-1).view(-1, 9)
            boxes_info.append(info)
        return torch.concat(boxes_info, dim=0)

    def forward(self, box_features, union_features, proposals, rel_pair_idxs, rel_labels, rel_weights=None):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [boxes_in_image["boxes"].shape[0] for boxes_in_image in proposals]

        proposal_labels = torch.concat([proposal["labels"] for proposal in proposals], dim=0)
        obj_logits = torch.concat([proposal["logits"] for proposal in proposals], dim=0)

        # --- 2. 初始对象编码 (Mamba) ---
        obj_embed = self.obj_embed(proposal_labels)
        pos_embed = self.bbox_embed(self.encode_box_info(proposals))
        obj_representation = torch.concat((box_features, obj_embed, pos_embed), -1)
        obj_representation = self.lin_obj(obj_representation)
        local_context = self.local_encoder(obj_representation, num_objs)

        local_contexts = local_context.split(num_objs, dim=0)
        pair_reps_list = []
        for pair_idx, local_context in zip(rel_pair_idxs, local_contexts):
            head = local_context[pair_idx[:, 0]]
            tail = local_context[pair_idx[:, 1]]
            pair_reps_list.append(torch.cat((head, tail), dim=-1))

        pair_rep = torch.cat(pair_reps_list, dim=0)
        pair_rep = torch.cat([pair_rep, union_features], dim=-1)
        rel_logits = self.relation_classifier(pair_rep)

        obj_logits = obj_logits.split(num_objs, dim=0)
        rel_logits = rel_logits.split(num_rels, dim=0)

        losses = {}
        result = (rel_logits, obj_logits)
        if self.training:
            loss_relation = relation_loss(rel_labels, rel_logits, rel_weights)
            losses = dict(loss_rel=loss_relation)
        return result, losses
