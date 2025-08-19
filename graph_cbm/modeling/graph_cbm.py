import torch
import numpy.random as npr
import torch.nn.functional as F
import torch.nn
from torchvision.ops import boxes as box_ops
from torch import nn

from graph_cbm.modeling.detection.backbone import build_vgg_backbone, build_resnet50_backbone, build_mobilenet_backbone, \
    build_efficientnet_backbone
from graph_cbm.modeling.detection.detector import FasterRCNN
from graph_cbm.modeling.detection.transform import resize_boxes
from graph_cbm.modeling.relation.predictor import Predictor
from graph_cbm.utils.boxes import box_union
from torch_geometric.nn import RGCNConv


class RGCNConvLayer(torch.nn.Module):
    def __init__(
            self,
            num_relations,
            input_dim=1280,
            hidden_dim=512,
            out_dim=256,
            num_layers=2,
            dropout=0.5,
            num_bases=None,
            pooling='mean'
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.lin_obj = nn.Linear(input_dim, 512)
        self.convs.append(RGCNConv(512, hidden_dim, num_relations, num_bases))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations, num_bases))
        self.dropout = dropout
        self.pooling = pooling

    def forward(self, x, edge_index, edge_type, num_nodes):
        x = self.lin_obj(x)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        x = self.graph_pooling(x, num_nodes)
        return x

    def graph_pooling(self, x, num_nodes):
        node_splits = torch.split(x, num_nodes, dim=0)
        pooled = []
        for nodes in node_splits:
            if self.pooling == 'mean':
                pooled.append(torch.mean(nodes, dim=0))
            elif self.pooling == 'sum':
                pooled.append(torch.sum(nodes, dim=0))
            elif self.pooling == 'max':
                pooled.append(torch.max(nodes, dim=0)[0])
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
        return torch.stack(pooled, dim=0)


def task_loss(y_logits, y, n_tasks, task_class_weights=None):
    loss_task = (torch.nn.CrossEntropyLoss(weight=task_class_weights)
                 if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(weight=task_class_weights))
    return loss_task(y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1), y)


class C2yModel(nn.Module):
    def __init__(self, num_classes, relation_classes, n_tasks, embedding_dim):
        super(C2yModel, self).__init__()
        self.num_classes = num_classes
        self.relation_classes = relation_classes
        self.n_tasks = n_tasks
        self.features = RGCNConvLayer(relation_classes)
        self.classifier = nn.Sequential(*[
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_tasks)
        ])

        self.obj_embed = nn.Embedding(num_classes, embedding_dim)

    def forward(self, box_features, relation_graphs, edge_index, edge_type, num_nodes, targets):
        num_nodes = [relation["pred_labels"].shape[0] for relation in relation_graphs]
        pred_labels = torch.concat([relation["pred_labels"] for relation in relation_graphs], dim=0)
        x = torch.concat((box_features, self.obj_embed(pred_labels.long())), dim=-1)
        edge_index = torch.concat([relation["rel_pair_idx"] for relation in relation_graphs], dim=0).permute(1, 0)
        edge_type = torch.concat([relation["rel_labels"] for relation in relation_graphs], dim=0)

        x = self.features(x, edge_index, edge_type, num_nodes)
        y_logits = self.classifier(x)
        y_logits = y_logits.split(num_nodes, dim=0)
        result = self.post_processor(y_logits, relation_graphs)
        loss = {}
        if self.training:
            y = torch.as_tensor([target['class_label'] for target in targets], dtype=torch.int64,
                                device=y_logits.device)
            loss_task = task_loss(x, y, self.n_tasks)
            loss['loss_task'] = loss_task
        return result, loss

    def post_processor(self, y_logits, relation_graphs):
        result = []
        for i, (y_logit, relation_graph) in enumerate(zip(y_logits, relation_graphs)):
            y_prob = F.softmax(y_logit, -1)
            relation_graph["y_prob"] = y_prob
            relation_graph["y_logit"] = y_logit
            result.append(relation_graph)
        return result


class GraphCBM(nn.Module):
    def __init__(
            self,
            detector: FasterRCNN,
            predictor: Predictor,
            num_classes: int,
            relation_classes: int,
            n_tasks: int,
            use_c2ymodel=False,
            batch_size_per_image=1024,
            positive_fraction=0.25,
            num_sample_per_gt_rel=4,
            fg_thres=0.5,
    ):
        super().__init__()
        self.box_roi_pool = detector.roi_heads.box_roi_pool
        # self.box_head = detector.roi_heads.box_head
        self.detector = detector
        self.predictor = predictor
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.fg_thres = fg_thres
        self.num_pos_per_img = int(batch_size_per_image * positive_fraction)
        self.num_sample_per_gt_rel = num_sample_per_gt_rel

        self.obj_embed = nn.Embedding(num_classes, self.predictor.embedding_dim)

        self.use_c2ymodel = use_c2ymodel

        resolution = self.box_roi_pool.output_size[0]
        in_channels = detector.backbone.out_channels
        self.rect_size = resolution * 4 - 1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

        self.union_fusion_layer = nn.Linear(in_channels * 2, in_channels)

        if self.use_c2ymodel:
            self.c2y_model = C2yModel(num_classes, relation_classes, n_tasks, self.predictor.embedding_dim)

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility):
        """
        prepare to sample fg relation triplet and bg relation triplet
        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]
        """
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                # binary rel only consider related or not, so its symmetric
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            # find matching pair in proposals (might be more than one)
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0:
                continue
            # all combination pairs
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            # construct corresponding proposal triplets corresponding to i_th gt relation
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device).view(-1,
                                                                                                                    1)
            fg_rel_i = torch.concat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(
                torch.int64)
            # select if too many corresponding proposal pairs to one pair of gt relationship triplet
            # NOTE that in original motif, the selection is based on a ious_score score
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel:
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

        # select fg relations
        if len(fg_rel_triplets) == 0:
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = torch.concat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img:
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        # select bg relations
        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = torch.concat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        # if both fg and bg is none
        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=device)

        return torch.concat((fg_rel_triplets, bg_rel_triplets), dim=0), binary_rel

    def select_training_samples(self, proposals, targets):
        device = proposals[0]["boxes"].device
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            prp_box = proposal["boxes"]
            prp_lab = proposal["labels"].long()
            tgt_box = target["boxes"]
            tgt_lab = target["labels"].long()
            tgt_rel_matrix = target["relation"]  # [tgt, tgt]
            # IoU matching
            # ious = boxlist_iou(target, proposal)  # [tgt, prp]
            ious = box_ops.box_iou(tgt_box, prp_box)
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)  # [tgt, prp]
            # Proposal self IoU to filter non-overlap
            prp_self_iou = box_ops.box_iou(prp_box, prp_box)  # [prp, prp]

            num_prp = prp_box.shape[0]
            rel_possibility = (torch.ones((num_prp, num_prp), device=device).long() -
                               torch.eye(num_prp, device=device).long())
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, binary_rel = self.motif_rel_fg_bg_sampling(
                device, tgt_rel_matrix, ious, is_match, rel_possibility)
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def gtbox_relsample(self, proposals, targets):
        device = proposals[0]["boxes"].device
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)

        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            prp_box = proposal["boxes"]
            tgt_box = target["boxes"]
            tgt_rel_matrix = target["relation"]
            prp_lab = proposal["labels"].long()

            iou_matrix = box_ops.box_iou(tgt_box, prp_box)
            is_match = iou_matrix.argmax(dim=1)
            max_iou = iou_matrix.max(dim=1).values
            is_match[max_iou < self.fg_thres] = -1
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

            num_tgt_rels = tgt_rel_labs.shape[0]
            num_prp = prp_box.shape[0]

            binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
            binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp head)
            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()

            rel_possibility = (torch.ones((num_prp, num_prp), device=device) - torch.eye(num_prp, device=device)).long()
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            rel_triplets = []
            fg_rel_triplets = []
            for i in range(num_tgt_rels):
                bi_match_head = binary_prp_head[i]
                bi_match_tail = binary_prp_tail[i]

                prp_head_idx = bi_match_head.squeeze()
                prp_tail_idx = bi_match_tail.squeeze()
                tgt_head_idx = tgt_head_idxs[i]
                tgt_tail_idx = tgt_tail_idxs[i]
                tgt_rel_lab = tgt_rel_labs[i]

                num_bi_head = (bi_match_head >= 0).sum()
                num_bi_tail = (bi_match_tail >= 0).sum()

                if num_bi_head <= 0 or num_bi_tail <= 0:
                    continue
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                binary_rel[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

                rel_possibility[prp_head_idx, prp_tail_idx] = 0
                rel_i = torch.tensor([[tgt_head_idx, tgt_tail_idx, tgt_rel_lab]], device=device, dtype=torch.int64)
                fg_rel_i = torch.tensor([[prp_head_idx, prp_tail_idx, tgt_rel_lab]], device=device, dtype=torch.int64)
                rel_triplets.append(rel_i)
                fg_rel_triplets.append(fg_rel_i)

            if len(fg_rel_triplets) == 0:
                fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
            else:
                fg_rel_triplets = torch.concat(fg_rel_triplets, dim=0).to(torch.int64)
                if fg_rel_triplets.shape[0] > num_pos_per_img:
                    perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:num_pos_per_img]
                    fg_rel_triplets = fg_rel_triplets[perm]
            bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
            bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
            bg_rel_triplets = torch.concat((bg_rel_inds, bg_rel_labs.view(-1, 1),), dim=-1).to(torch.int64)

            num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
            if bg_rel_triplets.shape[0] > 0:
                perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
                bg_rel_triplets = bg_rel_triplets[perm]
            else:
                bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

            img_rel_triplets = torch.concat((fg_rel_triplets, bg_rel_triplets), dim=0)
            rel_idx_pairs.append(img_rel_triplets[:, :2])  # (num_rel, 2),  (sub_idx, obj_idx)
            rel_labels.append(img_rel_triplets[:, 2])  # (num_rel, )
            rel_sym_binarys.append(binary_rel)
        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    # def gtbox_relsample(self, proposals, targets):
    #     device = proposals[0]["boxes"].device
    #     num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
    #     rel_idx_pairs = []
    #     rel_labels = []
    #     rel_sym_binarys = []
    #     for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
    #         prp_box = proposal["boxes"]
    #         tgt_box = target["boxes"]
    #         num_prp = prp_box.shape[0]
    #
    #         assert prp_box.shape[0] == tgt_box.shape[0]
    #         tgt_rel_matrix = target["relation"]
    #         tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
    #         assert tgt_pair_idxs.shape[1] == 2
    #         tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
    #         tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
    #         tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
    #
    #         # sym_binary_rels
    #         binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
    #         binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
    #         binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
    #         rel_sym_binarys.append(binary_rel)
    #
    #         rel_possibility = (torch.ones((num_prp, num_prp), device=device).long() -
    #                            torch.eye(num_prp, device=device).long())
    #         rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
    #         tgt_bg_idxs = torch.nonzero(rel_possibility > 0)
    #
    #         if tgt_pair_idxs.shape[0] > num_pos_per_img:
    #             perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
    #             tgt_pair_idxs = tgt_pair_idxs[perm]
    #             tgt_rel_labs = tgt_rel_labs[perm]
    #         num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)
    #         num_bg = self.batch_size_per_image - num_fg
    #         perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
    #         tgt_bg_idxs = tgt_bg_idxs[perm]
    #
    #         img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
    #         img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long()),
    #                                    dim=0).contiguous().view(-1)
    #         rel_idx_pairs.append(img_rel_idxs)
    #         rel_labels.append(img_rel_labels)
    #     return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def select_test_pairs(self, proposals):
        device = proposals[0]["boxes"].device
        rel_pair_idxs = []
        for p in proposals:
            n = p["boxes"].shape[0]
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    def union_feature_extractor(self, images, features, proposals, rel_pair_idxs):
        device = features["0"].device
        image_sizes = images.image_sizes
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx, sizes in zip(proposals, rel_pair_idxs, image_sizes):
            head_proposal = proposal["boxes"][rel_pair_idx[:, 0]]
            tail_proposal = proposal["boxes"][rel_pair_idx[:, 1]]
            union_proposal = box_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            num_rel = len(rel_pair_idx)

            dummy_x_range = (torch.arange(self.rect_size, device=device).view(1, 1, -1)
                             .expand(num_rel, self.rect_size, self.rect_size))
            dummy_y_range = (torch.arange(self.rect_size, device=device).view(1, -1, 1)
                             .expand(num_rel, self.rect_size, self.rect_size))

            head_proposal = resize_boxes(head_proposal, sizes, [self.rect_size, self.rect_size])
            tail_proposal = resize_boxes(tail_proposal, sizes, [self.rect_size, self.rect_size])

            head_rect = ((dummy_x_range >= head_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= head_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= head_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= head_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal[:, 0].floor().view(-1, 1, 1).long()) &
                         (dummy_x_range <= tail_proposal[:, 2].ceil().view(-1, 1, 1).long()) &
                         (dummy_y_range >= tail_proposal[:, 1].floor().view(-1, 1, 1).long()) &
                         (dummy_y_range <= tail_proposal[:, 3].ceil().view(-1, 1, 1).long())).float()

            rect_input = torch.stack((head_rect, tail_rect), dim=1)
            rect_inputs.append(rect_input)

        rect_inputs = torch.cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        union_vis_features = self.box_roi_pool(features, union_proposals, images.image_sizes)
        union_features = union_vis_features + rect_features
        # union_features = self.box_head(union_features)

        return union_features

    def forward(self, images, targets=None, class_weights=None):
        proposals, features, images, targets = self.detector(images, targets)
        if self.training:
            with torch.no_grad():
                proposals, rel_labels, rel_pair_idxs, rel_binarys = self.select_training_samples(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.select_test_pairs(proposals)

        roi_features = self.box_roi_pool(features, [t["boxes"] for t in proposals], images.image_sizes)
        union_features = self.union_feature_extractor(images, features, proposals, rel_pair_idxs)
        box_features, relation_graphs, predictor_losses = self.predictor(
            roi_features,
            proposals,
            rel_pair_idxs,
            union_features,
            rel_labels,
            class_weights
        )
        result = relation_graphs
        losses = {}
        losses.update(predictor_losses)
        if self.use_c2ymodel:
            cbm_logits, loss_task = self.c2y_model(box_features, relation_graphs, targets)
            losses.update(loss_task)
            result = cbm_logits
        if self.training:
            return losses
        return result


def build_Graph_CBM(
        backbone_name,
        num_classes,  # 目标检测的数量
        relation_classes,  # 关系的数量
        n_tasks,  # 分类的数量
        detector_weights_path="",
        weights_path="",
        use_c2ymodel=False
):
    if backbone_name == 'resnet50':
        backbone = build_resnet50_backbone(pretrained=False)
    elif backbone_name == 'mobilenet':
        backbone = build_mobilenet_backbone(pretrained=False)
    elif backbone_name == 'efficientnet':
        backbone = build_efficientnet_backbone(pretrained=False)
    elif backbone_name == 'squeezenet':
        backbone = build_vgg_backbone(pretrained=False)
    else:
        backbone = build_resnet50_backbone(pretrained=False)

    detector = FasterRCNN(backbone=backbone, num_classes=num_classes, use_relation=True)

    out_channels = backbone.out_channels
    resolution = detector.roi_heads.box_roi_pool.output_size[0]
    feature_extractor_dim = out_channels * resolution ** 2

    if detector_weights_path != "":
        detector_weights = torch.load(detector_weights_path, map_location='cpu', weights_only=True)
        detector_weights = detector_weights['model'] if 'model' in detector_weights else detector_weights
        detector.load_state_dict(detector_weights)

    predictor = Predictor(num_classes, relation_classes, feature_extractor_dim)

    model = GraphCBM(detector, predictor, num_classes, relation_classes, n_tasks, use_c2ymodel)
    if weights_path != "":
        weights_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
        model.load_state_dict(weights_dict)

    return model
