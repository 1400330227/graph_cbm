import numpy as np
from functools import reduce

from graph_cbm.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps

MODES = ('sgdet', 'sgcls', 'predcls')
np.set_printoptions(precision=3)


class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_precision'] = {20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_mean_precision'] = {20: [], 50: [], 100: []}
        self.multiple_preds = multiple_preds
        self.recall_means = {}
        self.precision_means = {}
        self.mean_precision_means = {}

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, iou_thresh=0.5):
        res = self.evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict, iou_thresh=iou_thresh)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('Predicate mode: ' + self.mode)
        for k, v in self.result_dict[self.mode + '_recall'].items():
            mean = np.mean(v)
            self.recall_means[k] = mean
            print(f' Average Recall     (AR) @{k:<3} = {mean:<10.3f}')
        for k, v in self.result_dict[self.mode + '_precision'].items():
            mean = np.mean(v)
            self.precision_means[k] = mean
            print(f' Average Precision  (AP) @{k:<3} = {mean:<10.3f}')
        for k, v in self.result_dict[self.mode + '_mean_precision'].items():
            v_filtered = [val for val in v if val is not None]
            if not v_filtered:
                mean = 0.0
            else:
                mean = np.mean(v_filtered)
            self.mean_precision_means[k] = mean
            print(f' Mean Precision     (mP) @{k:<3} = {mean:<10.3f}')


    def evaluate_from_dict(self, gt_entry, pred_entry, mode, result_dict, **kwargs):
        gt_rels = gt_entry['gt_relations']
        gt_boxes = gt_entry['gt_boxes']
        gt_classes = gt_entry['gt_classes']
        kwargs['gt_image_id'] = gt_entry['gt_image_id']

        pred_rel_inds = pred_entry['pred_rel_inds']
        rel_scores = pred_entry['rel_scores']
        pred_classes = pred_entry['pred_classes']
        pred_boxes = pred_entry['pred_boxes']
        obj_scores = pred_entry['obj_scores']

        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        predicate_scores = rel_scores[:, 1:].max(1)

        pred_to_gt, pred_5ples, rel_scores = self.evaluate_recall(
            gt_rels,
            gt_boxes,
            gt_classes,
            pred_rels,
            pred_boxes,
            pred_classes,
            predicate_scores,
            obj_scores,
            **kwargs,
        )

        self.calculate_mP_R_at_K(gt_rels.shape[0], pred_to_gt, result_dict, mode)
        # for k in result_dict[mode + '_recall']:
        #     match = reduce(np.union1d, pred_to_gt[:k])
        #     rec_i = float(len(match)) / float(gt_rels.shape[0])
        #     result_dict[mode + '_recall'][k].append(rec_i)
        return pred_to_gt, pred_5ples, rel_scores



    def calculate_mP_R_at_K(self, num_gt_relations, pred_to_gt, result_dict, mode, k_values=(20, 50, 100)):
        """
        计算并更新 Recall@K, Precision@K, 和 Mean Precision@K。
        """
        # is_match 的长度是实际的总预测数
        is_match = np.array([len(x) > 0 for x in pred_to_gt])
        num_predictions = len(is_match)

        # 遍历不同的 K 值
        for k in k_values:

            # --- 在这里进行关键修改 (1/1) ---
            # 确定我们实际要处理的范围，取 k 和实际预测数的较小值
            effective_k = min(k, num_predictions)

            # 如果有效预测数为0，所有指标都为0
            if effective_k == 0:
                result_dict[mode + '_recall'][k].append(0.0)
                result_dict[mode + '_precision'][k].append(0.0)
                result_dict[mode + '_mean_precision'][k].append(None)  # 或者 0.0
                continue  # 继续下一个 K 值的循环
            # --- 修改结束 ---

            # --- 计算 Recall@K ---
            # 只取前 effective_k 个预测
            pred_to_gt_k = pred_to_gt[:effective_k]
            match = reduce(np.union1d, pred_to_gt_k) if pred_to_gt_k else []
            rec_i = float(len(match)) / float(num_gt_relations)
            result_dict[mode + '_recall'][k].append(rec_i)

            # --- 计算 Precision@K ---
            # 在前 k 个预测中，正确的预测有多少个
            # 注意：这里我们仍然用 k 作为分母，因为 Precision@K 的定义就是“在前K个位置中”
            num_correct_k = np.sum(is_match[:effective_k])
            prec_i = float(num_correct_k) / float(k)
            result_dict[mode + '_precision'][k].append(prec_i)

            # --- 计算 Mean Precision@K ---
            if num_correct_k == 0:
                mean_prec_i = 0.0
            else:
                # 所有的计算现在都只在 effective_k 的范围内进行
                is_match_k = is_match[:effective_k]
                cumulative_correct = np.cumsum(is_match_k)
                # 分母的长度也应该是 effective_k
                instantaneous_precision = cumulative_correct / (np.arange(effective_k) + 1)
                mean_prec_i = np.sum(instantaneous_precision * is_match_k) / num_correct_k

            # 对于没有预测的图像，它的mP是None
            if num_predictions == 0:
                result_dict[mode + '_mean_precision'][k].append(None)
            else:
                result_dict[mode + '_mean_precision'][k].append(mean_prec_i)


    def evaluate_recall(self, gt_rels, gt_boxes, gt_classes, pred_rels, pred_boxes, pred_classes, rel_scores=None,
                        cls_scores=None, iou_thresh=0.5, **kwargs):
        if pred_rels.size == 0:
            return [[]], np.zeros((0, 5)), np.zeros(0)

        num_gt_boxes = gt_boxes.shape[0]
        num_gt_relations = gt_rels.shape[0]
        assert num_gt_relations != 0

        gt_triplets, gt_triplet_boxes, _ = self._triplet(gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes)
        num_boxes = pred_boxes.shape[0]
        assert pred_rels[:, :2].max() < pred_classes.shape[0]

        assert np.all(pred_rels[:, 2] > 0)

        pred_triplets, pred_triplet_boxes, relation_scores = self._triplet(
            pred_rels[:, 2],
            pred_rels[:, :2],
            pred_classes,
            pred_boxes,
            rel_scores,
            cls_scores
        )

        scores_overall = relation_scores.prod(1)
        if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
            print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))

        pred_to_gt = self._compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thresh,
        )

        pred_5ples = np.column_stack((
            pred_rels[:, :2],
            pred_triplets,
        ))

        return pred_to_gt, pred_5ples, relation_scores


    def _triplet(self, predicates, relations, classes, boxes, predicate_scores=None, class_scores=None):
        assert (predicates.shape[0] == relations.shape[0])

        sub_id, ob_id = relations[:, 0], relations[:, 1]
        triplets = np.column_stack((classes[sub_id], predicates, classes[ob_id]))
        triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

        triplet_scores = None
        if predicate_scores is not None and class_scores is not None:
            triplet_scores = np.column_stack((
                class_scores[sub_id],
                predicate_scores,
                class_scores[ob_id],
            ))

        return triplets, triplet_boxes, triplet_scores


    def _compute_pred_matches(self, gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False):
        keeps = intersect_2d(gt_triplets, pred_triplets)
        gt_has_match = keeps.any(1)
        pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
        for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match]):
            boxes = pred_boxes[keep_inds]
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

            for i in np.where(keep_inds)[0][inds]:
                pred_to_gt[i].append(int(gt_ind))
        return pred_to_gt
