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
        self.multiple_preds = multiple_preds
        self.recall_means = {}

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_entry, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict, iou_thresh=iou_thresh)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('Predicate mode: ' + self.mode)
        for k, v in self.result_dict[self.mode + '_recall'].items():
            mean = np.mean(v)
            self.recall_means[k] = mean
            print(' Average Recall     (AR) @%i: %f' % (k, mean))


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, **kwargs):
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

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
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

    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores


def evaluate_recall(gt_rels, gt_boxes, gt_classes, pred_rels, pred_boxes, pred_classes, rel_scores=None,
                    cls_scores=None, iou_thresh=0.5, **kwargs):
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:, :2].max() < pred_classes.shape[0]

    assert np.all(pred_rels[:, 2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(
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

    pred_to_gt = _compute_pred_matches(
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


def _triplet(predicates, relations, classes, boxes, predicate_scores=None, class_scores=None):
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


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False):
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
