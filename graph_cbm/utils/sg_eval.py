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

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in MODES}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict, viz_dict=viz_dict,
                                 iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        print('Predicate mode: ' + self.mode)
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print(' Average Recall     (AR) @%i: %f' % (k, np.mean(v)))


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False, viz_dict=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict: 
    :param viz_dict: 
    :param kwargs: 
    :return: 
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']
    gt_images = gt_entry['gt_images']

    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']

    if mode == 'predcls':
        pred_boxes = gt_boxes
        pred_classes = gt_classes
        obj_scores = np.ones(gt_classes.shape[0])
    elif mode == 'sgcls':
        pred_boxes = gt_boxes
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'sgdet' or mode == 'phrdet':
        pred_boxes = pred_entry['pred_boxes'].astype(float)
        pred_classes = pred_entry['pred_classes']
        obj_scores = pred_entry['obj_scores']
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]

        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:, 1:])
        rel_scores_sorted[:, 1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:, 0]], rel_scores_sorted[:, 1]))

        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    else:
        raise ValueError('invalid mode')

    if multiple_preds:
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1] + 1))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1] + 1]
    else:
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))
        predicate_scores = rel_scores[:, 1:].max(1)

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(gt_rels, gt_boxes, gt_classes, pred_rels, pred_boxes,
                                                         pred_classes, predicate_scores, obj_scores,
                                                         phrdet=mode == 'phrdet', gt_images=gt_images)

    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores


def evaluate_recall(gt_rels, gt_boxes, gt_classes, pred_rels, pred_boxes, pred_classes, rel_scores=None,
                    cls_scores=None, iou_thresh=0.5, phrdet=False, **kwargs):
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes)
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:, :2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,1])
    assert np.all(pred_rels[:, 2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes,
                                                                  pred_boxes, rel_scores, cls_scores)

    scores_overall = relation_scores.prod(1)
    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:, :2],
        pred_triplets[:, [0, 2, 1]],
    ))

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(predicates, relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    # sub_ob_classes = classes[relations[:, :2]]
    sub_id, ob_id = relations[:, 0], relations[:, 1]
    triplets = np.column_stack((classes[sub_id], predicates, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id],
            class_scores[ob_id],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_boxes[gt_has_match], keeps[gt_has_match], ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
