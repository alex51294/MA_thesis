import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split

from detection.datasets import LazyDDSMDataset, CacheINbreastDataset, \
    LazyINbreastDataset
import detection.datasets.ddsm.ddsm_utils as ddsm_utils
import detection.datasets.inbreast.inbreast_utils as inbreast_utils
import detection.datasets.utils as utils
from detection.retinanet import RetinaNet, Anchors
from detection.retinanet.retinanet_utils import box_iou, change_box_order, box_nms

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, \
    confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import signature
from scipy.stats import norm
from sklearn.utils.multiclass import unique_labels

def calc_tp_fn_fp(gt_bbox, crop_bbox, score_bbox, iou_thr = 0.2,
                  confidence_values = np.arange(0.5, 1, 0.05)):
    # bboxes and score must be tensors
    if isinstance(crop_bbox, list):
        crop_bbox = torch.stack(crop_bbox)

    if isinstance(score_bbox, list):
        score_bbox = torch.stack(score_bbox)

    tp_list = []
    fp_list = []
    fn_list = []
    iou_list = []
    gt_bbox = change_box_order(gt_bbox, order='xywh2xyxy')

    for j in confidence_values:
        current_bbox = crop_bbox[score_bbox > j]

        if len(current_bbox) == 0:
            tp_list.append(torch.tensor(0, device = current_bbox.device))
            fp_list.append(torch.tensor(0, device = current_bbox.device))
            fn_list.append(torch.tensor(gt_bbox.shape[0],
                                        device = current_bbox.device))
            continue
            # break

        iou_matrix = box_iou(gt_bbox,
                             change_box_order(current_bbox,
                                              order="xywh2xyxy"))
        hits = iou_matrix > iou_thr
        iou_values = iou_matrix[hits]
        iou_list.append(iou_values)

        # true positives are the lesions that are recognized
        # count only one detected box per lesion as positive
        tp = torch.clamp(torch.sum(hits, 1), 0, 1).sum()
        tp_list.append(tp)

        # false negatives are the lesions that are NOT recognized
        fn = gt_bbox.shape[0] - tp
        fn_list.append(fn)

        # number of false positives
        fp = (current_bbox.shape[0] - tp).type(torch.float32)
        fp_list.append(fp)

    return tp_list, fp_list, fn_list

# not finished yet
def calc_lrp(labels_list, scores_list, iou_hits_list, iou_thr = 0.2,
             confidence_values=np.arange(0.5, 1, 0.05)):
    label = [labels_list[i][j].cpu() for i in
                 range(len(labels_list)) for j
                 in range(len(labels_list[i]))]
    label = np.asarray(label)

    score = [scores_list[i][j].cpu() for i in
                 range(len(scores_list))
                 for j in range(len(scores_list[i]))]
    score = np.asarray(score)

    # calculate the Average Precision (AP_20) score
    ap = average_precision_score(label, score)

    return ap

def calc_detection_hits(gt_bbox, crop_bbox, score_bbox, iou_thr = 0.2, score_thr=0.):
    # bboxes and score must be tensors
    if isinstance(crop_bbox, list):
        crop_bbox = torch.stack(crop_bbox)

    if isinstance(score_bbox, list):
        score_bbox = torch.stack(score_bbox)

    # sort according to score and limit according to score_thr
    scores, indices = torch.sort(score_bbox, descending=True)
    boxes = crop_bbox[indices]

    boxes = boxes[scores > score_thr]
    scores = scores[scores > score_thr]

    # calculate iou
    iou_matrix = box_iou(change_box_order(gt_bbox, order='xywh2xyxy'),
                         change_box_order(boxes, order="xywh2xyxy"))

    # create label list, denoting at max one detection per lesion
    labels = torch.zeros_like(scores)
    iou_hits = torch.zeros_like(scores)
    for i in range(iou_matrix.shape[0]):
        hits = (iou_matrix[i] > iou_thr).nonzero()
        if len(hits) > 0:
            labels[hits[0].cpu()] = 1.
            iou_hits[hits[0].cpu()] = iou_matrix[i, hits[0].cpu()]

    return labels, scores, iou_hits

def classification(image_class_list, label_type="birads"):
    if isinstance(image_class_list, list):
        image_class_list = np.asarray(image_class_list)

    gt_labels = image_class_list[:, 0]
    pred_labels = image_class_list[:, 1]
    scores = image_class_list[:, 2]

    classes = unique_labels(gt_labels)
    subset_list = []
    for i in range(len(classes)):
        subset = gt_labels[gt_labels == classes[i]]
        subset_list.append(subset)

    if label_type == "birads":
        gt_labels = np.int32(gt_labels > 3)

    fpr, tpr, _ = roc_curve(gt_labels, scores)
    auroc = roc_auc_score(gt_labels, scores)

    return fpr, tpr, auroc

def conf_matrix(image_class_list, label_type="birads"):
    if isinstance(image_class_list, list):
        image_class_list = np.asarray(image_class_list)

    classes = unique_labels(image_class_list[:, 0])

    cm = confusion_matrix(image_class_list[:, 0],
                          image_class_list[:, 1],
                          labels=classes)

    return cm, classes


def calc_pr_values(labels_list, scores_list, number_gt,
            confidence_values=np.arange(0.05, 1, 0.01)):
    labels = torch.tensor([labels_list[i][j] for i in
                           range(len(labels_list)) for j
                           in range(len(labels_list[i]))])

    scores = torch.tensor([scores_list[i][j] for i in
                           range(len(scores_list))
                           for j in range(len(scores_list[i]))])

    precision_list = []
    recall_list = []

    for thr in confidence_values:
        current_labels = labels[scores > thr]
        tp = torch.sum(current_labels)
        fp = torch.tensor(len(current_labels)) - tp

        precision = tp / (tp + fp)
        precision_list.append(precision)

        recall = tp / torch.tensor(number_gt)
        recall_list.append(recall)

    precision_list = np.asarray(torch.tensor(precision_list))
    recall_list = np.asarray(torch.tensor(recall_list))

    return precision_list, recall_list

def calc_ap_MDT(labels_list, scores_list, all_p):
    """
        adapted from: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
        :param labels_list: dataframe containing class labels of predictions sorted in
                    descending manner by their prediction score.
        :param all_p: number of all ground truth objects.
                (for denominator of recall.)
        :return:

    """
    labels = torch.tensor([labels_list[i][j] for i in
                 range(len(labels_list)) for j
                 in range(len(labels_list[i]))])


    scores = torch.tensor([scores_list[i][j] for i in
                 range(len(scores_list))
                 for j in range(len(scores_list[i]))])

    # sort according to score
    scores, indices = torch.sort(scores, descending=True)
    labels = labels[indices]

    # convert to numpy
    labels = np.asarray(labels.cpu())
    scores = np.asarray(scores.cpu())


    tp = labels
    fp = (tp == 0) * 1
    # recall thresholds, where precision will be measured
    R = np.linspace(.0, 1, 101, endpoint=True)
    tp_sum = np.cumsum(tp)
    fp_sum = np.cumsum(fp)
    nd = len(tp)
    rc = tp_sum / all_p
    pr = tp_sum / (fp_sum + tp_sum)
    # initialize precision array over recall steps.
    q = np.zeros((len(R),))

    # numpy is slow without cython optimization for accessing elements
    # use python array gets significant speed improvement
    pr = pr.tolist()
    q = q.tolist()
    for i in range(nd - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]

    # discretize empiric recall steps with given bins.
    inds = np.searchsorted(rc, R, side='left')
    try:
        for ri, pi in enumerate(inds):
            q[ri] = pr[pi]
    except:
        pass

    return np.mean(q), np.asarray(q)

def calc_f1(tp_list, fp_list, fn_list):
    tp = np.sum(np.asarray(tp_list, dtype=np.float32), axis=0)
    fn = np.sum(np.asarray(fn_list, dtype=np.float32), axis=0)
    fp = np.sum(np.asarray(fp_list, dtype=np.float32), axis=0)

    f1 = 2 * tp / (2 * tp + fn + fp)

    return f1

def calc_f_beta(tp_list, fp_list, fn_list, beta=1):
    tp = np.sum(np.asarray(tp_list, dtype=np.float32), axis=0)
    fn = np.sum(np.asarray(fn_list, dtype=np.float32), axis=0)
    fp = np.sum(np.asarray(fp_list, dtype=np.float32), axis=0)

    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)

    f_beta = (1 + beta*beta) * tp / ((1 + beta*beta) * tp + beta*beta*fn + fp)

    return f_beta


def calc_froc(tp_list, fp_list, fn_list):
    # calculate FROC over all test images for one model if desired
    tp = np.sum(np.asarray(tp_list, dtype=np.float32), axis=0)
    fn = np.sum(np.asarray(fn_list, dtype=np.float32), axis=0)
    fp = np.sum(np.asarray(fp_list, dtype=np.float32), axis=0)

    tpr = tp / (tp + fn)
    fppi = fp / np.float32(len(fp_list))

    return tpr, fppi

def gt_overlap(image_bbox, crop_bbox, iou_thr = 0.2):
    # convert bbox format from [x, y, w, h] to [x_1, y_1, x_2, y_2] for IoU
    # calculation
    image_bbox = change_box_order(image_bbox, order='xywh2xyxy')
    crop_bbox = change_box_order(crop_bbox, order="xywh2xyxy")

    # determine overlap with ground truth bboxes
    iou_matrix = box_iou(image_bbox, crop_bbox)
    iou_matrix = iou_matrix > iou_thr

    # assign each detected bbox a label: 1 if its overlap with the ground
    # truth is higher than the given threshold, 0 otherwise
    box_class = torch.clamp(torch.sum(iou_matrix, 0), 0, 1)

    return box_class

def nms(bboxes, scores, labels=None, thr=0.2):
    # merge overlapping bounding boxes

    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        bboxes = torch.stack(bboxes)

    if isinstance(scores, list):
        scores = torch.stack(scores)

    # sort the score in descending order, adjust bboxes accordingly
    scores, indices = torch.sort(scores, descending=True)
    bboxes = bboxes[indices]
    if labels is not None:
        if isinstance(labels, list):
            labels = torch.tensor(labels)

        labels = labels[indices]

    # change the order from xywh to xyxy
    bboxes = change_box_order(bboxes, order="xywh2xyxy")

    # limit the amount of bboxes to 300 (or less)
    if len(scores) > 300:
        limit = 300
    else:
        limit = len(scores)
    scores = scores[0:limit]
    bboxes = bboxes[0:limit]
    if labels is not None:
        labels = labels[0:limit]

    # perform an image-wise NMS
    keep_ids = box_nms(bboxes, scores, threshold=thr)
    bboxes = bboxes[keep_ids]
    scores = scores[keep_ids]
    if labels is not None:
        labels = labels[keep_ids]
    else:
        labels = []

    # filter out overlapping bboxes
    #bboxes, scores, labels = rm_overlapping_boxes(bboxes, scores, labels=labels)

    # convert in xywh form again
    bboxes = change_box_order(bboxes, order="xyxy2xywh")

    return bboxes, scores, labels

def rm_overlapping_boxes(bboxes, scores, labels=None, order="xyxy"):
    if order == "xywh":
        bboxes = change_box_order(bboxes, order="xywh2xyxy")

    scores, indices = torch.sort(scores, descending=True)
    bboxes = bboxes[indices]
    if labels is not None:
        labels = labels[indices]
        result_labels = [labels[0]]

    # filter out overlapping bboxes
    result_bboxes = [bboxes[0]]
    result_scores = [scores[0]]

    for i in range(1, len(bboxes)):
        ignore = False
        for j in range(len(result_bboxes)):
            # if i == j:
            #     continue

            if box_overlap(bboxes[i], result_bboxes[j]):
                ignore = True
                break
        if not ignore:
            result_bboxes.append(bboxes[i])
            result_scores.append(scores[i])
            if labels is not None:
                result_labels.append(labels[i])

    result_bboxes = torch.stack(result_bboxes)
    result_scores = torch.stack(result_scores)

    if order == "xywh":
        result_bboxes = change_box_order(result_bboxes, order="xyxy2xywh")

    if labels is not None:
        return result_bboxes, result_scores, torch.stack(result_labels)
    else:
        return result_bboxes, result_scores, None


def box_overlap(box_1, box_2, overlap_thr=0.75):
    # check whether box_2 overlaps with box_1 with at least overlap_thr percent
    flag = False
    lt = torch.max(box_1[:2], box_2[:2])
    rb = torch.min(box_1[2:], box_2[2:])

    wh = (rb - lt + 1).clamp(min=0)
    overlap_area = wh[0] * wh[1]
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])
    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    overlap_box_2 = overlap_area / box_2_area
    overlap_box_1 = overlap_area / box_1_area
    if overlap_box_1 > overlap_thr or overlap_box_2 >= 1:
        flag = True

    return flag

def wbc(bboxes, scores, thr=0.2):
    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        device = bboxes[0].device
        bboxes = torch.stack(bboxes).to("cpu")
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")

    if isinstance(scores, list):
        scores = torch.stack(scores).to("cpu")
    else:
        scores = scores.to("cpu")

    # change the order from xywh to xyxy
    bboxes = np.asarray(change_box_order(bboxes, order="xywh2xyxy"))
    scores = np.asarray(scores)

    # perform (my version) of weighted box clustering
    wbc_bboxes, wbc_scores = weighted_box_clustering(bboxes, scores, thr=thr)

    # # filter out overlapping boxes
    # wbc_bboxes, wbc_scores = rm_overlapping_boxes(torch.tensor(wbc_bboxes, dtype=torch.float32),
    #                                               torch.tensor(wbc_scores, dtype=torch.float32))

    # convert the boxes in the "xywh" form again
    wbc_bboxes = change_box_order(torch.tensor(wbc_bboxes, dtype=torch.float32),
                                  order="xyxy2xywh").to(device)

    wbc_scores = torch.tensor(wbc_scores, dtype=torch.float32).to(device)

    return wbc_bboxes, wbc_scores

def weighted_box_clustering(boxes, scores, image_size=[600,600], thr=0.2):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # calculate the area of each box
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    # # determine the center of each box and its distance to the pacht center
    # box_centers = np.asarray([y2-y1, x2-x1]).transpose()
    # patch_center = np.ones(box_centers.shape)*np.asarray(image_size)/2.0
    # dist = np.linalg.norm(patch_center-box_centers, ord=2, axis=1)
    #
    # # calculate the down-weighting factor based on the distance from center
    # patch_center_factor = norm.pdf(dist, loc=0, scale=50) * np.sqrt(2*np.pi) * 50

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]

    keep = []
    keep_scores = []
    keep_bboxes = []

    while order.size > 0:
        i = order[0]  # higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # overall between currently highest scoring box and all boxes.
        iou = inter / (areas[i] + areas[order] - inter)

        # get all the predictions that match the current box to build one cluster.
        #criterium = (iou > thr) | (inter / areas[order] > 0.5)
        criterium = (iou > thr)
        matches = np.argwhere(criterium)

        match_iou = iou[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]
        #match_pcf = patch_center_factor[order[matches]]

        # weight all scores in cluster by patch factors, and size.
        match_score_weights = match_iou * match_areas #* match_pcf
        match_scores *= match_score_weights

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / np.sum(match_score_weights)

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores),
                      np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores)]

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.05:
            keep_scores.append(avg_score)
            keep_bboxes.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(iou <= thr)[0]
        order = order[inds]

    return keep_bboxes, keep_scores

def my_merging(bboxes, scores, crop_center_factor, heatmap_factor, thr=0.2):
    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        device = bboxes[0].device
        bboxes = torch.stack(bboxes).to("cpu")
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")

    if isinstance(scores, list):
        scores = np.asarray(torch.stack(scores).to("cpu"))

    if isinstance(crop_center_factor, list):
        crop_center_factor = np.asarray(crop_center_factor)

    if isinstance(heatmap_factor, list):
        heatmap_factor = np.asarray(heatmap_factor)

    # change the order from xywh to xyxy
    bboxes = np.asarray(change_box_order(bboxes, order="xywh2xyxy"))

    # order is the sorted index.  maps order to index o[1] = 24 (rank1, ix 24)
    order = scores.argsort()[::-1]

    # limit the amount of bboxes to 300 (or less)
    if len(order) > 300:
        limit = 300
    else:
        limit = len(order)
    order = order[:limit]
    bboxes = bboxes[order]
    scores = scores[order]
    crop_center_factor = crop_center_factor[order]
    heatmap_factor = heatmap_factor[order]
    order = scores.argsort()[::-1]

    # define list for bboxes to keep
    keep_scores = []
    keep_bboxes = []

    # seperate coordinates
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    # calculate the area of each box
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    while order.size > 0:
        i = order[0]  # higehst scoring element
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # overall between currently highest scoring box and all boxes.
        iou = inter / (areas[i] + areas[order] - inter)

        # get all the predictions that match the current box to build one cluster.
        #criterium = (iou > thr) | (inter / areas[order] > 0.5)
        criterium = (iou > thr)
        matches = np.argwhere(criterium)

        match_iou = iou[matches]
        match_areas = areas[order[matches]]
        match_scores = scores[order[matches]]
        match_ccf = crop_center_factor[order[matches]]
        match_hf = heatmap_factor[order[matches]]
        #print(match_scores)

        # weight all scores in cluster by patch factors, and size.
        match_score_weights =  match_iou * match_ccf # / match_hf # * match_areas
        #match_score_weights =  np.ones_like(match_scores)
        #match_score_weights = match_iou
        match_scores *= match_score_weights

        # compute weighted average score for the cluster
        avg_score = np.sum(match_scores) / np.sum(match_score_weights)
        #avg_score = np.sum(match_scores) / len(match_scores)

        # compute weighted average of coordinates for the cluster. now only take existing
        # predictions into account.
        avg_coords = [
            np.sum(x1[order[matches]] * match_scores) / np.sum(match_scores),
            np.sum(y1[order[matches]] * match_scores) / np.sum(match_scores),
            np.sum(x2[order[matches]] * match_scores) / np.sum(match_scores),
            np.sum(y2[order[matches]] * match_scores) / np.sum(match_scores)]

        # some clusters might have very low scores due to high amounts of missing predictions.
        # filter out the with a conservative threshold, to speed up evaluation.
        if avg_score > 0.05:
            keep_scores.append(avg_score)
            keep_bboxes.append(avg_coords)

        # get index of all elements that were not matched and discard all others.
        inds = np.where(~criterium)[0]
        order = order[inds]

    # convert the boxes in the "xywh" form and tensors again
    keep_bboxes = change_box_order(torch.tensor(keep_bboxes,
                                                dtype=torch.float32),
                                   order="xyxy2xywh").to(device)

    keep_scores = torch.tensor(keep_scores, dtype=torch.float32).to(device)

    return keep_bboxes, keep_scores

def my_merging_2(bboxes, scores, crop_center_factor, heatmap_factor, thr=0.2):
    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        device = bboxes[0].device
        bboxes = torch.stack(bboxes).to("cpu")
    else:
        device = bboxes.device
        bboxes = bboxes.to("cpu")

    if isinstance(scores, list):
        scores = torch.stack(scores).to("cpu")

    # sort the score in descending order, adjust bboxes accordingly
    scores, indices = torch.sort(scores, descending=True)
    bboxes = bboxes[indices]

    # change the order from xywh to xyxy
    bboxes = change_box_order(bboxes, order="xywh2xyxy")

    box_set_list = []
    for i in range(len(bboxes)):
        if len(box_set_list) == 0:
            box_set_list.append([[bboxes[i], scores[i]]])
        else:
            isAppended = False
            for box_set in box_set_list:
                for saved_box in box_set:
                    iou = box_iou(saved_box[0].view(1, -1),
                                  bboxes[i].view(1, -1),
                                  "xyxy")
                    isCovered = isInside(saved_box[0],
                                         bboxes[i])
                    if iou > thr or isCovered:
                        # if iou > 0.2:
                        box_set.append([bboxes[i], scores[i]])
                        isAppended = True
                        break

                if isAppended == True:
                    break

            if isAppended == False:
                box_set_list.append([[bboxes[i], scores[i]]])

    merged_box_list = []
    merged_score_list = []
    for i in range(len(box_set_list)):
        pos = [box_set_list[i][j][0] for j in range(len(box_set_list[i]))]
        scores = [box_set_list[i][j][1] for j in range(len(box_set_list[i]))]

        pos = torch.stack(pos)
        scores = torch.stack(scores)

        xx1 = torch.max(pos[0, 0], pos[:,0])
        yy1 = torch.max(pos[0, 1], pos[:,1])
        xx2 = torch.min(pos[0, 2], pos[:,2])
        yy2 = torch.min(pos[0, 3], pos[:,3])

        w = torch.max(torch.Tensor([0]), xx2 - xx1 + 1)
        h = torch.max(torch.Tensor([0]), yy2 - yy1 + 1)
        inter = w * h

        areas = (pos[:,2] - pos[:,0] + 1) * (pos[:,3] - pos[:,1] + 1)

        # overall between currently highest scoring box and all boxes.
        iou = inter / (areas[0] + areas - inter)

        match_weigths = iou
        match_scores = scores * match_weigths

        avg_score = torch.sum(match_scores) / torch.sum(match_weigths)
        merged_score_list.append(avg_score)

        avg_pos = torch.tensor([torch.sum(pos[:,0] * match_scores) / torch.sum(match_scores),
                                torch.sum(pos[:, 1] * match_scores) / torch.sum(match_scores),
                                torch.sum(pos[:, 2] * match_scores) / torch.sum(match_scores),
                                torch.sum(pos[:, 3] * match_scores) / torch.sum(match_scores)])
        merged_box_list.append(avg_pos)


    # convert the boxes in the "xywh" form and tensors again
    keep_bboxes = change_box_order(torch.stack(merged_box_list),
                                   order="xyxy2xywh").to(device)

    keep_scores = torch.tensor(merged_score_list, dtype=torch.float32).to(device)

    return keep_bboxes, keep_scores

def isInside(box_1, box_2):
    # check whether box_1 is inside box_2 or vice versa
    flag = False
    if box_1[0] > box_2[0]:
        if box_1[1] > box_2[1]:
            if box_1[2] < box_2[2]:
                if box_1[3] < box_2[3]:
                    flag = True

    if box_2[0] > box_1[0]:
        if box_2[1] > box_1[1]:
            if box_2[2] < box_1[2]:
                if box_2[3] < box_1[3]:
                    flag = True

    return flag


def merge_jung(bboxes, scores, merge_thr=0.2):
    def iteration(annotation_list):
        total_set_list = []
        merged_list = []
        for annotation in annotation_list:
            if len(total_set_list) == 0:
                total_set_list.append([annotation])
            else:
                isAppended = False
                for set in total_set_list:
                    for saved_annotation in set:
                        iou = box_iou(saved_annotation[0].view(1, -1),
                                      annotation[0].view(1, -1),
                                      "xyxy")
                        isCovered = isInside(saved_annotation[0],
                                             annotation[0])
                        #if iou > merge_thr or isCovered:
                        if iou > 0.2:
                            set.append(annotation)
                            isAppended = True
                            break

                    if isAppended == True:
                        break

                if isAppended == False:
                    total_set_list.append([annotation])

        for set in total_set_list:
            value = 0
            max_area = 0
            max_info = None
            max_value = 0
            for idx, saved_annotation in enumerate(set):
                area = (saved_annotation[0][3] - saved_annotation[0][1]) * \
                       (saved_annotation[0][2] - saved_annotation[0][0])
                value += saved_annotation[1]

                if area > max_area:
                    max_area = area
                    max_info = saved_annotation
                if float(saved_annotation[1]) > max_value:
                    max_value = float(saved_annotation[1])

            value = float(value) / len(set)
            merged_list.append((max_info[0], max_value))
        return merged_list

    # bboxes and score must be tensors
    if isinstance(bboxes, list):
        bboxes = torch.stack(bboxes)

    if isinstance(scores, list):
        #scores = np.asarray(torch.stack(scores))
        scores = torch.stack(scores)

    # change the order from xywh to xyxy
    bboxes = change_box_order(bboxes, order="xywh2xyxy")

    # save boxes and scores in required format
    raw_list = [[bboxes[i], scores[i]] for i in range(len(bboxes))]

    # remove overlapping boxes (first run)
    final_list = iteration(raw_list)

    # remove remaining overlapping boxes till no box can be merged anymore
    change = True
    while change:
        length = len(final_list)
        final_list = iteration(final_list)
        if length == len(final_list):
            change = False

    bboxes = [final_list[i][0] for i in range(len(final_list))]
    bboxes = torch.stack(bboxes)

    scores = [final_list[i][1] for i in range(len(final_list))]
    scores = torch.tensor(scores, dtype=torch.float32)

    # filter out overlapping bboxes
    #bboxes, scores = rm_overlapping_boxes(bboxes, scores)

    # change box order
    bboxes = change_box_order(bboxes, order="xyxy2xywh")

    # sort the score in descending order, adjust bboxes accordingly
    scores, indices = torch.sort(scores, descending=True)
    bboxes = bboxes[indices]

    return bboxes, scores