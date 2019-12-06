import torch
import torch.nn as nn
import numpy as np
import time

from detection.retinanet import Anchors
from detection.retinanet.retinanet_utils import box_iou, change_box_order
from detection.retinanet.losses import FocalLoss

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_curve, \
    average_precision_score, log_loss

class BoxAuroc(torch.nn.Module):
    """
       Accuracy, defined as #correct prediction / #total predictions
    """

    def __init__(self, iou_thr=0.2):
        """
        Implements the True Positive Rate
        Parameters
        ----------

        Returns
        -------
        tpr: float
        """
        super().__init__()
        self.iou_thr = iou_thr

    def forward(self, data, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size = loc_preds.shape[0]
        anchors = Anchors()
        labels = []
        preds = []

        for i in range(batch_size):

            pred_boxes, pred_labels, pred_score = anchors.generateBoxesFromAnchors(
                loc_preds[i],
                cls_preds[i],
                tuple(data[i].shape[1:]),
                cls_tresh=0.05)

            #start = time.time()
            target_boxes, target_labels = anchors.restoreBoxesFromAnchors(
                loc_targets[i],
                cls_targets[i],
                tuple(data[i].shape[1:]))
            #end = time.time()
            #print(end-start)

            if pred_boxes is None and target_boxes is None:
                continue

            if pred_boxes is None:
                preds.append(torch.zeros_like(target_labels))
                labels.append(target_labels)
                continue

            if target_boxes is None:
                preds.append(pred_labels)
                labels.append(torch.zeros_like(pred_labels))
                continue

            pred_boxes = change_box_order(pred_boxes, order='xywh2xyxy')
            target_boxes = change_box_order(target_boxes, order='xywh2xyxy')

            iou_matrix = box_iou(target_boxes, pred_boxes)
            iou_matrix = iou_matrix > self.iou_thr

            box_labels = torch.clamp(torch.sum(iou_matrix, 0), 0, 1)

            preds.append(pred_score)
            labels.append(box_labels)

        labels = torch.tensor([item for sublist in labels for item in sublist])\
            .type(torch.float32)
        preds = torch.tensor([item for sublist in preds for item in sublist])\
            .type(torch.float32)

        if not any(labels):
            return float_to_tensor(0.5)
        elif all(labels):
            return float_to_tensor(1)
        elif labels.dim() == 0:
            return float_to_tensor(0)
        else:
            return float_to_tensor(roc_auc_score(labels, preds))

class BoxCE(torch.nn.Module):
    """
       Accuracy, defined as #correct prediction / #total predictions
    """

    def __init__(self, iou_thr=0.2):
        """
        Implements the True Positive Rate
        Parameters
        ----------

        Returns
        -------
        tpr: float
        """
        super().__init__()
        self.iou_thr = iou_thr

    def forward(self, data, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size = loc_preds.shape[0]
        anchors = Anchors()
        labels = []
        preds = []

        for i in range(batch_size):

            pred_boxes, pred_labels, pred_score = anchors.generateBoxesFromAnchors(
                loc_preds[i],
                cls_preds[i],
                tuple(data[i].shape[1:]),
                cls_tresh=0.05)

            #start = time.time()
            target_boxes, target_labels = anchors.restoreBoxesFromAnchors(
                loc_targets[i],
                cls_targets[i],
                tuple(data[i].shape[1:]))
            #end = time.time()
            #print(end-start)

            if pred_boxes is None and target_boxes is None:
                continue

            if pred_boxes is None:
                preds.append(torch.zeros_like(target_labels))
                labels.append(target_labels)
                continue

            if target_boxes is None:
                preds.append(pred_labels)
                labels.append(torch.zeros_like(pred_labels))
                continue

            pred_boxes = change_box_order(pred_boxes, order='xywh2xyxy')
            target_boxes = change_box_order(target_boxes, order='xywh2xyxy')

            iou_matrix = box_iou(target_boxes, pred_boxes)
            iou_matrix = iou_matrix > self.iou_thr

            box_labels = torch.clamp(torch.sum(iou_matrix, 0), 0, 1)

            preds.append(pred_score)
            labels.append(box_labels)

        labels = torch.tensor([item for sublist in labels for item in sublist])\
            .type(torch.float32)
        preds = torch.tensor([item for sublist in preds for item in sublist])\
            .type(torch.float32)

        if not any(labels):
            return float_to_tensor(0.05)
        else:
            return float_to_tensor(log_loss(labels, preds,
                                            eps=1e-4, labels=[0,1]))


class AnchorCE1000(torch.nn.Module):
    def __init__(self, iou_thr=0.2):

        super().__init__()
        self.iou_thr = iou_thr

    def forward(self, data, loc_preds, loc_targets, cls_preds,
                cls_targets):

        cls_preds_val = torch.sigmoid(cls_preds).reshape(-1)
        cls_targets_val = cls_targets.reshape(-1)

        # remove the anchors with label -1
        pos = cls_targets_val > -1
        cls_preds_val = cls_preds_val[pos]
        cls_targets_val = cls_targets_val[pos]

        # sort according to score
        score, indices = torch.sort(cls_preds_val, descending=True)

        # take the top 1000
        score = score[:1000]
        indices = indices[:1000]

        cls_preds_val = cls_preds_val[indices]
        cls_targets_val = cls_targets_val[indices]

        ce = log_loss(cls_targets_val, cls_preds_val, eps=1e-4, labels=[0,1])

        return float_to_tensor(ce)

def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()

def float_to_tensor(f: float):
    return torch.from_numpy(np.array([f], dtype=np.float32))