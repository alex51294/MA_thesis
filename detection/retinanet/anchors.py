'''Encode object boxes and labels.'''
import math
import torch
import numpy as np
import time

from .retinanet_utils import meshgrid, box_iou, box_nms, change_box_order


class Anchors:
    def __init__(self):
        # Anchor sizes, ratios and additional scales according to
        # retinanet paper
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.
        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            # w/h = ar
            for ar in self.aspect_ratios:
                h = math.sqrt(s/ar)
                w = ar * h
                # scale
                for sr in self.scale_ratios:
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.
        Args:
          input_size: (tensor) model input size of (w,h).
        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size
          [#anchors,4], where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)

        # p3 -> p7 feature map sizes based on the input image size
        # e.g. 224 -> [28, 17, 7, 4, 2]
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in
                    range(num_fms)]

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])

            # [fm_h*fm_w, 2]
            # as far as I get it, meshgrid gives out a list of coordinates for
            # the single feature map positions
            xy = meshgrid(fm_w,fm_h) + 0.5

            # next, these coordinates are "extended" by the down-scale factor
            # to the original size and reshaped
            # ATTENTION: expand is used to save memory: on the one side, we
            # have one anchor position per feature map point; these positions
            # are equal for all different anchor types, hence can be shared;
            # on the other side, each feature map point has the same anchor
            # types that can be thus shared across the feature map
            # TODO: change the hard-coded 9 to #types of anchors
            xy = (xy.type(torch.float32)*grid_size)\
                .view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)

            # in the end we should have a tuple of this form [x,y,w,h], where
            # some values share their memory location
            box = torch.cat([xy,wh], 3)
            boxes.append(box.view(-1,4))

        return torch.cat(boxes, 0)

    def _get_num_anchor_boxes(self, input_size):
        num_fm = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fm)]
        num_anchors = [i[0]*i[1]*9 for i in fm_sizes]

        return torch.sum(torch.Tensor(num_anchors))

    def _get_feature_map_sizes(self, input_size):
        num_fm = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in
                    range(num_fm)]

        return fm_sizes

    def generateAnchorsFromBoxes(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (x_center,y_center,width,height),
                sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        # convert the input_size tuple to torch tensor
        input_size = torch.Tensor([input_size, input_size]) \
                    if isinstance(input_size, int) \
                    else torch.Tensor(input_size)

        # this give us the standard anchor_boxes for comparison
        anchor_boxes = self._get_anchor_boxes(input_size)

        # if empty image return all zeros
        if not any(labels):
            return torch.zeros_like(anchor_boxes), \
                   torch.zeros(len(anchor_boxes))

        # calculate for each combination of bboxes and anchors the IoU, i.e.
        # giving an tensor of the form [#anchors, #bboxes]
        ious = box_iou(anchor_boxes, boxes, order='xywh')

        # Now we have for each anchor the IoU with the given bboxes; since one
        # anchor cannot match more than one bbox, we consider only the bbox
        # with the maximum IoU
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        # saving the anchors in the form described above
        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)

        # save the according labels (label of bbox1 or bbox2 or ...)
        cls_targets = labels[max_ids]

        # the class label "0" is reserved for the background, check whether
        # it is used and shift the labels if required
        if labels.min(0) == 0:
            cls_targets = cls_targets + 1

        # define everything below an IoU of 0.5 as background
        cls_targets[max_ious < 0.5] = 0

        # ignore IoUs between [0.4,0.5]
        ignore = (max_ious > 0.4) & (max_ious < 0.5)

        # for now just mark ignored to -1
        cls_targets[ignore] = -1

        # second rule (was not implemented yet): if no anchor has an IoU over
        # 0.5, save the one with the maximum IoU
        if cls_targets.max() == 0:
            max_iou, max_id = max_ious.max(0)
            cls_targets[max_id] = labels[max_ids[max_id]]

        return loc_targets, cls_targets

    def generateBoxesFromAnchors(self, loc_preds, cls_preds, input_size,
                                 cls_tresh=0.05, nms_thresh=0.5):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels,
                    sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
          score: (tensor) confidence for each class label, sied [#obj, ]
        '''
        NMS_THRESH = nms_thresh

        # convert the input_size tuple to torch tensor
        if isinstance(input_size, int):
            input_size =  torch.Tensor([input_size, input_size])
        elif isinstance(input_size, tuple):
            input_size = torch.Tensor(input_size)

        # this give us the standard anchor_boxes for comparison;
        # set to the same type of device as predictions
        anchor_boxes = self._get_anchor_boxes(input_size).to(loc_preds.device)

        # convert the normalized parameters back to the form
        # (x_center, y_center, width, height)
        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]

        # save the boxes in the form [#anchors,4], where each box has the
        # format [x1,y1, x2, y2]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)

        # save the class labels in the form [#anchors,]
        score, labels = cls_preds.sigmoid().max(1)

        # the label "0" indicates background; therefore, we shift the labels
        labels = labels + 1

        # sort the score in descending order, adjust bboxes accordingly
        score, indices = torch.sort(score, descending=True)
        boxes = boxes[indices]
        labels = labels[indices]

        # take only the top-scoring 1000 predictions
        score = score[:1000]
        boxes = boxes[:1000]
        labels = labels[:1000]

        # save the IDs of anchors whose probability is high enough (e.g. 0.05)
        ids = score > cls_tresh

        #if no anchor has an prob over 0.05, return None as bbox, 0 as label
        # and the present maximum prob
        if not any(ids):

            # good for testing
            max_prob, max_id = score.max(0)
            # ids[max_id] = 1

            #return torch.zeros(1, 4), 0, torch.Tensor([max_prob])
            return None, 0, torch.Tensor([max_prob])

        # reduce by NMS the number of anchor boxes to the number of found
        # objects [#obj,]
        ids = ids.nonzero().squeeze()

        # if only one bbox available
        if ids.dim() == 0:
            boxes = change_box_order(boxes, "xyxy2xywh")
            return boxes[ids].view(1, -1), \
                   torch.Tensor([labels[ids]]).type(labels.dtype).to(loc_preds.device), \
                   torch.Tensor([score[ids]]).to(loc_preds.device)

        # else perform NMS
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)

        # change box order
        boxes = change_box_order(boxes, "xyxy2xywh")

        return boxes[ids][keep], labels[ids][keep], score[ids][keep]

    def restoreBoxesFromAnchors(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels,
                    sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
          score: (tensor) confidence for each class label, sied [#obj, ]
        '''
        NMS_THRESH = 0.5

        # convert the input_size tuple to torch tensor
        if isinstance(input_size, int):
            input_size =  torch.Tensor([input_size, input_size])
        elif isinstance(input_size, tuple):
            input_size = torch.Tensor(input_size)

        # if not any(cls_preds):
        #     return None, 0


        # this give us the standard anchor_boxes for comparison;
        # set to the same type of device as predictions
        anchor_boxes = self._get_anchor_boxes(input_size).to(loc_preds.device)

        # convert the normalized parameters back to the form
        # (x_center, y_center, width, height)
        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]

        # save the boxes in the form [#anchors,4], where each box has the
        # format [x1,y1, x2, y2]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)

        # save the IDs of anchors whose probability is high enough (e.g. 0.05)
        ids = cls_preds > 0

        # reduce by NMS the number of anchor boxes to the number of found
        # objects [#obj,]
        ids = ids.nonzero().squeeze()

        if ids.nelement() == 0:
            return None, 0

        # if only one bbox available
        if ids.dim() == 0:
            boxes = change_box_order(boxes, "xyxy2xywh")
            return boxes[ids].view(1, -1), cls_preds[ids].view(1, -1)


        # else perform NMS
        keep = box_nms(boxes[ids], cls_preds[ids], iou_thr=NMS_THRESH)


        # change box order
        boxes = change_box_order(boxes, "xyxy2xywh")

        return boxes[ids][keep], cls_preds[ids][keep]