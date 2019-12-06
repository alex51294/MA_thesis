import torch
import torch.nn as nn
import torch.nn.functional as F
from .retinanet_utils import one_hot_embedding

class FocalLoss(nn.Module):
    """
    Focal loss for binary case
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='sum'):
        """
        Implements Focal Loss for binary class case
        Parameters
        ----------
        alpha: float
            alpha has to be in range [0,1], assigns class weight
        gamma: float
            focusing parameter
        reduction: string
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)

        Returns
        -------
        loss: float
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    # up to now the used focal loss implementation: since identical to the one
    # in delira, it will be removed later
    def focal_loss_with_logits(self, cls_preds, cls_targets):
        '''Compute loss between (cls_preds, cls_targets).

        Args:
          cls_preds: (tensor) predicted class confidences,
                    sized [batch_size*(#anchors-#ignored_anchors), #classes].
          cls_targets: (tensor) encoded target labels in one-hot vector format,
                    sized [#anchors-#ignored_anchors), #classes]
        Returns:
          (tensor) loss
        '''
        bce_loss = F.binary_cross_entropy_with_logits(cls_preds,
                                                      cls_targets,
                                                      reduction='none')

        cls_preds = torch.sigmoid(cls_preds)

        if self.alpha is not None:
            # create weights for alpha
            alpha_weight = torch.ones(cls_targets.shape,
                                      device=cls_preds.device) * \
                           self.alpha
            alpha_weight = torch.where(torch.eq(cls_targets, 1.),
                                       alpha_weight,
                                       1 - alpha_weight)
        else:
            alpha_weight = torch.Tensor([1]).to(cls_preds.device)

        # create weights for focal loss
        focal_weight = 1 - torch.where(torch.eq(cls_targets, 1.),
                                       cls_preds,
                                       1 - cls_preds)
        focal_weight.pow_(self.gamma)
        focal_weight.to(cls_preds.device)

        # compute loss
        focal_loss = focal_weight * alpha_weight * bce_loss

        if self.reduction == 'elementwise_mean':
            return torch.mean(focal_loss)
        if self.reduction == 'none':
            return focal_loss
        if self.reduction == 'sum':
            return torch.sum(focal_loss)
        raise AttributeError('Reduction parameter unknown.')

    def multi_class_FL_loss(self, cls_preds, cls_targets):
        '''Compute loss between (cls_preds, cls_targets).

           Args:
             cls_preds: (tensor) predicted class confidences,
                       sized [batch_size*(#anchors-#ignored_anchors), #classes].
             cls_targets: (tensor) encoded target labels in one-hot vector format,
                       sized [#anchors-#ignored_anchors), ]
           Returns:
             (tensor) loss
       '''
        ce_loss = F.cross_entropy(cls_preds, cls_targets.type(torch.int64),
                                  reduction='none')

        cls_preds = torch.exp(-ce_loss)

        if self.alpha is not None:
            # create weights for alpha
            alpha_weight = torch.ones(cls_targets.shape,
                                      device=cls_preds.device) * \
                           self.alpha
            alpha_weight = torch.where(torch.gt(cls_targets, 0),
                                       alpha_weight,
                                       1 - alpha_weight)
        else:
            alpha_weight = torch.Tensor([1]).to(cls_preds.device)

        # create weights for focal loss
        focal_weight = 1 - torch.where(torch.gt(cls_targets, 0),
                                       cls_preds,
                                       1 - cls_preds)
        focal_weight.pow_(self.gamma)
        focal_weight.to(cls_preds.device)

        # compute loss
        focal_loss = focal_weight * alpha_weight * ce_loss

        if self.reduction == 'elementwise_mean':
            return torch.mean(focal_loss)
        if self.reduction == 'none':
            return focal_loss
        if self.reduction == 'sum':
            return torch.sum(focal_loss)
        raise AttributeError('Reduction parameter unknown.')

    @staticmethod
    def focal_loss(cls_preds, cls_targets, alpha=0.25, gamma=2,
                   reduction='sum'):
        '''Compute loss between (cls_preds, cls_targets).

        Args:
          cls_preds: (tensor) predicted class confidences,
                    sized [batch_size*(#anchors-#ignored_anchors), #classes].
          cls_targets: (tensor) encoded target labels in one-hot vector format,
                    sized [#anchors-#ignored_anchors), #classes]
        Returns:
          (tensor) loss
        '''

        # according to the retinanet paper, it is more numerical stable to
        # perform the sigmoid operation here
        p = cls_preds

        # interpretation of pt as used in paper: pt = p if t > 0 else 1-p
        pt = p * cls_targets + (1 - p) * (1 - cls_targets)

        # same for alpha: w = alpha if t > 0 else 1-alpha
        w = alpha * cls_targets + (1 - alpha) * (1 - cls_targets)

        # new weight consisting out of the focusing parameter gamma and the
        # class balance alpha
        w = w * (1 - pt).pow(gamma)

        # return the CE loss modified by the weight
        # note that the input is not normalized by the sigmoid function
        # (happens in the function below)
        return F.binary_cross_entropy(cls_preds, cls_targets, w,
                                      reduction=reduction)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and
                (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations,
                sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations,
                sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences,
                sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels,
                sized [batch_size, #anchors].

        returns:
          loss: (tensor) SmoothL1Loss(loc_preds, loc_targets) +
                FocalLoss(cls_preds, cls_targets).
        '''

        batch_size, num_boxes = cls_targets.size()
        num_classes = cls_preds.shape[2]
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        # cls_loss = FocalLoss(cls_preds, cls_targets)
        # filter out unassigned anchors
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, num_classes)
        cls_targets = cls_targets[pos_neg]

        # check whether we investigate a binary or multi-class case: if
        # multi-class, change the format into one-hot embedding
        if num_classes > 1:
            # Note that the background is considered as a seperate class and
            # needs to be clipped
            cls_targets = one_hot_embedding(cls_targets.type(torch.int64),
                                            num_classes + 1)[:, 1:]
            # cls_loss = self.multi_class_FL_loss(masked_cls_preds,
            #                                     cls_targets.view(-1,))
        #else:
        # general focal loss function
        cls_loss = self.focal_loss_with_logits(masked_cls_preds,
                                               cls_targets.view(-1, num_classes))

        # normalize by the number of detected anchors
        # or by the total number of anchors if no object presente
        if num_pos > 0:
            return cls_loss / num_pos.type(torch.float32)
        else:
            return cls_loss / torch.tensor(batch_size).type(torch.float32).to(num_pos.device)

class FocalMSELoss(nn.Module):
    """
    Focal loss for binary case
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='sum'):
        """
        Implements Focal Loss for binary class case
        Parameters
        ----------
        alpha: float
            alpha has to be in range [0,1], assigns class weight
        gamma: float
            focusing parameter
        reduction: string
            Specifies the reduction to apply to the output: ‘none’ |
            ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
            ‘elementwise_mean’: the sum of the output will be divided by the
            number of elements in the output, ‘sum’: the output will be summed
        (further information about parameters above can be found in pytorch
        documentation)

        Returns
        -------
        loss: float
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    # up to now the used focal loss implementation: since identical to the one
    # in delira, it will be removed later
    def focal_mse_loss_with_logits(self, cls_preds, cls_targets):
        '''Compute loss between (cls_preds, cls_targets).

        Args:
          cls_preds: (tensor) predicted class confidences,
                    sized [batch_size*(#anchors-#ignored_anchors), prob + class].
          cls_targets: (tensor) encoded target labels in one-hot vector format,
                    sized [#anchors-#ignored_anchors), #classes]
        Returns:
          (tensor) loss
        '''
        mse_loss = F.mse_loss(cls_preds[:, 1].view(-1,1), cls_targets,
                              reduction='none')

        cls_prob = torch.sigmoid(cls_preds[:,0].view(-1,1))

        if self.alpha is not None:
            # create weights for alpha
            alpha_weight = torch.ones(cls_targets.shape,
                                      device=cls_preds.device) * \
                           self.alpha
            alpha_weight = torch.where(torch.gt(cls_targets, 0.),
                                       alpha_weight,
                                       1 - alpha_weight)
        else:
            alpha_weight = torch.Tensor([1]).to(cls_preds.device)

        # create weights for focal loss
        #focal_weight = (1 - cls_prob).pow(self.gamma).to(cls_preds.device)

        # create weights for focal loss
        focal_weight = 1 - torch.where(torch.gt(cls_targets, 0.),
                                       cls_prob,
                                       1 - cls_prob)
        focal_weight.pow_(self.gamma)
        focal_weight.to(cls_preds.device)

        # compute loss
        focal_mse_loss = focal_weight * alpha_weight * mse_loss

        if self.reduction == 'elementwise_mean':
            return torch.mean(focal_mse_loss)
        if self.reduction == 'none':
            return focal_mse_loss
        if self.reduction == 'sum':
            return torch.sum(focal_mse_loss)
        raise AttributeError('Reduction parameter unknown.')

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and
                (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations,
                sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations,
                sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences,
                sized [batch_size, #anchors, prob. + class].
          cls_targets: (tensor) encoded target labels,
                sized [batch_size, #anchors].

        returns:
          loss: (tensor) FocalMSELoss
        '''

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        # filter out unassigned anchors
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, 2)
        cls_targets = cls_targets[pos_neg]

        # general focal loss function
        cls_loss = self.focal_mse_loss_with_logits(masked_cls_preds,
                                                   cls_targets.view(-1, 1))

        # normalize by the number of detected anchors
        # or by the total number of anchors if no object present
        if num_pos > 0:
            return cls_loss / num_pos.type(torch.float32)
        else:
            return cls_loss / torch.tensor(batch_size).type(torch.float32).to(num_pos.device)



class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss for bbix regression
    """
    def __init__(self, weight=10):

        super().__init__()
        self.weight = weight

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and
                (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations,
                sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations,
                sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences,
                sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels,
                sized [batch_size, #anchors].

        returns:
          loss: (tensor) SmoothL1Loss(loc_preds, loc_targets) +
                FocalLoss(cls_preds, cls_targets).
        '''

        batch_size, num_boxes = cls_targets.size()
        num_classes = cls_preds.shape[2]
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()


        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets,
                                    reduction='sum')

        if num_pos > 0:
            return self.weight * loc_loss / num_pos.type(torch.float32)
        else:
            return self.weight * loc_loss