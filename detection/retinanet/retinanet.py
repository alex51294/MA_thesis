import torch
import torch.nn as nn
import logging
import math
import numpy as np

from delira.models.abstract_network import AbstractPyTorchNetwork

from .anchors import Anchors
from detection.datasets.utils import create_crops
from detection.datasets.utils import bounding_box

file_logger = logging.getLogger(__name__)

class RetinaNet(AbstractPyTorchNetwork):
    # value recommended by the retinanet paper
    num_anchors = 9

    def __init__(self, in_channels: int, n_outputs: int,
                 pretrain: bool = False, resnet="RN50",
                 multistage: bool = False,
                 model_checkpoint: str = None,
                 **kwargs):
        """

        Parameters
        ----------
        in_channels: int
            number of input_channels
        n_outputs: int
            number of outputs (usually same as number of classes)
        """
        from detection.retinanet.models import FPN18, FPN34, FPN50, \
            FPN101, FPN152

        # check whether the model shall be initialized
        #pretrain = kwargs.pop("pretrain", False)

        # register params by passing them as kwargs to parent class __init__
        super().__init__(in_channels=in_channels,
                         n_outputs=n_outputs,
                         **kwargs)

        #self.backbone = self._build_backbone(in_channels, **kwargs)
        if resnet == "RN18":
            self.backbone = FPN18()
        elif resnet == "RN34":
            self.backbone = FPN34()
        elif resnet == "RN50":
            self.backbone = FPN50()
        elif resnet == "RN101":
            self.backbone = FPN101()
        elif resnet == "RN152":
            self.backbone = FPN152()
        else:
            raise KeyError("Unsupported ResNet version!")

        self.backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                       stride=2, padding=3, bias=False)

        self.bbox_subnet = self._build_subnet(self.num_anchors * 4)
        self.cls_subnet = self._build_subnet(self.num_anchors * n_outputs)

        for key, value in kwargs.items():
            setattr(self, key, value)

        # initialize according to the RetinaNet paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or \
                    isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.pi = 0.01
        nn.init.constant_(self.cls_subnet[-1].bias,
                          -math.log((1 - self.pi) / self.pi))

        self.start = 250

        # if desired, load ImageNet or other pre-trained model weights
        if pretrain:
            if resnet == "RN18":
                print('Loading pretrained ResNet18 model..')
                pretrained_dict = \
                    torch.load('/home/temp/moriz/data/weights/resnet18.pth')
            elif resnet == "RN34":
                print('Loading pretrained ResNet34 model..')
                pretrained_dict = \
                    torch.load('/home/temp/moriz/data/weights/resnet34.pth')
            elif resnet == "RN50":
                print('Loading pretrained ResNet50 model..')
                pretrained_dict = \
                    torch.load('/home/temp/moriz/data/weights/resnet50.pth')
            elif resnet == "RN101":
                print('Loading pretrained ResNet101 model..')
                pretrained_dict = \
                    torch.load('/home/temp/moriz/data/weights/resnet101.pth')
            else:
                print('Loading pretrained ResNet152 model..')
                pretrained_dict = \
                    torch.load('/home/temp/moriz/data/weights/resnet152.pth')

            required_dict = {}

            # rename the keys for single layer to fit to our FPN-layer names
            for key in pretrained_dict.keys():
                if key.startswith('layer'):
                    key_parts = key.split("layer")[1].split(".")
                    new_key = "conv" + str(int(key_parts[0]) + 1) + "." + \
                              key_parts[1] + "." + key_parts[2] + "." + key_parts[3]

                    required_dict[new_key] = pretrained_dict[key]
                else:
                    required_dict[key] = pretrained_dict[key]

            # load the pretrained weights into the backbone
            backbone_dict = self.backbone.state_dict()
            for k in required_dict.keys():
                if k in backbone_dict.keys():
                    # take care of the input layer
                    if k.startswith("conv1"):
                        if in_channels == 1:
                            backbone_dict['conv1.weight'] = \
                                torch.sum(backbone_dict['conv1.weight'],
                                          dim=1).unsqueeze(1)
                        else:
                            backbone_dict['conv1.weight'][:,0,:,:] = \
                                torch.sum(backbone_dict['conv1.weight'], dim=1)
                        continue

                    else:
                        backbone_dict[k] = required_dict[k]

            self.backbone.load_state_dict(backbone_dict)

        if multistage:
            if model_checkpoint is not None:
                print('Loading prior checkpoint...')
                checkpoint = torch.load(model_checkpoint)
                state_dict = checkpoint['state_dict']['model']
                backbone_dict = {}
                bbox_subnet_dict = {}
                cls_subnet_dict = {}
                for key in state_dict.keys():
                    if key.startswith("backbone"):
                        new_key = key.split("backbone.")[1]
                        backbone_dict[new_key] = state_dict[key]
                    elif key.startswith("bbox_subnet"):
                        new_key = key.split("bbox_subnet.")[1]
                        bbox_subnet_dict[new_key] = state_dict[key]
                    elif key.startswith("cls_subnet"):
                        new_key = key.split("cls_subnet.")[1]
                        cls_subnet_dict[new_key] = state_dict[key]

                self.backbone.load_state_dict(backbone_dict)
                self.bbox_subnet.load_state_dict(bbox_subnet_dict)
                self.cls_subnet.load_state_dict(cls_subnet_dict)
            else:
                raise KeyError("Weights required!")


    # ask Justus, did not work (did not found FPN50)
    # @staticmethod
    # def _build_backbone(in_channels):
    #     _backbone = FPN50()
    #     _backbone.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
    #                                    stride=2, padding=3, bias=False)
    #     return _backbone

    @staticmethod
    def _build_subnet(n_outputs):
        layers = []
        # four 3x3 conv layers with 256 channels each
        for _ in range(4):
            layers.append(
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(torch.nn.ReLU(True))

        # one 3x3 conv layer with n_outputs channels (classes * anchors or
        # anchors x coordinates tuple)
        layers.append(
            torch.nn.Conv2d(256, n_outputs, kernel_size=3, stride=1, padding=1))

        return torch.nn.Sequential(*layers)

    def forward(self, input_batch: torch.Tensor):
        # forward trough backbone (FPN50)
        backbone_preds = self.backbone(input_batch)

        # lists for bbox and class label predictions
        bbox_preds = []
        cls_preds = []

        for feature_maps in backbone_preds:
            bbox_pred = self.bbox_subnet(feature_maps)
            cls_pred = self.cls_subnet(feature_maps)

            # permutation: [N,9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(
                input_batch.size(0), -1,  4)

            # permutation: [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                input_batch.size(0), -1, self.init_kwargs["n_outputs"])

            bbox_preds.append(bbox_pred)
            cls_preds.append(cls_pred)

        return torch.cat(bbox_preds, 1), torch.cat(cls_preds, 1)

    @staticmethod
    def closure(model: AbstractPyTorchNetwork, data_dict: dict,
                optimizers: dict, criterions={}, metrics={}, fold=0, **kwargs):
        """
        closure method to do a single backpropagation step

        Parameters
        ----------
        model: ClassificationNetworkBase
            trainable model
        data_dict: dict
            dictionary containing the data
        optimizers: dict
            dictionary of optimizers to optimize model's parameters
        criterions: dict
            dict holding the criterions to calculate errors
            (gradients from different criterions will be accumulated)
        metrics: dict
            dict holding the metrics to calculate

        Returns
        -------
        dict: Metric values (with same keys as input dict metrics)
        dict: Loss values (with same keys as input dict criterions)
        list: Arbitrary number of predictions as torch.Tensor
        """

        assert (optimizers and criterions) or not optimizers, \
            "Criterion dict cannot be emtpy, if optimizers are passed"

        loss_vals = {}
        metric_vals = {}
        total_loss = 0

        # choose suitable context manager:
        if optimizers:
            context_man = torch.enable_grad
        else:
            context_man = torch.no_grad

        with context_man():
            # calculate the prediction through a forward pass through
            # the network
            loc_preds, cls_preds = model(data_dict["data"])

            if data_dict:
            #     if model.start > 0:
            #         label = torch.max(data_dict["cls_targets"])
            #         z = torch.zeros_like(data_dict["cls_targets"])
            #         l = torch.ones_like(data_dict["cls_targets"]) * label
            #         prob = torch.rand_like(data_dict["cls_targets"])
            #         init_anchors = torch.where(torch.ge(prob, 0.001), z, l)
            #         data_dict["cls_targets"] = \
            #             torch.clamp(data_dict["cls_targets"] + init_anchors,
            #                         min=0, max= label)
            #
            #         model.start = model.start - 1
            #
            #         if model.start == 0:
            #             print("Start-phase completed.")

                # computation of losses
                for key, crit_fn in criterions.items():
                #     # initialization for a more stabel training start according
                #     # to the original paper
                #     if key == 'FocalLoss' and model.start > 0:
                #         label = torch.max(data_dict["cls_targets"])
                #         z = torch.zeros_like(data_dict["cls_targets"])
                #         l = torch.ones_like(data_dict["cls_targets"]) * label
                #         prob = torch.rand_like(data_dict["cls_targets"])
                #         init_anchors = torch.where(torch.ge(prob, model.pi), z, l)
                #         init_anchors = \
                #             torch.clamp(data_dict["cls_targets"] + init_anchors,
                #                         min=0, max=label)
                #
                #         _loss_val = crit_fn(loc_preds,
                #                             data_dict['loc_targets'],
                #                             cls_preds,
                #                             init_anchors)
                #
                #         model.start = model.start - 1
                #
                #         if model.start == 0:
                #             print("Start-phase completed.")
                #     else:
                #         _loss_val = crit_fn(loc_preds,
                #                             data_dict['loc_targets'],
                #                             cls_preds,
                #                             data_dict['cls_targets'])
                    _loss_val = crit_fn(loc_preds,
                                        data_dict['loc_targets'],
                                        cls_preds,
                                        data_dict['cls_targets'])
                    loss_vals[key] = _loss_val.detach()
                    total_loss += _loss_val

                # computation of metric scores
                with torch.no_grad():
                    for key, metric_fn in metrics.items():

                        # average precision
                        metric_vals[key] = metric_fn(data_dict["data"],
                                                     loc_preds,
                                                     data_dict['loc_targets'],
                                                     cls_preds,
                                                     data_dict['cls_targets'])

        # this is the main functionality of the function: backprop and
        # actualisation step
        if optimizers:
            optimizers['default'].zero_grad()
            total_loss.backward()
            optimizers['default'].step()

        else:
            # add prefix "val" in validation mode
            eval_loss_vals, eval_metrics_vals = {}, {}
            for key in loss_vals.keys():
                eval_loss_vals["val_" + str(key)] = loss_vals[key]

            for key in metric_vals:
                eval_metrics_vals["val_" + str(key)] = metric_vals[key]

            loss_vals = eval_loss_vals
            metric_vals = eval_metrics_vals

        # logging.info({'scores': {**metric_vals, **loss_vals},
        #               #'images': {str(name): data_dict["data"][name] for name in range(1)},
        #               'fold': fold})

        for key, val in {**metric_vals, **loss_vals}.items():
            logging.info({"value": {"name": key,
                                    "value": val.item(),
                                    "env_appendix": "_%02d" % fold}})

        return metric_vals, loss_vals, [loc_preds, cls_preds]

    @staticmethod
    def prepare_batch(batch: dict, input_device: torch.device,
                          output_device: torch.device):
        """
        Converts a numpy batch of data and labels to torch.Tensors and pushes
        them to correct devices

        Parameters
        ----------
        batch: dict
            dictionary containing the batch (must have keys 'data' and 'label'
        input_device: torch.device
            device for network inputs
        output_device: torch.device
            device for network outputs

        Returns
        -------
        torch.Tensor: data tensor
        list: list of labels (as torch.Tensors)
        """

        # load the images, bboxes and labels from the batch
        imgs = batch['data'].astype(np.float32)
        if "crops" in batch.keys():
            crops = batch["crops"].astype(np.float32)
            imgs = np.concatenate((imgs, crops), axis=1)

        boxes = batch['bboxes']
        labels = batch["roi_labels"]

        # define required constructs/models
        anchors = Anchors()
        inputs = []
        loc_targets = []
        cls_targets = []

        num_imgs = len(imgs)
        for i in range(num_imgs):
            # get image size; the first value is the number of channels
            # (not required here)
            img_size = imgs[i].shape[1:]
            inputs.append(torch.Tensor(imgs[i]))

            # get anchors and according labels
            loc_target, cls_target = \
                anchors.generateAnchorsFromBoxes(torch.Tensor(boxes[i]),
                                                 torch.Tensor(labels[i]),
                                                 input_size=(imgs[i].shape[2],
                                                             imgs[i].shape[1]))
            # append in list
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)

        data_dict = {'data': torch.stack(inputs).to(input_device),
                     'loc_targets': torch.stack(loc_targets).to(output_device),
                     'cls_targets': torch.stack(cls_targets).to(output_device)}

        return data_dict

# implemented by Justus (for loading purposes)
# def getRetinaNet(load_file=None, **kwargs):
#     if load_file:
#         loaded_args = torch.load(load_file)
#
#         if not isinstance(loaded_args, dict) and not isinstance(loaded_args,
#                                                                 OrderedDict):
#             loaded_args = {"state_dict": loaded_args}
#
#     else:
#         loaded_args = {}
#
#     net = retinanet(**loaded_args.get("init_kwargs", {}), **kwargs)
#
#     if load_file:
#         net.load_state_dict(loaded_args["state_dict"])
#
#     return net
