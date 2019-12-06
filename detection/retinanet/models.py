import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models as t_models
import logging
from delira.models.abstract_network import AbstractNetwork

file_logger = logging.getLogger(__name__)

# copied from kuangliu

# implementation is close to the original (the one in pytorch), the only
# difference I see is that the downsampling was moved from _make_layer
# to the Bottleneck class
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        # self.bn1 = nn.GroupNorm(32, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        # self.bn2 = nn.GroupNorm(32, planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes))
                nn.InstanceNorm2d(self.expansion * planes))
                # nn.GroupNorm(32, self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        #self.bn1 = nn.GroupNorm(32, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        #self.bn2 = nn.GroupNorm(32, planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1,
                               bias=False)
        #self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.bn3 = nn.InstanceNorm2d(self.expansion*planes)
        #self.bn3 = nn.GroupNorm(32, self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                          #nn.BatchNorm2d(self.expansion*planes))
                          nn.InstanceNorm2d(self.expansion*planes))
                          #nn.GroupNorm(32, self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class FPN(torch.nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = nn.InstanceNorm2d(64)
        #self.bn1 = nn.GroupNorm(32, 64)

        # Bottom-up layers
        self.conv2 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # this two are already part of the retinanet model
        self.conv6 = nn.Conv2d(512 * block.expansion, 256, kernel_size=3,
                               stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512 * block.expansion, 256, kernel_size=1,
                                   stride=1,
                                   padding=0)
        self.latlayer2 = nn.Conv2d(256 * block.expansion, 256,
                                   kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128 * block.expansion, 256,
                                   kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                   padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                   padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # the up-sampling is to be checked
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # this two are already part of the retinanet model
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7

def FPN18():
    return FPN(BasicBlock, [2,2,2,2])

def FPN34():
    return FPN(BasicBlock, [3,4,6,3])

def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [3,4,23,3])

def FPN152():
    return FPN(Bottleneck, [3,8,36,3])
