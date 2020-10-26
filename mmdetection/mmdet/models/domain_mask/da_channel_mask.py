import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import DOMAINMASKS, build_loss
from .da_base_mask import DABaseMask
from mmdet.utils import GradReverse
from mmdet.utils import MPNCOV


@DOMAINMASKS.register_module()
class DAChannelMask(DABaseMask):
    """RepPoint head.

    Args:
        point_feat_channels (int): Number of channels of points features.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 att_dim,
                 **kwargs):
        self.in_channels = in_channels
        self.att_dim = att_dim

        super(DAChannelMask, self).__init__()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=False)
        self.conv_for_DR = nn.Conv2d(self.in_channels, self.att_dim, kernel_size=1, stride=4, bias=True)
        self.bn_for_DR = nn.BatchNorm2d(self.att_dim)
        self.row_bn = nn.BatchNorm2d(self.att_dim)
        self.row_conv_group = nn.Conv2d(self.att_dim, 4*self.att_dim, kernel_size=(self.att_dim, 1),
                                        groups=self.att_dim, bias=True)
        self.fc_adapt_channels = nn.Conv2d(4*self.att_dim, self.in_channels, kernel_size=1, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=4)
      
    def init_weights(self):
        """Initialize weights of the head."""
#        for m in self.cls_domain:
#            normal_init(m.conv, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat_s, feat_t):
        return multi_apply(self.forward_single, feat_s, feat_t)

    def forward_single(self, x_s, x_t):
        out_s = self.relu(x_s)
        out_s = self.conv_for_DR(out_s)
        out_s = self.bn_for_DR(out_s)
        out_s = self.relu(out_s)

        out_t = self.relu(x_t)
        out_t = self.conv_for_DR(out_t)
        out_t = self.bn_for_DR(out_t)
        out_t = self.relu(out_t)

        out_t = out_t.detach()
        out = MPNCOV.CovpoolLayer(out_s, out_t) # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous() # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out) # Nx512x1x1

        out = self.fc_adapt_channels(out) #NxCx1x1
        out = self.sigmoid(out) #NxCx1x1

        return out, torch.tensor([0])
    def loss(self):
        return torch.tensor([0])
