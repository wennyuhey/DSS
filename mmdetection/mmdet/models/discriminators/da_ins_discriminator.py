import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import DISCRIMINATORS, build_loss
from .da_base_discriminator import DABaseDiscriminator
from mmdet.utils import GradReverse


@DISCRIMINATORS.register_module()
class DAInsDiscriminator(DABaseDiscriminator):
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
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=0.01,
                     alpha=1,
                     loss_weight=1.0),
                 **kwargs):
        self.in_channels = in_channels
        self.loss_cls = loss_cls
        super(DAInsDiscriminator, self).__init__()
    def _init_layers(self):
        """Initialize layers of the head."""
        self.dropout = nn.Dropout()
        self.cls_linear = nn.Linear(100, 1)
        self.cls_domain = nn.ModuleList([nn.Linear(self.in_channels,100),
                                         nn.BatchNorm1d(100),
                                         nn.Linear(100,100),
                                         nn.BatchNorm1d(100)])
        self.gradreverse = GradReverse(1)
        self.loss_cls = build_loss(self.loss_cls) 
        self.relu = nn.ReLU()
        self.mse = nn.MSELoss()
    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_domain:
            normal_init(m, std=0.01)

    def forward(self, feats):
        return self.forward_single(feats)

    def forward_single(self, x):
        dis_feat = self.gradreverse.apply(x)
        for layer in self.cls_domain:
            dis_feat = layer(dis_feat)
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                dis_feat = self.relu(dis_feat)
                dis_feat = self.dropout(dis_feat)
        ins_dis_scores = self.cls_linear(dis_feat)

        return ins_dis_scores, torch.tensor([0])

    def loss_single(self, feat_dis_scores, gt_domain):
        # feature domain classification loss
        #target = [gt_domain[0].float() for i in range(len(feat_dis_scores))]
        ins_loss = self.mse(torch.mean(feat_dis_scores), gt_domain.float())
        return ins_loss, torch.tensor([0])

    def loss(self,
             feat_dis_scores,
             gt_domains):
        # compute loss
        """
        loss_feat, tempt = multi_apply(
        self.loss_single,
        feat_dis_scores,
        gt_domains)
        loss_dict_all = {
            'loss_ins': loss_feat}
        return loss_dict_all
        """
        loss_feat, tempt = self.loss_single(feat_dis_scores, gt_domains)
        loss_dict_all = {
            'loss_ins': loss_feat}
        return loss_dict_all
