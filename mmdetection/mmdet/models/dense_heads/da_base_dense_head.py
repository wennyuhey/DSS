from abc import ABCMeta, abstractmethod

import torch.nn as nn
import pdb
import math


class DABaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(DABaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x_s,
                      img_metas_s,
                      x_t,
                      img_metas_t,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs_s = self(x_s)
        #outs_t = self(x_t)
        if gt_labels is None:
            loss_inputs_s = outs_s + (gt_bboxes, img_metas_s)
        else:
            loss_inputs_s = outs_s + (gt_bboxes, gt_labels, img_metas_s)
        losses = self.loss(*loss_inputs_s, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list_s = self.get_bboxes(*outs_s, img_metas_s, cfg=proposal_cfg)
            #proposal_list_t = self.get_bboxes(*outs_t, img_metas_t, cfg=proposal_cfg)
            return losses, proposal_list_s#, proposal_t
