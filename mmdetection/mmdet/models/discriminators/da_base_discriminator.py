from abc import ABCMeta, abstractmethod

import torch.nn as nn
import math
import pdb


class DABaseDiscriminator(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(DABaseDiscriminator, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    def forward_train(self,
                      x,
                      gt_domains=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        outs = self(x) #discriminator.forward()
        loss_inputs = outs + (gt_domains)
        losses = self.loss(*loss_inputs)
