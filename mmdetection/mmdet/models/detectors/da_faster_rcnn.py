from ..builder import DETECTORS
from .da_two_stage import DATwoStageDetector


@DETECTORS.register_module()
class DAFasterRCNN(DATwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 feat_dis_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(DAFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            feat_dis_head=feat_dis_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
