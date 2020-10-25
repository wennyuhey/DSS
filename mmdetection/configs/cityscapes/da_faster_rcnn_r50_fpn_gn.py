_base_ = './da_faster_rcnn_fpn_1x_cityscapes.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    pretrained=None,
    backbone=dict(norm_cfg=norm_cfg),
    neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg)))
