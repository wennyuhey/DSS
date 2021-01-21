_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_cityscapes.py'
conv_cfg = dict(type='ConvWM')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)))
load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
