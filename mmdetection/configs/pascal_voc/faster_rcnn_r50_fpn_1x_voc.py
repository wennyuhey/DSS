_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc0712_test.py',
    '../_base_/default_runtime.py'
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            num_classes=20,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
total_epochs = 20  # actual epoch = 4 * 3 = 12
