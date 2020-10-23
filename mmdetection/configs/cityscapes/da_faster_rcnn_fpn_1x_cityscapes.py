_base_ = [
    '../_base_/models/da_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/da_cityscapes_detection.py',
    '../_base_/da_default_runtime.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        type='AuxResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    #neck=dict(norm_cfg=dict(type='GN', num_group=32, requires_grad=True),)
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
     #       norm_cfg=dict(type='GN', num_group=32, requires_grad=True),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
     feat_dis_head=dict(
        type='DAFeatDiscriminator',
        in_channels=256))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0001)
optimizer_backbone = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0001)
optimizer_discriminator = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[25])
total_epochs = 40  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
