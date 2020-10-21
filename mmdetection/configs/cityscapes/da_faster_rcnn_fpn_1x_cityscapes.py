_base_ = [
    '../_base_/models/da_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/da_cityscapes_detection.py',
    '../_base_/da_default_runtime.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        type='AuxResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),

    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
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
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0001, 
    paramwise_cfg = dict(
        custom_keys={
        'backbone': dict(lr_mult=1, decay_mult=1),
        'neck': dict(lr_mult=1, decay_mult=1),
        'rpn_head': dict(lr_mult=1, decay_mult=1),
        'roi_head': dict(lr_mult=1, decay_mult=1),
        'feat_dis_head': dict(lr_mult=0.05, decay_mult=1)}))

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[8, 28])
total_epochs = 30  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
