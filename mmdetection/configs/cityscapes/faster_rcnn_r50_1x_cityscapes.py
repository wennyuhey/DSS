_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_detection.py',
    '../_base_/default_runtime.py'
]
conv_cfg=dict(type='ConvWM')
model = dict(
    backbone=dict(conv_cfg=conv_cfg),
    neck=dict(conv_cfg=conv_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            conv_cfg=conv_cfg,
            in_channels=512,
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
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
total_epochs = 16  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
#load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_caffe_c4_1x-75ecfdfa_new.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
