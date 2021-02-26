_base_ = [
    '../_base_/models/da_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/da_cityscapes_detection.py',
    '../_base_/da_default_runtime.py'
]
conv_cfg = dict(type='ConvWM')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(
        type='AuxResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
        type='Shared4Conv1FCBBoxHead',
            num_classes=8,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)),
    feat_dis_head=dict(
        type='DAFeatDiscriminator',
        in_channels=256),
    ins_dis_head=None,
    domain_mask=None)
     #ins_dis_head=dict(
     #    type='DAInsDiscriminator',
     #   in_channels=256*7*7))
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
    step=[35])
total_epochs = 40  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
# load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
