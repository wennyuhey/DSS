_base_ = [
    '../_base_/models/da_faster_rcnn_r50_fpn.py', '../_base_/datasets/da_voc_clipart.py',
    '../_base_/da_default_runtime.py'
]
# optimizer
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet101_gn_ws',
    backbone=dict(
        type='AuxResNet',
        depth=101,
        frozen_stages=1,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        norm_eval=False,
        style='pytorch'),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            num_classes=20,
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)),
    feat_dis_head=dict(
        type='DAFeatDiscriminator',
        in_channels=256),
    ins_dis_head=None,
    domain_mask=None)
# optimizer
# lr is set for a batch size of 8
optimizer = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0005)
optimizer_backbone = dict(
    type='SGD',
    lr=0.01,
    weight_decay=0.0005)
optimizer_discriminator = dict(
    type='SGD',
    lr=0.001,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[3])
# runtime settings
total_epochs = 20  # actual epoch = 4 * 3 = 12
load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r101_fpn_gn_ws-all_1x_coco_20200205-a93b0d75.pth'
