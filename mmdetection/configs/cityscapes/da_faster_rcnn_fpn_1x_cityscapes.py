_base_ = [
    '../_base_/models/da_faster_rcnn_r50_fpn.py',
    '../_base_/datasets/da_cityscapes_detection.py',
    '../_base_/da_default_runtime.py'
]
model = dict(
#    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    #domain_mask=dict(
    #    type='DAChannelMask',
    #    in_channels=256,
    #    att_dim=128),
    domain_mask=None,
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
        in_channels=256),
#     feat_dis_head=None,
     ins_dis_head=None)
#     ins_dis_head=dict(
#         type='DAInsDiscriminator',
#         in_channels=256*7*7))
## optimizer
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
    step=[60])
total_epochs = 100  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
#load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/fasterrcnn_r50_fpn_gn_ws_cityscapes.pth'
#load_from = '/lustre/S/wangyu/aux_normon_ins_step_done/epoch_17.pth'
#load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
