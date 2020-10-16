_base_ = '../reppoints/reppoints_moment_r50_fpn_gn-neck+head_2x_cityscapes.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(
        depth=101,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
total_epochs = 15  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)

