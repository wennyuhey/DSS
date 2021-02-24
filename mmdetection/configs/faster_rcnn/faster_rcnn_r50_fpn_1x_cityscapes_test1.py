_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/cityscapes_voc_test_detection.py',
    '../_base_/default_runtime.py'
]
#norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# optimizer
# lr is set for a batch size of 8
model = dict(
#    pretrained='open-mmlab://detectron/resnet50_gn',
    roi_head=dict(
        bbox_head=dict(
            num_classes=8)))

optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[8])
total_epochs = 20  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
