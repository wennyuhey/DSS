_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/gta_voc_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
classes = ('car', )
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)))

load_from = '/lustre/S/wangyu/PretrainedModels/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
