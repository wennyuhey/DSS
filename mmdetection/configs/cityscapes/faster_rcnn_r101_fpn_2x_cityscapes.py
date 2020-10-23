_base_ = './faster_rcnn_r50_fpn_2x_cityscapes.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
