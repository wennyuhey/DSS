_base_ = './faster_rcnn_r50_fpn_1x_cityscapes.py'
classes = ('car', )
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

