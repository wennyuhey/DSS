_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_cityscapes.py'
lr_config = dict(step=[4, 6])
total_epochs = 12
