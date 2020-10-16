_base_ = './reppoints_moment_r50_fpn_gn-neck+head_1x_foggy_cityscapes.py'
lr_config = dict(step=[6, 9])
total_epochs = 12
