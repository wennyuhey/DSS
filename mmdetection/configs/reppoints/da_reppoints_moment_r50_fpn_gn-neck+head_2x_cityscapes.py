_base_ = './da_reppoints_moment_r50_fpn_gn-neck+head_1x_cityscapes.py'
lr_config = dict(step=[16, 24])
total_epochs = 30
