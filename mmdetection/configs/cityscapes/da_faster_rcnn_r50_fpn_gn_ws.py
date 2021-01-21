_base_ = './da_faster_rcnn_fpn_1x_cityscapes.py'
conv_cfg = dict(type='ConvWM')
#conv_cfg = dict(type='Conv2d')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
#norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://jhu/resnet50_gn_ws',
    backbone=dict(
        #type='AuxResNet',
        conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)))

#load_from = '/lustre/S/wangyu/PretrainedModels/fasterrcnn_r50_fpn_gn_ws_cityscapes.pth'
#load_from = '/lustre/S/wangyu/kitti_model/fpn/epoch_7.pth'
#load_from = '/lustre/S/wangyu/aux_gn_wm/head0001_dis001/epoch_3.pth'
#load_from = '/lustre/S/wangyu/city_model/gn/epoch_8.pth'
#load_from = '/lustre/S/wangyu/aux_gn+wa_feat/head0001_dis001/epoch_2.pth'
#load_from = '/lustre/S/wangyu/city_model/gnwm/fin.pth'
#load_from = '/lustre/S/wangyu/city_model/gnwm/fin.pth'
#load_from = '/lustre/S/wangyu/city_model/gnwm_resinit/lr01/epoch_8.pth' 
#load_from = '/lustre/S/wangyu/gta_model/resinit/epoch_6.pth'
#load_from = '/lustre/S/wangyu/kitti_model/resinit_nofrozen/latest.pth'
#load_from = '/lustre/S/wangyu/city_model/resinit_nofrozen/epoch_9.pth'
#load_from = '/lustre/S/wangyu/kitti_model/cocoinit/latest.pth'
#load_from = '/lustre/S/wangyu/gta_model/cocoinit/epoch_2.pth'
#load_from = '/lustre/S/wangyu/kitti_model/cocoinit_nf/latest.pth'
load_from = '/lustre/S/wangyu/city_model/cocoinit_nf/latest.pth'
