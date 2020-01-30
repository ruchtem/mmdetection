import warnings
warnings.filterwarnings("ignore")  # "ignore" or "error"

backbone_out = 32
img_size = (200, 200)   # w, h

# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ConvNet',
        in_channels=1,
        hidden_channels=64,
        out_channels=backbone_out),
    rpn_head=dict(
        type='RPNHead',
        in_channels=backbone_out,
        feat_channels=64,
        anchor_scales=[1],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[50],     # One for each stage outputted by the backbone, this is the w and h of the anchor. This is dependent on the size of the feature output
        #anchor_scales=[3],
        #anchor_ratios=[0.5, 1.0, 2.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=backbone_out,
        featmap_strides=[4]),
    bbox_head=dict(
        type='BBoxHead',
        with_reg=False,
        in_channels=backbone_out,
        roi_feat_size=7,
        num_classes=11,     # 10 digits + 1 for background
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=backbone_out,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=2,
        in_channels=backbone_out,
        conv_out_channels=64,
        num_classes=11,     # 10 digits + 1 for background
        class_agnostic=True,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=200,
        nms_post=200,
        max_num=200,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=200,
        nms_post=200,
        max_num=200,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'MovingMnistDataset'
data_root = 'data/moving-mnist/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='grayscale'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Pad', size=img_size),    # height, width, necessary for mask stacking see https://github.com/open-mmlab/mmdetection/issues/2026
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_shape', 'img_shape')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='grayscale'),
    dict(type='Pad', size=img_size),    # height, width, necessary for mask stacking see https://github.com/open-mmlab/mmdetection/issues/2026
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape')),

]
data = dict(
    imgs_per_gpu=5,    # batch size
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline)
    )
# optimizer
optimizer = dict(type="Adam", lr=0.00001)
optimizer_config = None

# Learning rate update policy for training. Parameters:
# policy, str.
#       Can be 'fixed', 'step', 'exp', 'poly', 'inv', or 'cosine'. For documentation see:
#       mmcv/runner/hooks/lr_updater.py
# 
# Optional parameters:
#
# step, list or int.
#       Valid for policy='step'. Unclear.
# gamma, float.
#       Valid for policy='step', 'exp', 'inv'. Unclear.
# power, float.
#       Valid fpr policy='poly', 'inv'. Unclear.
# min_lr, float.
#       Valid for policy='poly'. Unclear.
# larget_lr, float,
#       Valid for policy='cosine'. Unclear.
# 
#
lr_config = dict(
    policy="fixed"
)

# Save parameters every intervall=n epochs
checkpoint_config = dict(interval=50)

# The loggers which can be attached/hooked to executions. Parameters:
# interval: int. 
#       How often the loger should report with respect to epochs
# hooks: list of dicts.
#       Unclear. It seems only TextLoggerHook is supported

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='DistEvalmAPHook')   use --validate in tools/train.py
        # dict(type='TensorboardLoggerHook')
    ])

# Unclear

# yapf:enable
evaluation = dict(interval=1)


######################
#  runtime settings  #
######################

# Number of epochs to ran the total model.
total_epochs = 5000

# Unclear
dist_params = dict(backend='nccl')

# Unclear
log_level = 'INFO'

# Dir to save checkpoints and logs
work_dir = './work_dirs/mask_rcnn_r50_fpn_1x'

# Just loads the parameters of the network, but applies the current configuration in this file.
# Can be 'None' or 'path/to/checkpoint_file.pth', e.g. 'work_dirs/mask_rcnn_r50_fpn_1x/latest.pth'
load_from = './work_dirs/mask_rcnn_r50_fpn_1x/latest.pth'

# Loads everything including optimizer state and all configurations. Ignores changes made in this file.
# Resumes from the epoch specified and continues from there.
# Can be 'None' or 'path/to/checkpoint_file.pth', e.g. 'work_dirs/mask_rcnn_r50_fpn_1x/latest.pth'
resume_from = None#'./work_dirs/mask_rcnn_r50_fpn_1x/latest.pth'

# Unclear
workflow = [('train', 1)]
