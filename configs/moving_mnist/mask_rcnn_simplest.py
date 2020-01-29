import warnings
warnings.filterwarnings("ignore")  # "ignore" or "error"

# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ConvNet',
        in_channels=1,
        hidden_channels=8,
        out_channels=8),
    rpn_head=dict(
        type='RPNHead',
        in_channels=8,
        feat_channels=8,
        anchor_scales=[3],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=32,
        featmap_strides=[2]),
    bbox_head=dict(
        type='BBoxHead',
        in_channels=8,
        roi_feat_size=7,
        num_classes=11,     # 10 digits + 1 for background
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=8,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=2,
        in_channels=8,
        conv_out_channels=8,
        num_classes=11,     # 10 digits + 1 for background
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
        nms_pre=20,
        nms_post=20,
        max_num=20,
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
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
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
    dict(type='Pad', size=(50, 60)),    # height, width, necessary for mask stacking see https://github.com/open-mmlab/mmdetection/issues/2026
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_shape', 'img_shape')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='grayscale'),
    #dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Pad', size=(50, 60)),    # height, width, necessary for mask stacking see https://github.com/open-mmlab/mmdetection/issues/2026
    #dict(type='DefaultFormatBundle'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape', 'img_shape')),
    #dict(type='LoadImageFromFile', color_type='grayscale'),
    #dict(type='Pad', size=(50, 60)),    # height, width, necessary for mask stacking see https://github.com/open-mmlab/mmdetection/issues/2026
    #dict(type='ImageToTensor', keys=['img']),
    #dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_shape')),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(1333, 800),
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='ImageToTensor', keys=['img']),
    #         dict(type='Collect', keys=['img']),
    #     ])
]
data = dict(
    imgs_per_gpu=300,    # batch size
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline)
    )
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=1)
# runtime settings
total_epochs = 500
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/mask_rcnn_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
