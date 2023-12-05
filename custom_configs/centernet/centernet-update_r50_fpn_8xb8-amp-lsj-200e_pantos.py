

image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
    ),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

max_epochs = 25

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01 * 4, momentum=0.9, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0.))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.00025,
        by_epoch=False,
        begin=0,
        end=4000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

##########################################################################
# dataset settings
##########################################################################
dataset_type = 'PantosDatasetVer2'
data_root = '/data/lx_pantos/'
data_split = 'annotation_231113'


metainfo = {
    'classes':('slv_right', 
               'slv_wrong',
               'copper',
               'tube',
               'tape'
               ),
        'palette':
            [
                (0, 0, 142),
                (0, 60, 100),
                (220, 20, 60),
                (119, 11, 32),
                (106, 0, 228),
                ]
    }

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='%s/train_ver2.json'%data_split,
        data_prefix=dict(img='img_231113_ver2/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='%s/val_ver2.json'%data_split,
        data_prefix=dict(img='img_231113_ver2/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation_231113/val_ver2.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

