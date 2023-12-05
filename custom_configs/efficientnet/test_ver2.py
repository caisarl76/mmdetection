_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/pantos_detection.py', '../_base_/default_runtime.py'
]

##########################################################################
# model settings
##########################################################################

image_size = (896, 896)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_classes=5, num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

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


##########################################################################
# dataset settings
##########################################################################
dataset_type = 'PantosDatasetVer2'
data_root = '/data/lx_pantos/'
data_split = 'annotation_231115'

backend_args = None
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(batch_size=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader


test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation_231115/val_ver2.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='/data/lx_pantos/annotation_231115/')


##########################################################################
# schedule settings
##########################################################################

test_cfg = dict(type='TestLoop')

##########################################################################
# runtime settings
##########################################################################

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
