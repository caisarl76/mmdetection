_base_ = [
    './faster-rcnn_r50_fpn_1x_.py',
    '../_base_/default_runtime.py'
]

##########################################################################
# model settings
##########################################################################

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=6)))

metainfo = {
    'classes':('slv_front', 
               'slv_wrong',
               'slv_side',
               'copper',
               'tube',
               'tape',
               ),
        'palette':
            [
                (220, 20, 60),
                (119, 11, 32),
                (0, 0, 142),
                (0, 0, 230),
                (106, 0, 228),
                (0, 60, 100),
                ]
    }


##########################################################################
# dataset settings
##########################################################################
dataset_type = 'PantosDataset'
data_root = '/data/lx_pantos/'
data_split = 'annotation_231113'


metainfo = {
    'classes':('slv_front', 
               'slv_wrong',
               'slv_side',
               'copper',
               'tube',
               'tape'
               ),
        'palette':
            [
                (0, 0, 142),
                (0, 60, 100),
                (0, 0, 230),
                (220, 20, 60),
                (119, 11, 32),
                (106, 0, 228),
                ]
    }

backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='%s/val_ver1.json'%data_split,
        data_prefix=dict(img='img_231113/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation_231113/val_ver1.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='/data/lx_pantos/annotation_231113/')


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
