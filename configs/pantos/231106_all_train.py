_base_ = [
    './faster-rcnn_r50_fpn_1x_.py',
    '../_base_/datasets/pantos_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=6)))

# dataset settings
dataset_type = 'PantosDataset'
data_root = '/data/lx_pantos/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
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
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation_231106/all_train.json',
        data_prefix=dict(img='img_231106/all_cls/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[8, 11, 18],
        gamma=0.1)
]



vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='/data/lx_pantos/annotation_231106/')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50))
# Optional: set moving average window size
log_processor = dict(
    type='LogProcessor', window_size=50)


# data = dict(
#     train=dict(
#         dataset=dict(
#             classes=
#             ('1','압착슬리브_정면'),
#             ('2','압착슬리브_미압착'),
#             ('3','압착슬리브_측면'),
#             ('4','재사용'),
#             ('5','구리선'),
#             ('6','협의_필요'),
#             ('7','열수축튜브'),
#             ('8','절연테이프'),
#             )))