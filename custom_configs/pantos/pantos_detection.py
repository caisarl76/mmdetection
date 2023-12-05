_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'PantosDataset'
data_root = '/data/lx_pantos/'

metainfo = {
    'classes':('압착슬리브_정면', 
               '압착슬리브_미압착',
               '압착슬리브_측면',
               '재사용',
               '구리선',
               '협의_필요',
               '열수축튜브',
               '절연테이프'
               ),
        'palette':
            [
                (220, 20, 60),
                (119, 11, 32),
                (0, 0, 142),
                (0, 0, 230),
                (106, 0, 228),
                (0, 60, 100),
                (0, 80, 100),
                (0, 0, 70)
                ]
    }

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512,512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation_231031/all_annotations.json',
        data_prefix=dict(img='images/'),
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
        ann_file='annotation_231031/1/instances_default.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotation_231031/1/instances_default.json',
    metric=['bbox'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

evaluation = dict(metric=['bbox'])
