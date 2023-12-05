_base_ = [
    '../common/ms-poly_3x_pantos-instance.py',
    '../_base_/models/mask-rcnn_r50_fpn.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)
        ),
)

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