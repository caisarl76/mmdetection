_base_ = [
    '/root/mmdetection/configs/pantos/faster-rcnn_r50_fpn_1x_.py',
    '../_base_/datasets/pantos_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8)))

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