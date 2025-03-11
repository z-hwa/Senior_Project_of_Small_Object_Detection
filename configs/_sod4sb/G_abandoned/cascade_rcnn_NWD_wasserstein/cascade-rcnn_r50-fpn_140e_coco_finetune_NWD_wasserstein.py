import os

LIB_ROOT_DIR = '/root/Document/MVA2023SmallObjectDetection4SpottingBirds'

# _base_ = './centernet_resnet18_140e_coco.py'
_base_ = '../../cascade_rcnn/cascade-rcnn_r50_fpn_140e_coco_NWD_wasserstein.py'
data_root = LIB_ROOT_DIR + '/data/'

data = dict(
    train=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[15, 18])
runner = dict(max_epochs=20)
evaluation = dict(interval=20, metric='bbox')
# load_from = LIB_ROOT_DIR + '/work_dirs/centernet_resnet18_140e_coco/latest.pth'
load_from = LIB_ROOT_DIR + '/work_dirs/cascade-rcnn_r50_fpn_140e_coco_NWD_wasserstein/epoch_140.pth'
checkpoint_config = dict(interval=5)