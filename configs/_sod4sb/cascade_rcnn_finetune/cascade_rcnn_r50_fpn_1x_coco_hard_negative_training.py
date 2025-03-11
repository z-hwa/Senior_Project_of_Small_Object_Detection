import os
_base_ = './cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800.py'
data_root = "/root/Document/MVA2023SmallObjectDetection4SpottingBirds" + '/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'), 
    dict(type='LoadAnnotations', with_bbox=True), 
    dict(type='LoadHardNegatives'),  # --
    dict(type='RandomCrop', crop_size=(800, 800), allow_negative_crop=False),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        hard_negative_file='/root/Document/MVA2023SmallObjectDetection4SpottingBirds' + 'work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800/train_coco_hard_negative.json',  # ---
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_train_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    val=dict(
        ann_file=data_root + 'mva2023_sod4bird_train/annotations/split_val_coco.json',
        img_prefix=data_root + 'mva2023_sod4bird_train/images/',
    ),
    test=dict(
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
load_from = "work_dirs/cascade_rcnn_r50_fpn_1x_coco_lr_1024/41to50/epoch_10.pth"
