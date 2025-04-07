import os
dataset_type = 'DroneDataset'  
data_root = '/root/Document/data/'

# 修改增加多尺度輸入 2024.10.2
# 復原2024.10.3

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadPastAnnotations', 
        pred_path='work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_smot4sb_train.bbox.json',
        empty_path= data_root + 'SMOT4SB/annotations/train.json',
        stack_threshold=0.9,),
    dict(type='LoadAnnotations', with_bbox=True), 
    dict( 
        type='MVARandomCrop',
        crop_size=(800,800),
        must_include_bbox_ratio=0.6),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(800,800), keep_ratio=True, override=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'past_prediction'])
]

val_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadPastAnnotations', 
        pred_path= 'work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_smot4sb_val.bbox.json',
        empty_path= data_root + 'SMOT4SB/annotations/val.json',
        stack_threshold=0.9,),
    dict(
    type='MultiScaleFlipAug',
    scale_factor=1.0,
    flip=False,
    transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            meta_keys=('filename', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'img_norm_cfg', 'past_prediction'),
            keys=['img'])
    ])
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadPastAnnotations', 
        pred_path='work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_smot4sb_phase2.json',
        empty_path='/root/Document/data/MVA2025/annotations/test_coco.json',
        stack_threshold=0.9,),
    dict(
    type='MultiScaleFlipAug',
    scale_factor=1.0,
    flip=False,
    transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            meta_keys=('filename', 'ori_shape', 'img_shape',
                        'scale_factor', 'flip', 'flip_direction',
                        'img_norm_cfg', 'past_prediction'),
            keys=['img'])
    ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'SMOT4SB/annotations/train.json',
        img_prefix=data_root + 'SMOT4SB/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'SMOT4SB/annotations/val.json',
        img_prefix=data_root + 'SMOT4SB/val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file = '/root/Document/data/MVA2025/annotations/test_coco.json',
        img_prefix=
        '/root/Document/data/MVA2025/pub_test',  
        pipeline=test_pipeline))