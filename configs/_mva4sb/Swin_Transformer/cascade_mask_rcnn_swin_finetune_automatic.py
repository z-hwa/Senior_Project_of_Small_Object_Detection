_base_ = [
    '../../_base_/models/cascade_mask_rcnn_swin_fpn_nwd.py',
    './mva4sb_automatic.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth"

model = dict(
    backbone=dict(
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.2,
        patch_norm=True,
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)

# # do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=1,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 1000,
    step=[16, 19])

runner = dict(max_epochs=140)

# 學習率調整策略
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=5, metric='bbox')
checkpoint_config = dict(interval=1)

# load_from = "work_dirs/cascade_mask_rcnn_swin_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco/epoch_40.pth"
load_from = "work_dirs/cascade_mask_rcnn_swin_finetune/epoch_55.pth"
# resume_from = "work_dirs/cascade_mask_rcnn_swin_finetune/epoch_1.pth"