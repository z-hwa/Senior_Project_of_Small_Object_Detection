_base_ = '../cascade_rcnn_finetune/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800.py'
data_root = "/root/Document/MVA2023SmallObjectDetection4SpottingBirds" + '/data/'

data = dict(
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json',
        img_prefix=data_root + 'mva2023_sod4bird_pub_test/images/',
    ) 
)

