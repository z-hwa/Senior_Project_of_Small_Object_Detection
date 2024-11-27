from sahi.scripts.coco_evaluation import evaluate

evaluate(
    dataset_json_path="data/mva2023_sod4bird_train/annotations/split_val_coco.json",
    result_json_path="runs/predict/cascade_mask_rcnn_swin_finetune_rfla_double_paste/result.json",
    out_dir="runs/predict/cascade_mask_rcnn_swin_finetune_rfla_double_paste/",
    # iou_thrs=[0.25, 0.5, 0.75],
    max_detections=1000,
    areas=[1024, 9216, 10000000000],
    classwise=False,
    return_dict=True,
)