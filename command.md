# fine-tuning on data/mva2023_sod4bird_train
bash tools/train.sh configs/_smot4sb/Swin_4stage/cascade_rcnn_swin_rfla_smot4sb.py

# 比較結果指令
python tools/analysis_tools/analyze_results.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic_val.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/result_val/result_val.pkl work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/result_val/ --topk 100

python tools/analysis_tools/analyze_results.py work_dirs/cascade_mask_rcnn_swin_finetune_rfla/cascade_mask_rcnn_swin_finetune_rfla.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune/result_train/result_train.pkl work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune/result_train/ --topk 100

# 模型進行辨識後圖片展示的指令
python tools/test.py configs/_smot4sb/Time_Consider/stack_cascade_rcnn_swin_rfla_4stage.py work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/epoch_104.pth --show

# 查看訓練歷史
### 儲存為檔案 
python ./tools/analysis_tools/analyze_logs.py plot_curve work_dirs/flow_cascade_rcnn_cos_swin_rfla_4stage/20250324_121745.log.json --keys loss
### 直接展示
python ./tools/analysis_tools/analyze_logs.py work_dirs/cascade_rcnn_swin_rfla_4stage_focalLoss/20250312_222425.log --keys loss loss_rpn_cls loss_rpn_bbox

# 所有定位相關的loss
python ./tools/analysis_tools/analyze_logs.py plot_curve work_dirs/cascade_rcnn_swin_rfla_smot4sb/20250426_172642.log.json --keys loss loss_rpn_bbox s0.loss_bbox s1.loss_bbox s2.loss_bbox

### 生成centernet的pkl
python tools/test.py configs/_smot4sb/Time_Consider/flow_cascade_rcnn_swin_rfla_4stage.py work_dirs/flow_cascade_rcnn_swin_rfla_4stage/epoch_5.pth --out result_val.pkl
python tools/test.py work_dirs/flow_cascade_rcnn_swin_rfla_4stage/flow_cascade_rcnn_swin_rfla_4stage.py work_dirs/flow_cascade_rcnn_swin_rfla_4stage/epoch_20.pth $GPU_NUM --out result_val.pkl

### browse dataset
python tools/misc/browse_dataset.py configs/_MyPlan/cascade_rcnn_finetune/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800.py [--show-interval ${SHOW_INTERVAL}]
python tools/misc/browse_dataset.py configs/_smot4sb/Time_Consider/cascade_mask_rcnn_swin_finetune_rfla_4stage.py --show 0

### fusion
python tools/ensemble/ensemble.py data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json

### eval metric from pkl file
python tools/analysis_tools/eval_metric.py work_dirs/flow_cascade_rcnn_swin_rfla_4stage/flow_cascade_rcnn_swin_rfla_4stage.py result_val.pkl --eval bbox