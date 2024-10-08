#mva2
#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO 
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
GPU_NUM=1

###############################
# Step 1: normal training on data/drone2021
###############################
echo "###############################"
echo "Step 1: normal training on data/drone2021"
echo "###############################"
bash tools/dist_train.sh  configs/_MyPlan/cascade_rcnn/cascade_internimage_xl_fpn_100e_coco_nwd.py $GPU_NUM


###############################
# Step 2: fine-tuning on data/mva2023_sod4bird_train
###############################
echo "###############################"
echo "Step 2: fine-tuning on data/mva2023_sod4bird_train"
echo "###############################"
bash tools/dist_train.sh  configs/_MyPlan/cascade_rcnn_finetune/cascade_internimage_xl_fpn_100e_coco_nwd_finetune_800.py $GPU_NUM


###############################
# Step 3: Generate predictions on data/mva2023_sod4bird_train to select hard negatives examples
###############################
echo "###############################"
echo "Step 3: Generate predictions on data/mva2023_sod4bird_train to select hard negatives examples"
echo "###############################"
# CONFIG=configs/mva2023_baseline/centernet_resnet18_140e_coco_sample_hard_negative.py
#CHECKPOINT=work_dirs/centernet_resnet18_140e_coco_finetune/latest.pth
CONFIG=configs/_MyPlan/cascade_rcnn_mva/cascade_rcnn_r50_fpn_1x_coco_sample_hard_negative.py
CHECKPOINT=work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800/epoch_40.pth
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPU_NUM \
#     --master_port=$PORT \
#  hard_neg_example_tools/test_hard_neg_example.py \
#     --config $CONFIG \
#     --checkpoint $CHECKPOINT \
#     --launcher pytorch \
#     --generate-hard-negative-samples True \
#     --hard-negative-file work_dirs/centernet_resnet18_140e_coco_finetune/train_coco_hard_negative.json \
#     --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPU_NUM \
    --master_port=$PORT \
 hard_neg_example_tools/test_hard_neg_example.py \
    --config $CONFIG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    --generate-hard-negative-samples True \
    --hard-negative-file work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800/train_coco_hard_negative.json \
    --hard-negative-config num_max_det=10 pos_iou_thr=1e-5 score_thd=0.05 nms_thd=0.05
    
# The setting for 'hard-negative-config' is the default setting for generating the hard negative examples. Please feel free to modify it.
# --------------------------

###############################
# Step 4: Hard negative training  on data/mva2023_sod4bird_train
###############################
echo "###############################"
echo "Step 4: Hard negative training  on data/mva2023_sod4bird_train"
echo "###############################"
bash tools/dist_train.sh  configs/_MyPlan/cascade_rcnn_NWD_wasserstein/cascade-rcnn_r50-fpn_140e_coco_hard_negative_training.py $GPU_NUM


###############################
# Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json.
# 要上傳的是.json壓縮的檔案
###############################
echo "###############################"
echo "Step 5: To generate the predictions for submission, the result will be saved in results.bbox.json."
echo "###############################"
bash tools/dist_test.sh work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic2/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic2.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic2/epoch_72.pth \
1 --format-only --eval-options jsonfile_prefix=results

_time=`date +%Y%m%d%H%M`
mkdir -p submit/${_time}
SUBMIT_FILE=`echo ./submit/${_time}/results.json`
SUBMIT_ZIP_FILE=`echo ${SUBMIT_FILE//results.json/submit.zip}`
mv ./results.bbox.json $SUBMIT_FILE
zip $SUBMIT_ZIP_FILE $SUBMIT_FILE

###############################
# my code
###############################

# Single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--show]

# 同時顯示標註和預測結果
python tools/analysis_tools/analyze_results.py \
      ${CONFIG} \  
      ${PREDICTION_PATH} \
      ${SHOW_DIR} \
      [--show] \
      [--wait-time ${WAIT_TIME}] \
      [--topk ${TOPK}] \
      [--show-score-thr ${SHOW_SCORE_THR}] \
      [--cfg-options ${CFG_OPTIONS}]

# {CONFIG}： 是config文件路径
# {PREDICTION_PATH}： 是test.py得到的pkl文件路径
# {SHOW_DIR}： 是绘制的到的图片存放的目录
# --show： 决定是否显示，不指定的话，默认为不显示
# --wait-time时间的间隔，若为 0 表示持续显示
# --topk: 根据最高或最低 topk 概率排序保存的图片数量，若不指定，默认设置为 20
# --show-score-thr: 能够展示的概率阈值，默认为 0
# --cfg-options: 如果指定，可根据指定键值对覆盖更新配置文件的对应选项

# 比較結果指令
python tools/analysis_tools/analyze_results.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic_val.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/result_val/result_val.pkl work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/result_val/ --topk 100

python tools/analysis_tools/analyze_results.py configs/_MyPlan/cascade_rcnn_finetune/cascade_rcnn_r50_fpn_1x_coco_finetune.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune/result_train/result_train.pkl work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune/result_train/ --topk 100

# 模型進行辨識後圖片展示的指令
python tools/test.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic_val.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/epoch_100.pth --show

# 查看訓練歷史
### 儲存為檔案 
python ./tools/analysis_tools/analyze_logs.py plot_curve work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/20240629_131115.log.json --out vis_log --keys loss
### 直接展示
python ./tools/analysis_tools/analyze_logs.py plot_curve work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/20240629_131115.log.json --keys loss

# 所有定位相關的loss
python ./tools/analysis_tools/analyze_logs.py plot_curve work_dirs/cascade_internimage_xl_fpn_100e_coco_nwd/20240825_154123.log.json --keys loss loss_rpn_bbox s0.loss_bbox s1.loss_bbox s2.loss_bbox

### 生成centernet的pkl
python tools/test.py work_dirs/cascade_internimage_xl_fpn_100e_coco_nwd_finetune/cascade_internimage_xl_fpn_100e_coco_nwd_finetune.py work_dirs/cascade_internimage_xl_fpn_100e_coco_nwd_finetune/epoch_40.pth --out result.pkl
python tools/test.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic_val.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/epoch_100.pth --out result_val.pkl

### browse dataset
python tools/misc/browse_dataset.py configs/_MyPlan/cascade_rcnn_finetune/cascade_rcnn_r50_fpn_1x_coco_finetune_RC_800800.py [--show-interval ${SHOW_INTERVAL}]
python tools/misc/browse_dataset.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic/cascade_rcnn_r50_fpn_1x_coco_finetune_automatic_val.py --show 0
