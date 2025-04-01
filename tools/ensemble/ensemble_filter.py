from ensemble_boxes import weighted_boxes_fusion
import json
import numpy as np
import argparse
from pycocotools.coco import COCO
# need normalization for bbox coordination
# read in json file and turn it into bboxes lists, scores lists and label lists
# bbox_per_file = {1:[], 2:[], ... 9699:[]}
# bbox_lists = [bbox_per_file1, ...  ]

'''
修改 config.txt 檔案：
在你的 tools/ensemble/config.txt 檔案中，為你希望忽略小預測框的來源檔案路徑後面添加 # FILTER_SMALL 的標記。對於你希望保留所有尺寸預測框的來源，可以不添加標記，或者添加 # NO_FILTER。例如：

model1_predictions.json # NO_FILTER
model_with_small_errors.json # FILTER_SMALL
another_model.json # NO_FILTER
執行腳本：
使用與之前相同的方式執行你的 ensemble 腳本，並通過 --min_size 參數指定你認為的最小尺寸（例如 32）。

Bash

python tools/ensemble/ensemble.py /root/Document/data/MVA2025/annotations/test_coco.json --min_size 32
'''

def normalization(bbox, width=3840, height=2160):
    # Size: 3840x2160 for pub_test dataset
    bbox = [bbox[0]/(width*1.000000000000), bbox[1]/height, bbox[2]/(width*1.000000000), bbox[3]/height]
    return bbox

def denorm(bbox, width=3840, height=2160):
    bbox = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height]
    return bbox

def xywh2xyxy(xywh):
    bbox = [xywh[0], xywh[1], (xywh[0] + xywh[2]), (xywh[1] + xywh[3])]
    return bbox

def xyxy2xywh(xyxy):
    bbox = [xyxy[0], xyxy[1], (xyxy[2] - xyxy[0]), (xyxy[3]-xyxy[1])]
    return bbox

def bbox_formatting(bboxes, scores , image_id, output):
    for i in range(0, len(bboxes)):
        cur_box = dict()
        cur_box['image_id'] = image_id
        cur_box['bbox'] = xyxy2xywh(denorm(bboxes[i]))
        cur_box['score'] = scores[i]
        cur_box['category_id'] = 1
        output.append(cur_box)
    return output



#4567+0.5 #4567+0.55 #3567+0.55 #cascade+0.6
#2457
def ensemble(config_file, output_file, annotation_file, weights=[2,4,5,6,8], iou_thr=0.5, skip_box_thr= 0.001, sigma = 0.1, min_bbox_size=32):
    test_coco = COCO(annotation_file)
    test_img_num = len(test_coco.getImgIds())
    output = []
    files = []
    filter_flags = [] # 儲存每個檔案是否需要過濾的標誌
    json_file = []
    bbox_lists = []
    score_lists = []
    label_lists = []
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("#")
                file_path = parts[0].strip()
                filter_flag = "NO_FILTER"
                if len(parts) > 1:
                    filter_flag = parts[1].strip()
                files.append(file_path)
                filter_flags.append(filter_flag)

    for i, cur_file in enumerate(files):
        json_data = []
        print(cur_file)
        with open(cur_file) as k:
            json_data = json.load(k)
        json_file.append(json_data)

    for idx, jf in enumerate(json_file):
        bbox_per_file = {k: [] for k in test_coco.getImgIds() }
        score_per_file = {k: [] for k in test_coco.getImgIds() }
        label_per_file = {k: [] for k in test_coco.getImgIds() }

        do_filter = (filter_flags[idx] == "FILTER_SMALL")

        for data in jf:
            bbox_xyxy = xywh2xyxy(data['bbox'])
            width = bbox_xyxy[2] - bbox_xyxy[0]
            height = bbox_xyxy[3] - bbox_xyxy[1]

            if do_filter and width < min_bbox_size and height < min_bbox_size:
                continue # 忽略小框
            else:
                bbox_norm = normalization(bbox_xyxy)
                bbox_per_file[data['image_id']].append(bbox_norm)
                score_per_file[data['image_id']].append(data['score'])
                label_per_file[data['image_id']].append(data['category_id'])

        bbox_lists.append(bbox_per_file)
        score_lists.append(score_per_file)
        label_lists.append(label_per_file)

    # ensemble
    for i, image_id in enumerate(test_coco.getImgIds()):
        bbox_cmb, score_cmb, label_cmb = [], [], []
        for idx in range(len(files)):
            bbox_cmb.append(bbox_lists[idx][image_id])
            score_cmb.append(score_lists[idx][image_id])
            label_cmb.append(label_lists[idx][image_id])

        tot_box = 0
        for b in bbox_cmb:
            tot_box += len(b)

        if tot_box > 0:
            pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
                weighted_boxes_fusion( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        # print(pred_bboxes_per_image)
        print('Current Progress: {} / {}'.format(i, test_img_num), end='\r')
        output = bbox_formatting(pred_bboxes_per_image, pred_score_per_image, image_id, output)

    with open(output_file, "w") as f:
        json.dump(output, f)

    return output

parser = argparse.ArgumentParser(description='Ensemble Choices')
parser.add_argument("annotation_file", help="The path to annotation file")
parser.add_argument("--min_size", type=int, default=32, help="Minimum bbox size (width and height) to consider for filtering")
args = parser.parse_args()

ensemble('tools/ensemble/config.txt', 'results_selective_filter.json', args.annotation_file, weights=[2, 4], min_bbox_size=args.min_size)

# python tools/ensemble/ensemble.py data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json
# python tools/ensemble/ensemble.py /root/Document/MVA2025-SMOT4SB/datasets/SMOT4SB/annotations/test_coco.json