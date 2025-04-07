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
python tools/ensemble/ensemble.py /root/Document/data/MVA2025/annotations/test_coco.json

1. 引入函式庫：

ensemble_boxes.weighted_boxes_fusion: 用於執行加權框融合的核心函式。
json: 用於讀取和寫入 JSON 格式的檔案。
numpy: 用於數值計算。
argparse: 用於解析命令列參數。
pycocotools.coco.COCO: 用於處理 COCO 格式的 annotation 檔案。
2. 輔助函式：

normalization(bbox, width=3840, height=2160): 將邊界框的坐標從絕對像素值歸一化到 0-1 的範圍內。這通常是為了讓不同模型輸出的邊界框在相同的尺度上進行融合。程式碼中寫死的寬高是 3840x2160，這可能是針對特定的資料集。
denorm(bbox, width=3840, height=2160): 將歸一化後的邊界框坐標還原為絕對像素值。
xywh2xyxy(xywh): 將以 (x_center, y_center, width, height) 格式表示的邊界框轉換為 (x_min, y_min, x_max, y_max) 格式。
xyxy2xywh(xyxy): 將以 (x_min, y_min, x_max, y_max) 格式表示的邊界框轉換為 (x_center, y_center, width, height) 格式。
bbox_formatting(bboxes, scores , image_id, output): 將融合後的邊界框、分數和圖像 ID 整理成 COCO 格式的檢測結果字典，並添加到 output 列表中。
3. ensemble 函式：

config_file: 指定一個包含多個模型預測結果 JSON 檔案路徑的文字檔案。每一行代表一個檔案路徑，以 # 開頭的行會被忽略。
output_file: 指定融合後的結果要寫入的 JSON 檔案路徑。
annotation_file: COCO 格式的 annotation 檔案路徑，用於獲取測試圖像的 ID 列表。
weights: 一個列表，指定每個模型預測結果在融合時的權重。權重越高，該模型的預測結果在融合時的影響越大。預設權重是 [2, 4, 5, 6, 8]。
iou_thr: IoU (Intersection over Union) 閾值，用於判斷哪些預測框應該被視為同一個目標。
skip_box_thr: 低於這個分數的預測框將被忽略。
sigma: 這個參數在程式碼中雖然有定義，但在預設使用的 weighted_boxes_fusion 方法中並沒有直接用到。它可能是為另一個融合方法（註解中的 soft_nms）預留的參數。
ensemble 函式的處理流程：

讀取設定檔： 從 config_file 中讀取所有模型預測結果的檔案路徑。
讀取模型預測結果： 遍歷每個檔案路徑，讀取 JSON 格式的預測結果，並將每個模型的預測結果分別存儲在 bbox_lists、score_lists 和 label_lists 中。這些列表的結構是：
bbox_lists: 一個列表，每個元素是一個字典，字典的 key 是圖像 ID，value 是該圖像上該模型預測的所有歸一化後的邊界框列表。
score_lists: 結構與 bbox_lists 類似，存儲的是對應的預測分數。
label_lists: 結構與 bbox_lists 類似，存儲的是對應的類別 ID。
進行融合： 遍歷測試集中的每一張圖像 ID：
從 bbox_lists、score_lists 和 label_lists 中提取所有模型對當前圖像的預測結果。
使用 weighted_boxes_fusion 函式將這些預測結果融合成一組最終的預測框、分數和標籤。融合的過程會考慮到每個模型的權重和預測框之間的 IoU。
格式化輸出： 將融合後的預測結果（邊界框從歸一化還原為絕對像素值，並轉換為 xywh 格式）整理成 COCO 格式的字典，並添加到 output 列表中。
寫入輸出檔案： 將最終的融合結果以 JSON 格式寫入 output_file。
4. 命令列參數解析：

使用 argparse 建立一個命令列參數解析器，用於接收 annotation 檔案的路徑。
5. 呼叫 ensemble 函式：

在程式碼的最後，呼叫 ensemble 函式，並傳入設定檔路徑 ('tools/ensemble/config.txt')、輸出檔案名 ('results_smot4sb.json')、從命令列參數獲取的 annotation 檔案路徑 (args.annotation_file) 以及指定的權重 [2, 7, 8, 9]。
總結來說，這個腳本的作用是：

讀取多個目標檢測模型在測試集上的預測結果（以 COCO 格式的 JSON 檔案儲存）。
將這些預測結果中的邊界框歸一化。
針對每一張測試圖像，使用加權框融合算法（weighted_boxes_fusion）將來自不同模型的預測結果合併成一組更可靠的預測結果。 融合的過程會考慮到每個模型的權重以及預測框之間的重疊程度。
將融合後的預測結果反歸一化，並轉換為 COCO 格式。
將最終的融合結果儲存到一個 JSON 檔案中。
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
        cur_box['category_id'] = 0
        output.append(cur_box)
    return output



#4567+0.5 #4567+0.55 #3567+0.55 #cascade+0.6
#2457
def ensemble(config_file, output_file, annotation_file, weights=[2,4,5,6,8], iou_thr=0.5, skip_box_thr= 0.001, sigma = 0.1): #0.01
    test_coco = COCO(annotation_file)
    test_img_num = len(test_coco.getImgIds())
    output = [] # final result
    files = []
    json_file = []
    bbox_lists = []
    score_lists = []
    label_lists = []
    with open(config_file) as f:
        files = f.read().splitlines()
    
    files = [item for item in files if not item.startswith("#")]

    for cur_file in files:
        json_data = []
        print(cur_file)
        with open(cur_file) as k:
            json_data = json.load(k)
        json_file.append(json_data)

    for jf in json_file:
        bbox_per_file = {k: [] for k in test_coco.getImgIds() }
        score_per_file = {k: [] for k in test_coco.getImgIds() }
        label_per_file = {k: [] for k in test_coco.getImgIds() }

        for data in jf:
            bbox_norm = normalization(xywh2xyxy(data['bbox']))
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
        for i in range(0, len(bbox_cmb)):
            tot_box += len(bbox_cmb[i])

        if tot_box > 0:
            pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
                weighted_boxes_fusion( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            # elif method == 'snms':
            #     pred_bboxes_per_image, pred_score_per_image, pred_label_per_image = \
            #         soft_nms( bbox_cmb, score_cmb, label_cmb, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
        # print(pred_bboxes_per_image)
        print('Current Progress: {} / {}'.format(i, test_img_num), end='\r')
        output = bbox_formatting(pred_bboxes_per_image, pred_score_per_image, image_id, output)
    
    with open(output_file, "w") as f:
        json.dump(output, f)
    
    return output

parser = argparse.ArgumentParser(description='Ensemble Choices')
parser.add_argument("annotation_file", help="The path to annotation file")
args = parser.parse_args()


ensemble('tools/ensemble/config.txt', 'results_smot4sb.json', args.annotation_file, weights=[2, 4])