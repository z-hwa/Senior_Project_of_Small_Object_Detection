import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import defaultdict

def calculate_iou(bbox1, bbox2):
    """計算兩個 bounding box 的 IoU (Intersection over Union)。
    bbox 格式: [x_min, y_min, width, height]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection_area = intersection_x * intersection_y

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    if union_area == 0:
        return 0
    return intersection_area / union_area

def find_overlapping_predictions(prediction_file1, prediction_file2, output_file):
    """
    找出兩個 COCO 預測結果檔案中重疊的預測框，並將它們保存到一個新的 JSON 檔案中。
    重疊的判定標準是 IoU > 0，且只比較相同 image_id 的預測框。

    Args:
        prediction_file1 (str): 第一個 COCO 預測結果 JSON 檔案路徑。
        prediction_file2 (str): 第二個 COCO 預測結果 JSON 檔案路徑。
        output_file (str): 輸出包含交集預測結果的 JSON 檔案路徑。
    """
    with open(prediction_file1, 'r') as f:
        data1 = json.load(f)
    with open(prediction_file2, 'r') as f:
        data2 = json.load(f)

    predictions1 = data1
    predictions2 = data2

    # 按 image_id 分組預測結果
    grouped_predictions1 = defaultdict(list)
    for pred in predictions1:
        grouped_predictions1[pred['image_id']].append(pred)

    grouped_predictions2 = defaultdict(list)
    for pred in predictions2:
        grouped_predictions2[pred['image_id']].append(pred)

    overlapping_predictions = []
    filtered_count = 0

    for image_id, preds1 in grouped_predictions1.items():
        if image_id in grouped_predictions2:
            preds2 = grouped_predictions2[image_id]
            processed_indices2 = [False] * len(preds2)

            for pred1 in preds1:
                bbox1 = pred1['bbox']
                found_overlap = False

                for i, pred2 in enumerate(preds2):
                    if not processed_indices2[i]:
                        bbox2 = pred2['bbox']
                        iou = calculate_iou(bbox1, bbox2)
                        if iou > 0:
                            overlapping_predictions.append(pred1.copy())
                            processed_indices2[i] = True
                            found_overlap = True
                            break  # 找到一個重疊的框就跳出內層迴圈

                if not found_overlap:
                    filtered_count += 1
        else:
            filtered_count += len(preds1) # 如果 image_id 在第二個檔案中不存在，則所有預測都算作篩掉

    output_data = overlapping_predictions

    with open(output_file, 'w') as f:
        json.dump(output_data, f)

    total_predictions1 = len(predictions1)
    total_predictions2 = len(predictions2)

    return total_predictions1, filtered_count, total_predictions2

if __name__ == "__main__":
    prediction_file1 = "best_result/results_smot4sb_phase2.json"  # 替換為你的第一個預測結果檔案路徑
    # prediction_file2 = "work_dirs/cascade_rcnn_swin_rfla_smot4sb/lr_000005_with_old/results_smot4sb_phase2_finetune.bbox.json"  # 替換為你的第二個預測結果檔案路徑
    prediction_file2 = "work_dirs/cascade_rcnn_swin_rfla_smot4sb/lr_5e-5_with_old_phase2_sample/results_smot4sb_finetune_phase2_data.bbox.json"  # 替換為你的第二個預測結果檔案路徑
    output_file = "intersection_predictions_phase2sample.json"  # 替換為你想要保存交集結果的檔案路徑

    # prediction_file1 = "best_result/results_smot4sb_val.bbox.json"  # 替換為你的第一個預測結果檔案路徑
    # prediction_file2 = "work_dirs/cascade_rcnn_swin_rfla_smot4sb/lr_000005_with_old/results_val_finetune_5e5_old.bbox.json"  # 替換為你的第二個預測結果檔案路徑
    # output_file = "intersection_predictions_val.json"  # 替換為你想要保存交集結果的檔案路徑

    total_predictions1, filtered_predictions, total_predictions2 = find_overlapping_predictions(
        prediction_file1, prediction_file2, output_file
    )

    print(f"第一個預測檔案 ('{prediction_file1}') 的總預測框數: {total_predictions1}")
    print(f"第二個預測檔案 ('{prediction_file2}') 的總預測框數: {total_predictions2}")
    print(f"篩掉的預測框數 (來自 '{prediction_file1}'): {filtered_predictions}")
    print(f"找到的交集預測框數 (來自 '{prediction_file1}'): {len(json.load(open(output_file)))}")
    print(f"交集預測結果已保存到 '{output_file}'。")