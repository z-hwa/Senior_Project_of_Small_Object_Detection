'''
讀取預測json檔案
並針對當前處理的帧數 找到對應的前後數帧
每個bbox進行配對 針對前後數帧 每次配對成功給予正面 失敗給予負面
當負面比正面高的時候 將該bbox移除

並重新生成json預測檔案
'''
import json
import numpy as np
import os
import math

def load_json(file_path):
    """載入 JSON 檔案"""
    with open(file_path, "r") as f:
        return json.load(f)

def convert_coco_predictions(coco_predictions, empty_annotations):
    """
    將 COCO 預測結果轉換為 {file_name: list of numpy arrays} 結構。

    :param coco_predictions: COCO 格式的預測結果 (list of dicts)
    :param empty_annotations: 原始空標註資料，應包含 {"images": [{"id": X, "file_name": "path/to/image"}, ...]}
    :return: dict，key 為 file_name，value 為該圖片的預測結果 numpy array list
    """
    # 建立 image_id 到 file_name 的映射
    image_id_to_file = {img["id"]: img["file_name"] for img in empty_annotations["images"]}

    # 建立 {file_name: {category_id: list of numpy arrays}} 結構
    results = {}
    for pred in coco_predictions:
        image_id = pred["image_id"]
        file_name = image_id_to_file.get(image_id, None)
        if file_name is None:
            continue

        bbox = pred["bbox"]  # [x, y, w, h]
        score = pred["score"]
        category_id = pred["category_id"]

        # 轉換為 [x1, y1, x2, y2, score]，符合 MMDetection 格式
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        bbox_data = np.array([x1, y1, x2, y2, score])

        # 初始化該圖片的分類列表
        if file_name not in results:
            results[file_name] = {}

        # 把 bbox 存入對應類別的 list
        if category_id not in results[file_name]:
            results[file_name][category_id] = []

        results[file_name][category_id].append(bbox_data)

    # 把 list 轉成 numpy array，確保格式一致
    for file_name in results:
        for category_id in results[file_name]:
            results[file_name][category_id] = np.array(results[file_name][category_id])

    return results

def distance_compare(det_a, det_b, metric):
    # 超過單張圖片的大小
    d = 4000

    if metric == 'e_dis':
        a_center_x = (det_a[0] + det_a[2]) / 2
        a_center_y = (det_a[1] + det_a[3]) / 2
        b_center_x = (det_b[0] + det_b[2]) / 2
        b_center_y = (det_b[1] + det_b[3]) / 2

        d = math.sqrt((a_center_x - b_center_x)**2 + (a_center_y - b_center_y)**2)

    return d

def save_coco_predictions(predictions, output_path, image_id_to_name):
    """將過濾後的預測結果保存為 COCO JSON 格式"""
    coco_output = []
    for image_name, category_preds in predictions.items():
        if image_name in image_id_to_name:
            image_id = image_id_to_name[image_name]
            for category_id, bboxes in category_preds.items():
                for bbox_data in bboxes:
                    x1, y1, x2, y2, score = bbox_data
                    w = x2 - x1
                    h = y2 - y1
                    coco_output.append({
                        'image_id': image_id,
                        'bbox': [x1, y1, w, h],
                        'score': score,
                        'category_id': int(category_id)
                    })

    with open(output_path, 'w') as f:
        json.dump(coco_output, f)
    print(f"Filtered predictions saved to {output_path}")

pred_path = 'work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_smot4sb_val.bbox.json'
empty_path = '/root/Document/data/SMOT4SB/annotations/val.json'
output_path = 'filtered_results_smot4sb_val.json'

# pred_path = 'work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_smot4sb_phase2.json'
# empty_path = '/root/Document/data/MVA2025/annotations/test_coco.json'
# output_path = 'filtered_results.json'
voter_range = {'prev': 3, 'fut': 2}

if __name__ == '__main__':
    pred_anno = load_json(file_path=pred_path)
    empty_anno = load_json(file_path=empty_path)

    image_name_to_id = {img['file_name']: img['id'] for img in empty_anno['images']}
    id_to_image_name = {img['id']: img['file_name'] for img in empty_anno['images']}

    pred_dict = convert_coco_predictions(pred_anno, empty_anno)

    filtered_predictions_dict = {}

    for filename, category_preds in pred_dict.items():
        current_frame_num_str = os.path.splitext(filename.split('/')[-1])[0]
        try:
            current_frame_num = int(current_frame_num_str)
        except ValueError:
            print(f"Warning: Could not parse frame number from filename: {filename}")
            continue

        filtered_predictions_dict[filename] = {}
        for c_id, detections in category_preds.items():
            matched_detections = []
            for det_idx, det in enumerate(detections):
                positive_votes = 0
                negative_votes = 0

                # 前向檢查
                for i in range(1, voter_range['prev'] + 1):
                    prev_frame_num = current_frame_num - i
                    if prev_frame_num > 0:
                        prev_frame_num_str = f"{prev_frame_num:05d}"
                        sub_path_parts = filename.split('/')
                        sub_path_parts[-1] = prev_frame_num_str + os.path.splitext(filename.split('/')[-1])[1]
                        prev_filename = "/".join(sub_path_parts)

                        if prev_filename in pred_dict and c_id in pred_dict[prev_filename]:
                            found_match = False
                            for prev_det in pred_dict[prev_filename][c_id]:
                                n_w = (max(abs(det[0] - det[2]), abs(det[1] - det[3])) + max(abs(prev_det[0] - prev_det[2]), abs(prev_det[1] - prev_det[3]))) / 2 * i
                                d = distance_compare(det, prev_det, 'e_dis')
                                if d < n_w:
                                    positive_votes += 1
                                    found_match = True
                                    break
                            if not found_match:
                                negative_votes += 1
                        else:
                            negative_votes += 1  # 前一幀不存在

                # 後向檢查
                for i in range(1, voter_range['fut'] + 1):
                    fut_frame_num = current_frame_num + i
                    fut_frame_num_str = f"{fut_frame_num:05d}"
                    sub_path_parts = filename.split('/')
                    sub_path_parts[-1] = fut_frame_num_str + os.path.splitext(filename.split('/')[-1])[1]
                    fut_filename = "/".join(sub_path_parts)

                    if fut_filename in pred_dict and c_id in pred_dict[fut_filename]:
                        found_match = False
                        for fut_det in pred_dict[fut_filename][c_id]:
                            n_w = (max(abs(det[0] - det[2]), abs(det[1] - det[3])) + max(abs(fut_det[0] - fut_det[2]), abs(fut_det[1] - fut_det[3]))) / 2 * i
                            d = distance_compare(det, fut_det, 'e_dis')
                            if d < n_w:
                                positive_votes += 1
                                found_match = True
                                break
                        if not found_match:
                            negative_votes += 1
                    else:
                        negative_votes += 1  # 後一幀不存在

                if positive_votes >= negative_votes:
                    matched_detections.append(det.tolist())

            if matched_detections:
                filtered_predictions_dict[filename][c_id] = np.array(matched_detections)

            # breakpoint()

    save_coco_predictions(filtered_predictions_dict, output_path, image_name_to_id)