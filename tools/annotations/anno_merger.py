import json
import random
import os

def mix_coco_predictions_with_id_remap(filter_json_path, old_train_json_path, output_json_path, num_old_images_to_sample=None):
    """
    混合兩個 COCO 格式的 JSON 預測檔案，並重新維護 image 和 annotation 的 id。
    對於混合後的圖片，根據其來源調整 'file_name' 前綴。
    來自 old_train_json_path 且檔名開頭沒有目錄結構的，加上 'MVA2023/train/'。
    來自 filter_json_path 的圖片，加上 'MVA2025/train/'。
    來自 old_train_json_path 的標註，其 'category_id' 將被強制設定為 1。

    Args:
        filter_json_path (str): 包含模型易錯資料的 JSON 檔案路徑。
        old_train_json_path (str): 包含過去訓練資料的 JSON 檔案路徑。
        output_json_path (str): 輸出混合後 JSON 檔案的路徑。
        num_old_images_to_sample (int, optional): 從舊訓練資料中隨機抽取的圖片數量。
                                                    如果為 None，則使用所有舊訓練資料。
    """
    with open(filter_json_path, 'r') as f:
        filter_data = json.load(f)

    with open(old_train_json_path, 'r') as f:
        old_train_data = json.load(f)

    # 初始化混合資料
    mixed_data = {
        'info': filter_data.get('info', []),
        'licenses': filter_data.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': list(filter_data.get('categories', []))
    }

    old_images = old_train_data.get('images', [])
    old_annotations = old_train_data.get('annotations', [])
    filter_images = filter_data.get('images', [])
    filter_annotations = filter_data.get('annotations', [])

    sampled_old_images = []
    sampled_old_image_ids = set()
    sampled_old_annotations = []

    if num_old_images_to_sample is not None and num_old_images_to_sample < len(old_images):
        sampled_old_images = random.sample(old_images, num_old_images_to_sample)
        sampled_old_image_ids = {img['id'] for img in sampled_old_images}
        sampled_old_annotations = [ann for ann in old_annotations if ann['image_id'] in sampled_old_image_ids]
    else:
        sampled_old_images = old_images
        sampled_old_image_ids = {img['id'] for img in sampled_old_images}
        sampled_old_annotations = old_annotations

    # 添加 filter_data 的圖片和標註，並添加前綴 'MVA2025/train/'
    for img in filter_images:
        img['file_name'] = 'MVA2025/train/' + img['file_name']
        mixed_data['images'].append(img)
    mixed_filter_image_ids = {img['id'] for img in filter_images}
    for ann in filter_annotations:
        if ann['image_id'] in mixed_filter_image_ids:
            mixed_data['annotations'].append(ann)

    # 添加抽樣後的舊資料，並根據檔名結構添加前綴 'MVA2023/train/'，且標註類別改為 1
    for img in sampled_old_images:
        if img['id'] not in mixed_filter_image_ids:
            if '/' not in img['file_name']:
                img['file_name'] = 'MVA2023/train/images/' + img['file_name']
            mixed_data['images'].append(img)
    mixed_current_image_ids = {img['id'] for img in mixed_data['images']}
    for ann in sampled_old_annotations:
        if ann['image_id'] in mixed_current_image_ids and ann['image_id'] not in mixed_filter_image_ids:
            ann['category_id'] = 1  # 強制將舊資料的標註類別改為 1
            mixed_data['annotations'].append(ann)

    # 保留 filter_data 的 categories，並確保包含類別 1
    if mixed_data['categories'] is None:
        mixed_data['categories'] = []
    category_ids = {cat['id'] for cat in mixed_data['categories']}
    if 1 not in category_ids:
        mixed_data['categories'].append({'id': 1, 'name': 'your_old_category_name'}) # 你可能需要修改 'your_old_category_name'

    # 重新維護 image id
    image_id_map = {}
    new_images = []
    new_image_id_counter = 1
    for img in mixed_data['images']:
        old_id = img['id']
        img['id'] = new_image_id_counter
        image_id_map[old_id] = new_image_id_counter
        new_images.append(img)
        new_image_id_counter += 1
    mixed_data['images'] = new_images

    # 重新維護 annotation id 和 image_id
    new_annotations = []
    new_annotation_id_counter = 1
    for ann in mixed_data['annotations']:
        ann['id'] = new_annotation_id_counter
        ann['image_id'] = image_id_map.get(ann['image_id'], -1)
        if ann['image_id'] != -1:
            new_annotations.append(ann)
            new_annotation_id_counter += 1
    mixed_data['annotations'] = new_annotations

    # 寫入新的 JSON 檔案
    with open(output_json_path, 'w') as f:
        json.dump(mixed_data, f)

    print(f"混合後的圖片總張數: {len(mixed_data['images'])}")

if __name__ == "__main__":
    filter_json_path = "/root/Document/data/MVA2025/annotations_phase2/train_filter_only2.json"
    old_train_json_path = "/root/Document/data/MVA2023/train/annotations/split_train_coco.json"
    output_json_path = "/root/Document/data/SMOT4SB/annotations/mixed_train_coco_only2.json"  # 設定輸出的檔案路徑
    num_old_images_to_sample = 3000  # 設定要從舊資料中抽取的圖片數量

    # 確保輸出路徑的目錄存在
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mix_coco_predictions_with_id_remap(filter_json_path, old_train_json_path, output_json_path, num_old_images_to_sample)

    print(f"已成功將 '{filter_json_path}' 的所有內容與 '{old_train_json_path}' 中隨機抽取的 {num_old_images_to_sample} 張圖片及其標註混合，並重新維護了 image 和 annotation 的 id。來自舊資料的圖片（檔名無目錄結構）加上 'MVA2023/train/' 前綴，來自易錯資料的圖片加上 'MVA2025/train/' 前綴。來自舊資料的標註類別已強制設定為 1，保存到 '{output_json_path}'。")