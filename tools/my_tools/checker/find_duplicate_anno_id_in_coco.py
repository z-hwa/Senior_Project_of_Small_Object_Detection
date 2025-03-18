import json
from collections import Counter

def find_duplicate_ann_ids(ann_file):
    """找出 COCO 格式資料集中重複的 annotations ID。

    Args:
        ann_file (str): COCO 格式標註檔案的路徑。

    Returns:
        list[dict]: 包含重複 annotations ID 資訊的列表。
    """

    with open(ann_file, 'r') as f:
        data = json.load(f)

    annotations = data['annotations']
    images = data['images']

    ann_ids = [ann['id'] for ann in annotations]
    counts = Counter(ann_ids)
    duplicates = [ann_id for ann_id, count in counts.items() if count > 1]

    if duplicates:
        duplicate_info = []
        for ann_id in duplicates:
            same_id_anns = []
            for ann in annotations:
                if ann['id'] == ann_id:
                    image_info = next(img for img in images if img['id'] == ann['image_id'])
                    same_id_anns.append({
                        "ann_id": ann['id'],
                        "image_id": ann['image_id'],
                        "image_filename": image_info['file_name'],
                        "bbox": ann['bbox'],
                        "category_id": ann['category_id']
                    })
            duplicate_info.append(same_id_anns)
        return duplicate_info
    else:
        return None

# 範例用法
ann_file = "/home/zhwa/Document/data/MVA2023/train/annotations/split_val_coco.json"  # 將 "your_annotation.json" 替換為你的標註檔案路徑
duplicate_ids_info = find_duplicate_ann_ids(ann_file)

if duplicate_ids_info:
    print("找到重複的 annotations ID：")
    for duplicate_anns in duplicate_ids_info:
        for ann in duplicate_anns:
            print(f"  ann_id: {ann['ann_id']}, image_id: {ann['image_id']}, image_filename: {ann['image_filename']}, bbox: {ann['bbox']}, category_id: {ann['category_id']}")
else:
    print("沒有找到重複的 annotations ID。")