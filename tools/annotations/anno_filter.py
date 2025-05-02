import json

def filter_coco_by_video_ids(coco_annotation_file, video_ids_to_keep, output_file):
    """
    從 COCO 格式的標註檔案中篩選出指定 video_id 的標註，並處理指定編號不存在的情況。

    Args:
        coco_annotation_file (str): COCO 格式的標註檔案路徑。
        video_ids_to_keep (list): 需要保留的 video_id 編號列表 (整數)。
        output_file (str): 篩選後的 COCO 標註檔案輸出路徑。
    """
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    existing_video_ids = {video['id'] for video in coco_data.get('videos', [])}
    found_video_ids = set()

    filtered_images = []
    for img in coco_data['images']:
        if img['video_id'] in video_ids_to_keep and img['video_id'] in existing_video_ids:
            filtered_images.append(img)
            found_video_ids.add(img['video_id'])

    filtered_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in [img['id'] for img in filtered_images]
    ]

    filtered_videos = [
        vid for vid in coco_data['videos']
        if vid['id'] in video_ids_to_keep and vid['id'] in existing_video_ids
    ]

    filtered_coco_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'videos': filtered_videos,
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data.get('categories', [])
    }

    with open(output_file, 'w') as f:
        json.dump(filtered_coco_data, f)

    not_found_video_ids = set(video_ids_to_keep) - found_video_ids
    if not_found_video_ids:
        print(f"警告：指定的 video_id 中，以下編號在標註檔案中未找到：{list(not_found_video_ids)}")
    else:
        print(f"已成功篩選出 video_id 為 {video_ids_to_keep} 的標註並保存到 {output_file}")

if __name__ == "__main__":
    coco_file = "/root/Document/data/MVA2025/annotations/test_coco.json"  # 將你的 COCO 標註檔案路徑替換這裡
    video_ids = [30] # 包含一個可能不存在的編號
    output_coco_file = "/root/Document/data/MVA2025/annotations/test_filter_30.json"  # 篩選後的 COCO 標註檔案輸出路徑

    # coco_file = "/root/Document/data/MVA2025/annotations/test_coco.json"
    # video_ids = [5, 1000]
    # output_coco_file = "/root/Document/data/MVA2025/annotations/test_coco_filter_05.json"

    filter_coco_by_video_ids(coco_file, video_ids, output_coco_file)