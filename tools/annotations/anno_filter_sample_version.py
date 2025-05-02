import json
from collections import defaultdict

def filter_coco_by_video_ids_and_sampling(coco_annotation_file, video_ids_to_keep, sampling_rate, output_file):
    """
    從 COCO 格式的標註檔案中篩選出指定 video_id 的標註，並對每個影片每隔指定幀數取一張圖片的標註。
    同時處理指定編號不存在的情況，並統計篩選後的圖片數量以及每個影片取樣的圖片數量。

    Args:
        coco_annotation_file (str): COCO 格式的標註檔案路徑。
        video_ids_to_keep (list): 需要保留的 video_id 編號列表 (整數)。
        sampling_rate (int): 對每個影片取樣的幀率間隔 (例如，每隔 10 幀取 1 張)。
        output_file (str): 篩選後的 COCO 標註檔案輸出路徑。
    """
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    existing_video_ids = {video['id'] for video in coco_data.get('videos', [])}
    found_video_ids = set()
    sampled_image_ids = set()
    video_sample_counts = defaultdict(int)

    filtered_images = []
    for img in coco_data['images']:
        if img['video_id'] in video_ids_to_keep and img['video_id'] in existing_video_ids:
            video_frame_id = img.get('frame_id')
            if video_frame_id is not None and video_frame_id % sampling_rate == 0:
                filtered_images.append(img)
                found_video_ids.add(img['video_id'])
                sampled_image_ids.add(img['id'])
                video_sample_counts[img['video_id']] += 1
            elif video_frame_id is None:
                # 如果沒有 frame_id，則保留該圖片 (通常是第一幀或關鍵幀)
                filtered_images.append(img)
                found_video_ids.add(img['video_id'])
                sampled_image_ids.add(img['id'])
                video_sample_counts[img['video_id']] += 1

    filtered_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in sampled_image_ids
    ]

    filtered_videos = [
        vid for vid in coco_data['videos']
        if vid['id'] in video_ids_to_keep and vid['id'] in existing_video_ids
    ]

    # 更新 filtered_videos，使其包含每個影片的總幀數 (如果有的話)
    video_frame_counts = {}
    for img in coco_data['images']:
        if img['video_id'] in video_ids_to_keep and img['video_id'] in existing_video_ids and 'frame_id' in img:
            video_id = img['video_id']
            frame_id = img['frame_id']
            video_frame_counts[video_id] = max(video_frame_counts.get(video_id, 0), frame_id)

    for vid in filtered_videos:
        if vid['id'] in video_frame_counts:
            vid['total_frames'] = video_frame_counts[vid['id']]
        else:
            vid['total_frames'] = None # 如果沒有 frame_id 信息，則為 None

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

    print(f"已成功篩選出 video_id 為 {video_ids_to_keep}，並以每 {sampling_rate} 幀取樣的標註保存到 {output_file}")
    print(f"篩選後的圖片總數：{len(filtered_images)} 張")
    print("各影片取樣的圖片數量：")
    for video_id in video_ids_to_keep:
        print(f"  Video ID {video_id}: {video_sample_counts.get(video_id, 0)} 張")

if __name__ == "__main__":
    coco_file = "/root/Document/data/MVA2025/annotations_phase2/train.json"  # 將你的 COCO 標註檔案路徑替換這裡
    # 5, 6, 8, 10, 15, 16, 18, 19, 20, 22, 28, 40, 43, 44, 54, 65, 66, 70, 74, 75, 76, 84, 92, 95, 96, # phase1
    video_ids = [
                 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 117, 118, 119, # phase2
                  120, 121, 122, 123, 125, 126, 127, 128]  # 包含一個可能不存在的編號
    sampling_rate = 5  # 每隔 10 幀取 1 張
    output_coco_file = "/root/Document/data/MVA2025/annotations_phase2/train_filter_only2.json"  # 篩選後的 COCO 標註檔案輸出路徑

    # coco_file = "/root/Document/data/MVA2025/annotations/test_coco.json"
    # video_ids = [5, 1000]
    # sampling_rate = 5
    # output_coco_file = "/root/Document/data/MVA2025/annotations/test_coco_filter_sampled_05.json"

    filter_coco_by_video_ids_and_sampling(coco_file, video_ids, sampling_rate, output_coco_file)