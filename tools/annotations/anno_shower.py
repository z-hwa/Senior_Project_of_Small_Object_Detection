import json
import os
from PIL import Image
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def visualize_coco_annotations_to_video(coco_annotation_file, image_folder, output_video_path, video_id_to_visualize=None, fps=30):
    """
    從 COCO 格式的標註檔案中讀取指定影片的標註，並將帶有標註框的影片儲存到指定路徑。

    Args:
        coco_annotation_file (str): COCO 格式的標註檔案路徑。
        image_folder (str): 包含原始圖片的資料夾路徑。
        output_video_path (str): 輸出影片的完整路徑 (例如: /path/to/output.mp4)。
        video_id_to_visualize (int, optional): 需要視覺化的 video_id 編號 (整數)。
                                                如果為 None，則處理標註檔案中的第一個影片。默認為 None。
        fps (int, optional): 輸出影片的幀率 (frames per second)。默認為 30。
    """
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    videos = {v['id']: v for v in coco_data.get('videos', [])}
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data.get('annotations', [])
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}

    if video_id_to_visualize is None:
        if videos:
            video_id = list(videos.keys())[0]
            print(f"警告：未指定 video_id，將處理第一個找到的影片 ID: {video_id}")
        else:
            print("警告：標註檔案中沒有影片資訊。")
            return
    elif video_id_to_visualize not in videos:
        print(f"警告：指定的 video_id ({video_id_to_visualize}) 在標註檔案中未找到。")
        return
    else:
        video_id = video_id_to_visualize

    print(f"\n正在處理 Video ID: {video_id} - {videos[video_id].get('name', '')}")
    video_images = sorted([img for img in images.values() if img['video_id'] == video_id], key=lambda x: x.get('frame_id', x['id']))
    video_annotations = [ann for ann in annotations if images[ann['image_id']]['video_id'] == video_id]
    image_annotation_map = defaultdict(list)
    for ann in video_annotations:
        image_annotation_map[ann['image_id']].append(ann)

    if not video_images:
        print(f"警告：Video ID {video_id} 沒有對應的圖片。")
        return

    # 獲取第一幀的尺寸以初始化 VideoWriter
    first_image_path = os.path.join(image_folder, video_images[0]['file_name'])
    try:
        first_img = cv2.imread(first_image_path)
        if first_img is None:
            print(f"錯誤：無法讀取第一張圖片 {first_image_path}")
            return
        height, width, _ = first_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for img_info in tqdm(video_images, desc="處理幀"):
            image_path = os.path.join(image_folder, img_info['file_name'])
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告：無法讀取圖片 {image_path}")
                continue

            current_annotations = image_annotation_map.get(img_info['id'], [])
            for ann in current_annotations:
                bbox = ann['bbox']
                x, y, w, h = map(int, bbox)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 紅色框
                # 如果有類別資訊，可以添加文字標籤
                if 'category_id' in ann and ann['category_id'] in categories:
                    category_name = categories[ann['category_id']]['name']
                    cv2.putText(img, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            out.write(img)

        out.release()
        print(f"\n影片已儲存至: {output_video_path}")

    except FileNotFoundError:
        print(f"錯誤：找不到圖片或標註檔案。")
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")

if __name__ == "__main__":
    coco_file = "/root/Document/data/MVA2025/annotations_phase2/train.json"  # 將你的 COCO 標註檔案路徑替換這裡
    image_folder_path = "/root/Document/data/MVA2025/train"  # 將你的圖片資料夾路徑替換這裡
    output_video = "/root/Document/data/MVA2025/anno_show/output_video_97.mp4"  # 設定輸出影片的路徑和名稱
    video_id_to_show = 113  # 指定要處理的影片 ID
    output_fps = 10  # 設定輸出影片的幀率

    visualize_coco_annotations_to_video(coco_file, image_folder_path, output_video, video_id_to_show, output_fps)