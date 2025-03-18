import cv2
import json
import os
import numpy as np
import random

import cv2
import numpy as np
import random

'''
偽造當前圖片的上一幀畫面
用於mva4sb的資料集上
'''

def generate_previous_frame(image, bird_image, paste_x, paste_y, paste_w, paste_h, shift_range=(-40, 40), scale_range=(0.8, 1.2)):
    """生成新貼上鳥類的上一幀，使用隨機位移和 cv2.resize"""
    previous_frame = image.copy()

    # 隨機縮放鳥類圖片
    scale = random.uniform(scale_range[0], scale_range[1])
    new_paste_w = int(paste_w * scale)
    new_paste_h = int(paste_h * scale)
    bird_resized = cv2.resize(bird_image, (new_paste_w, new_paste_h))

    # 隨機位移貼上點
    shift_x = random.randint(shift_range[0], shift_range[1])
    shift_y = random.randint(shift_range[0], shift_range[1])
    new_paste_x = paste_x + shift_x
    new_paste_y = paste_y + shift_y

    # 處理邊界情況
    h, w, _ = image.shape
    new_paste_x = max(0, min(new_paste_x, w - new_paste_w))
    new_paste_y = max(0, min(new_paste_y, h - new_paste_h))

    # 取得透明度
    alpha_channel = bird_resized[:, :, 3] / 255.0

    # 將變換後的鳥類貼回到原始圖片中
    for c in range(0, 3):
        previous_frame[new_paste_y:new_paste_y + new_paste_h, new_paste_x:new_paste_x + new_paste_w, c] = (
            alpha_channel * bird_resized[:, :, c]
            + (1 - alpha_channel)
            * previous_frame[new_paste_y:new_paste_y + new_paste_h, new_paste_x:new_paste_x + new_paste_w, c]
        )

    return previous_frame

def random_copy_paste_bird(image, bird_image, min_size=5, max_size=80):
    """隨機 Copy-Paste 去背鳥類圖片，並新增標註，指定貼上大小區間"""
    h, w, _ = image.shape
    bird_h, bird_w, _ = bird_image.shape

    # 找出圖片較短的一邊
    short_side = min(w, h)

    # 隨機生成貼上的短邊大小
    paste_short = random.randint(min_size, min(max_size, short_side))

    # 隨機生成長寬比倍數
    aspect_ratio_factor = random.uniform(1, 9.0)  # 假設合理的長寬比倍數在 0.5 到 2.0 之間

    # 根據短邊和倍數計算長邊大小
    if w <= h:
        paste_w = paste_short
        paste_h = int(paste_short * aspect_ratio_factor)
    else:
        paste_h = paste_short
        paste_w = int(paste_short * aspect_ratio_factor)

    # 隨機生成貼上的大小
    paste_w = random.randint(min_size, min(max_size, bird_w))
    paste_h = random.randint(min_size, min(max_size, bird_h))

    # 調整鳥類圖片大小
    bird_resized = cv2.resize(bird_image, (paste_w, paste_h))

    # 隨機生成貼上的位置
    x = random.randint(0, w - paste_w)
    y = random.randint(0, h - paste_h)

    # 取得透明度
    alpha_channel = bird_resized[:, :, 3] / 255.0

    # Copy-Paste
    for c in range(0, 3):
        image[y:y + paste_h, x:x + paste_w, c] = (alpha_channel * bird_resized[:, :, c] + (1 - alpha_channel) * image[y:y + paste_h, x:x + paste_w, c])

    # 產生新的標註
    new_annotation = {
        'bbox': [x, y, paste_w, paste_h],
        'category_id': 0,  # 假設鳥類的 category_id 為 1
        'iscrowd': 0,
        'area': paste_w * paste_h  # 計算邊界框的面積
        # 其他必要的標註欄位
    }
    return image, new_annotation, x, y, paste_w, paste_h

def calculate_optical_flow(prev_frame, curr_frame):
    """計算光流"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def visualize_optical_flow(flow):
    """可視化光流場"""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

# 範例程式碼
image_dir = '/home/zhwa/Document/data/MVA2023/train/images'
bird_image_dir = '/home/zhwa/Document/data/MVA2023/birds'

coco_json_path = '/home/zhwa/Document/data/MVA2023/train/annotations/split_val_coco.json'
output_dir = '/home/zhwa/Document/data/fake_output_directory/val'
new_coco_json_path = os.path.join(output_dir, 'val.json') #新的json檔案路徑

os.makedirs(output_dir, exist_ok=True)

# 創建三個子資料夾
prev_dir = os.path.join(output_dir, 'prev_frames')
copy_paste_dir = os.path.join(output_dir, 'copy_paste_images')
flow_dir = os.path.join(output_dir, 'optical_flow')
# flow_visual_dir = os.path.join(output_dir, 'optical_flow_visual') #新增存放光流可視化圖片的資料夾

os.makedirs(prev_dir, exist_ok=True)
os.makedirs(copy_paste_dir, exist_ok=True)
os.makedirs(flow_dir, exist_ok=True)
# os.makedirs(flow_visual_dir, exist_ok=True)

# 打開標註檔案
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)
with open(coco_json_path.replace('split_val_coco', 'merged_train'), 'r') as f:
    coco_merge_data = json.load(f)

# 創建新標註檔案
new_coco_data = {'images': [], 'annotations': [], 'categories': coco_data['categories']} #創建新的json 字典
new_annotation_id = 29038 # merged train size is 29037

# 圖片處理變數
png_files = [f for f in os.listdir(bird_image_dir) if f.endswith('.png')] #過濾png檔案
processed_count = 0  # 初始化已處理圖片的計數器

if __name__ == "__main__":

    for image_info in coco_data['images']:
        # 讀取資料
        image_path = os.path.join(image_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]

        # 隨機生成偏移量
        shift_x = random.randint(-5, 5)
        shift_y = random.randint(-5, 5)

        # 應用偏移
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 隨機貼上 1 到 6 隻鳥類
        num_birds = random.randint(1, 6)
        temp_image = image.copy() #創建一個臨時的圖片，用於copy paste
        prev_frame = shifted_image.copy() #創建一個臨時的圖片，用於生成上一幀
        for _ in range(num_birds):
            if png_files:
                bird_file_name = random.choice(png_files)

                # 選擇一個去背的鳥的圖片
                bird_image = cv2.imread(os.path.join(bird_image_dir, bird_file_name), cv2.IMREAD_UNCHANGED)

                # 隨機 Copy-Paste 並新增標註
                temp_image, new_annotation, paste_x, paste_y, paste_w, paste_h = random_copy_paste_bird(temp_image, bird_image)

                # 生成新貼上鳥類的上一幀
                prev_frame = generate_previous_frame(prev_frame, bird_image, paste_x, paste_y, paste_w, paste_h)

                # 加入新的標註資訊 (只加入 new_annotation)
                new_annotation['id'] = new_annotation_id
                new_annotation['image_id'] = image_info['id']
                new_coco_data['annotations'].append(new_annotation)

                new_annotation_id += 1

        new_coco_data['annotations'].extend(annotations)

        # 計算光流
        flow = calculate_optical_flow(prev_frame, temp_image)
        
        # 保存結果
        prev_file_name = f'prev_{image_info["file_name"]}'
        copy_paste_file_name = f'copy_paste_{image_info["file_name"]}'
        # flow_visual_file_name = f'flow_visual_{image_info["file_name"]}.png' #新增光流可視化圖片的檔名

        cv2.imwrite(os.path.join(prev_dir, prev_file_name), prev_frame)
        cv2.imwrite(os.path.join(copy_paste_dir, copy_paste_file_name), temp_image)
        flow_name = image_info['file_name'].split('.')[0]
        # breakpoint()
        np.save(os.path.join(flow_dir, f'flow_{flow_name}.npy'), flow)
        # cv2.imwrite(os.path.join(flow_visual_dir, flow_visual_file_name), visualize_optical_flow(flow)) #保存光流可視化圖片


        # 加入新的圖片資訊
        new_coco_data['images'].append({
            'id': image_info['id'],
            'file_name': copy_paste_file_name,
            'width': image.shape[1],
            'height': image.shape[0],
        })

        processed_count += 1 #處理完成一張圖片，所以計數器加一

        # if processed_count > 5:
        #     break

        # 每處理完五張圖片，顯示一次進度條
        if processed_count % 5 == 0:
            print(f"已處理 {processed_count} / {len(coco_data['images'])} 張圖片")

    # 保存更新後的 JSON 檔案
    with open(new_coco_json_path, 'w') as f:
        json.dump(new_coco_data, f)

    print('資料集生成和新的 COCO JSON 檔案創建完成！')