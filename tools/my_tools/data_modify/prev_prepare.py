import cv2
import json
import os
import numpy as np
import random

import cv2
import numpy as np
import random

'''
準備當前圖片的上一幀畫面
用於mva4sb的資料集上
'''

def calculate_optical_flow(prev_frame, curr_frame, scale_factor=0.5, quantization_factor=10):
    """計算光流，降低解析度，並量化 1:5.42"""
    prev_resized = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor)
    curr_resized = cv2.resize(curr_frame, None, fx=scale_factor, fy=scale_factor)

    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    quantized_flow = np.round(flow * quantization_factor).astype(np.int16) # 量化

    return quantized_flow

def visualize_optical_flow(flow, quantization_factor=10):
    """可視化光流場，支援量化後的光流"""
    if flow is None or flow.ndim != 3 or flow.shape[-1] != 2:
        print("Error: Invalid optical flow data.")
        return None

    print(f"Flow dtype: {flow.dtype}")  # 檢查資料類型
    print(f"Flow shape: {flow.shape}")  # 檢查尺寸

    if flow.dtype == np.int16 and quantization_factor is not None:
        flow = flow.astype(np.float32) / quantization_factor  # 反量化並轉換為 float32
    elif flow.dtype != np.float32:
        flow = flow.astype(np.float32)  # 確保類型為 float32

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 處理 mag 全為零的情況
    if np.max(mag) > 0:
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    else:
        hsv[..., 2] = 0  # 如果 mag 全為零，則將 V 通道設為零

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

# 主機
# prefix = '/root/Document/data/'
# image_dir = prefix + 'MVA2023/train/images'
# bird_image_dir = prefix + 'MVA2023/birds'
# coco_json_path = prefix + 'MVA2023/train/annotations/split_train_coco.json'
# output_dir = prefix + 'output_directory/train'
# new_coco_json_path = os.path.join(output_dir, 'train.json') #新的json檔案路徑

# 筆電
prefix = '/home/zhwa/Document/data/'
image_dir = prefix + 'MVA2025/phase_1/unzip/train'
# coco_json_path = prefix + 'MVA2025/phase_1/unzip/annotations/test_coco.json'
coco_json_path = '/home/zhwa/Document/MVA2025-SMOT4SB/datasets/SMOT4SB/annotations/filtered_train.json'
output_dir = prefix + 'MVA2025/phase_1/unzip/train'

os.makedirs(output_dir, exist_ok=True)

# 創建三個子資料夾
flow_dir = os.path.join(output_dir, 'optical_flow')
flow_visual_dir = os.path.join(output_dir, 'optical_flow_visual') #新增存放光流可視化圖片的資料夾

os.makedirs(flow_dir, exist_ok=True)
os.makedirs(flow_visual_dir, exist_ok=True)

# 打開標註檔案
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

processed_count = 0  # 初始化已處理圖片的計數器

if __name__ == "__main__":

    for image_info in coco_data['images']:
        # 讀取資料
        image_path = os.path.join(image_dir, image_info['file_name'])
        curr_frame = cv2.imread(image_path)

        # 找出上一幀的檔案名稱
        curr_file_name = image_info['file_name']
        folder_name, frame_name = curr_file_name.split('/')
        frame_number = int(frame_name.split('.')[0])
        prev_frame_number = frame_number - 1

        if prev_frame_number > -1:
            # 第一幀的上一幀設為自己
            if prev_frame_number == 0:
                prev_frame_number = 1

            prev_file_name = f'{folder_name}/{prev_frame_number:05d}.jpg'  # 假設幀號為 5 位數字
            prev_image_path = os.path.join(image_dir, prev_file_name)

            if os.path.exists(prev_image_path):
                # 讀取上一幀
                prev_frame = cv2.imread(prev_image_path)

                # 計算光流
                flow = calculate_optical_flow(prev_frame, curr_frame)

                # 確保資料夾存在
                os.makedirs(os.path.join(flow_dir, f'{folder_name}/'), exist_ok=True)

                # 保存光流數據
                flow_name = frame_name.split('.')[0]
                np.save(os.path.join(flow_dir, f'{folder_name}/flow_{flow_name}.npy'), flow)

                # 可視化並保存光流
                # flow_visual = visualize_optical_flow(flow)
                # if flow_visual is not None:
                #     cv2.imwrite(os.path.join(flow_visual_dir, f'{folder_name}/flow_visual_{flow_name}.png'), flow_visual)

        processed_count += 1 #處理完成一張圖片，所以計數器加一

        # if processed_count > 5:
        #     break

        # 每處理完五張圖片，顯示一次進度條
        if processed_count % 5 == 0:
            print(f"已處理 {processed_count} / {len(coco_data['images'])} 張圖片")

    print('資料集生成和新的 COCO JSON 檔案創建完成！')