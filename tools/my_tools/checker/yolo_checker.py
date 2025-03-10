import mmcv
import torch
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

# 設定模型配置與權重
config_file = 'work_dirs/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco.py'  # 你的YOLOX配置檔案
checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco/epoch_55.pth'  # 訓練好的模型權重

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

def random_crop(img, crop_size=(640, 640)):
    """ 隨機裁剪圖片 """
    h, w, _ = img.shape
    ch, cw = crop_size

    if h < ch or w < cw:
        raise ValueError(f"圖片尺寸 ({w}x{h}) 小於裁剪尺寸 {crop_size}！")

    x1 = random.randint(0, w - cw)
    y1 = random.randint(0, h - ch)
    x2, y2 = x1 + cw, y1 + ch

    cropped_img = img[y1:y2, x1:x2]
    return cropped_img, (x1, y1, x2, y2)

def visualize_result(img, result, score_thr=0.6):
    """ 顯示推理結果 """
    img = img.copy()  # 避免修改原圖
    bboxes = np.vstack(result)
    labels = [np.full(b.shape[0], i, dtype=int) for i, b in enumerate(result)]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2, score = bbox
        if score < score_thr:
            continue

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f'{label}: {score:.2f}'
        cv2.putText(img, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# 讀取圖片並進行隨機裁剪
img_path = 'data/train/images/05971.jpg'  # 測試圖片路徑
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cropped_img, crop_coords = random_crop(img, (640, 640))

# 執行推理
result = inference_detector(model, cropped_img)

# 顯示結果
visualize_result(cropped_img, result)
