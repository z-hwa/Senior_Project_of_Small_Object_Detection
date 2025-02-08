import os
import cv2
import numpy as np
import gc  # 引入垃圾回收模組

# python tools/my_tools/data_modify/sharpen.py

def sharpen_image(img, alpha=1.0, beta=0.0, kernel_size=3):
    """對圖像進行銳利化處理"""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[(kernel_size - 1) // 2, :] = -1
    kernel[:, (kernel_size - 1) // 2] = -1
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = 5

    sharpened_img = cv2.filter2D(img, -1, kernel)
    img = cv2.addWeighted(sharpened_img, alpha, img, 1 - alpha, beta)
    return img

def process_dataset(input_dir, output_dir, alpha=1.2, beta=0, kernel_size=3):
    """批量處理數據集"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = os.listdir(input_dir)
    total = len(img_files)

    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        sharpened_img = sharpen_image(img, alpha=alpha, beta=beta, kernel_size=kernel_size)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, sharpened_img)

        # 釋放內存
        del img, sharpened_img
        gc.collect()

        # 每處理 10 張輸出一次資訊
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total} images...")

input_dir = 'data/mva2023_sod4bird_pub_test/images'
output_dir = 'data/SOD4SB_pub_sharpen/images'

process_dataset(input_dir, output_dir)
