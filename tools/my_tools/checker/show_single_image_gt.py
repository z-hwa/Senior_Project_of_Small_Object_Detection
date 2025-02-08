import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog
import argparse
from matplotlib.patches import Rectangle
import numpy as np

'''
# 可以載入指定的圖片進行確認

'''


def load_coco_annotations(file_path):
    """載入 COCO 格式的 JSON 檔案"""
    with open(file_path, "r") as f:
        return json.load(f)


def compute_iou(box1, box2):
    """計算兩個框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def show_image_with_gt_and_predictions(image_dir, annotations, predictions, image_name, iou_threshold=0.5):
    """
    顯示指定圖片的 GT 和模型預測框，並顯示分數
    """
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        image_name += '.jpg'

    image_data = next((img for img in annotations['images'] if img['file_name'] == image_name), None)
    if not image_data:
        print(f"Image '{image_name}' not found in annotations.")
        return

    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' does not exist.")
        return

    img = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()

    image_id = image_data['id']
    gt_bboxes = [ann['bbox'] for ann in annotations['annotations'] if ann['image_id'] == image_id]
    pred_bboxes = [
        pred['bbox'] + [pred['score']] for pred in predictions if pred['image_id'] == image_id
    ]

    matched_gt = set()
    
    # 檢查每個 GT 是否有匹配的預測框
    for gt_idx, gt in enumerate(gt_bboxes):
        matched = False
        for pred in pred_bboxes:
            iou = compute_iou(gt, pred[:4])
            if iou >= iou_threshold:
                matched = True
                matched_gt.add(gt_idx)
                break
        color = 'green' if matched else 'red'  # 綠色代表匹配，紅色代表未匹配
        x, y, w, h = gt
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', label="Unmatched GT" if color == 'red' else "Matched GT")
        ax.add_patch(rect)

    for bbox in pred_bboxes:
        x, y, w, h, score = bbox
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label="Prediction")
        ax.add_patch(rect)
        ax.text(x, y, f"{score:.2f}", color='blue', fontsize=10, weight='bold', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 顯示標籤
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title(f"GT and Predictions for {image_name}")
    plt.axis("off")
    plt.show()

def browse_file(file_type):
    """彈出視窗選擇檔案"""
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    file_path = filedialog.askopenfilename(
        title=f"選擇{file_type}",
        filetypes=[("JSON Files", "*.json")] if file_type == "JSON 檔案" else [("Image Files", "*.jpg;*.png;*.jpeg")]
    )
    return file_path

def select_image(annotations):
    """提供輸入視窗，讓使用者輸入圖片名稱"""
    image_names = [img['file_name'] for img in annotations['images']]
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    image_name = simpledialog.askstring(
        "輸入圖片名稱", f"請輸入圖片名稱（可選擇：{', '.join(image_names[:5])}...）"
    )

    # 自動補零到五位數
    if image_name and image_name.isdigit():
        image_name = image_name.zfill(5)

    return image_name


# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="檢視指定圖片的 GT 和預測框")
    parser.add_argument("--custom", action="store_true", help="啟用自定義路徑選擇")
    args = parser.parse_args()

    # 預設路徑
    default_annotations_file = "data/mva2023_sod4bird_train/annotations/split_val_coco.json"
    default_predictions_file = "work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_val.json"
    default_image_dir = "data/mva2023_sod4bird_train/images"

    # 如果使用 --custom，讓使用者自選檔案
    if args.custom:
        print("請選擇標註檔案...")
        annotations_file = browse_file("JSON 檔案")
        if not annotations_file:
            print("未選擇標註檔案，程式結束。")
            exit()

        print("請選擇預測檔案...")
        predictions_file = browse_file("JSON 檔案")
        if not predictions_file:
            print("未選擇預測檔案，程式結束。")
            exit()

        print("請選擇圖片目錄...")
        image_dir = filedialog.askdirectory(title="選擇圖片目錄")
        if not image_dir:
            print("未選擇圖片目錄，程式結束。")
            exit()
    else:
        annotations_file = default_annotations_file
        predictions_file = default_predictions_file
        image_dir = default_image_dir

    # 載入標註與預測檔案
    annotations = load_coco_annotations(annotations_file)
    predictions = load_coco_annotations(predictions_file)

    # 進入互動式模式
    while True:
        image_name = select_image(annotations)
        if not image_name:
            print("未輸入圖片名稱，程式結束。")
            break

        show_image_with_gt_and_predictions(image_dir, annotations, predictions, image_name)
