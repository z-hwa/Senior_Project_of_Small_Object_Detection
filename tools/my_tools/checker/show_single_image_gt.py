import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import tkinter as tk
from tkinter import filedialog, simpledialog
import argparse

# 可以載入指定的圖片進行確認

def load_coco_annotations(file_path):
    """載入 COCO 格式的 JSON 檔案"""
    with open(file_path, "r") as f:
        return json.load(f)

def show_image_with_gt_and_predictions(image_dir, annotations, predictions, image_name):
    """
    顯示指定圖片的 GT 和模型預測框，並顯示分數
    """
    # 檢查圖片名稱並補全副檔名
    if not image_name.endswith(('.jpg', '.png', '.jpeg')):
        image_name += '.jpg'

    # 找到對應的圖片資料
    image_data = next((img for img in annotations['images'] if img['file_name'] == image_name), None)
    if not image_data:
        print(f"Image '{image_name}' not found in annotations.")
        return

    image_path = os.path.join(image_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' does not exist.")
        return

    # 載入圖片
    img = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()

    # 繪製 GT 標註框
    image_id = image_data['id']
    gt_bboxes = [ann['bbox'] for ann in annotations['annotations'] if ann['image_id'] == image_id]
    for bbox in gt_bboxes:
        x, y, w, h = bbox
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', label="GT")
        ax.add_patch(rect)

    # 繪製模型預測框並顯示分數
    if predictions:
        pred_bboxes = [
            pred['bbox'] + [pred['score']] for pred in predictions if pred['image_id'] == image_id
        ]
        for bbox in pred_bboxes:
            x, y, w, h, score = bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--', label="Prediction")
            ax.add_patch(rect)
            # 在框的左上角顯示分數
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
