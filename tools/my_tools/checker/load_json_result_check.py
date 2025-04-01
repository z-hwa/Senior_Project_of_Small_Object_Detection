import numpy as np
import json
from mmcv import Config
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os
import pandas as pd
from pycocotools.coco import COCO
import cv2

# 資料分析用
# 載入json預測檔案

class ImageNavigator:
    def __init__(self, ann_file, json_file, img_prefix, confidence_threshold=0.05):
        self.ann_file = ann_file
        self.json_file = json_file
        self.img_prefix = img_prefix
        self.confidence_threshold = confidence_threshold
        self.index = 0
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.predictions = self._load_predictions()

        # 創建圖形和兩個子圖
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.update_image()  # 顯示當前圖片

        # 添加按鈕
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.075])  # 按鈕的位置
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)

    def _load_predictions(self):
        """載入 JSON 檔案中的預測結果"""
        with open(self.json_file, 'r') as f:
            predictions = json.load(f)
            predictions_df = pd.DataFrame(predictions)
        return predictions_df

    def search_by_image_id(self, image_id):
        '''
        根據 image_id 進行檢索
        '''
        result = self.predictions[self.predictions["image_id"] == image_id]
        if result.empty:
            return []
        return result

    def update_image(self):
        '''
        更新當前畫面顯示的圖片
        '''
        img_id = self.img_ids[self.index]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_prefix, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred_bboxes = self.search_by_image_id(img_id)
        pred_bboxes_formatted = []
        if not pred_bboxes.empty:
            pred_bboxes_formatted = [
                [x[0], x[1], x[0] + x[2], x[1] + x[3], score]
                for x, score in zip(pred_bboxes['bbox'], pred_bboxes['score'])
                if score > self.confidence_threshold
            ]

        # 清空上一張圖片的內容
        self.ax.clear()

        # 顯示帶框圖片
        self.plot_image_with_boxes(img, pred_bboxes_formatted, self.ax, title=f"Image ID: {img_id}")

        # 顯示圖片
        self.fig.canvas.draw()

    def plot_image_with_boxes(self, image, boxes, ax, title=""):
        '''
        為圖片添加bboxes 用於可視化
        '''
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

        if boxes:
            for box in boxes:
                x1, y1, x2, y2, score = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'{score:.2f}', color='black', fontsize=12, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.3))

    def next_image(self, event):
        '''切換到下一張圖片'''
        self.index += 1
        if self.index >= len(self.img_ids):
            self.index = 0  # 如果是最後一張，回到第一張
        self.update_image()

default_config_path = 'work_dirs/flow_cascade_rcnn_cos_swin_rfla_4stage_finetune/flow_cascade_rcnn_cos_swin_rfla_4stage_finetune.py'
default_json_file = 'coco_results.bbox.json'

# 載入配置檔案 (僅用於獲取 img_prefix)
cfg = Config.fromfile(default_config_path)

# 假設您的 config 中 'data.test.ann_file' 和 'data.test.img_prefix' 存在
annotations_file = cfg.data.test.ann_file
image_folder = cfg.data.test.img_prefix

# 初始化圖像導航器
navigator = ImageNavigator(
    annotations_file,
    default_json_file,
    image_folder,
    confidence_threshold=0.05
)

# 顯示圖形
plt.show()