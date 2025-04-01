import numpy as np
import json
import torch
import cv2
from mmcv import Config
from mmdet.datasets import build_dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mmcv.parallel import DataContainer
import os
import pandas as pd
import time
import argparse
import tkinter as tk
from tkinter import filedialog

# 資料分析用
# 載入json預測檔案
# 並與gt進行比較分析

# on the val dataset
# Processing image 767/767...
# total_correct_predictions: 2511
# total_bird_as_background: 354
# total_background_as_bird: 3904
# Time since last output: 2.97 seconds


# 用於解析布林值的函數
def str2bool(value):
    '''用於解析布林值的函數'''

    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setting_parser():
    '''設定命令列參數解析'''

    
    # 設定預設路徑和參數
    default_show_image = True
    default_confidence_threshold = 0.05
    default_process_info = True
    default_return_result = False
    default_get_unmatched = False
    default_get_gt = False
    default_get_background_as_bird = False

    # 設定命令列參數解析
    parser = argparse.ArgumentParser(description="Custom Image Navigator")
    parser.add_argument('--custom', action='store_true', help='Enable custom path selection')
    parser.add_argument('--show_image', type=str2bool, default=default_show_image, help='Whether to show image')
    parser.add_argument('--confidence_threshold', type=float, default=default_confidence_threshold, help='Confidence threshold')
    parser.add_argument('--process_info', type=str2bool, default=default_process_info, help='Whether to process info')
    parser.add_argument('--return_result', type=str2bool, default=default_return_result, help='Whether to return result')
    parser.add_argument('--get_unmatched', type=str2bool, default=default_get_unmatched, help='Whether to get unmatched info')
    parser.add_argument('--get_gt', type=str2bool, default=default_get_gt, help='Whether to get watched result')
    parser.add_argument('--get_background_as_bird', type=str2bool, default=default_get_background_as_bird, help='Whether to get false positive')

    # 解析命令列參數
    args = parser.parse_args()

    return args

class ImageNavigator:
    def __init__(self, dataset, json_file, cfg, show_image=True, confidence_threshold=0.05, process_info=True, return_result=False,
                 get_unwatched=False, get_gt=False, get_background_as_bird=False):
        self.dataset = dataset
        self.json_file = json_file
        self.cfg = cfg
        self.index = 0
        self.show_image = show_image  # 新增控制顯示圖片的參數
        self.confidence_threshold = confidence_threshold
        self.process_info = process_info
        self.return_result = return_result
        
        # 儲存未匹配的鳥類框大小
        self.gt_bird_sizes = []
        self.get_gt = get_gt

        self.unmatched_bird_sizes = []
        self.get_unmatched = get_unwatched
        
        self.background_as_bird_size = []
        self.get_background_as_bird = get_background_as_bird

        # 初始化累積結果
        self.total_gt_birds = 0
        self.total_correct_predictions = 0
        self.total_bird_as_background = 0
        self.total_background_as_bird = 0

        self.last_output_time = time.time()  # 初始化上次輸出的時間

        # 載入 JSON 檔案中的預測結果
        with open(json_file, 'r') as f:
            self.predictions = json.load(f)
            self.predictions = pd.DataFrame(self.predictions)
            # print(f"df data: {self.predictions.head}")

        if show_image:
            # 創建圖形和兩個子圖
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
            self.update_image()  # 顯示當前圖片

            # 添加按鈕
            ax_next = plt.axes([0.8, 0.02, 0.1, 0.075])  # 按鈕的位置
            self.btn_next = Button(ax_next, 'Next')
            self.btn_next.on_clicked(self.next_image)
        else:
            self.analyze_result()

    def calculate_iou(self, box1, box2):
        """
        計算兩個框的交並比 (IoU)
        :param box1: [x1, y1, x2, y2] 預測框
        :param box2: [x1, y1, x2, y2] 真實框
        :return: IoU 值
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # 計算交集的面積
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        # 計算預測框和真實框的面積
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 計算聯集面積
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def analyze_predictions(self, gt_bboxes, pred_bboxes, image_id=0):
        """
        根據預測框和真實框進行統計分析
        :param gt_bboxes: 真實框列表
        :param pred_bboxes: 預測框列表
        :return: 分析結果
        """
        correct_predictions = 0
        bird_as_background = 0
        background_as_bird = 0
        threshold_iou = 0.5  # 設定 IoU 門檻

        gt_bboxes = gt_bboxes.data.numpy()  # 假設 gt_bboxes 是 DataContainer

        # 使用 Numpy 加速計算
        gt_bboxes = np.array(gt_bboxes)  # 假設 gt_bboxes 已經是 numpy 格式

        gt_matched = np.zeros(len(gt_bboxes), dtype=bool)
        pred_matched = np.zeros(len(pred_bboxes), dtype=bool)

        for gt_idx, gt in enumerate(gt_bboxes): 
            # 計算 IoU 並找到最大值
            ious = np.array([self.calculate_iou(gt[:4], pred[:4]) for pred in pred_bboxes])
            
            # 檢查 ious 是否有有效的 IoU 值
            if ious.size > 0:
                max_iou_idx = np.argmax(ious)
                iou = ious[max_iou_idx]

                if iou > threshold_iou and not pred_matched[max_iou_idx]:
                    pred_matched[max_iou_idx] = True
                    correct_predictions += 1
                    gt_matched[gt_idx] = True
                else:
                    bird_as_background += 1
            else:
                bird_as_background += 1  # 沒有有效的預測框，當作是背景

            # 如果該真實框未匹配，計算其大小並儲存
            if not gt_matched[gt_idx]:
                width = gt[2] - gt[0]
                height = gt[3] - gt[1]
                area = width * height
                if self.get_unmatched:
                    self.unmatched_bird_sizes.append((width, height, image_id))

            if self.get_gt:
                width = gt[2] - gt[0]
                height = gt[3] - gt[1]
                area = width * height
                self.gt_bird_sizes.append((width, height, image_id))

        if self.get_background_as_bird == True:
            # 計算未匹配到真實框的預測框，即為背景被當作鳥類的情況
            for pred_idx, pred in enumerate(pred_bboxes):
                if not pred_matched[pred_idx]:
                    width = gt[2] - gt[0]
                    height = gt[3] - gt[1]
                    area = width * height
                    self.background_as_bird_size.append((width, height, image_id))

        background_as_bird = np.sum(~pred_matched)

        return {
            "correct_predictions": correct_predictions,
            "bird_as_background": bird_as_background,
            "background_as_bird": background_as_bird
        }

    def analyze_result(self):
        '''
        分析結果的主函數
        透過呼叫analyze_prediction來針對每張圖片的結果進行分析
        '''
        for self.index in range(0, len(self.dataset)):
            # 更新當前顯示的圖片
            self.sample = self.dataset[self.index]

            self.gt_bboxes = self.sample['gt_bboxes']

            # 檢查是否為 Tensor 類型，如果是則轉換為 numpy 陣列
            if isinstance(self.gt_bboxes, torch.Tensor):
                self.gt_bboxes = self.gt_bboxes.numpy()
            elif isinstance(self.gt_bboxes, list):
                self.gt_bboxes = np.array(self.gt_bboxes)

            image_id = os.path.splitext(self.sample['ori_filename'][0])[0]
            image_id = int(image_id)  # 將字串轉為整數後再轉回字串，去除開頭的零
            # print(f"image id: {image_id}")
            # print(f"pred result type: {type(self.predictions)}")

            self.pred_bboxes = self.search_by_image_id(image_id)
            self.pred_bboxes = self.predictions[self.predictions["image_id"] == image_id]
            if self.pred_bboxes.empty:
                self.pred_bboxes = []
            # print(f"pred result: {pred_bboxes}")
            
            # 將 score 加入每個 bbox 的第五個位置
            if isinstance(self.pred_bboxes, pd.DataFrame):
                self.pred_bboxes = [
                    [x[0], x[1], x[0] + x[2], x[1] + x[3], score]
                    for x, score in zip(self.pred_bboxes['bbox'], self.pred_bboxes['score'])
                    if score > self.confidence_threshold
                ]

            # 計算統計數據
            analysis_result = self.analyze_predictions(self.gt_bboxes[0], self.pred_bboxes, image_id)
            # print(f"Analysis result: {analysis_result}")

            self.total_gt_birds += analysis_result["correct_predictions"] + analysis_result["bird_as_background"]
            self.total_correct_predictions += analysis_result["correct_predictions"]
            self.total_bird_as_background += analysis_result["bird_as_background"]
            self.total_background_as_bird += analysis_result["background_as_bird"]

            # 每處理10張圖片顯示一次進度
            if self.index % 50 == 0:

                # 計算從上次輸出到現在的時間差
                current_time = time.time()
                elapsed_time = current_time - self.last_output_time
                self.last_output_time = current_time  # 更新上次輸出的時間

                if self.process_info:
                    print(f"Processing image {self.index + 1}/{len(self.dataset)}...")
                    print(f"total_gt_birds: {self.total_gt_birds}")
                    print(f"total_correct_predictions: {self.total_correct_predictions}")
                    print(f"total_bird_as_background: {self.total_bird_as_background}")
                    print(f"total_background_as_bird: {self.total_background_as_bird}")
                    print(f"Time since last output: {elapsed_time:.2f} seconds")
                    print()
            
        # 計算從上次輸出到現在的時間差
        current_time = time.time()
        elapsed_time = current_time - self.last_output_time
        self.last_output_time = current_time  # 更新上次輸出的時間

        if self.process_info:
            print(f"Processing image {self.index + 1}/{len(self.dataset)}...")
            print(f"total_gt_birds: {self.total_gt_birds}")
            print(f"total_correct_predictions: {self.total_correct_predictions}")
            print(f"total_bird_as_background: {self.total_bird_as_background}")
            print(f"total_background_as_bird: {self.total_background_as_bird}")
            print(f"Time since last output: {elapsed_time:.2f} seconds")
            print()

        if self.return_result:
            return self.total_gt_birds, self.total_correct_predictions, self.total_bird_as_background, self.total_background_as_bird

    def search_by_image_id(self, image_id):
        '''
        根據 image_id 進行檢索
        '''

        result = self.predictions[self.predictions["image_id"] == image_id]
        if result.empty:
            # 返回空列表，或其他預設的資料結構
            return []
        return result

    def update_image(self):
        '''
        更新當前畫面顯示的圖片
        '''
        
        # 更新當前顯示的圖片
        sample = self.dataset[self.index]
        img = sample['img']

        # 確認圖片是否為 list 類型
        if isinstance(img, list):
            img = img[0]  # 如果是 list，取第一個元素

        img = img.data.numpy().transpose(1, 2, 0)

        # 配置文件中的標準化參數
        img_norm_cfg = self.cfg.img_norm_cfg
        mean = img_norm_cfg['mean']
        std = img_norm_cfg['std']

        # 反標準化
        img = img * std + mean
        img = img.clip(0, 255).astype(np.uint8)

        self.gt_bboxes = sample['gt_bboxes']

        # 檢查是否為 Tensor 類型，如果是則轉換為 numpy 陣列
        if isinstance(self.gt_bboxes, torch.Tensor):
            self.gt_bboxes = self.gt_bboxes.numpy()
        elif isinstance(self.gt_bboxes, list):
            self.gt_bboxes = np.array(self.gt_bboxes)

        image_id = os.path.splitext(sample['ori_filename'][0])[0]
        image_id = int(image_id)  # 將字串轉為整數後再轉回字串，去除開頭的零
        # print(f"image id: {image_id}")
        # print(f"pred result type: {type(self.predictions)}")

        self.pred_bboxes = self.search_by_image_id(image_id)
        # print(f"pred result: {pred_bboxes}")
        
        # 將 score 加入每個 bbox 的第五個位置
        if isinstance(self.pred_bboxes, pd.DataFrame):
            self.pred_bboxes = [
                [x[0], x[1], x[0] + x[2], x[1] + x[3], score]
                for x, score in zip(self.pred_bboxes['bbox'], self.pred_bboxes['score'])
                if score > self.confidence_threshold
            ]

        # 計算統計數據
        analysis_result = self.analyze_predictions(self.gt_bboxes[0], self.pred_bboxes)
        print(f"Analysis result: {analysis_result}")

        # 清空上一張圖片的內容
        self.ax1.clear()
        self.ax2.clear()

        # 顯示帶框圖片
        self.plot_image_with_boxes(img, self.gt_bboxes[0], self.ax1, title="Ground Truth Boxes")
        self.plot_image_with_boxes(img, self.pred_bboxes, self.ax2, title="Predicted Boxes")
        
        # 顯示統計結果
        analysis_text = f"Correct Predictions: {analysis_result['correct_predictions']}\n" \
                        f"Bird as Background: {analysis_result['bird_as_background']}\n" \
                        f"Background as Bird: {analysis_result['background_as_bird']}"
        
        # 在圖形上顯示分析結果
        self.ax2.text(0.05, 0.95, analysis_text, transform=self.ax2.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # 顯示圖片
        self.fig.canvas.draw()

    def plot_image_with_boxes(self, image, boxes, ax, title=""):
        '''
        為圖片添加bboxes 用於可視化
        '''
        
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

        # print(f"boxes: {boxes}")

        if isinstance(boxes, DataContainer):
            boxes = boxes.data.numpy()

        for img_boxes in boxes:
            if isinstance(img_boxes, DataContainer):
                img_boxes = img_boxes.data.numpy()

            if len(img_boxes) == 5:
                x1, y1, x2, y2 = img_boxes[:4]
                score = img_boxes[4]
            else:
                x1, y1, x2, y2 = img_boxes[:4]
                score = None

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            if score is not None:
                ax.text(x1, y1, f'{score:.2f}', color='black', fontsize=12, weight='bold',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.3))

    def next_image(self, event):
        '''切換到下一張圖片'''
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0  # 如果是最後一張，回到第一張
        self.update_image()

def print_unwatched(navigator=None):
    '''
    輸出true negative的目標
    '''

    for i in range(0, len(navigator.watched_bird_sizes)):
        print(f"unwatch size: {navigator.watched_bird_sizes[i]}")

def output_analyse_data_file(data:list=None , file_name="analyse_data.txt"):
    '''將資訊輸出到文件'''

    # 將未檢測鳥類的大小資訊輸出到文件
    output_file = file_name
    with open(output_file, "w") as file:
        for i in range(len(data)):
            bird_size = data[i]
            file.write(f"data: {bird_size}\n")

def output_background_as_bird_to_file(navigator: ImageNavigator=None , file_name="background_as_bird.txt"):
    '''將未檢測鳥類的大小資訊輸出到文件'''

    # 將未檢測鳥類的大小資訊輸出到文件
    output_file = file_name
    with open(output_file, "w") as file:
        for i in range(len(navigator.background_as_bird_size)):
            bird_size = navigator.background_as_bird_size[i]
            file.write(f"background as bird size: {bird_size}\n")

def search_confidence(navigator=None):
    '''
        查詢不同置信度下
        哪一種效果最好
    '''
    result = navigator.analyze_result()
    for i in range(0, len(navigator.unmatched_bird_sizes)):
        print(f"unwatch size: {navigator.unmatched_bird_sizes[i]}")

    confidence_thr = [i for i in range(0, 100, 5)]
    for i in range(0, len(confidence_thr)):
        confidence_thr[i] = confidence_thr[i] / 100

    for i in confidence_thr:
        navigator = ImageNavigator(dataset, json_file, cfg, False, i, False, True)
        result = navigator.analyze_result()
        print(f"confidence thrshold: {i}...")
        print(f"total_gt_birds: {result[0]}")
        print(f"total_correct_predictions: {result[1]}")
        print(f"total_bird_as_background: {result[2]}")
        print(f"total_background_as_bird: {result[3]}")
        print()



default_config_path = 'work_dirs/flow_cascade_rcnn_cos_swin_rfla_4stage_finetune/flow_cascade_rcnn_cos_swin_rfla_4stage_finetune.py'
default_json_file = 'coco_results.bbox.json'
args = setting_parser()

# 根據 --custom 標誌決定是否手動選擇路徑
if args.custom:
    # 使用 tkinter 打開檔案選擇器
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗

    config_path = filedialog.askopenfilename(title="Select Config File", filetypes=[("Python files", "*.py")]) or args.config
    json_file = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON files", "*.json")]) or args.json
else:
    config_path = default_config_path
    json_file = default_json_file

# 載入配置檔案
cfg = Config.fromfile(config_path)

# 確保 dataset 被正確加載
dataset = build_dataset(cfg.data.val)

# 初始化圖像導航器
navigator = ImageNavigator(
    dataset,
    json_file,
    cfg,
    show_image=args.show_image,
    confidence_threshold=args.confidence_threshold,
    process_info=args.process_info,
    return_result=args.return_result,
    get_gt=args.get_gt,
    get_unwatched=args.get_unmatched,
    get_background_as_bird=args.get_background_as_bird
)

output_analyse_data_file(navigator.gt_bird_sizes, "gt_bbox_data.txt")
# print_unwatched(navigator)
# output_background_as_bird_to_file(navigator)
# search_confidence(navigator)

# 顯示圖形
plt.show()
