import numpy as np
import torch
import cv2
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mmcv.parallel import DataContainer

'''
# CV DEMO用的
# 載入模型 實時推理圖片 並顯示結果
'''

class ImageNavigator:
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.index = 0

        # 創建圖形和兩個子圖
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self.update_image()  # 顯示當前圖片

        # 添加按鈕
        ax_next = plt.axes([0.8, 0.02, 0.1, 0.075])  # 按鈕的位置
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)

    def update_image(self):
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

        gt_bboxes = sample['gt_bboxes']
        
        # 檢查是否為 Tensor 類型，如果是則轉換為 numpy 陣列
        if isinstance(gt_bboxes, torch.Tensor):
            gt_bboxes = gt_bboxes.numpy()
        elif isinstance(gt_bboxes, list):
            gt_bboxes = np.array(gt_bboxes)

        # 使用模型進行推理
        result = inference_detector(self.model, img)
        pred_bboxes = result[0]  # 假設是第一個類別的預測框

        # 只選擇置信度大於0.5的框
        pred_bboxes = pred_bboxes[pred_bboxes[:, -1] > 0.5]

        # 清空上一張圖片的內容
        self.ax1.clear()
        self.ax2.clear()

        # 顯示帶框圖片
        self.plot_image_with_boxes(img, gt_bboxes[0], self.ax1, title="Ground Truth Boxes")
        self.plot_image_with_boxes(img, pred_bboxes, self.ax2, title="Predicted Boxes")
        
        # 顯示圖片
        self.fig.canvas.draw()

    def plot_image_with_boxes(self, image, boxes, ax, title=""):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

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
        # 切換到下一張圖片
        self.index += 1
        if self.index >= len(self.dataset):
            self.index = 0  # 如果是最後一張，回到第一張
        self.update_image()

# 載入配置檔案
cfg = Config.fromfile('work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/cascade_mask_rcnn_swin_finetune_rfla_4stage.py')

# 確保 dataset 被正確加載
dataset = build_dataset(cfg.data.val)

# 載入模型
model = init_detector(cfg, 'work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/epoch_104.pth', device='cuda:0')

# 初始化圖像導航器
navigator = ImageNavigator(dataset, model, cfg)

# 顯示圖形
plt.show()
