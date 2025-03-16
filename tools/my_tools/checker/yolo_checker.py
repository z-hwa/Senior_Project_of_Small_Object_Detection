import mmcv
import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 設定模型配置與權重
config_file = 'work_dirs/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco.py'  # 你的YOLOX配置檔案
checkpoint_file = 'work_dirs/yolox_s_8x8_300e_coco/epoch_55.pth'  # 訓練好的模型權重

# 初始化模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

def sliding_window_inference(img, model, crop_size=(640, 640), stride=(320, 320), score_thr=0.3):
    """ 滑動窗口推理 """
    h, w, _ = img.shape
    ch, cw = crop_size
    sh, sw = stride
    results = []

    for y in range(0, h - ch + 1, sh):
        for x in range(0, w - cw + 1, sw):
            cropped_img = img[y:y + ch, x:x + cw]
            result = inference_detector(model, cropped_img)
            
            # 將結果轉換回原始圖片座標系
            for i, bboxes in enumerate(result):
                if len(bboxes) > 0:
                    for bbox in bboxes:
                        x1, y1, x2, y2, score = bbox
                        if score >= score_thr:
                            results.append((x1 + x, y1 + y, x2 + x, y2 + y, score, i))

    return results

def visualize_result(img, results):
    """ 顯示推理結果 """
    img_copy = img.copy()  # 避免修改原圖
    for x1, y1, x2, y2, score, label in results:
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f'{label}: {score:.2f}'
        cv2.putText(img_copy, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()

# 讀取圖片
img_path = 'data/train/images/09647.jpg'  # 測試圖片路徑
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 執行滑動窗口推理
results = sliding_window_inference(img, model)

# NMS (非最大抑制)
nms_bboxes, nms_inds = mmcv.ops.nms(  # 將輸出拆分為兩個張量
    torch.tensor([[x1, y1, x2, y2] for x1, y1, x2, y2, score, label in results], dtype=torch.float32),
    torch.tensor([score for x1, y1, x2, y2, score, label in results], dtype=torch.float32),
    iou_threshold=0.5
)

nms_results_final = []
for idx in nms_inds:  # 只使用索引張量
    nms_results_final.append(results[idx])

# 顯示結果
visualize_result(img, nms_results_final)