from mmcv import Config
from mmdet.datasets import build_dataset
import matplotlib.pyplot as plt

# 載入測試管道的圖片
# 用以確認推理時的圖片長相

# 加載配置文件
cfg = Config.fromfile('configs/_MyPlan/Swin_rfla_4stage/cascade_rcnn_swin_finetune_rfla_3stage_pa.py')

# 構建測試數據集
test_dataset = build_dataset(cfg.data.test)

# 檢查測試數據集是否為空
print(f"Number of test samples: {len(test_dataset)}")

if len(test_dataset) > 0:
    # 測試第一張圖像
    sample = test_dataset[2]
    print("Sample keys:", sample.keys())

    # 提取 img
    img_container = sample['img'][0]  # 取第一個 DataContainer
    img_tensor = img_container.data  # 提取張量數據

    # 從 (C, H, W) 轉換為 (H, W, C)
    img = img_tensor.numpy().transpose(1, 2, 0)

    # 輸出圖片尺寸
    print(f"Image shape (H, W, C): {img.shape}")

    # 提取標準化參數
    img_norm_cfg = cfg.img_norm_cfg
    mean = img_norm_cfg['mean']
    std = img_norm_cfg['std']

    # 反標準化
    img = img * std + mean

    # 確保值在有效範圍內
    img = img.clip(0, 255)

    # 可視化
    plt.imshow(img.astype('uint8'))  # 將數據轉為整數型
    plt.title("Sample Image")
    plt.axis('off')
    plt.show()
else:
    print("Test dataset is empty! Please check your configuration.")
