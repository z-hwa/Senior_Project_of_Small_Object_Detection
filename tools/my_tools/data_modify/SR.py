import torch
import cv2
import numpy as np
from basicsr.archs.swinir_arch import SwinIR

# 加載 SwinIR 模型（2 倍超解析度）
model = SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0)
model.load_state_dict(torch.load('swinir_sr_x2.pth'))
model.eval()

# 讀取低解析度圖片
img = cv2.imread('tools/my_tools/data_modify/in/06508.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

# 進行超解析度處理
with torch.no_grad():
    sr_img = model(img).squeeze(0).permute(1, 2, 0).numpy()

# 存儲結果
sr_img = (sr_img * 255).astype(np.uint8)
cv2.imwrite('high_res_bird.png', cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
