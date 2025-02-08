import os
import cv2
import numpy as np
from tqdm import tqdm

'''
這段程式的特點
✅ 支持透明背景 (RGBA 圖片)：能正確裁剪包含 alpha 通道的 PNG 圖片。
✅ 適應 RGB 圖片（無透明度）：若是黑色背景的 RGB 圖片，也能找到範圍裁剪。
✅ 自動建立輸出資料夾，避免手動建立的麻煩。
✅ 使用 tqdm 顯示進度條，適合處理大量圖片。

你可以修改 input_folder 和 output_folder 來適應你的需求，跑完後就能在 輸出資料夾 看到貼合邊界的 PNG 圖片了！ 🚀
'''

def crop_and_save_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for image_name in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_folder, image_name)
        output_path = os.path.join(output_folder, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"無法讀取 {image_name}，跳過...")
            continue

        if image.shape[2] == 4:  # 如果有 Alpha 通道 (透明背景)
            mask = image[:, :, 3] > 0
        else:  # 如果沒有 Alpha，則使用亮度檢查
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = gray > 0

        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            print(f"跳過空白圖片 {image_name}")
            continue

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

        cropped_image = image[y_min:y_max, x_min:x_max]

        cv2.imwrite(output_path, cropped_image)  # 直接寫入磁碟
        del image, cropped_image  # **手動釋放記憶體**
    
    print(f"所有圖片已處理完成，輸出至 {output_folder}")

input_folder = "data/birds_test_rmg"
output_folder = "data/birds_test_rmg"
crop_and_save_images(input_folder, output_folder)
