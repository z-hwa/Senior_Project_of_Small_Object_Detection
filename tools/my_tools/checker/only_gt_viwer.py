import cv2
import json
import os

def draw_ground_truth(image_path, annotations, categories):
    """根據標註檔案繪製 ground truth 邊界框"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}")
        return

    for ann in annotations:
        bbox = ann['bbox']
        category_id = ann['category_id']
        print(category_id)
        print(categories)
        category_name = [cat['name'] for cat in categories if cat['id'] == category_id][0]

        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 綠色邊框

        # 顯示類別名稱
        label = f"{category_name}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Ground Truth', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'gt_{os.path.basename(image_path)}', image)

path = '/home/zhwa/Document/data/fake_output_directory'

# 標註檔案路徑
annotation_file = os.path.join(path,  'val/new_coco_dataset.json')  # 將 'your_annotation.json' 替換為你的標註檔案路徑

# 圖片資料夾路徑
image_folder = os.path.join(path,  'val/copy_paste_images') #將 '/path/to/your/images' 替換為你的圖片資料夾路徑

# 載入標註檔案
with open(annotation_file, 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

# 繪製 ground truth
for image_info in images:
    image_id = image_info['id']
    image_file_name = image_info['file_name']
    image_path = os.path.join(image_folder, image_file_name)

    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    draw_ground_truth(image_path, image_annotations, categories)