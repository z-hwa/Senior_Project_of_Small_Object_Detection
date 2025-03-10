# arrange an instance segmentation model for test
from sahi.utils.mmdet import (
    download_mmdet_cascade_mask_rcnn_model,
    download_mmdet_config,
)

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download cascade mask rcnn model&config
model_path = "work_dirs/yolox_s_8x8_300e_coco/epoch_10.pth"
config_path = "work_dirs/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco.py"

# download test images into demo_data folder
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
# download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = AutoDetectionModel.from_pretrained(
    model_type='mmdet',
    model_path=model_path,
    config_path=config_path,
    confidence_threshold=0.5,
    image_size=(3840, 2160),
    device="cuda:0", # or 'cuda:0'
)

# result = get_prediction("sahi/demo_data/00012.jpg", detection_model)
result = get_sliced_prediction(
    "sahi/demo_data/00012.jpg",
    detection_model,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

output = result.to_coco_predictions(image_id=1)
print(output)

result.export_visuals(export_dir="sahi/demo_data/")

# Image("sahi/demo_data/prediction_visual.png")