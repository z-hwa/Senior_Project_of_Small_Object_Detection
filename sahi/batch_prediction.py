# import required functions, classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

model_type = "mmdet"
model_path = "work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/epoch_104.pth"
config_path = "work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/cascade_mask_rcnn_swin_finetune_rfla_4stage.py"
model_device = "cuda:0" # or 'cuda:0'
model_confidence_threshold = 0.7

slice_height = 800
slice_width = 800
overlap_height_ratio = 0.4
overlap_width_ratio = 0.4

source_image_dir = "data/mva2023_sod4bird_pub_test/images"
dataset_json_path = "data/mva2023_sod4bird_pub_test/annotations/public_test_coco_empty_ann.json"

predict(
    model_type=model_type,
    model_path=model_path,
    model_config_path=config_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
    dataset_json_path=dataset_json_path,
    name="cascade_mask_rcnn_swin_finetune_rfla_4stage_thr07",
    novisual=True,
    # export_whole_pickle=True
)