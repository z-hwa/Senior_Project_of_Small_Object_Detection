import pickle
import json
import os
from mmdet.datasets import CocoDataset
from pycocotools.coco import COCO

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(
        type='LoadPreviousFrameFromFile',
        to_float32=True,
        color_type='color',
        method='video'),
    dict(type='LoadOpticalFlowFromFile', method='video'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize_for_2img',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'),
                keys=['img'])
        ])
]

def convert_pkl_results_to_coco_json(pkl_file, ann_file, output_json_file):
    """
    將使用 pickle 儲存的檢測結果轉換為 COCO 格式的 JSON 文件。

    Args:
        pkl_file (str): pickle 文件的路徑，其中儲存了檢測結果 (results)。
        ann_file (str): COCO 格式的 annotations 文件路徑 (.json)。
        output_json_file (str): 輸出 COCO 格式 JSON 文件的路徑。
    """
    try:
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        print(f"成功載入 pickle 文件: {pkl_file}")
    except FileNotFoundError:
        print(f"錯誤: pickle 文件未找到: {pkl_file}")
        return
    except Exception as e:
        print(f"載入 pickle 文件時發生錯誤: {e}")
        return

    try:
        coco_gt = COCO(ann_file)
        print(f"成功載入 annotations 文件: {ann_file}")
    except FileNotFoundError:
        print(f"錯誤: annotations 文件未找到: {ann_file}")
        return
    except Exception as e:
        print(f"載入 annotations 文件時發生錯誤: {e}")
        return

    # 創建一個臨时的 CocoDataset 實例，用於使用其 results2json 方法
    # 我們只需要它的類別信息和圖像 ID 映射
    class DummyCocoDataset(CocoDataset):
        def __init__(self, ann_file):
            super().__init__(ann_file=ann_file, classes=None, pipeline=test_pipeline) # classes=None 會使用 ann_file 中的類別

        def __len__(self):
            return len(self.img_ids)

        def __getitem__(self, idx):
            raise NotImplementedError

    dummy_dataset = DummyCocoDataset(ann_file)
    dummy_dataset.coco = coco_gt # 確保使用載入的 COCO API
    dummy_dataset.img_ids = list(coco_gt.imgs.keys()) # 獲取圖像 IDs

    # 確保 results 的長度與圖像數量一致
    if len(results) != len(dummy_dataset):
        print(f"警告: results 的長度 ({len(results)}) 與 annotations 中的圖像數量 ({len(dummy_dataset)}) 不一致。")

    # 使用 results2json 方法將結果轉換為 COCO JSON 格式
    # 這裡我們假設 results 是 detection results (list[list])
    outfile_prefix = os.path.splitext(output_json_file)[0]
    result_files = dummy_dataset.results2json(results, outfile_prefix)

    if 'bbox' in result_files:
        print(f"檢測結果已儲存到: {result_files['bbox']}")
    if 'segm' in result_files:
        print(f"分割結果已儲存到: {result_files['segm']}")
    if 'proposal' in result_files:
        print(f"Proposal 結果已儲存到: {result_files['proposal']}")



if __name__ == "__main__":
    pkl_file = 'results.pkl'  # 替換為您的 pickle 文件路徑
    ann_file = '/root/Document/data/MVA2025/annotations/test_coco.json'  # 替換為您的 annotations 文件路徑
    output_json_file = 'coco_results.json'  # 替換為您希望輸出的 JSON 文件名

    convert_pkl_results_to_coco_json(pkl_file, ann_file, output_json_file)