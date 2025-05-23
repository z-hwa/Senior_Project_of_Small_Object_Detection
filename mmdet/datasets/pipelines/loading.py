# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None

import os.path as osp
import io
import cv2
import json
import os
import torch

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

@PIPELINES.register_module()
class LoadPastHard:

    def __init__(self,
                 pred_path=None,
                 empty_path=None,
                 past_range=1,
                 max_preds=20):
        
        self.past_range = past_range

        # 來源路徑
        self.pred_anno = self.load_json(file_path=pred_path)
        self.empty_anno = self.load_json(file_path=empty_path)

        self.converted_results = self.convert_coco_predictions(self.pred_anno, self.empty_anno)
        self.c_id = self.pred_anno[0]["category_id"]
        self.max_preds = max_preds  # 保存最大預測數

    def __call__(self, results):

        # 針對上一幀進行處理
        # 獲取上一批圖片得到的預測框位置(x1, y1, x2, y2)
        filename = results['img_info']['filename']

        # 查詢預測結果表 進行推理
        sub_path = "/".join(filename.split('/')[-2:])
        current_frame_num_str = os.path.splitext(sub_path.split('/')[-1])[0]

        if filename in self.converted_results:
            all_detections_bboxes = self.converted_results[filename][self.c_id]
            all_detections_bboxes = all_detections_bboxes[:, :4]
            all_detections_bboxes = torch.from_numpy(all_detections_bboxes)
        else:
            all_detections_bboxes = torch.empty(0, 4)  # 設定為空的 Tensor

        gt_bboxes = torch.from_numpy(results['gt_bboxes']) # 轉換為 Tensor

        if gt_bboxes.numel() > 0 and all_detections_bboxes.numel() > 0:
            # 計算所有檢測框和真實框之間的 IoU
            overlaps = self.bbox_overlaps(all_detections_bboxes, gt_bboxes)

            # 對於每一個檢測框，找到與之 IoU 最大的真實框及其 IoU 值
            max_overlaps, argmax_overlaps = overlaps.max(dim=1)

            # 將 IoU 小於一定閾值的檢測框視為沒有正確預測的框
            iou_threshold = 0.1  # 可以調整這個閾值
            incorrectly_predicted_mask = max_overlaps < iou_threshold

            # 獲取沒有正確預測的檢測框
            hard_examples_bboxes = all_detections_bboxes[incorrectly_predicted_mask]
            hard_examples = hard_examples_bboxes

        elif all_detections_bboxes.numel() > 0 and gt_bboxes.numel() == 0:
            # 如果有檢測框但沒有真實框，則所有檢測框都可能是假陽性，可以視為困難樣本
            hard_examples = all_detections_bboxes
        else:
            # 設為空
            hard_examples = all_detections_bboxes

        # print(all_detections_bboxes)
        # print(gt_bboxes)
        # print(max_overlaps)
        # print(hard_examples)
        # breakpoint()

        results['hard_example'] = { "hard": hard_examples.tolist() }

        return results
    
    def bbox_overlaps(self, bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
        """Calculate overlap between two set of bboxes.

        FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
        Note:
            Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
            there are some new generated variable when calculating IOU
            using bbox_overlaps function:

            1) is_aligned is False
                area1: M x 1
                area2: N x 1
                lt: M x N x 2
                rb: M x N x 2
                wh: M x N x 2
                overlap: M x N x 1
                union: M x N x 1
                ious: M x N x 1

                Total memory:
                    S = (9 x N x M + N + M) * 4 Byte,

                When using FP16, we can reduce:
                    R = (9 x N x M + N + M) * 4 / 2 Byte
                    R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                    Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                            N + 1 < 3 * N, when N or M is 1.

                Given M = 40 (ground truth), N = 400000 (three anchor boxes
                in per grid, FPN, R-CNNs),
                    R = 275 MB (one times)

                A special case (dense detection), M = 512 (ground truth),
                    R = 3516 MB = 3.43 GB

                When the batch size is B, reduce:
                    B x R

                Therefore, CUDA memory runs out frequently.

                Experiments on GeForce RTX 2080Ti (11019 MiB):

                |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
                |:----:|:----:|:----:|:----:|:----:|:----:|
                |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
                |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
                |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
                |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

            2) is_aligned is True
                area1: N x 1
                area2: N x 1
                lt: N x 2
                rb: N x 2
                wh: N x 2
                overlap: N x 1
                union: N x 1
                ious: N x 1

                Total memory:
                    S = 11 x N * 4 Byte

                When using FP16, we can reduce:
                    R = 11 x N * 4 / 2 Byte

            So do the 'giou' (large than 'iou').

            Time-wise, FP16 is generally faster than FP32.

            When gpu_assign_thr is not -1, it takes more time on cpu
            but not reduce memory.
            There, we can reduce half the memory and keep the speed.

        If ``is_aligned`` is ``False``, then calculate the overlaps between each
        bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
        pair of bboxes1 and bboxes2.

        Args:
            bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
            bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                B indicates the batch dim, in shape (B1, B2, ..., Bn).
                If ``is_aligned`` is ``True``, then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection over
                foreground) or "giou" (generalized intersection over union).
                Default "iou".
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.
            eps (float, optional): A value added to the denominator for numerical
                stability. Default 1e-6.

        Returns:
            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

        Example:
            >>> bboxes1 = torch.FloatTensor([
            >>>     [0, 0, 10, 10],
            >>>     [10, 10, 20, 20],
            >>>     [32, 32, 38, 42],
            >>> ])
            >>> bboxes2 = torch.FloatTensor([
            >>>     [0, 0, 10, 20],
            >>>     [0, 10, 10, 19],
            >>>     [10, 10, 20, 20],
            >>> ])
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
            >>> assert overlaps.shape == (3, 3)
            >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
            >>> assert overlaps.shape == (3, )

        Example:
            >>> empty = torch.empty(0, 4)
            >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
            >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
            >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
            >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
        """

        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        # Either the boxes are empty or the length of boxes' last dimension is 4
        assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
        assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

        # 修正 bboxes1
        bboxes1[:, [0, 2]] = torch.sort(bboxes1[:, [0, 2]], dim=1)[0]  # 確保 x_min ≤ x_max
        bboxes1[:, [1, 3]] = torch.sort(bboxes1[:, [1, 3]], dim=1)[0]  # 確保 y_min ≤ y_max

        # 修正 bboxes2
        bboxes2[:, [0, 2]] = torch.sort(bboxes2[:, [0, 2]], dim=1)[0]
        bboxes2[:, [1, 3]] = torch.sort(bboxes2[:, [1, 3]], dim=1)[0]

        assert (bboxes1[..., 2] >= bboxes1[..., 0]).all()
        assert (bboxes1[..., 3] >= bboxes1[..., 1]).all()
        assert (bboxes2[..., 2] >= bboxes2[..., 0]).all()
        assert (bboxes2[..., 3] >= bboxes2[..., 1]).all()

        # Batch dim must be the same
        # Batch dim: (B1, B2, ... Bn)
        assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
        batch_shape = bboxes1.shape[:-2]

        rows = bboxes1.size(-2)
        cols = bboxes2.size(-2)
        if is_aligned:
            assert rows == cols

        if rows * cols == 0:
            if is_aligned:
                return bboxes1.new(batch_shape + (rows, ))
            else:
                return bboxes1.new(batch_shape + (rows, cols))

        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

        if is_aligned:
            lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
            rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

            wh = fp16_clamp(rb - lt, min=0)
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1 + area2 - overlap
            else:
                union = area1
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
        else:
            lt = torch.max(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
            rb = torch.min(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

            wh = fp16_clamp(rb - lt, min=0)
            overlap = wh[..., 0] * wh[..., 1]

            if mode in ['iou', 'giou']:
                union = area1[..., None] + area2[..., None, :] - overlap
            else:
                union = area1[..., None]
            if mode == 'giou':
                enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                        bboxes2[..., None, :, :2])
                enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                        bboxes2[..., None, :, 2:])

        eps = union.new_tensor([eps])
        union = torch.max(union, eps)
        ious = overlap / union
        if mode in ['iou', 'iof']:
            return ious
        # calculate gious
        enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps)
        gious = ious - (enclose_area - union) / enclose_area

        # print(f"enclose area {enclose_area}")

        return gious

       
    def load_json(self, file_path):
        """載入 JSON 檔案"""
        with open(file_path, "r") as f:
            return json.load(f)

    def convert_coco_predictions(self, coco_predictions, empty_annotations):
        """
        將 COCO 預測結果轉換為 {file_name: tensor_list} 結構。
        
        :param coco_predictions: COCO 格式的預測結果 (list of dicts)
        :param empty_annotations: 原始空標註資料，應包含 {"images": [{"id": X, "file_name": "path/to/image"}, ...]}
        :return: dict，key 為 file_name，value 為該圖片的預測結果 tensor list
        """
        # 建立 image_id 到 file_name 的映射
        image_id_to_file = {img["id"]: img["file_name"] for img in empty_annotations["images"]}
        
        # 建立 {file_name: list of tensors} 結構
        results = {}
        for pred in coco_predictions:
            image_id = pred["image_id"]
            file_name = image_id_to_file.get(image_id, None)
            if file_name is None:
                continue
            
            bbox = pred["bbox"]  # [x, y, w, h]
            score = pred["score"]
            category_id = pred["category_id"]

            # 轉換為 [x1, y1, x2, y2, score]，符合 MMDetection 格式
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            bbox_data = np.array([x1, y1, x2, y2, score])

            # 初始化該圖片的分類列表
            if file_name not in results:
                results[file_name] = {}

            # 把 bbox 存入對應類別的 list
            if category_id not in results[file_name]:
                results[file_name][category_id] = []
            
            results[file_name][category_id].append(bbox_data)
        
        # 把 list 轉成 numpy array，確保格式一致
        for file_name in results:
            for category_id in results[file_name]:
                results[file_name][category_id] = np.array(results[file_name][category_id])
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'past_annotations_range={self.past_range})')
        return repr_str


@PIPELINES.register_module()
class LoadPastAnnotations:

    def __init__(self,
                 stack_threshold=None,
                 pred_path=None,
                 empty_path=None,
                 past_range=1,
                 max_preds=20):
        
        self.past_range = past_range

        # 來源路徑
        self.stack_threshold = 0.7 if stack_threshold == None else stack_threshold
        self.pred_anno = self.load_json(file_path=pred_path)
        self.empty_anno = self.load_json(file_path=empty_path)

        self.converted_results = self.convert_coco_predictions(self.pred_anno, self.empty_anno)
        self.c_id = self.pred_anno[0]["category_id"]
        self.max_preds = max_preds  # 保存最大預測數

    def __call__(self, results):

        # 針對上一幀進行處理
        # 獲取上一批圖片得到的預測框位置(x1, y1, x2, y2)
        filename = results['img_info']['filename']

        # 查詢預測結果表 進行推理
        sub_path = "/".join(filename.split('/')[-2:])
        current_frame_num_str = os.path.splitext(sub_path.split('/')[-1])[0]

        current_frame_num = int(current_frame_num_str)
        detections = []

        # past
        prev_frame_num = current_frame_num - 1
        if prev_frame_num > 0:
            frame_num_str = f"{prev_frame_num:05d}"  # 假設是5位數字
            sub_path_parts = sub_path.split('/')
            sub_path_parts[-1] = frame_num_str + os.path.splitext(sub_path.split('/')[-1])[1]
            sub_path = "/".join(sub_path_parts)

        if sub_path in self.converted_results and self.c_id in self.converted_results[sub_path]:
            preds = self.converted_results[sub_path][self.c_id]
            valid_preds = preds[preds[:, 4] > self.stack_threshold]
            num_valid = valid_preds.shape[0]
            if num_valid < self.max_preds:
                padding = np.zeros((self.max_preds - num_valid, 5), dtype=valid_preds.dtype)
                valid_preds_padded = np.concatenate([valid_preds, padding], axis=0)
            elif num_valid > self.max_preds:
                valid_preds_padded = valid_preds[:self.max_preds]
            else:
                valid_preds_padded = valid_preds
            detections.append(valid_preds_padded)
        else:
            detections.append(np.zeros((self.max_preds, 5)))  # 填充 20 個分數為 0 的框

        # future
        fut_frame_num = current_frame_num + 1
        if fut_frame_num < len(self.converted_results):
            frame_num_str = f"{fut_frame_num:05d}"  # 假設是5位數字
            sub_path_parts = sub_path.split('/')
            sub_path_parts[-1] = frame_num_str + os.path.splitext(sub_path.split('/')[-1])[1]
            sub_path = "/".join(sub_path_parts)

        if sub_path in self.converted_results and self.c_id in self.converted_results[sub_path]:
            preds = self.converted_results[sub_path][self.c_id]
            valid_preds = preds[preds[:, 4] > self.stack_threshold]
            num_valid = valid_preds.shape[0]
            if num_valid < self.max_preds:
                padding = np.zeros((self.max_preds - num_valid, 5), dtype=valid_preds.dtype)
                valid_preds_padded = np.concatenate([valid_preds, padding], axis=0)
            elif num_valid > self.max_preds:
                valid_preds_padded = valid_preds[:self.max_preds]
            else:
                valid_preds_padded = valid_preds
            detections.append(valid_preds_padded)
        else:
            detections.append(np.zeros((self.max_preds, 5)))  # 填充 20 個分數為 0 的框

        # Stack the previous frame image with the original image
        results['past_prediction'] = { "past_det": detections # list 每一個位置相當於過去k帧 現在只支援一帧
                                       }
        # # Add breakpoint here
        # breakpoint()

        return results
       
    def load_json(self, file_path):
        """載入 JSON 檔案"""
        with open(file_path, "r") as f:
            return json.load(f)

    def convert_coco_predictions(self, coco_predictions, empty_annotations):
        """
        將 COCO 預測結果轉換為 {file_name: tensor_list} 結構。
        
        :param coco_predictions: COCO 格式的預測結果 (list of dicts)
        :param empty_annotations: 原始空標註資料，應包含 {"images": [{"id": X, "file_name": "path/to/image"}, ...]}
        :return: dict，key 為 file_name，value 為該圖片的預測結果 tensor list
        """
        # 建立 image_id 到 file_name 的映射
        image_id_to_file = {img["id"]: img["file_name"] for img in empty_annotations["images"]}
        
        # 建立 {file_name: list of tensors} 結構
        results = {}
        for pred in coco_predictions:
            image_id = pred["image_id"]
            file_name = image_id_to_file.get(image_id, None)
            if file_name is None:
                continue
            
            bbox = pred["bbox"]  # [x, y, w, h]
            score = pred["score"]
            category_id = pred["category_id"]

            # 轉換為 [x1, y1, x2, y2, score]，符合 MMDetection 格式
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            bbox_data = np.array([x1, y1, x2, y2, score])

            # 初始化該圖片的分類列表
            if file_name not in results:
                results[file_name] = {}

            # 把 bbox 存入對應類別的 list
            if category_id not in results[file_name]:
                results[file_name][category_id] = []
            
            results[file_name][category_id].append(bbox_data)
        
        # 把 list 轉成 numpy array，確保格式一致
        for file_name in results:
            for category_id in results[file_name]:
                results[file_name][category_id] = np.array(results[file_name][category_id])
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'past_annotations_range={self.past_range})')
        return repr_str

@PIPELINES.register_module()
class LoadOpticalFlowFromFile:
    """Load optical flow from file.

    Required keys are "img_info" (a dict that must contain the key "filename").
    Adds the optical flow (ndarray) to the "optical_flow" key.

    Args:
        file_dir (str): Directory containing the original images.
        file_prefix (str): Prefix of the original image filenames.
        flow_dir (str): Directory containing the optical flow files.
        flow_prefix (str): Prefix of the optical flow filenames.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 file_dir="copy_paste_images",
                 file_prefix="copy_paste_",
                 flow_dir="optical_flow",
                 flow_prefix="flow_",
                 file_client_args=dict(backend='disk'),
                 quantization_factor=10,
                 method='fake_dir'):
        self.file_dir = file_dir
        self.file_prefix = file_prefix
        self.flow_dir = flow_dir
        self.flow_prefix = flow_prefix
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.quantization_factor = quantization_factor  # 保存量化因子
        self.method = method

    def visualize_optical_flow(self, flow):
        import cv2

        """可視化光流場"""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Optical Flow', bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __call__(self, results):
        """呼叫函數從檔案載入光流。

        Args:
            results (dict): 來自 :obj:`mmdet.CustomDataset` 的結果字典。

        Returns:
            dict: 包含光流 (ndarray) 的字典。
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        flow_filename = ''
        if self.method == 'fake_dir':
            # 構建光流檔案名稱
            flow_filename = filename.replace(self.file_dir, self.flow_dir).replace(self.file_prefix, self.flow_prefix).replace(".jpg", ".npy")
        elif self.method == 'video':
            # 找出上一幀的檔案名稱
            curr_file_name = results['img_info']['filename']
            folder_name, frame_name = curr_file_name.split('/')
            frame_number = int(frame_name.split('.')[0])

            flow_file_name = f'optical_flow/{folder_name}/flow_{frame_number:05d}.npy'  # 假設幀號為 5 位數字
            flow_filename = osp.join(results['img_prefix'], flow_file_name)

        flow_bytes = self.file_client.get(flow_filename)
        flow = np.load(io.BytesIO(flow_bytes)) #直接讀取

        # 反量化
        if flow.dtype == np.int16:
            flow = flow.astype(np.float32) / self.quantization_factor

        # 調整光流尺寸
        original_h, original_w = results['img'].shape[:2]
        flow = cv2.resize(flow, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        results['img'] = np.concatenate((results['img'], flow), axis=2)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"file_dir='{self.file_dir}', "
                    f"file_prefix='{self.file_prefix}', "
                    f"flow_dir='{self.flow_dir}', "
                    f"flow_prefix='{self.flow_prefix}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadPreviousFrameFromFile:
    """Load previous frame image from file and stack it with the original image.

    Required keys are "img" and "img_info" (a dict that must contain the
    key "filename"). The "img" key should already contain the loaded original
    image (ndarray). Adds the previous frame image to the "img" key.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk'),
                 file_dir="copy_paste_images",
                 file_prefix="copy_paste_",
                 last_frame_dir="prev_frames",
                 last_frame_prefix="prev_",
                 method='fake_dir'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.method = method

        # 來源路徑
        self.file_dir = file_dir
        self.file_prefix = file_prefix
        self.last_frame_dir = last_frame_dir
        self.last_frame_prefix = last_frame_prefix

    def __call__(self, results):
        """Call functions to load previous frame image and stack it with the original image.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains the original image stacked with the previous frame image.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        prev_file_name = ''
        if self.method == 'video':
            # 找出上一幀的檔案名稱
            curr_file_name = results['img_info']['filename']
            folder_name, frame_name = curr_file_name.split('/')
            frame_number = int(frame_name.split('.')[0])
            prev_frame_number = frame_number - 1

            if prev_frame_number > -1:
                # 第一幀的上一幀設為自己
                if prev_frame_number == 0:
                    prev_frame_number = 1

                prev_file_name = f'{folder_name}/{prev_frame_number:05d}.jpg'  # 假設幀號為 5 位數字
                prev_filename = osp.join(results['img_prefix'], prev_file_name)
        elif self.method == 'fake_dir':
            # Construct the previous frame filename
            prev_filename = filename.replace(self.file_dir, self.last_frame_dir).replace(self.file_prefix, self.last_frame_prefix)

        img_bytes = self.file_client.get(prev_filename)
        prev_img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            prev_img = prev_img.astype(np.float32)

        # Stack the previous frame image with the original image
        results['img'] = np.concatenate((results['img'], prev_img), axis=2)
    
        # # Add breakpoint here
        # breakpoint()

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

@PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromWebcam(LoadImageFromFile):
    """Load an image from webcam.

    Similar with :obj:`LoadImageFromFile`, but the image read from webcam is in
    ``results['img']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadMultiChannelImageFromFiles:
    """Load multi-channel images from a list of separate channel files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", which is expected to be a list of filenames).
    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='unchanged',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load multiple images and get images meta
        information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']

        img = []
        for name in filename:
            img_bytes = self.file_client.get(name)
            img.append(mmcv.imfrombytes(img_bytes, flag=self.color_type))
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        denorm_bbox (bool): Whether to convert bbox from relative value to
            absolute value. Only used in OpenImage Dataset.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.denorm_bbox = denorm_bbox
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask annotations.
                If ``self.poly2mask`` is set ``True``, `gt_mask` will contain
                :obj:`PolygonMasks`. Otherwise, :obj:`BitmapMasks` is used.
        """

        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        results['gt_semantic_seg'] = mmcv.imfrombytes(
            img_bytes, flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'poly2mask={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadHardNegatives:
    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """
        results['bbox_fields'].append('hard_negatives')
        return results


@PIPELINES.register_module()
class LoadPanopticAnnotations(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: True.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=True,
                 with_seg=True,
                 file_client_args=dict(backend='disk')):
        if rgb2id is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(LoadPanopticAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=True,
            denorm_bbox=False,
            file_client_args=file_client_args)

    def _load_masks_and_semantic_segs(self, results):
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from `0` to
        `num_things - 1`, the background label is from `num_things` to
        `num_things + num_stuff - 1`, 255 means the ignored label (`VOID`).

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded mask and semantic segmentation
                annotations. `BitmapMasks` is used for mask annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename = osp.join(results['seg_prefix'],
                            results['ann_info']['seg_map'])
        img_bytes = self.file_client.get(filename)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + 255  # 255 as ignore

        for mask_info in results['ann_info']['masks']:
            mask = (pan_png == mask_info['id'])
            gt_seg = np.where(mask, mask_info['category'], gt_seg)

            # The legal thing masks
            if mask_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

        if self.with_seg:
            results['gt_semantic_seg'] = gt_seg
            results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        """Call function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            results = self._load_masks_and_semantic_segs(results)

        return results


@PIPELINES.register_module()
class LoadProposals:
    """Load proposal pipeline.

    Required key is "proposals". Updated keys are "proposals", "bbox_fields".

    Args:
        num_max_proposals (int, optional): Maximum number of proposals to load.
            If not specified, all proposals will be loaded.
    """

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        """Call function to load proposals from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded proposal annotations.
        """

        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                f'but found {proposals.shape}')
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(num_max_proposals={self.num_max_proposals})'


@PIPELINES.register_module()
class FilterAnnotations:
    """Filter invalid annotations.

    Args:
        min_gt_bbox_wh (tuple[int]): Minimum width and height of ground truth
            boxes.
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Default: True
    """

    def __init__(self, min_gt_bbox_wh, keep_empty=True):
        # TODO: add more filter options
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def __call__(self, results):
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])
        if not keep.any():
            if self.keep_empty:
                return None
            else:
                return results
        else:
            keys = ('gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg')
            for key in keys:
                if key in results:
                    results[key] = results[key][keep]
            return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(min_gt_bbox_wh={self.min_gt_bbox_wh},' \
               f'always_keep={self.always_keep})'
