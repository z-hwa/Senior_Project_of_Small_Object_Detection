# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import json
import os



@DETECTORS.register_module()
class StackCascadeRCNNFromJson(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 stack_threshold=None,
                 pred_path=None,
                 empty_path=None):
        super(StackCascadeRCNNFromJson, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.stack_threshold = 0.7 if stack_threshold == None else stack_threshold
        self.pred_anno = self.load_json(file_path=pred_path)
        self.empty_anno = self.load_json(file_path=empty_path)

        self.converted_results = self.convert_coco_predictions(self.pred_anno, self.empty_anno)
        self.c_id = self.pred_anno[0]["category_id"]

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(StackCascadeRCNNFromJson, self).show_result(data, result, **kwargs)
    
    def count_mask(self, img_metas, img):
        # 針對上一幀進行處理
        # 獲取上一批圖片得到的預測框位置(x1, y1, x2, y2)
        filename = img_metas[0]['filename']

        # 查詢預測結果表 進行推理
        sub_path = "/".join(filename.split('/')[-2:])
        current_frame_num_str = os.path.splitext(sub_path.split('/')[-1])[0]

        current_frame_num = int(current_frame_num_str)
        prev_frame_num = current_frame_num - 1
        if prev_frame_num >= 0:
            prev_frame_num_str = f"{prev_frame_num:05d}"  # 假設是5位數字
            prev_sub_path_parts = sub_path.split('/')
            prev_sub_path_parts[-1] = prev_frame_num_str + os.path.splitext(sub_path.split('/')[-1])[1]
            prev_sub_path = "/".join(prev_sub_path_parts)

            if prev_sub_path in self.converted_results and self.c_id in self.converted_results[prev_sub_path]:
                detections = [self.converted_results[prev_sub_path][self.c_id]]
            else:
                detections = []
        else:
            detections = []

        r = [detections] # self.test_fun(img=prev_frame, img_metas=img_metas, proposals=proposals)

        h, w = img.size()[2], img.size()[3]
        mask = torch.ones((h, w), dtype=torch.uint8, device=img.device) # 創建一個初始值為1的mask張量

        if r and r[0]: # 確保上一幀有預測框
            for bboxes in r[0]:
                # 假設每個類別的 bboxs 是 [x1, y1, x2, y2, score] 格式的張量
                valid_bboxes = bboxes # [bboxes[:, 4] > self.stack_threshold]
                for box in valid_bboxes:
                    # 使用 .item() 從單元素張量中提取 Python 數值
                    x1_orig = int(box[0].item())
                    y1_orig = int(box[1].item())
                    x2_orig = int(box[2].item())
                    y2_orig = int(box[3].item())

                    # 計算原始框的中心點
                    center_x = (x1_orig + x2_orig) // 2
                    center_y = (y1_orig + y2_orig) // 2

                    # 計算原始框的寬度和高度
                    width_orig = x2_orig - x1_orig
                    height_orig = y2_orig - y1_orig

                    # 計算擴展後的寬度和高度 (2倍)
                    width_expanded = width_orig * 2
                    height_expanded = height_orig * 2

                    # 計算擴展後的邊界框坐標
                    x1_expanded = max(0, center_x - width_expanded // 2)
                    y1_expanded = max(0, center_y - height_expanded // 2)
                    x2_expanded = min(w, center_x + (width_expanded + 1) // 2)
                    y2_expanded = min(h, center_y + (height_expanded + 1) // 2)

                    # 注意：PyTorch 的張量索引是 [row, column]，對應到圖像是 [y, x]
                    # 因此，我們需要使用 y 的範圍作為第一個索引，x 的範圍作為第二個索引
                    mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 2
        
        # self.visualize_  

        return mask

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        mask = self.count_mask(img_metas=img_metas, img=img)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.stack_extract_feat(img, mask)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def test_fun(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        # breakpoint()

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def stack_extract_feat(self, img, mask):
        backbone_features = self.backbone(img)

        if not isinstance(backbone_features, tuple) and not isinstance(backbone_features, list):
            backbone_features = [backbone_features]  # 轉換為列表方便統一處理

        enhanced_features = []
        for i, feature_map in enumerate(backbone_features):
            # 添加批次維度
            mask_with_batch = mask.unsqueeze(0)
            resized_mask = F.interpolate(mask_with_batch.float().unsqueeze(1), size=feature_map.shape[2:], mode='nearest').byte()
            enhancement = torch.ones_like(feature_map)

            # 擴展 resized_mask 的通道維度以匹配 enhancement
            expanded_mask = resized_mask.expand(-1, enhancement.shape[1], -1, -1)

            # 特徵強化和歸一化
            enhancement[expanded_mask == 2] = 2
            # enhancement[expanded_mask == 1] = 1  # 原本沒有標記的地方除以 2

            enhanced_feature_map = feature_map * enhancement
            enhanced_features.append(enhanced_feature_map)

        if self.with_neck:
            x = self.neck(tuple(enhanced_features)) # 將增強後的特徵圖送入 neck
            return x
        else:
            return tuple(enhanced_features)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        mask = self.count_mask(img_metas=img_metas, img=img)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.stack_extract_feat(img, mask)
        # breakpoint()

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

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

    def visualize_mask(self, mask: torch.Tensor, image: torch.Tensor, color_map: dict = {1: (255, 255, 255), 2: (0, 255, 0)}, alpha: float = 0.5) -> np.ndarray:
        """
        將 mask 可視化在背景圖像上，不同的數值對應不同的顏色並與圖像疊加。

        Args:
            mask (torch.Tensor): 形狀為 (H, W) 的整型張量，包含要可視化的 mask 數值。
            image (torch.Tensor): 形狀為 (B, C, H, W) 的 RGB 圖像張量，作為背景。
            color_map (dict): 一個字典，鍵是 mask 中的數值，值是對應的 RGB 顏色元組。
                            預設情況下，1 對應白色，2 對應綠色。
            alpha (float): Mask 疊加的透明度，取值範圍為 0 到 1。

        Returns:
            np.ndarray: 疊加了 mask 的 NumPy 圖像數組 (H, W, 3)，範圍在 0-255。
        """
        
        mask_np = mask.cpu().numpy().astype(np.uint8)
        # 從批次中取出第一張圖像作為背景
        image_np = image[0].permute(1, 2, 0).cpu().numpy()  # 將 (B, C, H, W) 轉換為 (H, W, C)
        image_np = (image_np * 255).astype(np.uint8) # 將圖像歸一化到 0-255

        overlay = np.zeros_like(image_np, dtype=np.uint8)
        for value, color in color_map.items():
            indices = (mask_np == value)
            overlay[indices] = color

        # 使用透明度疊加 mask
        blended = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)

        # 在計算完 mask 和 image 後直接顯示
        blended_image = blended
        plt.imshow(blended_image)
        plt.title("Mask Visualization")
        plt.axis('off')
        plt.show()