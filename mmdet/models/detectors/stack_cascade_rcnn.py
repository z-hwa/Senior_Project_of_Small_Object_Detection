# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F



@DETECTORS.register_module()
class StackCascadeRCNN(TwoStageDetector):
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
                 init_cfg=None):
        super(StackCascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

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
        return super(StackCascadeRCNN, self).show_result(data, result, **kwargs)
    
    def test_fun(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.ori_extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def ori_extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat(self, img, mask):
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

        prev_frame = img[:, 3:6, :, :]
        frame = img[:, 0:3, :, :]

        # 針對上一幀進行處理
        # 獲取上一批圖片得到的預測框位置(x1, y1, x2, y2)
        r = self.test_fun(img=prev_frame, img_metas=img_metas, proposals=proposals, rescale=rescale)

        h, w = img.size()[2], img.size()[3]
        mask = torch.ones((h, w), dtype=torch.uint8, device=img.device) # 創建一個初始值為1的mask張量

        if r and r[0]: # 確保上一幀有預測框
            for bboxs in r[0]:
                for box in bboxs:
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

        # DEBUG mask
        # img = self.visualize_mask(mask, frame)
        # plt.imshow(img)
        # plt.show()

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(frame, mask)
        breakpoint()

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def set_epoch(self, epoch, epochs):
        self.roi_head.epoch = epoch
        self.roi_head.epochs = epochs

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
        return blended