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
class StackCascadeRCNNV2(TwoStageDetector):
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
        super(StackCascadeRCNNV2, self).__init__(
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
        return super(StackCascadeRCNNV2, self).show_result(data, result, **kwargs)
    
    def count_past_mask(self, img_metas, img, expand_ratio=1.5, mask_value=1, past_prediction=None):
        """
        針對一個批次中的每張圖片，根據其過去的預測結果生成擴展的 mask，
        並將每個圖片的 mask 合併成一個批次的 mask。

        Args:
            img_metas (list[dict]): 包含每個圖片元數據的列表。每個 meta 應包含 'past_prediction'。
                                    'past_prediction' 是一個字典，包含 'past_range' (過去幀的索引列表)
                                    和 'past_det' (一個列表，每個元素是過去一幀的檢測結果，格式為 [N, 5] 的張量)。
            img (torch.Tensor): 形狀為 (B, C, H, W) 的當前批次圖像張量。
            expand_ratio (int or float): 邊界框擴展的倍率。預設為 2。
            mask_value (int): mask 中擴展區域每次偵測到添加的值。
            past_presiction: 包含過去預測結果的字典檔案

        Returns:
            torch.Tensor: 形狀為 (B, H, W) 的批次 mask 張量，其中擴展的預測框區域值為 `mask_value`，其餘為 1。
        """
        batch_size = len(img_metas)
        h, w = img.size()[2], img.size()[3]
        batch_mask = torch.zeros((batch_size, h, w), dtype=torch.uint8, device=img.device) # 創建一個初始值為1的批次 mask

        for b in range(batch_size):
            meta = img_metas[b]
            detections = []
            mask = torch.zeros((h, w), dtype=torch.uint8, device=img.device) # 創建當前圖片的 mask

            if meta.get('past_prediction') is not None:
                # test 沒有batch問題 都是單張測試
                past_pre = meta['past_prediction']['past_det']

                for i in range(0, len(past_pre)):
                    if past_pre[i] is not None:
                        for item in past_pre[i]:  # 遍歷內層列表
                            detections.append(torch.from_numpy(item).to(img.device))
                            # breakpoint()
            elif past_prediction is not None:
                # train
                past_pre = past_prediction['past_det']

                for i in range(0, len(past_pre)):
                    if past_pre[i][b] is not None:
                        det = past_pre[i][b][past_pre[i][b][:, 4] > 0]
                        for item in det:  # 遍歷內層列表
                                detections.append(item)

            
            if detections:
                if detections is not None:
                    # 假設每個類別的 bboxs 是 [x1, y1, x2, y2, score] 格式的張量

                    for box in detections:
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

                        # 計算擴展後的寬度和高度 (使用自訂的 expand_ratio)
                        width_expanded = int(width_orig * expand_ratio)
                        height_expanded = int(height_orig * expand_ratio)

                        # 計算擴展後的邊界框坐標
                        x1_expanded = max(0, center_x - width_expanded // 2)
                        y1_expanded = max(0, center_y - height_expanded // 2)
                        x2_expanded = min(w, center_x + (width_expanded + 1) // 2)
                        y2_expanded = min(h, center_y + (height_expanded + 1) // 2)

                        # 注意：PyTorch 的張量索引是 [row, column]，對應到圖像是 [y, x]
                        # 因此，我們需要使用 y 的範圍作為第一個索引，x 的範圍作為第二個索引
                        mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] += mask_value

            batch_mask[b] = mask

        # self.visualize_mask(mask=batch_mask[0], image=img)
        # breakpoint()
        return batch_mask

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      past_prediction=None,
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

        mask = self.count_past_mask(img_metas=img_metas, img=img, past_prediction=past_prediction)

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

        # breakpoint()
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
            resized_mask = F.interpolate(mask.float().unsqueeze(1), size=feature_map.shape[2:], mode='nearest').byte()
            enhancement = torch.ones_like(feature_map)

            # 擴展 resized_mask 的通道維度以匹配 enhancement
            expanded_mask = resized_mask.expand(-1, enhancement.shape[1], -1, -1)

            # 特徵強化和歸一化
            enhancement[expanded_mask == 2] = 1.5    # 上下帧重疊區域
            enhancement[expanded_mask == 1] = 1.25  # 只有上下帧其中之一覆蓋到 2

            enhanced_feature_map = feature_map * enhancement
            enhanced_features.append(enhanced_feature_map)

        if self.with_neck:
            x = self.neck(tuple(enhanced_features)) # 將增強後的特徵圖送入 neck
            return x
        else:
            return tuple(enhanced_features)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        mask = self.count_past_mask(img_metas=img_metas, img=img)

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.stack_extract_feat(img, mask)
        # breakpoint()

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

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