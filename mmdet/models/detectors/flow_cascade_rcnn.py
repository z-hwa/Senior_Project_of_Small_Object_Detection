# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2


@DETECTORS.register_module()
class FlowCascadeRCNN(TwoStageDetector):
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
        super(FlowCascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def visualize_feature_map_difference(self, feature_map1, feature_map2, curr_img, prev_img, channel_index, title="Feature Map Difference", scale_factor=4):
        """
        可視化兩個特徵圖的單個通道差異，並顯示原始圖像。

        Args:
            self: 類別實例。
            feature_map1 (torch.Tensor): 第一個特徵圖，形狀為 (1, C, H, W)。
            feature_map2 (torch.Tensor): 第二個特徵圖，形狀為 (1, C, H, W)。
            curr_img (torch.Tensor): 當前幀圖像，形狀為 (1, 3, H, W)。
            prev_img (torch.Tensor): 上一幀圖像，形狀為 (1, 3, H, W)。
            channel_index (int): 要可視化的通道索引。
            title (str): 顯示視窗的標題。
            scale_factor (int): 特徵圖的縮放倍數。
        """

        # 確保 feature_map1 和 feature_map2 是 PyTorch 張量
        if not isinstance(feature_map1, torch.Tensor) or not isinstance(feature_map2, torch.Tensor):
            raise TypeError("feature_map1 和 feature_map2 必須是 torch.Tensor 類型。")

        # 確保 curr_img 和 prev_img 是 PyTorch 張量
        if not isinstance(curr_img, torch.Tensor) or not isinstance(prev_img, torch.Tensor):
            raise TypeError("curr_img 和 prev_img 必須是 torch.Tensor 類型。")

        # 確保 feature_map1 和 feature_map2 的形狀為 (1, C, H, W)
        if len(feature_map1.shape) != 4 or feature_map1.shape[0] != 1 or len(feature_map2.shape) != 4 or feature_map2.shape[0] != 1:
            raise ValueError("feature_map1 和 feature_map2 必須是形狀為 (1, C, H, W) 的張量。")

        # 確保 curr_img 和 prev_img 的形狀為 (1, 3, H, W)
        if len(curr_img.shape) != 4 or curr_img.shape[0] != 1 or curr_img.shape[1] != 3 or len(prev_img.shape) != 4 or prev_img.shape[0] != 1 or prev_img.shape[1] != 3:
            raise ValueError("curr_img 和 prev_img 必須是形狀為 (1, 3, H, W) 的張量。")

        # 確保 channel_index 在有效範圍內
        if channel_index < 0 or channel_index >= feature_map1.shape[1] or channel_index < 0 or channel_index >= feature_map2.shape[1]:
            raise ValueError(f"channel_index 必須在 0 到 {min(feature_map1.shape[1], feature_map2.shape[1]) - 1} 的範圍內。")

        # 提取選定的通道，並使用 detach()
        channel_data1 = feature_map1[0, channel_index, :, :].detach().cpu().numpy()
        channel_data2 = feature_map2[0, channel_index, :, :].detach().cpu().numpy()

        # 計算差異
        difference = channel_data1 - channel_data2

        # 縮放差異圖
        height, width = difference.shape
        scaled_height, scaled_width = height * scale_factor, width * scale_factor
        scaled_difference = cv2.resize(difference, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        # 轉換圖像為 NumPy 陣列
        curr_img_np = curr_img[0].permute(1, 2, 0).detach().cpu().numpy()
        prev_img_np = prev_img[0].permute(1, 2, 0).detach().cpu().numpy()

        # 顯示圖像和差異圖
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 創建三個子圖

        axes[0].imshow(curr_img_np)
        axes[0].set_title("Current Image")

        axes[1].imshow(prev_img_np)
        axes[1].set_title("Previous Image")

        im = axes[2].imshow(scaled_difference, cmap='RdBu')  # 使用 'RdBu' 色彩映射
        axes[2].set_title(title)
        fig.colorbar(im, ax=axes[2])  # 顯示顏色條

        plt.show()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # 分離輸入資料

        curr_img = img[:, :3, :, :]  # 當前幀 RGB
        prev_img = img[:, 3:6, :, :]  # 上一幀 RGB
        flow = img[:, 6:, :, :]  # 光流 (2通道)

        # 特徵提取
        curr_feats = self.backbone(curr_img)
        prev_feats = self.backbone(prev_img)
        
        # breakpoint()

        # 光流扭曲
        warped_prev_feats = []
        for prev_feat, curr_feat in zip(prev_feats, curr_feats):
            # 調整光流尺寸
            flow_resized = F.interpolate(flow, size=prev_feat.shape[2:], mode='bilinear', align_corners=False)

            # 使用 grid_sample 進行扭曲
            grid = self.flow_to_grid(flow_resized)  # 將光流轉換為 grid
            warped_prev_feat = F.grid_sample(prev_feat, grid, mode='bilinear', align_corners=False)
            warped_prev_feats.append(warped_prev_feat)

        # 釋放光流張量
        del flow

        # print(type(curr_feats[0]))
        # self.visualize_feature_map_difference(curr_feats[0], prev_feats[0], curr_img, prev_img, 0)
        # breakpoint()

        # 特徵融合
        enhanced_curr_feats = []
        for warped_prev_feat, curr_feat in zip(warped_prev_feats, curr_feats):
            # 這裡可以使用不同的融合方式，例如元素級相加
            enhanced_curr_feat = warped_prev_feat + curr_feat
            enhanced_curr_feats.append(enhanced_curr_feat)

        # breakpoint()

        # Neck 網路
        if self.with_neck:
            x = self.neck(enhanced_curr_feats)
        else:
            x = enhanced_curr_feats

        return x
    
    def flow_to_grid(self, flow):
        """將光流轉換為 grid，用於 grid_sample."""
        b, _, h, w = flow.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(flow.device)  # (h, w, 2)
        grid = grid.unsqueeze(0).expand(b, h, w, 2)  # (b, h, w, 2)
        grid = grid + flow.permute(0, 2, 3, 1)  # (b, h, w, 2)
        grid[:, :, :, 0] = 2 * grid[:, :, :, 0] / (w - 1) - 1
        grid[:, :, :, 1] = 2 * grid[:, :, :, 1] / (h - 1) - 1
        return grid


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
        return super(CascadeRCNN, self).show_result(data, result, **kwargs)

    def set_epoch(self, epoch, epochs):
        self.roi_head.epoch = epoch
        self.roi_head.epochs = epochs