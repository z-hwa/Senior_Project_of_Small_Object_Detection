# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import torch
import torch.nn.functional as F


@DETECTORS.register_module()
class CascadeRCNN(TwoStageDetector):
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
        super(CascadeRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # 分離輸入資料
        curr_img = img[:, :3, :, :]  # 當前幀 RGB
        prev_img = img[:, 3:6, :, :]  # 上一幀 RGB
        flow = img[:, 6:, :, :]  # 光流 (2通道)

        # 特徵提取
        curr_feats = self.backbone(curr_img)
        prev_feats = self.backbone(prev_img)

        # 光流扭曲
        warped_prev_feats = []
        for prev_feat, curr_feat in zip(prev_feats, curr_feats):
            # 使用 grid_sample 進行扭曲
            grid = self.flow_to_grid(flow)  # 將光流轉換為 grid
            warped_prev_feat = F.grid_sample(prev_feat, grid, mode='bilinear', align_corners=False)
            warped_prev_feats.append(warped_prev_feat)

        # 特徵融合
        enhanced_curr_feats = []
        for warped_prev_feat, curr_feat in zip(warped_prev_feats, curr_feats):
            # 這裡可以使用不同的融合方式，例如元素級相加
            enhanced_curr_feat = warped_prev_feat / 2 + curr_feat
            enhanced_curr_feats.append(enhanced_curr_feat)

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