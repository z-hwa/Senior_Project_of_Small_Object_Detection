from .smooth_l1_loss import L1Loss, SmoothL1Loss, l1_loss, smooth_l1_loss
from .kld_loss import KLDLoss

import mmcv
import torch
import torch.nn as nn

class Adaptive_KLD_Smooth_Loss(nn.Module):
    def __init__(self, smooth_l1_weight=1.0, kld_weight=1.0, threshold=50):
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.kld_weight = kld_weight
        self.threshold = threshold  # 控制平滑 L1 的影響程度
        self.smooth_l1 = SmoothL1Loss()
        self.kld_loss = KLDLoss()

    def forward(self, pred, target):
        # 計算 bbox L1 誤差 (可用 IoU 或其他距離)
        bbox_diff = torch.abs(pred - target).mean(dim=1)  # shape: (N,)

        # 計算動態權重：當 bbox 差異較大時，Smooth L1 權重較高
        alpha = torch.clamp(1 - bbox_diff / self.threshold, min=0)
        beta = 1 - alpha

        # 計算 loss
        loss_smooth_l1 = self.smooth_l1(pred, target)
        loss_kld = self.kld_loss(pred, target)

        # 動態加權
        loss = beta.mean() * self.smooth_l1_weight * loss_smooth_l1 + alpha.mean() * self.kld_weight * loss_kld
        return loss
