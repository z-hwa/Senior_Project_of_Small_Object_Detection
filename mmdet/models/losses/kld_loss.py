import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class KLDLoss(nn.Module):
    """Kullback-Leibler Divergence Loss for BBox."""

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KLDLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): Predicted bboxes, shape (N, 4), format (cx, cy, w, h)
            target (Tensor): Target bboxes, shape (N, 4), format (cx, cy, w, h)
        """
        pred_mu, pred_sigma = pred[:, :2], pred[:, 2:] / 2
        target_mu, target_sigma = target[:, :2], target[:, 2:] / 2

        # 計算 KLD
        sigma_ratio = (target_sigma / pred_sigma).clamp(min=1e-6)
        trace_term = (sigma_ratio**2).sum(dim=1)
        mahalanobis_term = ((target_mu - pred_mu) ** 2 / pred_sigma**2).sum(dim=1)

        kld = 0.5 * (torch.log(sigma_ratio.prod(dim=1)) - 2 + trace_term + mahalanobis_term)

        # 加權
        loss = self.loss_weight * kld

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
