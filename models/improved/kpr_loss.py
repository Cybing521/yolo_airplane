"""
KPRLoss - 自适应融合回归损失函数
KPR (Kalman ProbIoU Regression) Loss for OBB Detection

本损失函数针对旋转目标检测的回归问题设计:
1. ProbIoU: 将旋转框建模为高斯分布，计算分布间的相似度
2. KFIoU: 基于卡尔曼滤波思想的旋转IoU
3. KPRLoss: 自适应权重融合ProbIoU和KFIoU

优势:
- 解决角度周期性问题(0°和180°等价)
- 对不同宽高比的目标自适应
- 收敛速度快，训练稳定
- 适合小样本场景下的训练

对应论文图4.11: 使用KPRLoss后损失曲线更平滑，收敛更快
- 基线在75个Epoch后才逐渐收敛
- KPRLoss在60个Epoch后就逐渐收敛
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


def xy_wh_r_to_gaussian(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将旋转框参数(x, y, w, h, angle)转换为二维高斯分布
    
    Args:
        pred: (N, 5) [x_center, y_center, width, height, angle_rad]
    
    Returns:
        mu: (N, 2) 均值
        sigma: (N, 2, 2) 协方差矩阵
    """
    x, y, w, h, angle = pred.unbind(-1)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # 旋转矩阵
    R = torch.stack([cos_a, -sin_a, sin_a, cos_a], dim=-1).reshape(-1, 2, 2)

    # 对角矩阵 (宽高的方差)
    S = torch.zeros_like(R)
    S[..., 0, 0] = (w / 2).pow(2) / 12  # 均匀分布的方差
    S[..., 1, 1] = (h / 2).pow(2) / 12

    # 协方差矩阵: Sigma = R @ S @ R^T
    sigma = R @ S @ R.transpose(-1, -2)
    mu = torch.stack([x, y], dim=-1)

    return mu, sigma


def probiou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    ProbIoU损失
    将旋转框建模为二维高斯分布，通过Bhattacharyya距离计算相似度
    
    Args:
        pred: (N, 5) 预测框 [x, y, w, h, angle]
        target: (N, 5) 目标框 [x, y, w, h, angle]
    
    Returns:
        loss: (N,) ProbIoU损失
    """
    mu_p, sigma_p = xy_wh_r_to_gaussian(pred)
    mu_t, sigma_t = xy_wh_r_to_gaussian(target)

    # Bhattacharyya距离
    sigma_sum = (sigma_p + sigma_t) / 2
    det_sum = sigma_sum[..., 0, 0] * sigma_sum[..., 1, 1] - sigma_sum[..., 0, 1] * sigma_sum[..., 1, 0]
    det_p = sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] * sigma_p[..., 1, 0]
    det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] * sigma_t[..., 1, 0]

    # 马氏距离项
    diff_mu = (mu_p - mu_t).unsqueeze(-1)
    sigma_sum_inv = torch.zeros_like(sigma_sum)
    sigma_sum_inv[..., 0, 0] = sigma_sum[..., 1, 1]
    sigma_sum_inv[..., 1, 1] = sigma_sum[..., 0, 0]
    sigma_sum_inv[..., 0, 1] = -sigma_sum[..., 0, 1]
    sigma_sum_inv[..., 1, 0] = -sigma_sum[..., 1, 0]
    sigma_sum_inv = sigma_sum_inv / (det_sum.unsqueeze(-1).unsqueeze(-1) + eps)

    mahal = diff_mu.transpose(-1, -2) @ sigma_sum_inv @ diff_mu
    mahal = mahal.squeeze(-1).squeeze(-1)

    # 行列式项
    det_term = 0.5 * torch.log(det_sum + eps) - 0.25 * torch.log(det_p + eps) - 0.25 * torch.log(det_t + eps)

    # Bhattacharyya距离
    bd = 0.125 * mahal + det_term

    # ProbIoU
    prob_iou = torch.exp(-bd)
    loss = 1 - prob_iou

    return loss.clamp(min=0, max=2)


def kfiou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    KFIoU损失
    基于卡尔曼滤波的旋转IoU计算
    
    通过建模预测框和真实框的高斯分布交集来近似IoU
    """
    mu_p, sigma_p = xy_wh_r_to_gaussian(pred)
    mu_t, sigma_t = xy_wh_r_to_gaussian(target)

    # 计算高斯分布的交集面积近似
    sigma_sum = sigma_p + sigma_t
    det_sum = sigma_sum[..., 0, 0] * sigma_sum[..., 1, 1] - sigma_sum[..., 0, 1] ** 2
    det_p = sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1] ** 2
    det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1] ** 2

    # KF-IoU
    t1 = torch.sqrt((4 * (det_p * det_t).clamp(min=eps)) / det_sum.clamp(min=eps))

    diff = mu_p - mu_t
    sigma_sum_inv_00 = sigma_sum[..., 1, 1] / (det_sum + eps)
    sigma_sum_inv_11 = sigma_sum[..., 0, 0] / (det_sum + eps)
    sigma_sum_inv_01 = -sigma_sum[..., 0, 1] / (det_sum + eps)

    exponent = -(diff[..., 0] ** 2 * sigma_sum_inv_00 +
                 diff[..., 1] ** 2 * sigma_sum_inv_11 +
                 2 * diff[..., 0] * diff[..., 1] * sigma_sum_inv_01) / 2

    t2 = torch.exp(exponent.clamp(min=-50, max=50))
    kf_iou = t1 * t2

    loss = 1 - kf_iou.clamp(min=0, max=1)

    return loss


class KPRLoss(nn.Module):
    """
    KPRLoss: 自适应融合ProbIoU和KFIoU的回归损失
    
    根据训练阶段和目标特性动态调整ProbIoU和KFIoU的权重:
    - 训练初期: 侧重ProbIoU(梯度稳定，收敛快)
    - 训练后期: 侧重KFIoU(精度高，定位准确)
    
    loss = alpha * ProbIoU + (1 - alpha) * KFIoU
    """

    def __init__(self, alpha: float = 0.6, dynamic_weight: bool = True,
                 total_epochs: int = 200):
        super().__init__()
        self.base_alpha = alpha
        self.dynamic_weight = dynamic_weight
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """更新当前epoch用于动态权重调整"""
        self.current_epoch = epoch

    def get_dynamic_alpha(self) -> float:
        """
        动态权重调度
        训练初期alpha较大(偏向ProbIoU)，后期逐渐降低(偏向KFIoU)
        """
        if not self.dynamic_weight:
            return self.base_alpha

        progress = self.current_epoch / max(self.total_epochs, 1)
        # 余弦退火: alpha从base_alpha逐渐降到base_alpha/2
        alpha = self.base_alpha * (1 + math.cos(math.pi * progress)) / 2
        alpha = max(alpha, self.base_alpha * 0.3)  # 最低权重
        return alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算KPRLoss
        
        Args:
            pred: (N, 5) 预测 [x, y, w, h, angle]
            target: (N, 5) 目标 [x, y, w, h, angle]
        
        Returns:
            loss: 标量损失值
        """
        alpha = self.get_dynamic_alpha()

        prob_loss = probiou_loss(pred, target)
        kf_loss = kfiou_loss(pred, target)

        # 自适应融合
        combined_loss = alpha * prob_loss + (1 - alpha) * kf_loss

        return combined_loss.mean()


class RotatedBBoxLoss(nn.Module):
    """
    完整的旋转框损失函数
    整合分类损失、回归损失(KPRLoss)和DFL损失
    """

    def __init__(self, num_classes: int = 1, use_kpr: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_kpr = use_kpr

        # 分类损失
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')

        # 回归损失
        if use_kpr:
            self.box_loss = KPRLoss(alpha=0.6, dynamic_weight=True)
        else:
            self.box_loss = None

        # 损失权重
        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5

    def forward(self, pred_boxes, pred_cls, target_boxes, target_cls):
        """计算总损失"""
        # 分类损失
        cls_loss = self.cls_loss(pred_cls, target_cls).mean()

        # 回归损失
        if self.use_kpr and self.box_loss is not None:
            box_loss = self.box_loss(pred_boxes, target_boxes)
        else:
            box_loss = probiou_loss(pred_boxes, target_boxes).mean()

        # 总损失
        total_loss = (self.box_weight * box_loss +
                     self.cls_weight * cls_loss)

        return total_loss, box_loss, cls_loss


if __name__ == '__main__':
    # 测试KPRLoss
    torch.manual_seed(42)

    pred = torch.randn(10, 5)
    target = torch.randn(10, 5)

    # ProbIoU
    prob_l = probiou_loss(pred, target)
    print(f"ProbIoU Loss: {prob_l.mean().item():.4f}")

    # KFIoU
    kf_l = kfiou_loss(pred, target)
    print(f"KFIoU Loss: {kf_l.mean().item():.4f}")

    # KPRLoss
    kpr = KPRLoss(alpha=0.6, dynamic_weight=True, total_epochs=200)
    for epoch in [0, 50, 100, 150, 200]:
        kpr.set_epoch(epoch)
        loss = kpr(pred, target)
        alpha = kpr.get_dynamic_alpha()
        print(f"Epoch {epoch:>3d}: KPRLoss={loss.item():.4f}, alpha={alpha:.3f}")
