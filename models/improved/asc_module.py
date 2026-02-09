"""
ASC注意力模块 - Attention + Spatial + Channel
ASC (Attention-Spatial-Channel) Module for Enhanced Feature Extraction

本模块针对遥感飞机检测中的弱特征问题设计:
1. 通道注意力(CA): 增强目标通道响应，抑制背景通道
2. 空间注意力(SA): 聚焦目标区域，降低背景干扰
3. 自注意力(Self-Attn): 捕获长距离依赖，增强小目标特征

遥感图像中飞机目标的特点:
- 占比小(像素比例低)
- 方向任意(需要旋转不变性)
- 背景复杂(机场、停机坪等)
- 轮廓细节与背景对比度低
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """
    通道注意力机制
    通过全局平均池化和最大池化聚合空间信息，
    利用共享MLP学习通道间的依赖关系
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    空间注意力机制
    通过通道维度上的平均池化和最大池化生成空间描述符，
    经过卷积生成空间注意力图
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


class CoordinateAttention(nn.Module):
    """
    坐标注意力机制
    将位置信息嵌入通道注意力中，能够更精确地定位目标
    特别适合旋转目标检测场景
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 水平和垂直方向的编码
        x_h = self.pool_h(x)           # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, C, W, 1) -> (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class ASCModule(nn.Module):
    """
    ASC综合注意力模块
    整合通道注意力、空间注意力和坐标注意力
    
    流程: Input -> CA -> SA -> CoordAttn -> Output + Residual
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)
        self.coord_attn = CoordinateAttention(channels, reduction=32)

        # 残差连接的权重学习
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        residual = x
        out = self.channel_attn(x)
        out = self.spatial_attn(out)
        out = self.coord_attn(out)
        return self.alpha * out + (1 - self.alpha) * residual


class C2f_ASC(nn.Module):
    """
    集成ASC注意力的C2f模块
    在YOLOv8的C2f基础上添加ASC注意力增强
    
    用于替换backbone和neck中的特定C2f层
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)

        self.bottlenecks = nn.ModuleList(
            [self._make_bottleneck(self.c, self.c, shortcut, g) for _ in range(n)]
        )
        self.asc = ASCModule(c2)

    def _make_bottleneck(self, c1, c2, shortcut, g):
        return nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        out = self.cv2(torch.cat(y, 1))
        return self.asc(out)


if __name__ == '__main__':
    # 测试ASC模块
    x = torch.randn(2, 256, 32, 32)
    asc = ASCModule(256)
    out = asc(x)
    print(f"ASC Module - Input: {x.shape}, Output: {out.shape}")

    c2f_asc = C2f_ASC(256, 256, n=2)
    out2 = c2f_asc(x)
    print(f"C2f_ASC - Input: {x.shape}, Output: {out2.shape}")

    # 参数量
    params = sum(p.numel() for p in asc.parameters())
    print(f"ASC Parameters: {params / 1e6:.3f}M")

    params2 = sum(p.numel() for p in c2f_asc.parameters())
    print(f"C2f_ASC Parameters: {params2 / 1e6:.3f}M")
