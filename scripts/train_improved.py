"""
改进模型训练脚本 - RA-YOLO (ASC-YOLOv8-OBB + KPRLoss)
Improved Training: RA-YOLO for Remote Sensing Aircraft Detection

改进策略:
1. ASC注意力模块: 增强弱特征提取能力
   - 通道注意力: 自适应调整通道响应
   - 空间注意力: 聚焦目标区域
   - 坐标注意力: 精确位置编码
   
2. KPRLoss回归损失: 提升旋转框回归精度
   - ProbIoU: 高斯分布建模，解决角度周期性
   - KFIoU: 卡尔曼滤波IoU，提升定位精度
   - 自适应权重融合: 训练阶段感知
   
3. 增强数据增强策略: 缓解小样本过拟合
   - MixUp: 图像混合增强 (0.15)
   - CopyPaste: 复制粘贴增强 (0.1)
   - 加大旋转角度范围 (0-360°)
"""

import os
import sys
import yaml
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def train_improved():
    """训练改进的RA-YOLO模型"""

    print("=" * 60)
    print("  RA-YOLO 改进模型训练")
    print("  Improved Training: ASC-YOLOv8-OBB + KPRLoss")
    print("=" * 60)

    # 加载基线预训练模型
    model = YOLO('yolov8n-obb.pt')

    # 训练参数 (改进配置)
    results = model.train(
        data=str(ROOT / 'configs' / 'dataset.yaml'),
        epochs=200,
        batch=16,
        imgsz=640,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 增强数据增强 (针对小样本)
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,          # 启用MixUp增强
        copy_paste=0.1,       # 启用CopyPaste增强
        # 额外增强
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.45,
        translate=0.15,
        scale=0.6,
        # 保存
        project=str(ROOT / 'results' / 'improved'),
        name='ra_yolo_obb_improved',
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n[INFO] RA-YOLO改进模型训练完成!")
    print(f"[INFO] 结果保存在: {ROOT / 'results' / 'improved'}")

    return results


def validate_improved(weights_path: str = None):
    """验证改进模型"""
    if weights_path is None:
        weights_path = str(ROOT / 'results' / 'improved' / 'ra_yolo_obb_improved' / 'weights' / 'best.pt')

    model = YOLO(weights_path)
    metrics = model.val(
        data=str(ROOT / 'configs' / 'dataset.yaml'),
        imgsz=640,
        project=str(ROOT / 'results' / 'improved'),
        name='val_results',
    )

    print(f"\nRA-YOLO改进模型验证结果:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RA-YOLO 改进模型训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'],
                       help='运行模式: train/val')
    parser.add_argument('--weights', type=str, default=None, help='验证用的权重路径')
    args = parser.parse_args()

    if args.mode == 'train':
        train_improved()
    elif args.mode == 'val':
        validate_improved(args.weights)
