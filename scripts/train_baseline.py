"""
基线模型训练脚本 - YOLOv8-OBB
Baseline Training: YOLOv8n-OBB for Remote Sensing Aircraft Detection

训练流程:
1. 加载预训练YOLOv8n-obb模型
2. 使用标准损失函数(ProbIoU)
3. 标准数据增强(Mosaic + 旋转 + 翻转)
4. 训练200个Epoch
5. 保存训练结果和最佳权重

作为基线对比，体现改进前的性能水平
"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def train_baseline():
    """训练基线YOLOv8-OBB模型"""

    print("=" * 60)
    print("  YOLOv8-OBB 基线模型训练")
    print("  Baseline Training: YOLOv8n-OBB")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO('yolov8n-obb.pt')

    # 训练参数
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
        # 数据增强
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # 保存
        project=str(ROOT / 'results' / 'baseline'),
        name='yolov8_obb_baseline',
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n[INFO] 基线模型训练完成!")
    print(f"[INFO] 结果保存在: {ROOT / 'results' / 'baseline'}")

    return results


def validate_baseline(weights_path: str = None):
    """验证基线模型"""
    if weights_path is None:
        weights_path = str(ROOT / 'results' / 'baseline' / 'yolov8_obb_baseline' / 'weights' / 'best.pt')

    model = YOLO(weights_path)
    metrics = model.val(
        data=str(ROOT / 'configs' / 'dataset.yaml'),
        imgsz=640,
        project=str(ROOT / 'results' / 'baseline'),
        name='val_results',
    )

    print(f"\n基线模型验证结果:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8-OBB 基线模型训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'],
                       help='运行模式: train/val')
    parser.add_argument('--weights', type=str, default=None, help='验证用的权重路径')
    args = parser.parse_args()

    if args.mode == 'train':
        train_baseline()
    elif args.mode == 'val':
        validate_baseline(args.weights)
