"""
真实数据训练脚本
在air-cj数据集上训练基线和改进模型
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def train_baseline():
    """训练基线 YOLOv8-OBB"""
    print("=" * 60)
    print("  Baseline Training: YOLOv8n-OBB")
    print("=" * 60)

    model = YOLO('yolov8n-obb.pt')
    results = model.train(
        data=str(ROOT / 'configs' / 'dataset_real.yaml'),
        epochs=100,
        batch=16,
        imgsz=640,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        project=str(ROOT / 'runs'),
        name='baseline',
        save=True,
        save_period=20,
        plots=True,
        verbose=True,
        device=0,
    )
    print("[INFO] Baseline training done!")
    return results


def train_improved():
    """训练改进 RA-YOLO (增强数据增强策略)"""
    print("=" * 60)
    print("  Improved Training: RA-YOLO")
    print("=" * 60)

    model = YOLO('yolov8n-obb.pt')
    results = model.train(
        data=str(ROOT / 'configs' / 'dataset_real.yaml'),
        epochs=100,
        batch=16,
        imgsz=640,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        degrees=180.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.45,
        translate=0.15,
        scale=0.6,
        project=str(ROOT / 'runs'),
        name='improved',
        save=True,
        save_period=20,
        plots=True,
        verbose=True,
        device=0,
    )
    print("[INFO] Improved training done!")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both', choices=['baseline', 'improved', 'both'])
    args = parser.parse_args()

    if args.mode in ('baseline', 'both'):
        train_baseline()
    if args.mode in ('improved', 'both'):
        train_improved()
