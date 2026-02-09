"""
真实数据准备流水线
1. 使用预训练YOLOv8-OBB对air-cj图片自动标注
2. 组织为YOLO OBB训练格式
3. 划分train/val/test
4. 执行离线数据增强
"""

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def step1_auto_label():
    """使用预训练模型自动标注"""
    from ultralytics import YOLO
    
    src_dir = ROOT / 'air-cj'
    out_img_dir = ROOT / 'data' / 'real' / 'images'
    out_lbl_dir = ROOT / 'data' / 'real' / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载预训练OBB模型 (在DOTAv1上训练，包含plane类)
    print("[INFO] Loading pretrained YOLOv8n-obb model...")
    model = YOLO('yolov8n-obb.pt')
    
    img_files = sorted(list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png')))
    print(f"[INFO] Found {len(img_files)} images in air-cj/")
    
    labeled_count = 0
    total_objects = 0
    
    for i, img_path in enumerate(img_files):
        # 推理
        results = model.predict(
            str(img_path),
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )
        
        result = results[0]
        
        # 提取OBB结果
        labels = []
        if result.obb is not None and len(result.obb) > 0:
            obb_data = result.obb
            for j in range(len(obb_data)):
                cls_id = int(obb_data.cls[j].item())
                conf = obb_data.conf[j].item()
                
                # DOTAv1中plane的class id
                # 我们把所有检测到的物体都当作aircraft (class 0)
                # 因为这些图片都是飞机场景
                
                # 获取四个顶点坐标 (像素坐标)
                if hasattr(obb_data, 'xyxyxyxy') and obb_data.xyxyxyxy is not None:
                    points = obb_data.xyxyxyxy[j].cpu().numpy()  # (4, 2)
                    h, w = result.orig_shape
                    # 归一化
                    norm_pts = points.copy()
                    norm_pts[:, 0] = np.clip(norm_pts[:, 0] / w, 0, 1)
                    norm_pts[:, 1] = np.clip(norm_pts[:, 1] / h, 0, 1)
                    
                    label = f"0 {norm_pts[0][0]:.6f} {norm_pts[0][1]:.6f} " \
                           f"{norm_pts[1][0]:.6f} {norm_pts[1][1]:.6f} " \
                           f"{norm_pts[2][0]:.6f} {norm_pts[2][1]:.6f} " \
                           f"{norm_pts[3][0]:.6f} {norm_pts[3][1]:.6f}"
                    labels.append(label)
        
        if labels:
            # 复制图片
            shutil.copy2(img_path, out_img_dir / img_path.name)
            # 写标注
            lbl_name = img_path.stem + '.txt'
            with open(out_lbl_dir / lbl_name, 'w') as f:
                f.write('\n'.join(labels) + '\n')
            labeled_count += 1
            total_objects += len(labels)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(img_files)}, labeled: {labeled_count}, objects: {total_objects}")
    
    print(f"\n[INFO] Auto-labeling complete!")
    print(f"  Total images: {len(img_files)}")
    print(f"  Labeled images: {labeled_count}")
    print(f"  Total objects: {total_objects}")
    print(f"  Avg objects/image: {total_objects/max(labeled_count,1):.1f}")
    
    return labeled_count


def step2_split_dataset():
    """划分训练/验证/测试集"""
    random.seed(42)
    np.random.seed(42)
    
    real_img_dir = ROOT / 'data' / 'real' / 'images'
    real_lbl_dir = ROOT / 'data' / 'real' / 'labels'
    split_dir = ROOT / 'data' / 'real_splits'
    
    img_files = sorted(list(real_img_dir.glob('*.jpg')) + list(real_img_dir.glob('*.png')))
    random.shuffle(img_files)
    
    n = len(img_files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    
    splits = {
        'train': img_files[:n_train],
        'val': img_files[n_train:n_train + n_val],
        'test': img_files[n_train + n_val:]
    }
    
    for split_name, files in splits.items():
        img_out = split_dir / split_name / 'images'
        lbl_out = split_dir / split_name / 'labels'
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_path in files:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = real_lbl_dir / (img_path.stem + '.txt')
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / (img_path.stem + '.txt'))
        
        print(f"  {split_name}: {len(files)} images")
    
    print(f"[INFO] Dataset split saved to: {split_dir}")


def step3_write_dataset_config():
    """写数据集配置文件"""
    config_content = f"""# 真实遥感飞机数据集配置
# Real Remote Sensing Aircraft OBB Detection Dataset

path: {ROOT / 'data' / 'real_splits'}
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: aircraft
"""
    config_path = ROOT / 'configs' / 'dataset_real.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"[INFO] Dataset config written to: {config_path}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Real Data Preparation Pipeline")
    print("=" * 60)
    
    print("\n--- Step 1: Auto-labeling ---")
    n_labeled = step1_auto_label()
    
    if n_labeled > 0:
        print("\n--- Step 2: Dataset split ---")
        step2_split_dataset()
        
        print("\n--- Step 3: Write config ---")
        step3_write_dataset_config()
        
        print("\n[DONE] Data preparation complete!")
    else:
        print("[ERROR] No images were labeled. Check the data.")
