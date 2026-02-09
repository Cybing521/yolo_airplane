"""
数据增强模块 - 针对小样本遥感飞机旋转目标检测
Data Augmentation for Small-Scale Remote Sensing Aircraft OBB Detection

数据量只有500多张，属于小样本场景，需要充分利用数据增强技术:
1. 几何变换: 随机旋转(0-360°)、水平/垂直翻转、仿射变换
2. 颜色空间: HSV调整、亮度对比度变化
3. 高级增强: Mosaic拼接、MixUp混合、CopyPaste复制粘贴
4. 离线增强: 预先扩充数据集至2-3倍
"""

import os
import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class DataAugmentor:
    """
    面向小样本OBB数据集的离线数据增强器
    通过离线增强将500张图像扩充至1500-2000张，有效缓解过拟合
    """

    def __init__(self, img_dir: str, label_dir: str, output_img_dir: str, output_label_dir: str):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.output_img_dir = Path(output_img_dir)
        self.output_label_dir = Path(output_label_dir)
        self.output_img_dir.mkdir(parents=True, exist_ok=True)
        self.output_label_dir.mkdir(parents=True, exist_ok=True)

    def rotate_obb_point(self, x: float, y: float, cx: float, cy: float, angle_rad: float) -> Tuple[float, float]:
        """旋转OBB标注中的单个顶点"""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dx = x - cx
        dy = y - cy
        new_x = dx * cos_a - dy * sin_a + cx
        new_y = dx * sin_a + dy * cos_a + cy
        return new_x, new_y

    def rotate_obb_label(self, label: str, img_w: int, img_h: int, angle_deg: float) -> Optional[str]:
        """
        旋转OBB标注
        YOLO OBB格式: class x1 y1 x2 y2 x3 y3 x4 y4 (归一化坐标)
        """
        parts = label.strip().split()
        if len(parts) != 9:
            return None

        cls_id = parts[0]
        points = []
        for i in range(4):
            x = float(parts[1 + i * 2]) * img_w
            y = float(parts[2 + i * 2]) * img_h
            points.append((x, y))

        cx = img_w / 2.0
        cy = img_h / 2.0
        angle_rad = math.radians(angle_deg)

        new_points = []
        for px, py in points:
            nx, ny = self.rotate_obb_point(px, py, cx, cy, angle_rad)
            nx = np.clip(nx / img_w, 0, 1)
            ny = np.clip(ny / img_h, 0, 1)
            new_points.append((nx, ny))

        coords_str = ' '.join([f'{p[0]:.6f} {p[1]:.6f}' for p in new_points])
        return f'{cls_id} {coords_str}'

    def flip_obb_label(self, label: str, flip_code: int) -> Optional[str]:
        """
        翻转OBB标注
        flip_code: 1=水平翻转, 0=垂直翻转, -1=水平+垂直翻转
        """
        parts = label.strip().split()
        if len(parts) != 9:
            return None

        cls_id = parts[0]
        new_parts = [cls_id]

        for i in range(4):
            x = float(parts[1 + i * 2])
            y = float(parts[2 + i * 2])

            if flip_code == 1:      # 水平翻转
                x = 1.0 - x
            elif flip_code == 0:    # 垂直翻转
                y = 1.0 - y
            elif flip_code == -1:   # 双翻转
                x = 1.0 - x
                y = 1.0 - y

            new_parts.append(f'{x:.6f}')
            new_parts.append(f'{y:.6f}')

        return ' '.join(new_parts)

    def augment_rotation(self, img: np.ndarray, labels: List[str], angle: float) -> Tuple[np.ndarray, List[str]]:
        """旋转增强"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

        new_labels = []
        for label in labels:
            new_label = self.rotate_obb_label(label, w, h, -angle)
            if new_label:
                new_labels.append(new_label)

        return rotated_img, new_labels

    def augment_flip(self, img: np.ndarray, labels: List[str], flip_code: int) -> Tuple[np.ndarray, List[str]]:
        """翻转增强"""
        flipped_img = cv2.flip(img, flip_code)
        new_labels = []
        for label in labels:
            new_label = self.flip_obb_label(label, flip_code)
            if new_label:
                new_labels.append(new_label)
        return flipped_img, new_labels

    def augment_brightness_contrast(self, img: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
        """亮度对比度调整"""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def augment_hsv(self, img: np.ndarray, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4) -> np.ndarray:
        """HSV颜色空间增强"""
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

    def augment_noise(self, img: np.ndarray, noise_level: float = 15.0) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy_img

    def run_offline_augmentation(self, augment_factor: int = 3):
        """
        离线数据增强主函数
        将原始数据集扩充augment_factor倍

        策略:
        - 原始图像保留 (1x)
        - 随机旋转90°/180°/270° (1x)
        - 水平翻转 (0.5x)
        - 垂直翻转 (0.5x)
        - HSV + 噪声组合 (剩余部分)
        """
        img_files = sorted(list(self.img_dir.glob('*.jpg')) +
                          list(self.img_dir.glob('*.png')) +
                          list(self.img_dir.glob('*.jpeg')) +
                          list(self.img_dir.glob('*.bmp')) +
                          list(self.img_dir.glob('*.tif')))

        if not img_files:
            print(f"[WARNING] 未找到图像文件: {self.img_dir}")
            return

        print(f"[INFO] 找到 {len(img_files)} 张原始图像")
        print(f"[INFO] 目标增强倍数: {augment_factor}x")
        print(f"[INFO] 预期增强后数量: ~{len(img_files) * augment_factor} 张")

        total_count = 0

        for img_path in img_files:
            stem = img_path.stem
            suffix = img_path.suffix
            label_path = self.label_dir / f'{stem}.txt'

            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARNING] 无法读取: {img_path}")
                continue

            # 读取标注
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]

            # 1. 保留原始图像
            shutil.copy2(img_path, self.output_img_dir / img_path.name)
            if label_path.exists():
                shutil.copy2(label_path, self.output_label_dir / label_path.name)
            total_count += 1

            # 2. 旋转增强
            for angle in [90, 180, 270]:
                aug_img, aug_labels = self.augment_rotation(img, labels, angle)
                aug_name = f'{stem}_rot{angle}'
                cv2.imwrite(str(self.output_img_dir / f'{aug_name}{suffix}'), aug_img)
                with open(self.output_label_dir / f'{aug_name}.txt', 'w') as f:
                    f.write('\n'.join(aug_labels) + '\n' if aug_labels else '')
                total_count += 1

            # 3. 翻转增强
            for flip_code, flip_name in [(1, 'hflip'), (0, 'vflip')]:
                aug_img, aug_labels = self.augment_flip(img, labels, flip_code)
                aug_name = f'{stem}_{flip_name}'
                cv2.imwrite(str(self.output_img_dir / f'{aug_name}{suffix}'), aug_img)
                with open(self.output_label_dir / f'{aug_name}.txt', 'w') as f:
                    f.write('\n'.join(aug_labels) + '\n' if aug_labels else '')
                total_count += 1

            # 4. 颜色增强组合
            if augment_factor > 3:
                # HSV增强
                aug_img = self.augment_hsv(img)
                aug_name = f'{stem}_hsv'
                cv2.imwrite(str(self.output_img_dir / f'{aug_name}{suffix}'), aug_img)
                if label_path.exists():
                    shutil.copy2(label_path, self.output_label_dir / f'{aug_name}.txt')
                total_count += 1

                # 噪声增强
                aug_img = self.augment_noise(img)
                aug_name = f'{stem}_noise'
                cv2.imwrite(str(self.output_img_dir / f'{aug_name}{suffix}'), aug_img)
                if label_path.exists():
                    shutil.copy2(label_path, self.output_label_dir / f'{aug_name}.txt')
                total_count += 1

        print(f"[INFO] 增强完成! 共生成 {total_count} 张图像")
        print(f"[INFO] 输出目录: {self.output_img_dir}")

    @staticmethod
    def split_dataset(img_dir: str, label_dir: str, output_dir: str,
                      train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                      seed: int = 42):
        """
        划分数据集为训练集/验证集/测试集

        对于500张小样本数据集:
        - 训练集: 70% (~350张, 增强后~1050张)
        - 验证集: 20% (~100张)
        - 测试集: 10% (~50张)
        """
        random.seed(seed)
        np.random.seed(seed)

        img_dir = Path(img_dir)
        label_dir = Path(label_dir)
        output_dir = Path(output_dir)

        img_files = sorted(list(img_dir.glob('*.jpg')) +
                          list(img_dir.glob('*.png')) +
                          list(img_dir.glob('*.jpeg')) +
                          list(img_dir.glob('*.bmp')) +
                          list(img_dir.glob('*.tif')))

        random.shuffle(img_files)
        n = len(img_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': img_files[:n_train],
            'val': img_files[n_train:n_train + n_val],
            'test': img_files[n_train + n_val:]
        }

        for split_name, files in splits.items():
            split_img_dir = output_dir / split_name / 'images'
            split_lbl_dir = output_dir / split_name / 'labels'
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy2(img_path, split_img_dir / img_path.name)
                label_path = label_dir / f'{img_path.stem}.txt'
                if label_path.exists():
                    shutil.copy2(label_path, split_lbl_dir / f'{img_path.stem}.txt')

            print(f"[INFO] {split_name}: {len(files)} 张图像")

        print(f"[INFO] 数据集划分完成, 保存至: {output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='数据增强工具')
    parser.add_argument('--img-dir', type=str, required=True, help='原始图像目录')
    parser.add_argument('--label-dir', type=str, required=True, help='原始标注目录')
    parser.add_argument('--output-img-dir', type=str, required=True, help='增强后图像输出目录')
    parser.add_argument('--output-label-dir', type=str, required=True, help='增强后标注输出目录')
    parser.add_argument('--factor', type=int, default=3, help='增强倍数')
    args = parser.parse_args()

    augmentor = DataAugmentor(args.img_dir, args.label_dir, args.output_img_dir, args.output_label_dir)
    augmentor.run_offline_augmentation(args.factor)
