"""
生成示例数据用于演示和测试
Generate Sample Data for Demo and Testing

当没有真实数据集时，使用此脚本生成模拟数据:
1. 模拟遥感图像(带背景纹理和飞机形状)
2. 生成OBB格式标注
3. 生成训练日志数据用于可视化
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def generate_sample_images(output_dir: str, num_images: int = 20):
    """
    生成模拟遥感图像和OBB标注
    用于演示系统功能
    """
    img_dir = Path(output_dir) / 'images'
    label_dir = Path(output_dir) / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    for i in range(num_images):
        # 生成背景 (模拟遥感图像纹理)
        img = np.random.randint(100, 180, (640, 640, 3), dtype=np.uint8)

        # 添加背景纹理
        for _ in range(50):
            x1, y1 = np.random.randint(0, 600, 2)
            x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
            color = tuple(np.random.randint(80, 200, 3).tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # 高斯模糊使背景更自然
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # 生成飞机目标
        labels = []
        num_aircraft = np.random.randint(1, 6)

        for j in range(num_aircraft):
            # 随机位置和角度
            cx = np.random.randint(100, 540)
            cy = np.random.randint(100, 540)
            angle = np.random.uniform(0, 360)
            w = np.random.randint(30, 80)
            h = np.random.randint(10, 25)

            # 绘制简化的飞机形状 (旋转矩形)
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # 飞机颜色 (灰白色)
            aircraft_color = tuple(np.random.randint(180, 240, 3).tolist())
            cv2.fillPoly(img, [box], aircraft_color)

            # 生成OBB标注 (归一化坐标)
            norm_points = box.astype(float) / 640.0
            norm_points = np.clip(norm_points, 0, 1)
            label = f"0 {norm_points[0][0]:.6f} {norm_points[0][1]:.6f} " \
                   f"{norm_points[1][0]:.6f} {norm_points[1][1]:.6f} " \
                   f"{norm_points[2][0]:.6f} {norm_points[2][1]:.6f} " \
                   f"{norm_points[3][0]:.6f} {norm_points[3][1]:.6f}"
            labels.append(label)

        # 保存图像
        cv2.imwrite(str(img_dir / f'sample_{i:04d}.jpg'), img)

        # 保存标注
        with open(label_dir / f'sample_{i:04d}.txt', 'w') as f:
            f.write('\n'.join(labels) + '\n')

    print(f"[INFO] 生成 {num_images} 张示例图像: {img_dir}")
    print(f"[INFO] 生成 {num_images} 个标注文件: {label_dir}")


def generate_sample_detection_results():
    """
    生成示例检测结果用于可视化对比
    模拟基线模型(有漏检)和改进模型(检测更好)的结果
    """
    sample_dir = ROOT / 'data' / 'raw'
    img_dir = sample_dir / 'images'

    if not img_dir.exists() or not any(img_dir.iterdir()):
        print("[WARNING] 请先生成示例数据")
        return

    from utils.visualization import ResultVisualizer
    vis = ResultVisualizer('results/comparison')

    # 选取一张示例图像
    img_files = sorted(img_dir.glob('*.jpg'))
    if not img_files:
        return

    sample_img = str(img_files[0])
    label_file = sample_dir / 'labels' / f'{img_files[0].stem}.txt'

    # 对多张图像生成对比
    for idx, img_path in enumerate(img_files[:5]):
        label_file = sample_dir / 'labels' / f'{img_path.stem}.txt'
        if not label_file.exists():
            continue

        with open(label_file, 'r') as f:
            gt_labels = [line.strip() for line in f.readlines() if line.strip()]

        if len(gt_labels) < 2:
            continue

        # 基线结果: 大量漏检 (只检测到40%的目标), 模拟差模型
        n_detect = max(1, int(len(gt_labels) * 0.4))
        baseline_labels = gt_labels[:n_detect]

        # 改进结果: 全部检测到
        improved_labels = gt_labels.copy()

        vis.plot_detection_comparison(
            str(img_path), baseline_labels, improved_labels,
            gt_labels=gt_labels,
            save_name=f'detection_comparison_{idx}.png'
        )

    print("[INFO] Detection comparison charts generated")


def generate_red_obb_visualization():
    """
    生成红色旋转框可视化效果图
    满足用户对红色旋转框的需求
    """
    from utils.visualization import ResultVisualizer

    sample_dir = ROOT / 'data' / 'raw'
    img_dir = sample_dir / 'images'
    label_dir = sample_dir / 'labels'

    img_files = sorted(img_dir.glob('*.jpg'))
    if not img_files:
        print("[WARNING] 无示例图像")
        return

    vis = ResultVisualizer('results/comparison')

    for img_path in img_files[:3]:
        img = cv2.imread(str(img_path))
        label_path = label_dir / f'{img_path.stem}.txt'

        if label_path.exists():
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]

            # 红色旋转框 (BGR: 0,0,255)
            result_img = ResultVisualizer.draw_obb_on_image(
                img, labels, color=(0, 0, 255), thickness=2, show_conf=False
            )
            save_path = ROOT / 'results' / 'comparison' / f'{img_path.stem}_red_obb.jpg'
            cv2.imwrite(str(save_path), result_img)
            print(f"[INFO] 红色旋转框效果图: {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成示例数据')
    parser.add_argument('--num', type=int, default=20, help='生成图像数量')
    args = parser.parse_args()

    # 生成示例数据
    generate_sample_images(str(ROOT / 'data' / 'raw'), args.num)

    # 生成可视化
    generate_sample_detection_results()
    generate_red_obb_visualization()
