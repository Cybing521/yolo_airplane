"""
可视化模块 - 遥感飞机旋转目标检测结果可视化
Visualization for Remote Sensing Aircraft OBB Detection

功能:
1. 红色旋转框检测结果绘制
2. 损失函数收敛曲线对比
3. F1分数曲线对比
4. 检测效果对比图 (good vs bad)
5. 混淆矩阵、PR曲线等
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """检测结果可视化器"""

    def __init__(self, output_dir: str = 'results/comparison'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def draw_obb_on_image(img: np.ndarray, obb_labels: List[str],
                          color: Tuple[int, int, int] = (0, 0, 255),
                          thickness: int = 2,
                          show_conf: bool = True) -> np.ndarray:
        """
        在图像上绘制红色旋转框
        
        Args:
            img: 输入图像
            obb_labels: OBB标注列表, 格式: "class x1 y1 x2 y2 x3 y3 x4 y4 [conf]"
            color: 框颜色 BGR格式, 默认红色(0,0,255)
            thickness: 线宽
            show_conf: 是否显示置信度
        """
        result = img.copy()
        h, w = result.shape[:2]

        for label in obb_labels:
            parts = label.strip().split()
            if len(parts) < 9:
                continue

            cls_id = int(parts[0])
            points = []
            for i in range(4):
                x = float(parts[1 + i * 2]) * w
                y = float(parts[2 + i * 2]) * h
                points.append([int(x), int(y)])

            pts = np.array(points, dtype=np.int32)
            cv2.polylines(result, [pts], True, color, thickness)

            # 绘制顶点
            for pt in points:
                cv2.circle(result, tuple(pt), 3, color, -1)

            # 显示置信度
            if show_conf and len(parts) > 9:
                conf = float(parts[9])
                text = f'{conf:.2f}'
                text_pos = (points[0][0], points[0][1] - 5)
                cv2.putText(result, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 1, cv2.LINE_AA)

        return result

    def plot_loss_comparison(self, baseline_losses: Dict, improved_losses: Dict,
                             save_name: str = 'loss_comparison.png'):
        """
        绘制损失函数改进前后对比图
        
        对应论文图4.11: 改进前后的训练回归损失曲线对比
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 回归损失对比
        ax1 = axes[0]
        epochs_b = range(1, len(baseline_losses['box_loss']) + 1)
        epochs_i = range(1, len(improved_losses['box_loss']) + 1)

        ax1.plot(epochs_b, baseline_losses['box_loss'], 'b-', label='Before (Baseline)', linewidth=1.5, alpha=0.8)
        ax1.plot(epochs_i, improved_losses['box_loss'], 'r-', label='After (KPRLoss)', linewidth=1.5, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Comparison of Regression Loss Curves', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 总损失对比
        ax2 = axes[1]
        if 'total_loss' in baseline_losses and 'total_loss' in improved_losses:
            ax2.plot(epochs_b, baseline_losses['total_loss'], 'b-', label='Before (Baseline)', linewidth=1.5, alpha=0.8)
            ax2.plot(epochs_i, improved_losses['total_loss'], 'r-', label='After (RA-YOLO)', linewidth=1.5, alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Comparison of Total Loss Curves', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 损失对比图已保存: {save_path}")

    def plot_f1_comparison(self, baseline_f1: List[float], improved_f1: List[float],
                            save_name: str = 'f1_comparison.png'):
        """
        绘制F1分数曲线对比图

        对应论文图4.12: 改进前后的F1分数曲线对比
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs_b = range(1, len(baseline_f1) + 1)
        epochs_i = range(1, len(improved_f1) + 1)

        ax.plot(epochs_b, baseline_f1, 'b-', label='Before (Baseline)', linewidth=1.5, alpha=0.8)
        ax.plot(epochs_i, improved_f1, 'r-', label='After (RA-YOLO)', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Comparison of F1 Score Curves', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] F1对比图已保存: {save_path}")

    def plot_metrics_comparison_table(self, metrics: Dict[str, Dict],
                                       save_name: str = 'metrics_table.png'):
        """
        绘制指标对比表格
        
        包含: mAP50, mAP50-95, Precision, Recall, F1, 参数量, FPS
        """
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')

        columns = ['Model', 'mAP50(%)', 'mAP50-95(%)', 'Precision(%)', 'Recall(%)',
                   'F1(%)', 'Params(M)', 'FPS']
        rows = []
        for model_name, m in metrics.items():
            rows.append([
                model_name,
                f"{m.get('mAP50', 0)*100:.1f}",
                f"{m.get('mAP50_95', 0)*100:.1f}",
                f"{m.get('precision', 0)*100:.1f}",
                f"{m.get('recall', 0)*100:.1f}",
                f"{m.get('f1', 0)*100:.1f}",
                f"{m.get('params', 0):.2f}",
                f"{m.get('fps', 0):.1f}"
            ])

        table = ax.table(cellText=rows, colLabels=columns, loc='center',
                         cellLoc='center', colWidths=[0.18] + [0.12]*7)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 设置表头样式
        for j in range(len(columns)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # 交替行颜色
        for i in range(1, len(rows) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[i, j].set_facecolor('#D6E4F0')

        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 指标对比表已保存: {save_path}")

    def plot_detection_comparison(self, img_path: str,
                                   baseline_labels: List[str],
                                   improved_labels: List[str],
                                   gt_labels: Optional[List[str]] = None,
                                   save_name: str = 'detection_comparison.png'):
        """
        绘制检测效果对比图 (输入 / 基线结果 / 改进结果)
        
        对应论文图3.17: 不同网络模型检测结果图
        展示效果差(漏检误检)与效果好的对比
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] 无法读取图像: {img_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ncols = 3 if gt_labels is not None else 2
        fig, axes = plt.subplots(1, ncols + 1, figsize=(6 * (ncols + 1), 6))

        # (a) 原始输入
        axes[0].imshow(img_rgb)
        axes[0].set_title('(a) Input Image', fontsize=12)
        axes[0].axis('off')

        # (b) 基线模型结果 - 蓝色框 (效果较差)
        baseline_img = self.draw_obb_on_image(img, baseline_labels,
                                                color=(255, 0, 0), thickness=2)
        axes[1].imshow(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('(b) YOLOv8-OBB (Baseline)', fontsize=12)
        axes[1].axis('off')

        # (c) 改进模型结果 - 红色框 (效果好)
        improved_img = self.draw_obb_on_image(img, improved_labels,
                                                color=(0, 0, 255), thickness=2)
        axes[2].imshow(cv2.cvtColor(improved_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('(c) RA-YOLO (Improved)', fontsize=12)
        axes[2].axis('off')

        # (d) Ground Truth (如果提供)
        if gt_labels is not None:
            gt_img = self.draw_obb_on_image(img, gt_labels,
                                             color=(0, 255, 0), thickness=2)
            axes[3].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
            axes[3].set_title('(d) Ground Truth', fontsize=12)
            axes[3].axis('off')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 检测效果对比图已保存: {save_path}")

    def plot_ablation_study(self, ablation_data: Dict[str, Dict],
                             save_name: str = 'ablation_study.png'):
        """
        绘制消融实验结果
        分析各改进模块的独立贡献
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('off')

        columns = ['Configuration', 'ASC', 'KPRLoss', 'Aug+', 'mAP50(%)', 'mAP50-95(%)', 'Delta']
        rows = []
        for config_name, data in ablation_data.items():
            rows.append([
                config_name,
                '✓' if data.get('asc', False) else '✗',
                '✓' if data.get('kprloss', False) else '✗',
                '✓' if data.get('aug_plus', False) else '✗',
                f"{data.get('mAP50', 0)*100:.1f}",
                f"{data.get('mAP50_95', 0)*100:.1f}",
                f"+{data.get('delta', 0)*100:.1f}" if data.get('delta', 0) > 0 else '-'
            ])

        table = ax.table(cellText=rows, colLabels=columns, loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        for j in range(len(columns)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')

        plt.title('Ablation Study Results', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 消融实验表已保存: {save_path}")


def generate_demo_comparison_plots():
    """
    生成演示用的对比图表
    模拟论文中的损失曲线、F1曲线、指标对比等
    """
    np.random.seed(42)
    vis = ResultVisualizer('results/comparison')

    # === 1. 损失函数对比 ===
    epochs = 200
    x = np.arange(1, epochs + 1)

    # 基线损失 (收敛较慢, 波动较大)
    baseline_box = 3.2 * np.exp(-0.025 * x) + 0.48 + 0.15 * np.random.randn(epochs) * np.exp(-0.01 * x)
    baseline_total = 4.5 * np.exp(-0.03 * x) + 0.65 + 0.2 * np.random.randn(epochs) * np.exp(-0.01 * x)

    # 改进后损失 (收敛快, 波动小)
    improved_box = 3.0 * np.exp(-0.035 * x) + 0.45 + 0.08 * np.random.randn(epochs) * np.exp(-0.015 * x)
    improved_total = 4.2 * np.exp(-0.04 * x) + 0.55 + 0.12 * np.random.randn(epochs) * np.exp(-0.015 * x)

    baseline_losses = {
        'box_loss': np.maximum(baseline_box, 0.3).tolist(),
        'total_loss': np.maximum(baseline_total, 0.4).tolist()
    }
    improved_losses = {
        'box_loss': np.maximum(improved_box, 0.25).tolist(),
        'total_loss': np.maximum(improved_total, 0.35).tolist()
    }

    vis.plot_loss_comparison(baseline_losses, improved_losses)

    # === 2. F1分数对比 ===
    baseline_f1 = (1 - np.exp(-0.06 * x)) * 0.85 + 0.02 * np.random.randn(epochs) * np.exp(-0.02 * x)
    improved_f1 = (1 - np.exp(-0.08 * x)) * 0.90 + 0.015 * np.random.randn(epochs) * np.exp(-0.02 * x)

    baseline_f1 = np.clip(baseline_f1, 0, 1).tolist()
    improved_f1 = np.clip(improved_f1, 0, 1).tolist()

    vis.plot_f1_comparison(baseline_f1, improved_f1)

    # === 3. 指标对比表 ===
    metrics = {
        'YOLOv8n-OBB (Baseline)': {
            'mAP50': 0.823, 'mAP50_95': 0.512, 'precision': 0.841,
            'recall': 0.795, 'f1': 0.817, 'params': 3.2, 'fps': 142.5
        },
        'YOLOv8n-OBB + ASC': {
            'mAP50': 0.856, 'mAP50_95': 0.541, 'precision': 0.869,
            'recall': 0.821, 'f1': 0.844, 'params': 3.8, 'fps': 128.3
        },
        'YOLOv8n-OBB + KPRLoss': {
            'mAP50': 0.847, 'mAP50_95': 0.535, 'precision': 0.858,
            'recall': 0.818, 'f1': 0.838, 'params': 3.2, 'fps': 141.8
        },
        'RA-YOLO (Full)': {
            'mAP50': 0.891, 'mAP50_95': 0.573, 'precision': 0.897,
            'recall': 0.862, 'f1': 0.879, 'params': 3.8, 'fps': 126.7
        }
    }
    vis.plot_metrics_comparison_table(metrics)

    # === 4. 消融实验 ===
    ablation = {
        'Baseline': {'asc': False, 'kprloss': False, 'aug_plus': False,
                     'mAP50': 0.823, 'mAP50_95': 0.512, 'delta': 0},
        'Baseline + ASC': {'asc': True, 'kprloss': False, 'aug_plus': False,
                           'mAP50': 0.856, 'mAP50_95': 0.541, 'delta': 0.033},
        'Baseline + KPRLoss': {'asc': False, 'kprloss': True, 'aug_plus': False,
                               'mAP50': 0.847, 'mAP50_95': 0.535, 'delta': 0.024},
        'Baseline + Aug+': {'asc': False, 'kprloss': False, 'aug_plus': True,
                            'mAP50': 0.839, 'mAP50_95': 0.528, 'delta': 0.016},
        'RA-YOLO (All)': {'asc': True, 'kprloss': True, 'aug_plus': True,
                          'mAP50': 0.891, 'mAP50_95': 0.573, 'delta': 0.068},
    }
    vis.plot_ablation_study(ablation)

    # === 5. 综合性能雷达图 ===
    plot_radar_chart(metrics, vis.output_dir)

    print("[INFO] 所有演示对比图表生成完成!")


def plot_radar_chart(metrics: Dict[str, Dict], output_dir: Path):
    """绘制模型性能雷达图"""
    categories = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FF0000']
    for idx, (model_name, m) in enumerate(metrics.items()):
        values = [m['mAP50'], m['mAP50_95'], m['precision'], m['recall'], m['f1']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = output_dir / 'radar_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 雷达图已保存: {save_path}")


if __name__ == '__main__':
    generate_demo_comparison_plots()
