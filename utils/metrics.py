"""
指标分析模块 - 遥感飞机旋转目标检测评估
Metrics Analysis for Remote Sensing Aircraft OBB Detection

功能:
1. 从训练日志解析指标
2. 计算mAP、Precision、Recall、F1
3. 生成对比分析报告
4. 支持不同模型间的横向对比
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MetricsAnalyzer:
    """训练指标分析器"""

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(results_dir)

    def parse_yolo_results_csv(self, csv_path: str) -> Dict[str, List[float]]:
        """
        解析YOLO训练输出的results.csv文件
        
        返回字典包含:
        - epoch, train/box_loss, train/cls_loss, train/dfl_loss
        - metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50-95
        - val/box_loss, val/cls_loss, val/dfl_loss
        """
        data = {}
        csv_path = Path(csv_path)

        if not csv_path.exists():
            print(f"[WARNING] 文件不存在: {csv_path}")
            return data

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    key = key.strip()
                    if key not in data:
                        data[key] = []
                    try:
                        data[key].append(float(value.strip()))
                    except (ValueError, AttributeError):
                        data[key].append(0.0)

        return data

    def compute_f1_scores(self, precision_list: List[float], recall_list: List[float]) -> List[float]:
        """计算F1分数序列"""
        f1_scores = []
        for p, r in zip(precision_list, recall_list):
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
            f1_scores.append(f1)
        return f1_scores

    def get_best_epoch_metrics(self, results: Dict[str, List[float]]) -> Dict[str, float]:
        """获取最佳epoch的指标"""
        if 'metrics/mAP50(B)' in results:
            mAP50_key = 'metrics/mAP50(B)'
            mAP50_95_key = 'metrics/mAP50-95(B)'
            precision_key = 'metrics/precision(B)'
            recall_key = 'metrics/recall(B)'
        else:
            mAP50_key = 'metrics/mAP50'
            mAP50_95_key = 'metrics/mAP50-95'
            precision_key = 'metrics/precision'
            recall_key = 'metrics/recall'

        if mAP50_key not in results:
            return {}

        mAP50_list = results[mAP50_key]
        best_epoch = int(np.argmax(mAP50_list))

        precision = results.get(precision_key, [0.0] * len(mAP50_list))[best_epoch]
        recall = results.get(recall_key, [0.0] * len(mAP50_list))[best_epoch]
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'best_epoch': best_epoch + 1,
            'mAP50': mAP50_list[best_epoch],
            'mAP50_95': results.get(mAP50_95_key, [0.0] * len(mAP50_list))[best_epoch],
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def compare_models(self, model_results: Dict[str, str]) -> Dict[str, Dict]:
        """
        对比多个模型的性能
        
        Args:
            model_results: {模型名称: results.csv路径}
        
        Returns:
            对比结果字典
        """
        comparison = {}
        for model_name, csv_path in model_results.items():
            results = self.parse_yolo_results_csv(csv_path)
            if results:
                best_metrics = self.get_best_epoch_metrics(results)
                comparison[model_name] = best_metrics

        return comparison

    def generate_comparison_report(self, comparison: Dict[str, Dict],
                                     output_path: str = 'results/comparison/metrics_report.json'):
        """生成指标对比报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'title': '遥感飞机旋转目标检测 - 模型性能对比报告',
            'models': comparison,
            'analysis': {}
        }

        if len(comparison) >= 2:
            models = list(comparison.keys())
            baseline = comparison[models[0]]
            improved = comparison[models[-1]]

            if baseline and improved:
                report['analysis'] = {
                    'mAP50_improvement': f"+{(improved.get('mAP50', 0) - baseline.get('mAP50', 0)) * 100:.1f}%",
                    'mAP50_95_improvement': f"+{(improved.get('mAP50_95', 0) - baseline.get('mAP50_95', 0)) * 100:.1f}%",
                    'precision_improvement': f"+{(improved.get('precision', 0) - baseline.get('precision', 0)) * 100:.1f}%",
                    'recall_improvement': f"+{(improved.get('recall', 0) - baseline.get('recall', 0)) * 100:.1f}%",
                    'f1_improvement': f"+{(improved.get('f1', 0) - baseline.get('f1', 0)) * 100:.1f}%",
                }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 指标对比报告已保存: {output_path}")
        return report

    @staticmethod
    def print_comparison_table(comparison: Dict[str, Dict]):
        """在终端打印对比表格"""
        header = f"{'Model':<35} {'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        print("=" * 85)
        print(header)
        print("-" * 85)

        for model_name, metrics in comparison.items():
            row = (f"{model_name:<35} "
                   f"{metrics.get('mAP50', 0)*100:>7.1f}% "
                   f"{metrics.get('mAP50_95', 0)*100:>9.1f}% "
                   f"{metrics.get('precision', 0)*100:>7.1f}% "
                   f"{metrics.get('recall', 0)*100:>7.1f}% "
                   f"{metrics.get('f1', 0)*100:>7.1f}%")
            print(row)

        print("=" * 85)


if __name__ == '__main__':
    analyzer = MetricsAnalyzer()

    # 演示用模拟数据
    demo_comparison = {
        'YOLOv8n-OBB (Baseline)': {
            'best_epoch': 185, 'mAP50': 0.823, 'mAP50_95': 0.512,
            'precision': 0.841, 'recall': 0.795, 'f1': 0.817
        },
        'RA-YOLO (Improved)': {
            'best_epoch': 168, 'mAP50': 0.891, 'mAP50_95': 0.573,
            'precision': 0.897, 'recall': 0.862, 'f1': 0.879
        }
    }

    MetricsAnalyzer.print_comparison_table(demo_comparison)
    analyzer.generate_comparison_report(demo_comparison)
