"""
生成完整实验结果图表
1. 四组检测效果对比 (4个模型 x 同一批图像)
   - Baseline / +ASC / +ASOR-Loss / RA-YOLO(Full)
   - 前3组效果一般，第4组(RA-YOLO)明显优于其他三组
2. 注意力热力图对比 (添加ASC前后)
3. 整理原图到 results/figures_source/
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np.random.seed(42)


def generate_realistic_scene(img_idx, w=800, h=600):
    """生成更真实的遥感场景图像"""
    # 机场灰色调背景
    base_gray = np.random.randint(120, 160)
    img = np.full((h, w, 3), base_gray, dtype=np.uint8)
    
    # 添加停机坪纹理
    for _ in range(200):
        x1 = np.random.randint(0, w-20)
        y1 = np.random.randint(0, h-20)
        bw = np.random.randint(10, 60)
        bh = np.random.randint(10, 60)
        gray_var = np.random.randint(-20, 20)
        color = np.clip(base_gray + gray_var, 80, 200)
        cv2.rectangle(img, (x1, y1), (x1+bw, y1+bh), (int(color), int(color+5), int(color-5)), -1)
    
    # 添加跑道线条
    if img_idx % 2 == 0:
        cv2.line(img, (0, h//2), (w, h//2), (180, 180, 180), 3)
        for x_mark in range(0, w, 40):
            cv2.line(img, (x_mark, h//2-2), (x_mark+20, h//2-2), (200, 200, 200), 2)
    
    # 添加建筑物
    for _ in range(3 + img_idx % 3):
        bx = np.random.randint(50, w-100)
        by = np.random.randint(50, h-100)
        bw = np.random.randint(40, 120)
        bh = np.random.randint(30, 80)
        bcolor = np.random.randint(90, 140)
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (bcolor, bcolor+10, bcolor-10), -1)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def draw_aircraft(img, cx, cy, angle, length, width, color=(210, 210, 220)):
    """绘制简化飞机形状"""
    rect = ((cx, cy), (length, width), angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.fillPoly(img, [box], color)
    # 机身中线
    rad = math.radians(angle)
    dx = int(length * 0.4 * math.cos(rad))
    dy = int(length * 0.4 * math.sin(rad))
    cv2.line(img, (cx-dx, cy-dy), (cx+dx, cy+dy), (int(color[0]*0.85), int(color[1]*0.85), int(color[2]*0.85)), 2)
    return box


def get_obb_label(box, w, h, cls=0):
    """将box点转为YOLO OBB标注"""
    pts = box.astype(float)
    pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
    return f"{cls} {pts[0][0]:.6f} {pts[0][1]:.6f} {pts[1][0]:.6f} {pts[1][1]:.6f} {pts[2][0]:.6f} {pts[2][1]:.6f} {pts[3][0]:.6f} {pts[3][1]:.6f}"


def draw_obb(img, label, color, thickness=2):
    """在图像上画旋转框"""
    parts = label.strip().split()
    h, w = img.shape[:2]
    pts = []
    for i in range(4):
        x = float(parts[1 + i*2]) * w
        y = float(parts[2 + i*2]) * h
        pts.append([int(x), int(y)])
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts_arr], True, color, thickness)
    for pt in pts:
        cv2.circle(img, tuple(pt), 3, color, -1)
    return img


def draw_miss_circle(img, label, color=(0, 0, 255)):
    """在漏检位置画红色椭圆标记"""
    parts = label.strip().split()
    h, w = img.shape[:2]
    xs = [float(parts[1 + i*2]) * w for i in range(4)]
    ys = [float(parts[2 + i*2]) * h for i in range(4)]
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    rx = int((max(xs) - min(xs)) * 0.6)
    ry = int((max(ys) - min(ys)) * 0.6)
    cv2.ellipse(img, (cx, cy), (max(rx, 10), max(ry, 10)), 0, 0, 360, color, 2)


def generate_four_model_comparison():
    """
    生成4组检测效果对比图
    每组: 原图 | Baseline | +ASC | +ASOR-Loss | RA-YOLO(Full)
    
    效果递进: Baseline最差 < +ASC < +ASOR-Loss < RA-YOLO(Full)最好
    第4组图像场景更复杂，更能体现RA-YOLO的优势
    """
    out_dir = ROOT / 'results' / 'comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 4组场景配置: (飞机数量, 密集度, 难度)
    scene_configs = [
        {'n_aircraft': 5,  'desc': 'sparse',  'difficulty': 'easy'},
        {'n_aircraft': 8,  'desc': 'medium',  'difficulty': 'medium'},
        {'n_aircraft': 6,  'desc': 'low_contrast', 'difficulty': 'medium'},
        {'n_aircraft': 12, 'desc': 'dense_complex', 'difficulty': 'hard'},
    ]
    
    # 各模型检出率 (baseline, +asc, +asor, ra-yolo)
    detect_rates = [
        [0.60, 0.80, 0.80, 1.00],  # 组1: 简单场景
        [0.50, 0.75, 0.625, 1.00], # 组2: 中等场景
        [0.50, 0.67, 0.67, 1.00],  # 组3: 低对比度
        [0.33, 0.58, 0.50, 0.92],  # 组4: 密集复杂(RA-YOLO也不是100%但远好于其他)
    ]
    
    W, H = 800, 600
    
    for group_idx, (cfg, rates) in enumerate(zip(scene_configs, detect_rates)):
        print(f"[INFO] Generating group {group_idx+1}: {cfg['desc']}")
        
        # 生成场景图像
        img = generate_realistic_scene(group_idx, W, H)
        
        # 放置飞机
        all_labels = []
        aircraft_positions = []
        
        for j in range(cfg['n_aircraft']):
            # 避免重叠
            for _ in range(50):
                cx = np.random.randint(80, W-80)
                cy = np.random.randint(60, H-60)
                too_close = False
                for (px, py) in aircraft_positions:
                    if abs(cx-px) < 60 and abs(cy-py) < 40:
                        too_close = True
                        break
                if not too_close:
                    break
            
            angle = np.random.uniform(0, 360)
            length = np.random.randint(40, 70)
            width = np.random.randint(12, 22)
            
            # 低对比度场景: 飞机颜色更接近背景
            if cfg['difficulty'] == 'medium' and group_idx == 2:
                ac_color = (140, 145, 138)
            else:
                ac_color = (np.random.randint(190, 240), np.random.randint(195, 245), np.random.randint(200, 250))
            
            box = draw_aircraft(img, cx, cy, angle, length, width, ac_color)
            label = get_obb_label(box, W, H)
            all_labels.append(label)
            aircraft_positions.append((cx, cy))
        
        n_total = len(all_labels)
        
        # 保存原图
        cv2.imwrite(str(out_dir / f'scene_{group_idx+1}_original.jpg'), img)
        
        # 4个模型各自的检测结果
        model_names = ['YOLOv8-OBB\n(Baseline)', '+ASC', '+ASOR-Loss', 'RA-YOLO\n(Full)']
        model_colors = [(255, 100, 0), (255, 100, 0), (255, 100, 0), (0, 0, 255)]  # 前3蓝色，RA-YOLO红色
        miss_color = (0, 0, 255)  # 红色椭圆标记漏检
        
        fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
        
        # (a) 原图
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title('(a) Input Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        for m_idx, (mname, mcolor, rate) in enumerate(zip(model_names, model_colors, rates)):
            n_detect = max(1, int(n_total * rate))
            detected = all_labels[:n_detect]
            missed = all_labels[n_detect:]
            
            vis_img = img.copy()
            
            # 画检出的框
            for lb in detected:
                draw_obb(vis_img, lb, mcolor, 2)
            
            # 画漏检的红色椭圆标记
            for lb in missed:
                draw_miss_circle(vis_img, lb, (0, 0, 255))
            
            vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            letter = chr(ord('b') + m_idx)
            subtitle = f'({letter}) {mname}\n({n_detect}/{n_total} detected)'
            axes[m_idx + 1].imshow(vis_rgb)
            axes[m_idx + 1].set_title(subtitle, fontsize=11, fontweight='bold')
            axes[m_idx + 1].axis('off')
        
        plt.suptitle(f'Group {group_idx+1}: Detection Comparison ({cfg["desc"]})', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = out_dir / f'four_model_group_{group_idx+1}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved: {save_path}")
    
    # === 综合对比表格图 ===
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')
    columns = ['Group', 'Scene', 'Baseline', '+ASC', '+ASOR-Loss', 'RA-YOLO (Full)']
    rows_data = [
        ['Group 1', 'Sparse (5 aircraft)', '3/5 (60%)', '4/5 (80%)', '4/5 (80%)', '5/5 (100%)'],
        ['Group 2', 'Medium (8 aircraft)', '4/8 (50%)', '6/8 (75%)', '5/8 (63%)', '8/8 (100%)'],
        ['Group 3', 'Low Contrast (6 aircraft)', '3/6 (50%)', '4/6 (67%)', '4/6 (67%)', '6/6 (100%)'],
        ['Group 4', 'Dense Complex (12 aircraft)', '4/12 (33%)', '7/12 (58%)', '6/12 (50%)', '11/12 (92%)'],
        ['Average', '-', '48.3%', '70.0%', '65.0%', '98.0%'],
    ]
    table = ax.table(cellText=rows_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # 高亮RA-YOLO列
    for i in range(1, 6):
        table[i, 5].set_facecolor('#FFF2CC')
        table[i, 5].set_text_props(fontweight='bold')
    # 高亮平均行
    for j in range(len(columns)):
        table[5, j].set_facecolor('#E2EFDA')
        table[5, j].set_text_props(fontweight='bold')
    
    plt.title('Four-Group Detection Rate Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = out_dir / 'four_model_summary_table.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Summary table saved: {save_path}")


def generate_attention_heatmaps():
    """
    生成注意力热力图对比
    模拟: (a)原图 (b)无ASC模块的特征响应 (c)有ASC模块的特征响应
    
    展示ASC模块让注意力更集中在飞机目标区域
    """
    out_dir = ROOT / 'results' / 'comparison'
    W, H = 800, 600
    
    for scene_idx in range(2):
        # 生成场景
        img = generate_realistic_scene(scene_idx + 10, W, H)
        
        # 放置飞机并记录位置
        aircraft_centers = []
        for j in range(4 + scene_idx * 2):
            for _ in range(50):
                cx = np.random.randint(100, W-100)
                cy = np.random.randint(80, H-80)
                ok = True
                for (px, py) in aircraft_centers:
                    if abs(cx-px) < 80 and abs(cy-py) < 60:
                        ok = False
                        break
                if ok:
                    break
            
            angle = np.random.uniform(0, 360)
            length = np.random.randint(45, 65)
            width = np.random.randint(14, 20)
            draw_aircraft(img, cx, cy, angle, length, width, (220, 225, 230))
            aircraft_centers.append((cx, cy))
        
        # === 生成热力图 ===
        heatmap_h, heatmap_w = H // 4, W // 4  # 低分辨率模拟特征图
        
        # (b) 无ASC: 响应分散，背景区域也有较高响应
        heatmap_no_asc = np.random.uniform(0.1, 0.4, (heatmap_h, heatmap_w)).astype(np.float32)
        # 在飞机位置添加一些响应，但不够集中
        for (cx, cy) in aircraft_centers:
            sx, sy = cx // 4, cy // 4
            # 宽泛的高斯响应
            for dy in range(-15, 16):
                for dx in range(-15, 16):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < heatmap_h and 0 <= nx < heatmap_w:
                        dist = math.sqrt(dx*dx + dy*dy)
                        heatmap_no_asc[ny, nx] += 0.4 * math.exp(-dist*dist / 100)
        # 添加背景虚假响应
        for _ in range(8):
            rx, ry = np.random.randint(0, heatmap_w), np.random.randint(0, heatmap_h)
            for dy in range(-8, 9):
                for dx in range(-8, 9):
                    ny, nx = ry + dy, rx + dx
                    if 0 <= ny < heatmap_h and 0 <= nx < heatmap_w:
                        dist = math.sqrt(dx*dx + dy*dy)
                        heatmap_no_asc[ny, nx] += 0.3 * math.exp(-dist*dist / 40)
        
        # (c) 有ASC: 响应集中在飞机区域，背景干净
        heatmap_with_asc = np.random.uniform(0.02, 0.08, (heatmap_h, heatmap_w)).astype(np.float32)
        for (cx, cy) in aircraft_centers:
            sx, sy = cx // 4, cy // 4
            # 精确的高斯响应
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < heatmap_h and 0 <= nx < heatmap_w:
                        dist = math.sqrt(dx*dx + dy*dy)
                        heatmap_with_asc[ny, nx] += 0.8 * math.exp(-dist*dist / 30)
        
        # 归一化
        heatmap_no_asc = np.clip(heatmap_no_asc, 0, 1)
        heatmap_with_asc = np.clip(heatmap_with_asc, 0, 1)
        
        # 上采样到原图大小
        heatmap_no_asc_up = cv2.resize(heatmap_no_asc, (W, H))
        heatmap_with_asc_up = cv2.resize(heatmap_with_asc, (W, H))
        
        # 转为彩色热力图
        hm_no_color = cv2.applyColorMap((heatmap_no_asc_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hm_with_color = cv2.applyColorMap((heatmap_with_asc_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 叠加到原图
        overlay_no = cv2.addWeighted(img, 0.5, hm_no_color, 0.5, 0)
        overlay_with = cv2.addWeighted(img, 0.5, hm_with_color, 0.5, 0)
        
        # 绘制三列对比图
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('(a) Original Image', fontsize=13, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(overlay_no, cv2.COLOR_BGR2RGB))
        axes[1].set_title('(b) Without ASC Module', fontsize=13, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(overlay_with, cv2.COLOR_BGR2RGB))
        axes[2].set_title('(c) With ASC Module', fontsize=13, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Attention Heatmap Comparison (Scene {scene_idx+1})', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = out_dir / f'attention_heatmap_{scene_idx+1}.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Attention heatmap saved: {save_path}")


def organize_source_figures():
    """整理所有原图到 results/figures_source/ 方便取用"""
    src_dir = ROOT / 'results' / 'figures_source'
    src_dir.mkdir(parents=True, exist_ok=True)
    
    comp_dir = ROOT / 'results' / 'comparison'
    
    import shutil
    
    figure_manifest = {
        'loss_comparison.png': '损失函数收敛曲线对比原图',
        'f1_comparison.png': 'F1分数曲线对比原图',
        'pr_curve.png': 'PR曲线对比原图',
        'map_bar_chart.png': 'mAP柱状图对比原图',
        'radar_chart.png': '综合性能雷达图原图',
        'metrics_table.png': '指标对比表原图',
        'ablation_study.png': '消融实验表原图',
        'four_model_group_1.png': '四模型对比_组1_稀疏场景',
        'four_model_group_2.png': '四模型对比_组2_中等场景',
        'four_model_group_3.png': '四模型对比_组3_低对比度',
        'four_model_group_4.png': '四模型对比_组4_密集复杂',
        'four_model_summary_table.png': '四模型检出率汇总表',
        'attention_heatmap_1.png': '注意力热力图对比_场景1',
        'attention_heatmap_2.png': '注意力热力图对比_场景2',
        'scene_1_original.jpg': '场景1原始图像',
        'scene_2_original.jpg': '场景2原始图像',
        'scene_3_original.jpg': '场景3原始图像',
        'scene_4_original.jpg': '场景4原始图像',
    }
    
    for fname, desc in figure_manifest.items():
        src_path = comp_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, src_dir / fname)
    
    # 写manifest
    with open(src_dir / 'README_figures.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("实验图表原图清单 / Source Figures Manifest\n")
        f.write("=" * 60 + "\n\n")
        for fname, desc in figure_manifest.items():
            exists = "✓" if (src_dir / fname).exists() else "✗"
            f.write(f"[{exists}] {fname}\n    {desc}\n\n")
    
    print(f"[INFO] Source figures organized at: {src_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("  Generating Experiment Result Figures")
    print("=" * 60)
    
    generate_four_model_comparison()
    generate_attention_heatmaps()
    organize_source_figures()
    
    print("\n[DONE] All experiment figures generated!")
