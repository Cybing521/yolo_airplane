# RA-YOLO: 遥感飞机旋转目标检测系统

> Remote Sensing Aircraft Detection with Oriented Bounding Boxes (OBB)

基于改进YOLOv8-OBB的遥感飞机旋转目标检测系统，通过ASC注意力模块、ASOR-Loss损失函数和数据增强策略优化，实现高精度旋转框检测。

---

## 项目概述

本项目针对遥感图像中的飞机旋转目标检测任务，采用"双阶段对比"研究范式：

| 阶段 | 模型 | 说明 |
|------|------|------|
| 阶段一 | YOLOv8n-OBB (Baseline) | 基线系统，建立性能基准 |
| 阶段二 | RA-YOLO (Improved) | ASC注意力 + ASOR-Loss + 增强策略 |

### 数据集

- **来源**: `air-cj/` 目录，534张遥感飞机图像（640x640）
- **标注**: 使用预训练YOLOv8-OBB自动标注，共检测8111个飞机目标
- **划分**: 训练集373张 / 验证集106张 / 测试集54张

### 核心改进

1. **ASC注意力模块** - 通道+空间+坐标三重注意力，增强弱特征提取
2. **ASOR-Loss损失函数** - ProbIoU与KFIoU自适应融合，提升旋转框回归精度
3. **数据增强策略** - MixUp(0.15) + CopyPaste(0.1) + 增强HSV/旋转

### 训练方式

使用 tmux 进行后台训练：
```bash
# 启动训练
tmux new-session -d -s train "python3 scripts/train_real.py --mode improved"

# 查看训练状态
tmux attach -t train

# 后台运行后断开
Ctrl+B, D
```

---

## 项目结构

```
yolo_320/
├── air-cj/                     # 原始遥感飞机图像 (534张, gitignore)
├── configs/                    # 配置文件
│   ├── dataset_real.yaml      # 真实数据集配置
│   ├── baseline_train.yaml    # 基线训练配置
│   ├── improved_train.yaml    # 改进模型训练配置
│   └── ra_yolo_obb.yaml       # RA-YOLO模型架构
├── models/                     # 模型定义
│   ├── baseline/              # 基线模型
│   └── improved/              # 改进模型
│       ├── asc_module.py      # ASC注意力模块
│       └── kpr_loss.py        # ASOR-Loss损失函数
├── scripts/                    # 可执行脚本
│   ├── prepare_real_data.py   # 真实数据准备 (自动标注+划分)
│   ├── train_real.py          # 真实数据训练 (baseline/improved)
│   ├── train_baseline.py      # 基线训练脚本
│   ├── train_improved.py      # 改进模型训练脚本
│   └── generate_experiment_figures.py # 实验图表生成
├── utils/                      # 工具模块
│   ├── augmentation.py        # 数据增强
│   ├── visualization.py       # 可视化
│   └── metrics.py             # 指标分析
├── data/                       # 数据目录 (gitignore)
│   ├── real/                  # 标注后的完整数据
│   └── real_splits/           # 训练/验证/测试划分
├── runs/                       # 训练输出 (gitignore)
│   ├── baseline/              # 基线训练结果
│   └── improved/              # 改进模型训练结果
├── results/comparison/         # 对比分析图表
├── latex/                      # LaTeX报告
│   ├── report_main.pdf        # 整体内容报告 (25页)
│   ├── report_why.pdf         # 技术决策说明 (15页)
│   └── figures/               # 报告用图
├── requirements.txt
└── README.md
```

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 1. 数据准备

将遥感飞机图像放入 `air-cj/` 目录，运行自动标注和划分：

```bash
python3 scripts/prepare_real_data.py
```

### 2. 训练 (tmux)

```bash
# 基线模型
tmux new-session -d -s baseline "python3 scripts/train_real.py --mode baseline"

# 改进模型
tmux new-session -d -s improved "python3 scripts/train_real.py --mode improved"

# 同时训练两个
tmux new-session -d -s train "python3 scripts/train_real.py --mode both"
```

### 3. 查看训练状态

```bash
tmux attach -t train    # 进入tmux查看
# Ctrl+B, D             # 退出tmux (训练继续)
tmux ls                 # 列出所有tmux会话
```

### 4. 编译LaTeX报告

```bash
cd latex
xelatex report_main.tex && xelatex report_main.tex
xelatex report_why.tex && xelatex report_why.tex
```

---

## 技术细节

### ASOR-Loss损失函数

```
ASOR-Loss = α(t) * ProbIoU_Loss + (1-α(t)) * KFIoU_Loss
```

- **ProbIoU**: 高斯分布建模，解决角度周期性，梯度稳定
- **KFIoU**: 卡尔曼滤波IoU，定位精确
- **α(t)**: 余弦退火调度，训练初期→ProbIoU，后期→KFIoU

### ASC注意力模块

```
Input -> ChannelAttention -> SpatialAttention -> CoordinateAttention -> Output + Residual
```

---

## 工作节点

- [x] 项目结构搭建
- [x] 真实数据准备 (air-cj 534张, 自动标注8111个目标)
- [x] 数据集划分 (373/106/54)
- [x] 基线模型训练 (YOLOv8n-OBB, 100 epochs)
- [x] 改进模型训练 (RA-YOLO, 100 epochs, tmux)
- [x] ASC注意力模块实现
- [x] ASOR-Loss损失函数实现
- [x] 可视化工具
- [x] LaTeX报告 (2份PDF)
- [x] README文档
- [ ] 训练完成后的结果对比分析
- [ ] 报告更新 (真实训练数据)
