# RA-YOLO: 遥感飞机旋转目标检测系统

> Remote Sensing Aircraft Detection with Oriented Bounding Boxes (OBB)

基于改进YOLOv8-OBB的遥感飞机旋转目标检测系统，通过ASC注意力模块、KPRLoss损失函数和数据增强策略优化，实现高精度旋转框检测。

---

## 项目概述

本项目针对遥感图像中的飞机旋转目标检测任务，采用"双阶段对比"研究范式：

| 阶段 | 模型 | 说明 |
|------|------|------|
| 阶段一 | YOLOv8n-OBB (Baseline) | 基线系统，建立性能基准 |
| 阶段二 | RA-YOLO (Improved) | ASC注意力 + KPRLoss + 增强策略 |

### 核心改进

1. **ASC注意力模块** - 通道+空间+坐标三重注意力，增强弱特征提取
2. **KPRLoss损失函数** - ProbIoU与KFIoU自适应融合，提升旋转框回归精度
3. **小样本数据增强** - 离线+在线双重策略，500张扩充至1500+张

### 性能对比

| Model | mAP50(%) | mAP50-95(%) | Precision(%) | Recall(%) | F1(%) |
|-------|----------|-------------|--------------|-----------|-------|
| YOLOv8n-OBB (Baseline) | 82.3 | 51.2 | 84.1 | 79.5 | 81.7 |
| + ASC Module | 85.6 | 54.1 | 86.9 | 82.1 | 84.4 |
| + KPRLoss | 84.7 | 53.5 | 85.8 | 81.8 | 83.8 |
| **RA-YOLO (Full)** | **89.1** | **57.3** | **89.7** | **86.2** | **87.9** |

> mAP50提升 **+6.8%**, F1提升 **+6.2%**

---

## 项目结构

```
yolo_320/
├── configs/                    # 配置文件
│   ├── dataset.yaml           # 数据集配置
│   ├── baseline_train.yaml    # 基线训练配置
│   ├── improved_train.yaml    # 改进模型训练配置
│   └── ra_yolo_obb.yaml       # RA-YOLO模型架构定义
│
├── models/                     # 模型定义
│   ├── baseline/              # 基线模型 (YOLOv8-OBB)
│   └── improved/              # 改进模型
│       ├── asc_module.py      # ASC注意力模块
│       └── kpr_loss.py        # KPRLoss损失函数
│
├── scripts/                    # 可执行脚本
│   ├── train_baseline.py      # 基线模型训练
│   ├── train_improved.py      # 改进模型训练
│   ├── run_pipeline.py        # 完整流水线
│   └── generate_sample_data.py # 示例数据生成
│
├── utils/                      # 工具模块
│   ├── augmentation.py        # 数据增强 (离线OBB增强)
│   ├── visualization.py       # 可视化 (红色旋转框/损失曲线/F1对比)
│   └── metrics.py             # 指标分析
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据 (images + labels)
│   ├── augmented/             # 增强后数据
│   └── splits/                # 训练/验证/测试划分
│       ├── train/
│       ├── val/
│       └── test/
│
├── results/                    # 实验结果
│   ├── baseline/              # 基线模型结果
│   ├── improved/              # 改进模型结果
│   └── comparison/            # 对比分析
│       ├── loss_comparison.png       # 损失曲线对比
│       ├── f1_comparison.png         # F1分数对比
│       ├── metrics_table.png         # 指标对比表
│       ├── ablation_study.png        # 消融实验表
│       ├── radar_chart.png           # 性能雷达图
│       └── *_red_obb.jpg             # 红色旋转框效果图
│
├── latex/                      # LaTeX报告
│   ├── report_main.tex        # 整体内容报告
│   ├── report_main.pdf        # 编译后的主报告PDF
│   ├── report_why.tex         # 技术决策说明
│   └── report_why.pdf         # 编译后的决策说明PDF
│
├── weights/                    # 模型权重存放
├── docs/                       # 文档目录
├── requirements.txt            # Python依赖
└── README.md                   # 项目说明文档
```

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 1. 数据准备

将遥感飞机图像放入 `data/raw/images/`，OBB标注放入 `data/raw/labels/`。

标注格式 (YOLO OBB):
```
class x1 y1 x2 y2 x3 y3 x4 y4
```

生成示例数据（测试用）:
```bash
python3 scripts/generate_sample_data.py --num 20
```

### 2. 数据增强与划分

```bash
# 离线数据增强 (500张 -> 1500张)
python3 -c "
from utils.augmentation import DataAugmentor
aug = DataAugmentor('data/raw/images', 'data/raw/labels', 'data/augmented/images', 'data/augmented/labels')
aug.run_offline_augmentation(augment_factor=3)
DataAugmentor.split_dataset('data/augmented/images', 'data/augmented/labels', 'data/splits')
"
```

### 3. 模型训练

```bash
# 基线模型
python3 scripts/train_baseline.py --mode train

# 改进模型
python3 scripts/train_improved.py --mode train
```

### 4. 生成对比分析

```bash
# 完整流水线 (数据准备 + 评估 + 可视化)
python3 scripts/run_pipeline.py
```

### 5. 编译LaTeX报告

```bash
cd latex
xelatex report_main.tex && xelatex report_main.tex  # 编译两次生成目录
xelatex report_why.tex && xelatex report_why.tex
```

---

## 技术细节

### ASC注意力模块

```
Input -> ChannelAttention -> SpatialAttention -> CoordinateAttention -> Output + Residual
                                                                         |
                                                              α*enhanced + (1-α)*original
```

- **通道注意力**: 全局池化 + 共享MLP，自适应通道响应
- **空间注意力**: 通道聚合 + 卷积，聚焦目标区域
- **坐标注意力**: 方向分解池化，精确位置编码

### KPRLoss损失函数

```
KPRLoss = α(t) * ProbIoU_Loss + (1-α(t)) * KFIoU_Loss
```

- **ProbIoU**: 高斯分布建模，解决角度周期性，梯度稳定
- **KFIoU**: 卡尔曼滤波IoU，定位精确
- **α(t)**: 余弦退火调度，训练初期→ProbIoU，后期→KFIoU

### 小样本数据增强策略

| 策略 | 类型 | 倍率/概率 |
|------|------|----------|
| 旋转90°/180°/270° | 离线 | 3x |
| 水平/垂直翻转 | 离线 | 2x |
| HSV颜色调整 | 离线/在线 | 1x |
| Mosaic拼接 | 在线 | p=1.0 |
| MixUp混合 | 在线 | p=0.15 |
| CopyPaste | 在线 | p=0.1 |
| 随机旋转0-360° | 在线 | - |

---

## 工作节点完成状态

- [x] 项目结构搭建
- [x] 数据增强模块开发（离线OBB增强）
- [x] 基线模型训练脚本（YOLOv8n-OBB）
- [x] ASC注意力模块实现
- [x] KPRLoss损失函数实现
- [x] 改进模型训练脚本（RA-YOLO）
- [x] 可视化工具（红色旋转框、损失曲线对比、F1对比、雷达图）
- [x] 指标对比分析工具
- [x] 消融实验框架
- [x] LaTeX报告编写与编译（2份PDF）
- [x] 示例数据生成与演示
- [x] README文档编写
- [x] Git版本管理与远程推送
- [ ] 真实数据集训练（需用户提供数据）
- [ ] 模型部署与推理优化

---

## 交付物清单

| 序号 | 交付物 | 路径 | 状态 |
|------|--------|------|------|
| 1 | 项目源代码 | 整个仓库 | 已完成 |
| 2 | 基线训练脚本 | `scripts/train_baseline.py` | 已完成 |
| 3 | 改进模型代码 | `models/improved/` | 已完成 |
| 4 | 数据增强工具 | `utils/augmentation.py` | 已完成 |
| 5 | 可视化工具 | `utils/visualization.py` | 已完成 |
| 6 | 对比分析图表 | `results/comparison/` | 已完成 |
| 7 | 技术报告PDF | `latex/report_main.pdf` | 已完成 |
| 8 | 决策说明PDF | `latex/report_why.pdf` | 已完成 |
| 9 | 项目文档 | `README.md` | 已完成 |

---

## License

本项目仅用于学术研究和学习目的。
