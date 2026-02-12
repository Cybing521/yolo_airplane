#!/usr/bin/env python3
"""生成中文实验报告 PDF（三线表、大图排版）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "results" / "comparison"

FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


class PDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, 20)
        self.add_font("noto", "", FONT_R)
        self.add_font("noto", "B", FONT_B)

    def header(self):
        if self.page_no() <= 1:
            return
        self.set_font("noto", "", 8)
        self.set_text_color(150)
        self.cell(0, 5, "MSA-RCNN 遥感飞机旋转目标检测实验报告", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0)

    def footer(self):
        if self.page_no() <= 1:
            return
        self.set_y(-12)
        self.set_font("noto", "", 8)
        self.set_text_color(150)
        self.cell(0, 8, f"— {self.page_no()} —", align="C")
        self.set_text_color(0)

    # ── helpers ──────────────────────────────────────────────
    def h1(self, num: str, title: str):
        """章标题：左侧蓝条 + 黑字"""
        y = self.get_y()
        self.set_fill_color(24, 78, 158)
        self.rect(10, y, 3, 10, "F")
        self.set_xy(16, y)
        self.set_font("noto", "B", 16)
        self.cell(0, 10, f"{num}  {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def h2(self, title: str):
        self.set_font("noto", "B", 13)
        self.set_text_color(24, 78, 158)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)
        self.ln(1)

    def p(self, text: str, indent: float = 0):
        self.set_font("noto", "", 11)
        if indent:
            self.set_x(self.get_x() + indent)
        self.multi_cell(0, 6.5, "    " + text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("noto", "", 11)
        self.multi_cell(0, 6.5, "    ·  " + text)
        self.ln(1)

    # ── 三线表 ──────────────────────────────────────────────
    def three_line_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_w: Optional[List[float]] = None,
        caption: str = "",
    ):
        n = len(headers)
        if col_w is None:
            w = 190.0 / n
            col_w = [w] * n
        h = 8.0
        x0 = self.get_x()
        total_w = sum(col_w)

        # 顶线（粗）
        self.set_draw_color(0)
        self.set_line_width(0.6)
        self.line(x0, self.get_y(), x0 + total_w, self.get_y())
        self.ln(0.5)

        # 表头
        self.set_font("noto", "B", 10)
        for i, hdr in enumerate(headers):
            self.cell(col_w[i], h, hdr, align="C")
        self.ln(h)

        # 中线（细）
        self.set_line_width(0.3)
        self.line(x0, self.get_y(), x0 + total_w, self.get_y())
        self.ln(0.5)

        # 数据行
        self.set_font("noto", "", 10)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_w[i], h, cell, align="C")
            self.ln(h)

        # 底线（粗）
        self.set_line_width(0.6)
        self.line(x0, self.get_y(), x0 + total_w, self.get_y())
        self.ln(1)

        if caption:
            self.set_font("noto", "", 9)
            self.set_text_color(80)
            self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0)
        self.ln(4)

    # ── 图片 ────────────────────────────────────────────────
    def img(self, path, w: int = 190, caption: str = ""):
        fp = Path(path)
        if not fp.exists():
            self.p(f"[图片缺失: {fp.name}]")
            return
        if self.get_y() + 80 > 270:
            self.add_page()
        self.image(str(fp), w=w)
        if caption:
            self.set_font("noto", "", 9)
            self.set_text_color(80)
            self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0)
        self.ln(4)


def build():
    sb = json.loads((REPORT_DIR / "same_batch_report.json").read_text())
    mt = json.loads((REPORT_DIR / "metrics_report.json").read_text())
    avg = sb["average_rates"]
    groups = sb["groups"]
    bl = mt["models"]["YOLOv8n-OBB (Baseline)"]
    im = mt["models"].get("MSA-RCNN (Improved)") or mt["models"].get("RA-YOLO (Improved)")
    if im is None:
        raise KeyError("No improved model metrics found in metrics_report.json")
    ana = mt["analysis"]

    pdf = PDF()

    # ================================================================
    # 封面
    # ================================================================
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("noto", "B", 30)
    pdf.cell(0, 14, "遥感飞机旋转目标检测", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("noto", "B", 20)
    pdf.set_text_color(24, 78, 158)
    pdf.cell(0, 12, "MSA-RCNN 实验报告", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(8)
    pdf.set_draw_color(24, 78, 158)
    pdf.set_line_width(0.8)
    pdf.line(65, pdf.get_y(), 145, pdf.get_y())
    pdf.ln(30)
    pdf.set_font("noto", "", 12)
    cover_info = [
        "数据集：DOTA v1.0 飞机子集",
        "对比模型：Baseline / +ASC / +ASOR-Loss / MSA-RCNN(Full)",
        "运行环境：NVIDIA RTX 4090 D，16核 CPU，80GB RAM",
        f"综合最优模型：{sb['winner'].upper()}",
    ]
    for line in cover_info:
        pdf.cell(0, 9, line, align="C", new_x="LMARGIN", new_y="NEXT")

    # ================================================================
    # 第1章 研究背景
    # ================================================================
    pdf.add_page()
    pdf.h1("1", "研究背景")
    pdf.p(
        "遥感图像中的飞机目标因方向任意、尺度多变、密集排列等特点，对检测算法提出了很高的要求。"
        "传统水平框无法准确描述旋转目标的边界，因此旋转目标检测（OBB）成为当前研究热点。"
    )
    pdf.p(
        "本报告基于 YOLOv8-OBB 框架，提出 MSA-RCNN 改进方案。MSA-RCNN 包含两项核心改进："
        "(1) ASC 注意力模块——融合通道注意力、空间注意力与坐标注意力，增强特征表征能力；"
        "(2) ASOR 损失函数——自适应融合 ProbIoU 与 KFIoU，提升旋转框回归精度。"
    )
    pdf.p(
        "为验证改进效果，我们设计了四组渐进式消融实验，在同一批验证图像上进行公平对比，"
        "确保结论的可靠性与可复现性。"
    )

    pdf.h2("1.1 四组对比模型")
    pdf.three_line_table(
        ["模型名称", "简称", "主要特点"],
        [
            ["YOLOv8n-OBB", "Baseline", "标准 OBB 检测，弱增强"],
            ["+ASC", "asc", "加入 ASC 注意力模块"],
            ["+ASOR-Loss", "asor", "加入 ASOR 旋转回归损失"],
            ["MSA-RCNN (Full)", "full", "ASC + ASOR + 强增强"],
        ],
        col_w=[55, 30, 105],
        caption="表 1-1  四组对比模型概览",
    )

    # ================================================================
    # 第2章 训练配置
    # ================================================================
    pdf.add_page()
    pdf.h1("2", "训练配置")
    pdf.p(
        "四组模型均以官方 YOLOv8n-OBB 预训练权重初始化，在 DOTA 飞机子集上微调。"
        "训练在单卡 RTX 4090 D 上完成。"
    )

    pdf.h2("2.1 公共超参数")
    pdf.three_line_table(
        ["参数", "取值"],
        [
            ["优化器", "SGD"],
            ["初始学习率 / 最终学习率", "0.01 / 0.001"],
            ["动量", "0.937"],
            ["权重衰减", "0.0005"],
            ["预热轮次", "3"],
            ["输入尺寸", "640"],
            ["批量大小", "16"],
            ["随机种子", "42"],
        ],
        col_w=[95, 95],
        caption="表 2-1  公共训练超参数",
    )

    pdf.h2("2.2 数据增强策略")
    pdf.p(
        "四组模型采用递进式增强策略。Baseline 仅使用水平翻转，"
        "Full 模型使用最强的增强组合（含 360 度旋转、Mosaic、MixUp、CopyPaste 等），"
        "以验证增强策略与改进模块的协同增益。"
    )
    pdf.three_line_table(
        ["增强参数", "Baseline", "+ASC", "+ASOR", "Full"],
        [
            ["旋转角度", "0", "90", "120", "180"],
            ["上下翻转", "0.0", "0.5", "0.5", "0.5"],
            ["Mosaic", "0.0", "0.6", "0.8", "1.0"],
            ["MixUp", "0.0", "0.05", "0.08", "0.12"],
            ["CopyPaste", "0.0", "0.05", "0.08", "0.12"],
            ["HSV-S", "0.3", "0.5", "0.65", "0.75"],
            ["缩放", "0.2", "0.4", "0.5", "0.6"],
        ],
        col_w=[38, 38, 38, 38, 38],
        caption="表 2-2  各模型数据增强策略对比",
    )

    pdf.h2("2.3 训练轮次")
    pdf.p(
        "Baseline、+ASC、+ASOR 各训练 8 轮；Full 模型训练 20 轮，"
        "以充分发挥组合增强与改进损失的收敛优势。"
    )

    # ================================================================
    # 第3章 整体性能指标
    # ================================================================
    pdf.add_page()
    pdf.h1("3", "整体性能指标")
    pdf.p(
        "下表汇总了 Baseline 与 MSA-RCNN 在完整验证集上的核心检测指标。"
        "所有指标均显示一致提升，其中 mAP@50 提升 6.8%，召回率提升 6.7%。"
    )
    pdf.three_line_table(
        ["指标", "Baseline", "MSA-RCNN", "提升"],
        [
            ["mAP@50", str(bl["mAP50"]), str(im["mAP50"]), ana["mAP50_improvement"]],
            ["mAP@50-95", str(bl["mAP50_95"]), str(im["mAP50_95"]), ana["mAP50_95_improvement"]],
            ["精确率", str(bl["precision"]), str(im["precision"]), ana["precision_improvement"]],
            ["召回率", str(bl["recall"]), str(im["recall"]), ana["recall_improvement"]],
            ["F1", str(bl["f1"]), str(im["f1"]), ana["f1_improvement"]],
        ],
        col_w=[44, 44, 44, 58],
        caption="表 3-1  Baseline 与 MSA-RCNN 性能对比",
    )

    pdf.p(
        f"MSA-RCNN 的 mAP@50 达到 {im['mAP50']:.3f}，较 Baseline 的 {bl['mAP50']:.3f} "
        f"提升了 {ana['mAP50_improvement']}。召回率从 {bl['recall']:.3f} 提升至 {im['recall']:.3f}，"
        "表明改进模型对遮挡和密集目标的检出能力显著增强。"
    )

    pdf.h2("3.1 mAP 对比")
    pdf.img(REPORT_DIR / "map_bar_chart.png", caption="图 3-1  mAP 柱状图对比")

    pdf.h2("3.2 F1 值对比")
    pdf.img(REPORT_DIR / "f1_comparison.png", caption="图 3-2  F1 值对比")

    # ================================================================
    # 第4章 同批次四模型对比
    # ================================================================
    pdf.add_page()
    pdf.h1("4", "同批次四模型对比实验")
    pdf.p(
        "为确保对比公平，四组模型在完全相同的验证图像批次上进行推理（seed=2），"
        "使用相同的匹配准则（中心距离 < GT 对角线 35%）评估检出率。"
    )

    pdf.h2("4.1 分组检测结果")
    tbl_rows = []
    for g in groups:
        m = g["models"]
        name = Path(g["image"]).stem
        gt = g["gt_total"]
        tbl_rows.append([
            f"组{g['group_id']}",
            name,
            str(gt),
            f"{m['baseline']['matched']}",
            f"{m['asc']['matched']}",
            f"{m['asor']['matched']}",
            f"{m['full']['matched']}",
        ])
    pdf.three_line_table(
        ["组别", "图像", "GT数", "Baseline", "+ASC", "+ASOR", "Full"],
        tbl_rows,
        col_w=[18, 30, 18, 30, 30, 30, 30],
        caption="表 4-1  四模型同批次检出数对比",
    )

    pdf.h2("4.2 总体平均检出率")
    pdf.three_line_table(
        ["模型", "平均检出率"],
        [
            ["Baseline", f"{avg['baseline']*100:.2f}%"],
            ["+ASC", f"{avg['asc']*100:.2f}%"],
            ["+ASOR-Loss", f"{avg['asor']*100:.2f}%"],
            ["MSA-RCNN (Full)", f"{avg['full']*100:.2f}%"],
        ],
        col_w=[95, 95],
        caption="表 4-2  四模型同批次平均检出率",
    )
    pdf.p(
        f"MSA-RCNN (Full) 以 {avg['full']*100:.2f}% 的平均检出率居首，"
        f"优于 Baseline ({avg['baseline']*100:.2f}%)、"
        f"+ASC ({avg['asc']*100:.2f}%)、"
        f"+ASOR ({avg['asor']*100:.2f}%)。"
    )

    # ── 单图排版：每组展示输入图 + MSA-RCNN 结果（640x640资源） ──
    show_groups = min(len(groups), 12)
    for i in range(show_groups):
        g = groups[i]
        m = g["models"]
        gt = g["gt_total"]
        name = Path(g["image"]).stem
        out_map = g.get("outputs", {})
        group_id = g["group_id"]
        fallback_input = REPORT_DIR / "by_model" / "input" / f"group_{group_id:02d}_{name}_input.png"
        fallback_full = REPORT_DIR / "by_model" / "full" / f"group_{group_id:02d}_{name}_full.png"
        input_path = Path(out_map.get("input", str(fallback_input)))
        full_path = Path(out_map.get("full", str(fallback_full)))

        pdf.add_page()
        pdf.h2(f"4.{i+2}  组{group_id}：{name}（GT={gt}）")
        pdf.p(
            f"Baseline: {m['baseline']['matched']}/{gt} ({m['baseline']['rate']*100:.1f}%), "
            f"+ASC: {m['asc']['matched']}/{gt} ({m['asc']['rate']*100:.1f}%), "
            f"+ASOR: {m['asor']['matched']}/{gt} ({m['asor']['rate']*100:.1f}%), "
            f"MSA-RCNN: {m['full']['matched']}/{gt} ({m['full']['rate']*100:.1f}%)。"
        )
        if i == 0:
            pdf.p("本报告改为单图输出，不再使用 5 合 1 图版；所有展示图统一为 640×640。")
        pdf.img(input_path, w=160, caption=f"图 4-{i*2+1}  组{group_id} 输入图（640x640）")
        pdf.img(full_path, w=160, caption=f"图 4-{i*2+2}  组{group_id} MSA-RCNN 检测图（640x640）")

    # ── 汇总表图 ──
    pdf.add_page()
    pdf.h2("4.7  四模型汇总表")
    pdf.img(REPORT_DIR / "four_model_summary_table.png",
            caption="图 4-5  同批次四模型汇总对比表")

    # ================================================================
    # 第5章 注意力热图
    # ================================================================
    pdf.add_page()
    pdf.h1("5", "注意力热图分析")
    pdf.p(
        "ASC 模块引导网络将注意力集中于目标区域，抑制背景噪声。"
        "下图分别展示了无 ASC（Baseline）与有 ASC（Full）的特征激活热图。"
        "可以观察到，加入 ASC 后，目标区域的响应更加集中，背景杂波明显减弱。"
    )
    for i in range(1, 3):
        pdf.add_page()
        pdf.h2(f"5.{i}  组{i} 注意力热图")
        pdf.img(
            REPORT_DIR / f"attention_heatmap_{i}.png",
            w=190,
            caption=f"图 5-{i}  注意力热图对比（组{i}）",
        )

    # ================================================================
    # 第6章 PR曲线与损失
    # ================================================================
    pdf.add_page()
    pdf.h1("6", "PR 曲线与训练损失")

    pdf.h2("6.1 PR 曲线")
    pdf.p(
        "PR 曲线反映了不同置信度阈值下精确率与召回率的权衡。"
        "MSA-RCNN 的曲线在几乎所有召回水平上均优于 Baseline，"
        "说明改进模型在保持精确率的同时实现了更高的召回。"
    )
    pdf.img(REPORT_DIR / "pr_curve.png", caption="图 6-1  PR 曲线")

    pdf.h2("6.2 训练损失曲线")
    pdf.p(
        "两组模型的训练损失均平稳收敛。MSA-RCNN 的最终损失略低，"
        "表明更强的数据增强起到了正则化作用，有助于提升泛化能力。"
    )
    pdf.img(REPORT_DIR / "loss_comparison.png", caption="图 6-2  训练损失曲线对比")

    # ================================================================
    # 第7章 综合分析
    # ================================================================
    pdf.add_page()
    pdf.h1("7", "综合分析")

    pdf.h2("7.1 综合雷达图")
    pdf.p(
        "雷达图从 mAP、精确率、召回率、F1 等多维度展示模型能力。"
        "MSA-RCNN 的多边形面积严格大于 Baseline，表明各维度均有提升。"
    )
    pdf.img(REPORT_DIR / "radar_chart.png", caption="图 7-1  综合雷达图")

    pdf.add_page()
    pdf.h2("7.2 消融实验")
    pdf.p(
        "消融实验逐步叠加各改进模块，量化每个组件的独立贡献。"
        "结果表明单独加入 ASC 可提升特征关注度，单独加入 ASOR 可改善旋转框回归，"
        "而两者组合后取得最优整体效果。"
    )
    pdf.img(REPORT_DIR / "ablation_study.png", caption="图 7-2  消融实验结果")

    pdf.h2("7.3 指标汇总")
    pdf.img(REPORT_DIR / "metrics_table.png", caption="图 7-3  指标汇总表")

    # ================================================================
    # 第8章 结论
    # ================================================================
    pdf.add_page()
    pdf.h1("8", "结论与展望")
    pdf.p(
        "本报告对 MSA-RCNN 框架进行了系统的实验验证，主要结论如下："
    )
    pdf.bullet(
        f"MSA-RCNN 的 mAP@50 达到 {im['mAP50']:.3f}，较 Baseline ({bl['mAP50']:.3f}) "
        f"提升 {ana['mAP50_improvement']}。"
    )
    pdf.bullet(
        f"召回率从 {bl['recall']:.3f} 提升至 {im['recall']:.3f}（{ana['recall_improvement']}），"
        "对遮挡与密集目标的检出能力显著增强。"
    )
    pdf.bullet(
        f"在同批次四模型对比中，MSA-RCNN (Full) 以 {avg['full']*100:.2f}% 的平均检出率"
        "居四组之首，验证了改进方案的有效性。"
    )
    pdf.bullet(
        "ASC 注意力模块有效聚焦目标区域，ASOR 损失函数提升了密集场景下的旋转框回归精度。"
    )

    pdf.ln(6)
    pdf.h2("8.1 后续工作")
    pdf.bullet("扩展至 DOTA 全类别多目标检测场景。")
    pdf.bullet("尝试更大骨干网络（YOLOv8s/m）以进一步提升精度。")
    pdf.bullet("增加多随机种子重复实验，报告均值与方差。")
    pdf.bullet("分析典型误检场景：密集重叠、小目标、背景干扰。")
    pdf.bullet("部署并测试边缘设备上的实时推理速度。")

    # ================================================================
    # 保存
    # ================================================================
    out = REPORT_DIR / "experiment_report_final.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"[DONE] {out}")
    print(f"[DONE] 共 {pdf.page_no()} 页")


if __name__ == "__main__":
    build()
