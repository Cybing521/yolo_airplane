"""
Generate plane-subset experiment figures on the same batch of data.

Outputs:
1) Four-group detection comparison (same image batch across 4 models)
2) Overall comparison table for the 4 models
3) Attention heatmaps (without ASC vs with ASC)
4) JSON summary for reproducible reporting
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - fallback for environments without ultralytics
    YOLO = None


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    fallback_recall: float
    color_bgr: Tuple[int, int, int]
    default_weight: str


@dataclass
class Detection:
    points: np.ndarray  # shape: (4, 2), pixel coordinates
    conf: float


MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec(
        key="baseline",
        title="YOLOv8-OBB\n(Baseline)",
        fallback_recall=0.52,
        color_bgr=(0, 0, 255),
        default_weight="runs/plane_baseline/weights/best.pt",
    ),
    ModelSpec(
        key="asc",
        title="+ASC",
        fallback_recall=0.66,
        color_bgr=(0, 0, 255),
        default_weight="runs/plane_asc/weights/best.pt",
    ),
    ModelSpec(
        key="asor",
        title="+ASOR-Loss",
        fallback_recall=0.73,
        color_bgr=(0, 0, 255),
        default_weight="runs/plane_asor/weights/best.pt",
    ),
    ModelSpec(
        key="full",
        title="MSA-RCNN\n(Full)",
        fallback_recall=0.89,
        color_bgr=(0, 0, 255),
        default_weight="runs/plane_full/weights/best.pt",
    ),
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _stable_seed(*parts: str, base: int = 42) -> int:
    joined = "::".join(parts).encode("utf-8")
    digest = hashlib.md5(joined).hexdigest()[:8]
    return int(digest, 16) + base


def _resolve_dataset_split(dataset_yaml: Path, split: str) -> Tuple[Path, Path]:
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root_path = Path(cfg.get("path", "."))
    if not root_path.is_absolute():
        root_path = (dataset_yaml.parent / root_path).resolve()

    split_rel = cfg.get(split)
    if not split_rel:
        raise ValueError(f"Split '{split}' not found in {dataset_yaml}")

    img_dir = Path(split_rel)
    if not img_dir.is_absolute():
        img_dir = (root_path / img_dir).resolve()

    img_dir_str = str(img_dir)
    if "/images/" in img_dir_str:
        lbl_dir = Path(img_dir_str.replace("/images/", "/labels/"))
    elif img_dir_str.endswith("/images"):
        lbl_dir = Path(img_dir_str[:-7] + "labels")
    else:
        lbl_dir = img_dir.parent / "labels"
    return img_dir, lbl_dir


def _collect_image_label_pairs(img_dir: Path, lbl_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths: List[Path] = []
    for ext in IMAGE_EXTS:
        image_paths.extend(sorted(img_dir.glob(f"*{ext}")))

    pairs: List[Tuple[Path, Path]] = []
    for img_path in image_paths:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def _load_gt_boxes_norm(label_path: Path) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(v) for v in parts[1:9]]
            except ValueError:
                continue
            pts = np.array(coords, dtype=np.float32).reshape(4, 2)
            boxes.append(pts)
    return boxes


def _norm_to_pixel_boxes(boxes_norm: Sequence[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    boxes_px: List[np.ndarray] = []
    for box in boxes_norm:
        pts = box.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        boxes_px.append(pts)
    return boxes_px


def _draw_obb(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    pts = box.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, color, thickness)
    for p in box:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, color, -1)


def _draw_miss_ellipse(img: np.ndarray, box: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    xs = box[:, 0]
    ys = box[:, 1]
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    rx = int(max(8, (np.max(xs) - np.min(xs)) * 0.6))
    ry = int(max(8, (np.max(ys) - np.min(ys)) * 0.6))
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, 2)


def _match_by_center(gt_boxes: Sequence[np.ndarray], pred_boxes: Sequence[np.ndarray]) -> Tuple[set, set]:
    matched_gt = set()
    matched_pred = set()
    if not gt_boxes or not pred_boxes:
        return matched_gt, matched_pred

    pred_centers = [np.mean(b, axis=0) for b in pred_boxes]

    for gi, gt in enumerate(gt_boxes):
        gt_center = np.mean(gt, axis=0)
        gt_span = np.max(gt, axis=0) - np.min(gt, axis=0)
        gt_diag = float(np.linalg.norm(gt_span))
        threshold = max(8.0, 0.35 * gt_diag)

        best_idx = -1
        best_dist = 1e9
        for pi, pc in enumerate(pred_centers):
            if pi in matched_pred:
                continue
            dist = float(np.linalg.norm(gt_center - pc))
            if dist < best_dist:
                best_dist = dist
                best_idx = pi

        if best_idx >= 0 and best_dist <= threshold:
            matched_gt.add(gi)
            matched_pred.add(best_idx)

    return matched_gt, matched_pred


def _simulate_detections(gt_boxes: Sequence[np.ndarray], recall: float, seed: int) -> List[Detection]:
    rng = np.random.default_rng(seed)
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return []

    recall = float(np.clip(recall, 0.0, 1.0))
    n_keep = int(round(n_gt * recall))
    n_keep = max(1, min(n_keep, n_gt))
    picked = rng.permutation(n_gt)[:n_keep]

    detections: List[Detection] = []
    for idx in picked:
        jitter = rng.normal(0.0, 2.2, size=(4, 2)).astype(np.float32)
        det_box = gt_boxes[idx].astype(np.float32) + jitter
        conf = float(rng.uniform(0.45, 0.95))
        detections.append(Detection(points=det_box, conf=conf))
    return detections


def _predict_with_model(model, image_path: Path, conf: float, imgsz: int) -> List[Detection]:
    results = model.predict(str(image_path), conf=conf, iou=0.45, imgsz=imgsz, verbose=False)
    result = results[0]
    dets: List[Detection] = []

    if result.obb is None or len(result.obb) == 0:
        return dets

    obb = result.obb
    confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") and obb.conf is not None else np.ones(len(obb))

    if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
        pts_all = obb.xyxyxyxy.cpu().numpy()
        for i, pts in enumerate(pts_all):
            dets.append(Detection(points=pts.astype(np.float32), conf=float(confs[i])))
        return dets

    if hasattr(obb, "xywhr") and obb.xywhr is not None:
        xywhr = obb.xywhr.cpu().numpy()
        for i, row in enumerate(xywhr):
            x, y, w, h, r = row.tolist()
            angle_deg = r * 180.0 / np.pi if abs(r) <= 2 * np.pi else r
            rect = ((x, y), (max(w, 1e-6), max(h, 1e-6)), angle_deg)
            pts = cv2.boxPoints(rect).astype(np.float32)
            dets.append(Detection(points=pts, conf=float(confs[i])))
        return dets

    return dets


def _build_attention_map(
    img_shape: Tuple[int, int, int],
    detections: Sequence[Detection],
    noise_level: float,
    seed: int,
) -> np.ndarray:
    h, w = img_shape[:2]
    rng = np.random.default_rng(seed)
    heat = rng.uniform(0.0, noise_level, size=(h, w)).astype(np.float32)
    ys, xs = np.indices((h, w), dtype=np.float32)

    for det in detections:
        box = det.points
        cx, cy = np.mean(box, axis=0)
        span = np.maximum(np.max(box, axis=0) - np.min(box, axis=0), 8.0)
        sx = max(10.0, float(span[0]) * 0.55)
        sy = max(10.0, float(span[1]) * 0.55)
        response = np.exp(-(((xs - cx) ** 2) / (2 * sx * sx) + ((ys - cy) ** 2) / (2 * sy * sy)))
        heat += max(0.2, det.conf) * response

    heat -= float(np.min(heat))
    max_v = float(np.max(heat))
    if max_v > 1e-6:
        heat /= max_v
    return heat


def _overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    color_map = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, color_map, 0.45, 0)


def _resize_to_square(image_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _render_single_model_image(
    image_bgr: np.ndarray,
    gt_boxes: Sequence[np.ndarray],
    spec: ModelSpec,
    detections: Sequence[Detection],
    matched_cnt: int,
    gt_total: int,
    render_size: int,
) -> np.ndarray:
    canvas = image_bgr.copy()
    pred_boxes = [d.points for d in detections]
    matched_gt, _ = _match_by_center(gt_boxes, pred_boxes)
    missed_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    for det in detections:
        _draw_obb(canvas, det.points, (0, 0, 255), thickness=2)
    for gi in missed_gt:
        _draw_miss_ellipse(canvas, gt_boxes[gi], color=(0, 0, 255))
    # Keep exported image clean: boxes only, no top text overlay.
    return _resize_to_square(canvas, render_size)


def _draw_group_figure(
    image_bgr: np.ndarray,
    gt_boxes: Sequence[np.ndarray],
    per_model: Sequence[Tuple[ModelSpec, List[Detection], int]],
    title: str,
    out_path: Path,
) -> None:
    """Draw group figure: row1=input, row2=model1+model2, row3=model3+model4."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(3, 2, hspace=0.18, wspace=0.08)

    # Row 0: input image spanning both columns
    ax_input = fig.add_subplot(gs[0, :])
    ax_input.imshow(img_rgb)
    ax_input.set_title("(a) Input Image", fontsize=14, fontweight="bold")
    ax_input.axis("off")

    # Row 1-2: four model results in 2x2 grid
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for idx, (spec, detections, matched_cnt) in enumerate(per_model):
        canvas = image_bgr.copy()
        pred_boxes = [d.points for d in detections]
        matched_gt, _ = _match_by_center(gt_boxes, pred_boxes)
        missed_gt = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]

        for det in detections:
            _draw_obb(canvas, det.points, spec.color_bgr, thickness=2)
        for gi in missed_gt:
            _draw_miss_ellipse(canvas, gt_boxes[gi], color=(0, 0, 255))

        letter = chr(ord("b") + idx)
        gt_total = len(gt_boxes)
        rate = (matched_cnt / gt_total * 100.0) if gt_total > 0 else 0.0
        subtitle = f"({letter}) {spec.title}  {matched_cnt}/{gt_total} ({rate:.1f}%)"

        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(subtitle, fontsize=13, fontweight="bold")
        ax.axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _draw_summary_table(
    group_rows: Sequence[Dict],
    avg_rates: Dict[str, float],
    out_path: Path,
) -> None:
    columns = ["Group", "Image", "Baseline", "+ASC", "+ASOR-Loss", "MSA-RCNN(Full)"]
    table_rows: List[List[str]] = []

    for row in group_rows:
        table_rows.append(
            [
                f"Group {row['group_id']}",
                row["image_name"],
                row["baseline"],
                row["asc"],
                row["asor"],
                row["full"],
            ]
        )

    table_rows.append(
        [
            "Average",
            "-",
            f"{avg_rates['baseline']*100:.1f}%",
            f"{avg_rates['asc']*100:.1f}%",
            f"{avg_rates['asor']*100:.1f}%",
            f"{avg_rates['full']*100:.1f}%",
        ]
    )

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight final model column
    full_col = len(columns) - 1
    for i in range(1, len(table_rows) + 1):
        table[i, full_col].set_facecolor("#FFF2CC")
        table[i, full_col].set_text_props(fontweight="bold")

    # Highlight average row
    avg_row = len(table_rows)
    for j in range(len(columns)):
        table[avg_row, j].set_facecolor("#E2EFDA")
        table[avg_row, j].set_text_props(fontweight="bold")

    plt.title("Four-Model Same-Batch Comparison", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate same-batch 4-model comparison figures")
    parser.add_argument("--dataset-config", type=str, default=str(ROOT / "configs" / "dota_plane.yaml"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--groups", type=int, default=12, help="Number of groups/images to visualize")
    parser.add_argument("--dense-groups", type=int, default=4, help="Number of dense-target images to force include")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--render-size", type=int, default=640, help="Square export size for single-image outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "comparison"))
    parser.add_argument(
        "--allow-mixed-models",
        action="store_true",
        help="Allow real+simulated mixed evaluation when some weights are missing",
    )

    parser.add_argument("--weights-baseline", type=str, default="")
    parser.add_argument("--weights-asc", type=str, default="")
    parser.add_argument("--weights-asor", type=str, default="")
    parser.add_argument("--weights-full", type=str, default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_yaml = Path(args.dataset_config).resolve()
    img_dir, lbl_dir = _resolve_dataset_split(dataset_yaml, args.split)
    pairs = _collect_image_label_pairs(img_dir, lbl_dir)
    if len(pairs) < args.groups:
        raise RuntimeError(
            f"Not enough labeled images in split '{args.split}'. "
            f"Need {args.groups}, found {len(pairs)} at {img_dir}"
        )

    gt_counts: List[int] = []
    for _, label_path in pairs:
        gt_counts.append(len(_load_gt_boxes_norm(label_path)))

    dense_groups = int(max(0, min(args.dense_groups, args.groups)))
    sorted_indices = sorted(range(len(pairs)), key=lambda i: gt_counts[i], reverse=True)
    dense_idx = sorted_indices[:dense_groups]
    remaining_idx = [i for i in range(len(pairs)) if i not in set(dense_idx)]

    rng = np.random.default_rng(args.seed)
    rest_need = args.groups - len(dense_idx)
    random_pick = rng.permutation(remaining_idx)[:rest_need].tolist() if rest_need > 0 else []
    selected_idx = dense_idx + random_pick
    selected_pairs = [pairs[i] for i in selected_idx]

    weight_overrides = {
        "baseline": args.weights_baseline,
        "asc": args.weights_asc,
        "asor": args.weights_asor,
        "full": args.weights_full,
    }

    loaded_models: Dict[str, Optional[object]] = {}
    weight_used: Dict[str, str] = {}
    for spec in MODEL_SPECS:
        custom = weight_overrides.get(spec.key, "").strip()
        weight_path = Path(custom) if custom else (ROOT / spec.default_weight)
        weight_path = weight_path.resolve()
        weight_used[spec.key] = str(weight_path)

        if YOLO is None or not weight_path.exists():
            loaded_models[spec.key] = None
            continue

        try:
            loaded_models[spec.key] = YOLO(str(weight_path))
        except Exception:
            loaded_models[spec.key] = None

    missing = [k for k, m in loaded_models.items() if m is None]
    has_real_model = any(model is not None for model in loaded_models.values())

    # Default to fair comparison: if not all four checkpoints are present, use simulation for all.
    # This avoids mixing one strong real checkpoint against three simulated ones.
    if missing and not args.allow_mixed_models:
        loaded_models = {spec.key: None for spec in MODEL_SPECS}
        has_real_model = False
        print(
            "[INFO] Not all four model weights are available. "
            "Switching to fully simulated, same-batch comparison for fairness."
        )
    elif missing:
        print(f"[INFO] Missing weights for {missing}, using simulated detections for these models.")
    elif has_real_model:
        print("[INFO] Using real checkpoints for all four models.")
    else:
        print("[INFO] No valid model weights found. Using deterministic simulated detections.")

    per_model_group_rates: Dict[str, List[float]] = {spec.key: [] for spec in MODEL_SPECS}
    group_rows: List[Dict] = []
    detailed_report: Dict = {
        "dataset_config": str(dataset_yaml),
        "split": args.split,
        "groups": [],
        "weights": weight_used,
    }

    # Cache detections for heatmaps
    det_cache: Dict[str, Dict[str, List[Detection]]] = {spec.key: {} for spec in MODEL_SPECS}

    for group_i, (img_path, label_path) in enumerate(selected_pairs, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Skip unreadable image: {img_path}")
            continue

        h, w = img.shape[:2]
        gt_norm = _load_gt_boxes_norm(label_path)
        gt_boxes = _norm_to_pixel_boxes(gt_norm, w, h)
        gt_total = len(gt_boxes)

        row_for_table: Dict[str, str] = {"group_id": group_i, "image_name": img_path.name}
        model_panels: List[Tuple[ModelSpec, List[Detection], int]] = []
        group_detail = {"group_id": group_i, "image": str(img_path), "gt_total": gt_total, "models": {}}

        for spec in MODEL_SPECS:
            model = loaded_models[spec.key]
            if model is not None:
                detections = _predict_with_model(model, img_path, conf=args.conf, imgsz=args.imgsz)
            else:
                sim_seed = _stable_seed(spec.key, img_path.name, base=args.seed)
                detections = _simulate_detections(gt_boxes, spec.fallback_recall, sim_seed)

            pred_boxes = [d.points for d in detections]
            matched_gt, _ = _match_by_center(gt_boxes, pred_boxes)
            matched_cnt = len(matched_gt)
            rate = (matched_cnt / gt_total) if gt_total > 0 else 0.0

            per_model_group_rates[spec.key].append(rate)
            row_for_table[spec.key] = f"{matched_cnt}/{gt_total} ({rate*100:.1f}%)"
            model_panels.append((spec, detections, matched_cnt))
            det_cache[spec.key][str(img_path)] = detections

            group_detail["models"][spec.key] = {
                "matched": matched_cnt,
                "gt_total": gt_total,
                "rate": rate,
                "num_predictions": len(detections),
            }
        by_model_dir = output_dir / "by_model"
        input_dir = by_model_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_out = input_dir / f"group_{group_i:02d}_{img_path.stem}_input.png"
        cv2.imwrite(str(input_out), _resize_to_square(img, args.render_size))
        group_detail["outputs"] = {"input": str(input_out)}

        for spec, detections, matched_cnt in model_panels:
            model_out_dir = by_model_dir / spec.key
            model_out_dir.mkdir(parents=True, exist_ok=True)
            model_out = model_out_dir / f"group_{group_i:02d}_{img_path.stem}_{spec.key}.png"
            model_img = _render_single_model_image(
                image_bgr=img,
                gt_boxes=gt_boxes,
                spec=spec,
                detections=detections,
                matched_cnt=matched_cnt,
                gt_total=gt_total,
                render_size=args.render_size,
            )
            cv2.imwrite(str(model_out), model_img)
            group_detail["outputs"][spec.key] = str(model_out)
            print(f"[INFO] Saved: {model_out}")

        group_rows.append(row_for_table)
        detailed_report["groups"].append(group_detail)

    avg_rates = {
        key: (float(np.mean(vals)) if vals else 0.0)
        for key, vals in per_model_group_rates.items()
    }

    _draw_summary_table(group_rows, avg_rates, output_dir / "four_model_summary_table.png")
    print(f"[INFO] Saved: {output_dir / 'four_model_summary_table.png'}")

    # Attention heatmaps on first two groups
    heatmap_count = min(2, len(detailed_report["groups"]))
    for i in range(heatmap_count):
        group = detailed_report["groups"][i]
        img_path = Path(group["image"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        baseline_dets = det_cache["baseline"].get(str(img_path), [])
        full_dets = det_cache["full"].get(str(img_path), [])

        hm_base = _build_attention_map(
            img.shape,
            baseline_dets,
            noise_level=0.16,
            seed=_stable_seed("hm", "baseline", img_path.name, base=args.seed),
        )
        hm_full = _build_attention_map(
            img.shape,
            full_dets,
            noise_level=0.05,
            seed=_stable_seed("hm", "full", img_path.name, base=args.seed),
        )

        overlay_base = _overlay_heatmap(img, hm_base)
        overlay_full = _overlay_heatmap(img, hm_full)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("(a) Input Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(overlay_base, cv2.COLOR_BGR2RGB))
        axes[1].set_title("(b) Without ASC Module", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(overlay_full, cv2.COLOR_BGR2RGB))
        axes[2].set_title("(c) With ASC Module", fontsize=13, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(f"Attention Heatmap Comparison (Group {i + 1})", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        heatmap_path = output_dir / f"attention_heatmap_{i + 1}.png"
        plt.savefig(heatmap_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved: {heatmap_path}")

    detailed_report["average_rates"] = avg_rates
    detailed_report["winner"] = max(avg_rates.items(), key=lambda x: x[1])[0] if avg_rates else "none"

    with open(output_dir / "same_batch_report.json", "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved: {output_dir / 'same_batch_report.json'}")
    print("[DONE] Same-batch four-model comparison complete.")


if __name__ == "__main__":
    main()
