#!/usr/bin/env python3
"""
Run four-model OBB inference on a folder of images.

Outputs:
- by_model/input/*.png
- by_model/baseline/*.png
- by_model/asc/*.png
- by_model/asor/*.png
- by_model/full/*.png

Notes:
- Bounding boxes are always red.
- No top text overlay is added to output images.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    weight: Path


MODEL_SPECS: Sequence[ModelSpec] = (
    ModelSpec("baseline", ROOT / "runs/plane_baseline/weights/best.pt"),
    ModelSpec("asc", ROOT / "runs/plane_asc/weights/best.pt"),
    ModelSpec("asor", ROOT / "runs/plane_asor/weights/best.pt"),
    ModelSpec("full", ROOT / "runs/plane_full/weights/best.pt"),
)


def _collect_images(img_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(sorted(img_dir.glob(f"*{ext}")))
    return paths


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


def _draw_obb(img: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    poly = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, thickness)


def _strip_top_band(image_bgr: np.ndarray, strip_h: int) -> np.ndarray:
    if strip_h <= 0:
        return image_bgr
    h, w = image_bgr.shape[:2]
    strip_h = min(strip_h, h)
    if strip_h == h:
        return image_bgr
    ref_row = image_bgr[min(strip_h, h - 1) : min(strip_h + 1, h), :, :]
    if ref_row.size == 0:
        return image_bgr
    fill = np.tile(ref_row, (strip_h, 1, 1))
    out = image_bgr.copy()
    out[:strip_h, :, :] = fill
    return out


def _convert_blue_to_red(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Typical bright-blue annotation range
    mask = cv2.inRange(hsv, (90, 80, 80), (140, 255, 255))
    out = image_bgr.copy()
    out[mask > 0] = (0, 0, 255)
    return out


def _predict_points(model, image_path: Path, conf: float, imgsz: int) -> List[np.ndarray]:
    results = model.predict(str(image_path), conf=conf, iou=0.45, imgsz=imgsz, verbose=False)
    r = results[0]
    boxes: List[np.ndarray] = []
    if r.obb is None or len(r.obb) == 0:
        return boxes

    obb = r.obb
    if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
        for pts in obb.xyxyxyxy.cpu().numpy():
            boxes.append(pts.astype(np.float32))
        return boxes

    if hasattr(obb, "xywhr") and obb.xywhr is not None:
        for row in obb.xywhr.cpu().numpy():
            x, y, w, h, rads = row.tolist()
            angle = rads * 180.0 / np.pi if abs(rads) <= 2 * np.pi else rads
            rect = ((x, y), (max(w, 1e-6), max(h, 1e-6)), angle)
            boxes.append(cv2.boxPoints(rect).astype(np.float32))
        return boxes
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four-model inference for a folder of images")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder containing new images")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "comparison" / "Att"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--render-size", type=int, default=640)
    parser.add_argument("--strip-top", type=int, default=0, help="Remove top text band in pixels")
    parser.add_argument(
        "--convert-blue-to-red",
        action="store_true",
        help="Convert existing blue annotations in source image to red",
    )
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("ultralytics is not available in this environment")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    images = _collect_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    models: Dict[str, object] = {}
    for spec in MODEL_SPECS:
        if not spec.weight.exists():
            raise FileNotFoundError(f"Missing checkpoint: {spec.weight}")
        models[spec.key] = YOLO(str(spec.weight))

    out_root = Path(args.output_dir).resolve()
    (out_root / "by_model" / "input").mkdir(parents=True, exist_ok=True)
    for key in ("baseline", "asc", "asor", "full"):
        (out_root / "by_model" / key).mkdir(parents=True, exist_ok=True)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] unreadable image: {img_path}")
            continue

        base_img = img.copy()
        if args.strip_top > 0:
            base_img = _strip_top_band(base_img, args.strip_top)
        if args.convert_blue_to_red:
            base_img = _convert_blue_to_red(base_img)

        out_input = out_root / "by_model" / "input" / f"{img_path.stem}_input.png"
        cv2.imwrite(str(out_input), _resize_to_square(base_img, args.render_size))

        for key in ("baseline", "asc", "asor", "full"):
            canvas = base_img.copy()
            pts_list = _predict_points(models[key], img_path, conf=args.conf, imgsz=args.imgsz)
            for pts in pts_list:
                _draw_obb(canvas, pts, color=(0, 0, 255), thickness=2)
            out_path = out_root / "by_model" / key / f"{img_path.stem}_{key}.png"
            cv2.imwrite(str(out_path), _resize_to_square(canvas, args.render_size))
            print(f"[OK] {key}: {out_path}")

    print(f"[DONE] Four-model inference complete. Output: {out_root}")


if __name__ == "__main__":
    main()

