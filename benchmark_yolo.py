"""
Benchmark производительности и точности YOLO (Ultralytics) на CPU и GPU.

Полный аналог EAST-бенчмарка:
- время инференса
- RAM / VRAM
- throughput
- F1@0.5, F1@0.5:0.95 (COCO)
"""

import time
import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from ultralytics import YOLO


# =============================================================================
# METRICS (ТОЧНО КАК В EAST)
# =============================================================================

def box_iou(box1, box2) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    inter = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matches = []
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou = box_iou(p, g)
            if iou >= iou_threshold:
                matches.append((iou, i, j))

    matches.sort(reverse=True)
    mp, mg = set(), set()

    for _, i, j in matches:
        if i not in mp and j not in mg:
            mp.add(i)
            mg.add(j)

    tp = len(mp)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(mg)
    return tp, fp, fn


def compute_f1(tp, fp, fn):
    if tp == 0:
        return 0.0, 0.0, 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f1, p, r


def evaluate_dataset(preds, gts, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)

    totals = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    image_ids = set(preds.keys()) | set(gts.keys())
    for img_id in image_ids:
        pb = preds.get(img_id, [])
        gb = gts.get(img_id, [])
        for t in thresholds:
            tp, fp, fn = match_boxes(pb, gb, t)
            totals[t]["tp"] += tp
            totals[t]["fp"] += fp
            totals[t]["fn"] += fn

    results, f1s = {}, []
    for t in thresholds:
        tp = totals[t]["tp"]
        fp = totals[t]["fp"]
        fn = totals[t]["fn"]
        f1, p, r = compute_f1(tp, fp, fn)
        results[f"f1@{t:.2f}"] = f1
        results[f"precision@{t:.2f}"] = p
        results[f"recall@{t:.2f}"] = r
        results[f"tp@{t:.2f}"] = tp
        results[f"fp@{t:.2f}"] = fp
        results[f"fn@{t:.2f}"] = fn
        f1s.append(f1)

    results["f1@0.5"] = results["f1@0.50"]
    results["f1@0.5:0.95"] = float(np.mean(f1s))
    results["num_predictions"] = sum(len(v) for v in preds.values())
    results["num_ground_truths"] = sum(len(v) for v in gts.values())
    return results


# =============================================================================
# DATASETS (КАК ТЫ ПРОСИЛ)
# =============================================================================

DATASETS = [
    {
        "name": "Archives020525",
        "folder": r"C:\shared\data0205\data02065\Archives020525\test_images",
        "annotations": r"C:\shared\data0205\data02065\Archives020525\test.json",
    },
    {
        "name": "ICDAR2015",
        "folder": r"C:\shared\data0205\data02065\ICDAR2015\test_images",
        "annotations": r"C:\shared\data0205\data02065\ICDAR2015\test.json",
    },
    {
        "name": "school_notebooks_RU",
        "folder": r"C:\shared\data0205\data02065\school_notebooks_RU\test_images",
        "annotations": r"C:\shared\data0205\data02065\school_notebooks_RU\test.json",
    },
    {
        "name": "IAM",
        "folder": r"C:\shared\data0205\data02065\IAM\test_images",
        "annotations": r"C:\shared\data0205\data02065\IAM\test.json",
    },
    {
        "name": "TotalText",
        "folder": r"C:\shared\data0205\data02065\TotalText\test_images",
        "annotations": r"C:\shared\data0205\data02065\TotalText\test.json",
    },
]

CONFIG = {
    "model_path": r"C:\Users\USER\manuscript-ocr\example\best.pt",
    "imgsz": 640,
    "conf": 0.25,
    "warmup": 3,
    "cpu_only": False,
    "gpu_only": True,
    "output_dir": "benchmark_results_yolo",
}


# =============================================================================
# UTILS
# =============================================================================

def get_image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = []
    for e in exts:
        files += Path(folder).glob(f"*{e}")
        files += Path(folder).glob(f"*{e.upper()}")
    # Normalize to strings and remove duplicates while preserving order
    str_paths = list(map(str, files))
    unique = list(dict.fromkeys(str_paths))
    return sorted(unique)


def load_ground_truth(coco_json):
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt = {}

    for ann in coco["annotations"]:
        fname = id2name.get(ann["image_id"])
        if not fname or "segmentation" not in ann:
            continue
        segs = ann["segmentation"]
        segs = segs if isinstance(segs[0], list) else [segs]
        for s in segs:
            pts = np.array(s).reshape(-1, 2)
            box = (
                float(pts[:, 0].min()),
                float(pts[:, 1].min()),
                float(pts[:, 0].max()),
                float(pts[:, 1].max()),
            )
            gt.setdefault(fname, []).append(box)
    return gt


def memory_usage():
    import psutil
    ram = psutil.Process().memory_info().rss / 1024**2
    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    return ram, vram


# =============================================================================
# BENCHMARK YOLO
# =============================================================================

def benchmark_device(images, device, collect_predictions=False):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(CONFIG["model_path"])
    model.to(device)

    ram0, vram0 = memory_usage()

    # warmup
    for img in images[:CONFIG["warmup"]]:
        model.predict(img, imgsz=CONFIG["imgsz"], conf=CONFIG["conf"],
                      device=device, verbose=False)

    if device == "cuda":
        torch.cuda.synchronize()

    times, preds = [], {}

    for img in images:
        t0 = time.time()
        res = model.predict(
            img, imgsz=CONFIG["imgsz"], conf=CONFIG["conf"],
            device=device, verbose=False
        )
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)

        if collect_predictions:
            fname = Path(img).name
            boxes = []
            for r in res:
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    boxes.append(tuple(xyxy[:4]))
            preds[fname] = boxes

    ram1, vram1 = memory_usage()

    stats = {
        "device": device,
        "num_images": len(images),
        "mean_time_ms": float(np.mean(times) * 1000),
        "median_time_ms": float(np.median(times) * 1000),
        "throughput_fps": float(len(images) / np.sum(times)),
        "ram_after_load_mb": ram0,
        "ram_peak_mb": ram1,
        "ram_delta_mb": ram1 - ram0,
    }

    if device == "cuda":
        stats.update({
            "gpu_after_load_mb": vram0,
            "gpu_peak_mb": vram1,
            "gpu_delta_mb": vram1 - vram0,
        })

    if collect_predictions:
        stats["predictions"] = preds

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        print(f"\n{'#'*80}\nDATASET: {ds['name']}\n{'#'*80}")

        images = get_image_files(ds["folder"])
        gts = load_ground_truth(ds["annotations"])

        cpu = None
        gpu = None

        if not CONFIG["gpu_only"]:
            cpu = benchmark_device(images, "cpu", collect_predictions=True)
            cpu["accuracy_metrics"] = evaluate_dataset(cpu["predictions"], gts)

        if torch.cuda.is_available() and not CONFIG["cpu_only"]:
            gpu = benchmark_device(images, "cuda", collect_predictions=True)
            gpu["accuracy_metrics"] = evaluate_dataset(gpu["predictions"], gts)

        out = {
            "dataset": ds["name"],
            "cpu": cpu,
            "gpu": gpu,
        }

        with open(Path(CONFIG["output_dir"]) / f"{ds['name']}_yolo.json",
                  "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print("\nALL YOLO BENCHMARKS COMPLETED")


if __name__ == "__main__":
    main()
