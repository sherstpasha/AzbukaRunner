"""
Бенчмарк производительности и точности EAST детектора на CPU и GPU.

Измеряет:
- Среднее время инференса на изображение
- Использование памяти (RAM для CPU, VRAM для GPU)
- Throughput (изображений в секунду)
- Точность детекции (F1@0.5, F1@0.5:0.95) при наличии аннотаций

Конфигурация:
    Измените список DATASETS ниже для настройки датасетов
"""

import sys
import time
import gc
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import cv2

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors import EAST


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def box_iou(box1: tuple, box2: tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Parameters
    ----------
    box1, box2 : tuple
        Boxes in format (x_min, y_min, x_max, y_max)
        
    Returns
    -------
    float
        IoU value in range [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Compute intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_boxes(
    pred_boxes: List[tuple],
    gt_boxes: List[tuple],
    iou_threshold: float = 0.5,
) -> tuple:
    """
    Match predicted boxes to ground truth boxes using IoU threshold.
    
    Returns
    -------
    tuple
        (tp, fp, fn) - true positives, false positives, false negatives
    """
    # Безопасность: если не список, делаем пустой список
    if not isinstance(pred_boxes, list):
        pred_boxes = []
    if not isinstance(gt_boxes, list):
        gt_boxes = []
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)
    
    # Compute IoU matrix (safe casts to int sizes)
    try:
        n_pred = int(len(pred_boxes))
        n_gt = int(len(gt_boxes))
    except Exception:
        # Fallback: treat as empty
        n_pred = 0
        n_gt = 0

    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            try:
                iou_matrix[i, j] = box_iou(pred, gt)
            except Exception as e:
                print(f"[ERROR] box_iou failed for pred[{i}] or gt[{j}]: {e}")
                iou_matrix[i, j] = 0.0
    
    # Greedy matching
    matches = []
    for i in range(n_pred):
        for j in range(n_gt):
            try:
                if iou_matrix[i, j] >= iou_threshold:
                    matches.append((float(iou_matrix[i, j]), int(i), int(j)))
            except Exception as e:
                print(f"[ERROR] IoU matrix access failed at ({i},{j}): {e}")
                continue
    
    matches.sort(reverse=True)
    
    matched_pred = set()
    matched_gt = set()
    
    for iou_val, i, j in matches:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
    
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn


def compute_f1_score(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple:
    """Compute F1 score, precision, and recall."""
    if true_positives == 0:
        return 0.0, 0.0, 0.0
    
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    
    if precision + recall == 0:
        return 0.0, precision, recall
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def evaluate_dataset(
    predictions: Dict[str, List[tuple]],
    ground_truths: Dict[str, List[tuple]],
    iou_thresholds: Optional[List[float]] = None,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate object detection on a dataset with multiple IoU thresholds.
    
    Parameters
    ----------
    predictions : dict
        Dictionary mapping image IDs to lists of predicted boxes.
    ground_truths : dict
        Dictionary mapping image IDs to lists of ground truth boxes.
    iou_thresholds : list of float, optional
        IoU thresholds to evaluate at. Default: [0.5, 0.55, ..., 0.95].
    verbose : bool, default=True
        If True, show progress.
    n_jobs : int, optional
        Number of parallel workers (not used in this simple version).
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    
    # Initialize accumulators
    total_tp = {th: 0 for th in iou_thresholds}
    total_fp = {th: 0 for th in iou_thresholds}
    total_fn = {th: 0 for th in iou_thresholds}
    
    # Get all image IDs
    all_image_ids = list(set(list(predictions.keys()) + list(ground_truths.keys())))
    
    # Evaluate each image
    for image_id in all_image_ids:
        pred_boxes = predictions.get(image_id, []) or []
        gt_boxes = ground_truths.get(image_id, []) or []
        
        for threshold in iou_thresholds:
            tp, fp, fn = match_boxes(pred_boxes, gt_boxes, iou_threshold=threshold)
            total_tp[threshold] += tp
            total_fp[threshold] += fp
            total_fn[threshold] += fn
    
    # Compute metrics for each threshold
    results = {}
    f1_scores = []
    
    for threshold in iou_thresholds:
        tp = total_tp[threshold]
        fp = total_fp[threshold]
        fn = total_fn[threshold]
        
        f1, precision, recall = compute_f1_score(tp, fp, fn)
        
        results[f"f1@{threshold:.2f}"] = f1
        results[f"precision@{threshold:.2f}"] = precision
        results[f"recall@{threshold:.2f}"] = recall
        results[f"tp@{threshold:.2f}"] = tp
        results[f"fp@{threshold:.2f}"] = fp
        results[f"fn@{threshold:.2f}"] = fn
        
        f1_scores.append(f1)
    
    # Add summary metrics
    results["f1@0.5"] = results.get("f1@0.50", 0.0)
    results["precision@0.5"] = results.get("precision@0.50", 0.0)
    results["recall@0.5"] = results.get("recall@0.50", 0.0)
    results["f1@0.5:0.95"] = float(np.mean(f1_scores))
    results["num_images"] = len(all_image_ids)
    results["num_predictions"] = sum(len(boxes) for boxes in predictions.values())
    results["num_ground_truths"] = sum(len(boxes) for boxes in ground_truths.values())
    
    return results


# =============================================================================
# КОНФИГУРАЦИЯ ДАТАСЕТОВ
# =============================================================================

# Список датасетов для тестирования
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

# Общие параметры
CONFIG = {
    "target_size": 1280,       # Размер входного изображения
    "score_thresh": 0.6,       # Порог уверенности
    "warmup_runs": 3,          # Количество прогревочных запусков
    "cpu_only": True,         # Только CPU
    "gpu_only": False,         # Только GPU
    "output_dir": "benchmark_results",  # Папка для сохранения результатов
}


def get_image_files(folder: str) -> List[str]:
    """
    Находит все изображения в папке.

    Returns
    -------
    list
        Список путей к изображениям
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = set()  # Используем set для избежания дубликатов

    for ext in extensions:
        # Ищем с учетом регистра (case-insensitive)
        image_files.update(folder_path.glob(f"*{ext}"))
        image_files.update(folder_path.glob(f"*{ext.upper()}"))

    return [str(f) for f in sorted(image_files)]


def load_ground_truth(annotation_file: str, images_folder: str) -> Dict[str, List]:
    """
    Загружает ground truth аннотации из COCO формата.

    Parameters
    ----------
    annotation_file : str
        Путь к JSON файлу с аннотациями в COCO формате
    images_folder : str
        Путь к папке с изображениями

    Returns
    -------
    dict
        Словарь {filename: list of boxes}, где boxes в формате (x_min, y_min, x_max, y_max)
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Создаем маппинг image_id -> filename
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Собираем аннотации по filename
    ground_truths = {}

    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in images_info:
            continue

        filename = images_info[image_id]["file_name"]

        # Извлекаем segmentation и конвертируем в bbox
        seg = ann.get("segmentation")
        if not seg or len(seg) == 0:
            continue

        # segmentation может быть списком полигонов
        seg_parts = seg if isinstance(seg[0], list) else [seg]

        for seg_poly in seg_parts:
            if len(seg_poly) < 8:  # Минимум 4 точки (8 координат)
                continue

            # Конвертируем полигон в bbox
            pts = np.array(seg_poly, dtype=np.float32).reshape(-1, 2)
            x_min = float(np.min(pts[:, 0]))
            y_min = float(np.min(pts[:, 1]))
            x_max = float(np.max(pts[:, 0]))
            y_max = float(np.max(pts[:, 1]))

            box = (x_min, y_min, x_max, y_max)

            if filename not in ground_truths:
                ground_truths[filename] = []
            ground_truths[filename].append(box)

    return ground_truths


def get_memory_usage() -> Dict[str, float]:
    """
    Получает текущее использование памяти.

    Returns
    -------
    dict
        {"ram_mb": float, "gpu_mb": float or None}
    """
    import psutil

    # RAM
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    # GPU VRAM
    gpu_mb = None
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024

    return {"ram_mb": ram_mb, "gpu_mb": gpu_mb}


def benchmark_device(
    image_files: List[str],
    device: str,
    target_size: int = 1280,
    warmup_runs: int = 3,
    score_thresh: float = 0.6,
    collect_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Бенчмарк EAST на указанном устройстве.

    Parameters
    ----------
    image_files : list
        Список путей к изображениям
    device : str
        Устройство ("cpu" или "cuda")
    target_size : int
        Размер входного изображения
    warmup_runs : int
        Количество прогревочных запусков
    score_thresh : float
        Порог уверенности
    collect_predictions : bool
        Собирать предсказания для оценки точности

    Returns
    -------
    dict
        Статистика бенчмарка
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device.upper()}")
    print(f"{'='*60}")

    # Очищаем память
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Создаём детектор
    print(f"Initializing EAST detector on {device}...")
    detector = EAST(device=device)

    # Измеряем память после загрузки модели
    mem_after_load = get_memory_usage()
    print(f"Memory after model load:")
    print(f"  RAM: {mem_after_load['ram_mb']:.2f} MB")
    if mem_after_load["gpu_mb"] is not None:
        print(f"  GPU VRAM: {mem_after_load['gpu_mb']:.2f} MB")

    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    warmup_images = image_files[: min(warmup_runs, len(image_files))]
    for img_path in warmup_images:
        _ = detector.predict(img_path)

    if device == "cuda":
        torch.cuda.synchronize()

    # Бенчмарк
    print(f"\nRunning benchmark on {len(image_files)} images...")

    inference_times = []
    detection_counts = []
    peak_memory = get_memory_usage()
    predictions = {} if collect_predictions else None

    for i, img_path in enumerate(image_files, 1):
        # Замеряем время
        start_time = time.time()

        result = detector.predict(img_path)

        if device == "cuda":
            torch.cuda.synchronize()

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Подсчитываем детекции (по lines.words)
        num_detections = 0
        for block in result["page"].blocks:
            for line in getattr(block, "lines", []):
                num_detections += len(getattr(line, "words", []))
        detection_counts.append(num_detections)

        # Собираем предсказания для оценки точности
        if collect_predictions:
            filename = Path(img_path).name
            boxes = []

            # Дополнительная отладка: смотрим содержимое блоков
            if i <= 2:
                for b_idx, block in enumerate(result["page"].blocks[:2]):
                    print(f"    [DEBUG] Block {b_idx}: type={type(block)}")
                    print(f"      Block attributes: {dir(block)}")
                    if hasattr(block, "words"):
                        print(f"      Block.words: {block.words}")
                    if hasattr(block, "lines"):
                        print(f"      Block.lines: {block.lines}")
                    if hasattr(block, "bbox"):
                        print(f"      Block.bbox: {block.bbox}")
                    if hasattr(block, "polygon"):
                        print(f"      Block.polygon: {block.polygon}")

            # Новый способ: собираем боксы из block.lines[*].words[*]
            for block in result["page"].blocks:
                for line in getattr(block, "lines", []):
                    for word in getattr(line, "words", []):
                        box = None
                        if hasattr(word, "polygon") and word.polygon and len(word.polygon) > 0:
                            xs = [pt[0] for pt in word.polygon]
                            ys = [pt[1] for pt in word.polygon]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            box = (x_min, y_min, x_max, y_max)
                        elif hasattr(word, "bbox") and word.bbox:
                            box = tuple(word.bbox)
                        elif hasattr(word, "geometry") and word.geometry:
                            if hasattr(word.geometry, "bbox"):
                                box = tuple(word.geometry.bbox)
                        if box:
                            boxes.append(box)
            predictions[filename] = boxes
            
            # Отладка для первых нескольких изображений
            if i <= 3:
                print(f"  [DEBUG] Image {filename}: {len(boxes)} boxes collected (from block.lines.words)")

        # Отслеживаем пиковую память
        current_mem = get_memory_usage()
        if current_mem["ram_mb"] > peak_memory["ram_mb"]:
            peak_memory["ram_mb"] = current_mem["ram_mb"]
        if current_mem["gpu_mb"] is not None and peak_memory["gpu_mb"] is not None:
            if current_mem["gpu_mb"] > peak_memory["gpu_mb"]:
                peak_memory["gpu_mb"] = current_mem["gpu_mb"]

        # Прогресс
        if i % 10 == 0 or i == len(image_files):
            avg_time = (
                np.mean(inference_times[-10:])
                if len(inference_times) >= 10
                else np.mean(inference_times)
            )
            print(
                f"  [{i}/{len(image_files)}] Avg time (last 10): {avg_time*1000:.2f} ms"
            )

    # Статистика
    inference_times = np.array(inference_times)
    detection_counts = np.array(detection_counts)

    print(f"\n[INFO] Detection statistics:")
    print(f"  Total detections across all images: {int(np.sum(detection_counts))}")
    print(f"  Average detections per image: {float(np.mean(detection_counts)):.1f}")
    print(f"  Images with 0 detections: {int(np.sum(detection_counts == 0))}/{len(detection_counts)}")

    stats = {
        "device": device,
        "num_images": len(image_files),
        "target_size": target_size,
        # Время
        "mean_time_ms": float(np.mean(inference_times) * 1000),
        "median_time_ms": float(np.median(inference_times) * 1000),
        "std_time_ms": float(np.std(inference_times) * 1000),
        "min_time_ms": float(np.min(inference_times) * 1000),
        "max_time_ms": float(np.max(inference_times) * 1000),
        "total_time_s": float(np.sum(inference_times)),
        "throughput_fps": float(len(image_files) / np.sum(inference_times)),
        # Детекции
        "mean_detections": float(np.mean(detection_counts)),
        "total_detections": int(np.sum(detection_counts)),
        # Память
        "ram_after_load_mb": mem_after_load["ram_mb"],
        "ram_peak_mb": peak_memory["ram_mb"],
        "ram_delta_mb": peak_memory["ram_mb"] - mem_after_load["ram_mb"],
    }

    if device == "cuda":
        stats["gpu_after_load_mb"] = mem_after_load["gpu_mb"]
        stats["gpu_peak_mb"] = peak_memory["gpu_mb"]
        stats["gpu_delta_mb"] = peak_memory["gpu_mb"] - mem_after_load["gpu_mb"]

    if collect_predictions:
        stats["predictions"] = predictions

    return stats


def print_stats(stats: Dict[str, Any]):
    """Красиво печатает статистику."""
    print(f"\n{'='*60}")
    print(f"Results for {stats['device'].upper()}")
    print(f"{'='*60}")

    print(f"\nDataset:")
    print(f"  Images: {stats['num_images']}")
    print(f"  Target size: {stats['target_size']}x{stats['target_size']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Avg detections/image: {stats['mean_detections']:.1f}")

    print(f"\nInference Time:")
    print(f"  Mean: {stats['mean_time_ms']:.2f} ms")
    print(f"  Median: {stats['median_time_ms']:.2f} ms")
    print(f"  Std: {stats['std_time_ms']:.2f} ms")
    print(f"  Min: {stats['min_time_ms']:.2f} ms")
    print(f"  Max: {stats['max_time_ms']:.2f} ms")
    print(f"  Total: {stats['total_time_s']:.2f} s")
    print(f"  Throughput: {stats['throughput_fps']:.2f} FPS")

    print(f"\nMemory Usage (RAM):")
    print(f"  After load: {stats['ram_after_load_mb']:.2f} MB")
    print(f"  Peak: {stats['ram_peak_mb']:.2f} MB")
    print(f"  Delta: {stats['ram_delta_mb']:.2f} MB")

    if "gpu_after_load_mb" in stats:
        print(f"\nMemory Usage (GPU VRAM):")
        print(f"  After load: {stats['gpu_after_load_mb']:.2f} MB")
        print(f"  Peak: {stats['gpu_peak_mb']:.2f} MB")
        print(f"  Delta: {stats['gpu_delta_mb']:.2f} MB")

    # Печатаем метрики точности, если они есть
    if "accuracy_metrics" in stats:
        metrics = stats["accuracy_metrics"]
        print(f"\nAccuracy Metrics:")
        print(f"  F1@0.5: {metrics['f1@0.5']:.4f}")
        print(f"  Precision@0.5: {metrics['precision@0.5']:.4f}")
        print(f"  Recall@0.5: {metrics['recall@0.5']:.4f}")
        print(f"  F1@0.5:0.95: {metrics['f1@0.5:0.95']:.4f}")
        print(f"\n  Total Predictions: {metrics['num_predictions']}")
        print(f"  Total Ground Truths: {metrics['num_ground_truths']}")
        print(f"  TP@0.5: {metrics.get('tp@0.50', 0)}")
        print(f"  FP@0.5: {metrics.get('fp@0.50', 0)}")
        print(f"  FN@0.5: {metrics.get('fn@0.50', 0)}")


def compare_devices(cpu_stats: Dict[str, Any], gpu_stats: Dict[str, Any]):
    """Сравнивает производительность CPU и GPU."""
    print(f"\n{'='*60}")
    print("CPU vs GPU Comparison")
    print(f"{'='*60}")

    speedup = cpu_stats["mean_time_ms"] / gpu_stats["mean_time_ms"]
    throughput_gain = gpu_stats["throughput_fps"] / cpu_stats["throughput_fps"]

    print(f"\nSpeed:")
    print(f"  CPU mean time: {cpu_stats['mean_time_ms']:.2f} ms")
    print(f"  GPU mean time: {gpu_stats['mean_time_ms']:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    print(f"\nThroughput:")
    print(f"  CPU: {cpu_stats['throughput_fps']:.2f} FPS")
    print(f"  GPU: {gpu_stats['throughput_fps']:.2f} FPS")
    print(f"  Gain: {throughput_gain:.2f}x")

    print(f"\nMemory:")
    print(f"  CPU RAM peak: {cpu_stats['ram_peak_mb']:.2f} MB")
    print(f"  GPU RAM peak: {gpu_stats['ram_peak_mb']:.2f} MB")
    print(f"  GPU VRAM peak: {gpu_stats['gpu_peak_mb']:.2f} MB")

    print(f"\nRecommendation:")
    if speedup > 2:
        print(f"  GPU is {speedup:.1f}x faster - strongly recommended for production")
    elif speedup > 1.5:
        print(f"  GPU is {speedup:.1f}x faster - recommended if available")
    else:
        print(f"  GPU is only {speedup:.1f}x faster - CPU may be sufficient")


def save_results(
    cpu_stats: Dict[str, Any], 
    gpu_stats: Optional[Dict[str, Any]], 
    output_file: str,
    dataset_name: str = None
):
    """Сохраняет результаты в JSON файл."""
    results = {
        "dataset_name": dataset_name,
        "cpu": cpu_stats,
    }

    if gpu_stats:
        results["gpu"] = gpu_stats
        results["comparison"] = {
            "speedup": cpu_stats["mean_time_ms"] / gpu_stats["mean_time_ms"],
            "throughput_gain": gpu_stats["throughput_fps"]
            / cpu_stats["throughput_fps"],
        }

    # Удаляем predictions из сохранения (слишком большой объем данных)
    if "predictions" in results.get("cpu", {}):
        del results["cpu"]["predictions"]
    if gpu_stats and "predictions" in results.get("gpu", {}):
        del results["gpu"]["predictions"]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to: {output_file}")


def run_benchmark_for_dataset(dataset_config: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Запускает бенчмарк для одного датасета."""
    dataset_name = dataset_config["name"]
    folder = dataset_config["folder"]
    annotations = dataset_config.get("annotations")
    
    print(f"\n\n{'#'*80}")
    print(f"# DATASET: {dataset_name}")
    print(f"{'#'*80}\n")
    
    # Проверяем папку
    if not Path(folder).exists():
        print(f"Error: Folder not found: {folder}")
        return None
    
    # Находим изображения
    print("Searching for images...")
    image_files = get_image_files(folder)
    
    if len(image_files) == 0:
        print(f"Error: No images found in {folder}")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Загружаем ground truth, если указан
    ground_truths = None
    if annotations:
        if not Path(annotations).exists():
            print(f"Warning: Annotations file not found: {annotations}")
        else:
            print(f"Loading ground truth annotations from {annotations}...")
            ground_truths = load_ground_truth(annotations, folder)
            print(f"Loaded annotations for {len(ground_truths)} images")
    
    # Проверяем доступность CUDA
    cuda_available = torch.cuda.is_available()
    if config["gpu_only"] and not cuda_available:
        print("Error: GPU requested but CUDA not available")
        return None
    
    if cuda_available:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, will benchmark only on CPU")
    
    # Бенчмарк
    cpu_stats = None
    gpu_stats = None
    
    # CPU
    if not config["gpu_only"]:
        cpu_stats = benchmark_device(
            image_files,
            device="cpu",
            target_size=config["target_size"],
            warmup_runs=config["warmup_runs"],
            score_thresh=config["score_thresh"],
            collect_predictions=(ground_truths is not None),
        )
        
        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in cpu_stats:
            print("\nEvaluating accuracy on CPU predictions...")
            accuracy_metrics = evaluate_dataset(
                cpu_stats["predictions"], ground_truths, verbose=True, n_jobs=1
            )
            cpu_stats["accuracy_metrics"] = accuracy_metrics
        
        print_stats(cpu_stats)
    
    # GPU
    if cuda_available and not config["cpu_only"]:
        gpu_stats = benchmark_device(
            image_files,
            device="cuda",
            target_size=config["target_size"],
            warmup_runs=config["warmup_runs"],
            score_thresh=config["score_thresh"],
            collect_predictions=(ground_truths is not None),
        )
        
        # Оцениваем точность, если есть ground truth
        if ground_truths and "predictions" in gpu_stats:
            print("\nEvaluating accuracy on GPU predictions...")
            accuracy_metrics = evaluate_dataset(
                gpu_stats["predictions"], ground_truths, verbose=True, n_jobs=1
            )
            gpu_stats["accuracy_metrics"] = accuracy_metrics
        
        print_stats(gpu_stats)
    
    # Сравнение
    if cpu_stats and gpu_stats:
        compare_devices(cpu_stats, gpu_stats)
    
    # Сохранение результатов для каждого датасета
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_dir) / f"{dataset_name}_results.json"
        save_results(cpu_stats, gpu_stats, str(output_file), dataset_name)
        print(f"[INFO] Results for {dataset_name} saved to {output_file}")
    
    return {
        "dataset_name": dataset_name,
        "cpu_stats": cpu_stats,
        "gpu_stats": gpu_stats,
    }


def main():
    """Запускает бенчмарк для всех датасетов из конфигурации."""
    print(f"{'='*80}")
    print("EAST Detector Benchmark")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Target size: {CONFIG['target_size']}")
    print(f"  Score threshold: {CONFIG['score_thresh']}")
    print(f"  Warmup runs: {CONFIG['warmup_runs']}")
    print(f"  CPU only: {CONFIG['cpu_only']}")
    print(f"  GPU only: {CONFIG['gpu_only']}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"\nDatasets to benchmark: {len(DATASETS)}")
    for i, ds in enumerate(DATASETS, 1):
        print(f"  {i}. {ds['name']}")
    
    # Запускаем бенчмарк для каждого датасета
    all_results = []
    for dataset_config in DATASETS:
        result = run_benchmark_for_dataset(dataset_config, CONFIG, CONFIG["output_dir"])
        if result:
            all_results.append(result)
    
    # Сводная таблица результатов
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if all_results:
        print(f"{'Dataset':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'CPU F1@0.5':<12} {'GPU F1@0.5':<12}")
        print("-" * 95)
        
        for result in all_results:
            name = result["dataset_name"]
            cpu = result["cpu_stats"]
            gpu = result["gpu_stats"]
            
            cpu_time = f"{cpu['mean_time_ms']:.2f}" if cpu else "N/A"
            gpu_time = f"{gpu['mean_time_ms']:.2f}" if gpu else "N/A"
            speedup = f"{cpu['mean_time_ms'] / gpu['mean_time_ms']:.2f}x" if (cpu and gpu) else "N/A"
            
            cpu_f1 = f"{cpu.get('accuracy_metrics', {}).get('f1@0.5', 0):.4f}" if cpu and 'accuracy_metrics' in cpu else "N/A"
            gpu_f1 = f"{gpu.get('accuracy_metrics', {}).get('f1@0.5', 0):.4f}" if gpu and 'accuracy_metrics' in gpu else "N/A"
            
            print(f"{name:<25} {cpu_time:<12} {gpu_time:<12} {speedup:<10} {cpu_f1:<12} {gpu_f1:<12}")
    
    print(f"\n{'='*80}")
    print("All benchmarks completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
