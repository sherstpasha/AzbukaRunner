# -*- coding: utf-8 -*-
# EasyOCR Speed Benchmark — измерение скорости распознавания
import os
import csv
import yaml
import time
from pathlib import Path
from easyocr import Reader
import cv2
import statistics


# ----------------- USER SETTINGS -----------------

CONFIG_PATH = r"C:\Users\USER\AzbukaBoard\config.yaml"
DATA_ROOT   = r"C:\shared\ocr_datasets"
OUTPUT_DIR  = "speed_results"  # папка для результатов
NUM_SAMPLES = 1000  # количество изображений для теста (None = все)


# ----------------- MODELS AVAILABLE -----------------

MODEL_LIST = [
    ("1", "cyrillic_g1"),
    ("2", "cyrillic_g2"),
]


# ----------------- CONFIG LOADER -----------------

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_images(csv_path, image_col=0, has_header=True):
    images = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)

        for row in reader:
            if len(row) <= image_col:
                continue
            images.append(row[image_col])
    return images


# ----------------- EASYOCR MODEL LOADER -----------------

_MODEL_CACHE = {}


def load_easyocr_model(gen: str, use_gpu: bool = True):
    """Создаёт EasyOCR reader для g1 / g2. Кэширует."""

    if gen in _MODEL_CACHE:
        return _MODEL_CACHE[gen]

    print(f"\n========== INITIALIZING MODEL {gen} ==========")

    if gen == "1":
        recog = "cyrillic_g1"
    elif gen == "2":
        recog = "cyrillic_g2"
    else:
        raise ValueError(f"Unknown model key: {gen}")

    reader = Reader(
        ["ru"],
        gpu=use_gpu,
        detector=False,
        recognizer=True,
        recog_network=recog
    )

    def recognize_fn(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return "", 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        horizontal_list = [[0, w, 0, h]]
        free_list = []

        start_time = time.perf_counter()
        out = reader.recognize(
            gray,
            horizontal_list=horizontal_list,
            free_list=free_list,
            detail=1
        )
        elapsed = time.perf_counter() - start_time

        text = out[0][1] if out else ""
        return text, elapsed

    _MODEL_CACHE[gen] = recognize_fn
    return recognize_fn


# ----------------- SPEED BENCHMARK -----------------

def benchmark_speed(model_key: str, display_name: str, dataset_name: str, ds_cfg: dict, data_root: str, output_dir: Path, num_samples=None):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {display_name}  |  DATASET: {dataset_name}")
    print(f"{'='*60}")

    csv_path = os.path.join(data_root, ds_cfg["csv"])
    images_dir = os.path.join(data_root, ds_cfg["images_dir"])

    # ---------- SAFETY CHECKS ----------
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found → skipping dataset '{dataset_name}'")
        print(f"       Path: {csv_path}")
        return None

    if not os.path.isdir(images_dir):
        print(f"[WARN] images_dir not found → skipping dataset '{dataset_name}'")
        print(f"       Path: {images_dir}")
        return None
    # -----------------------------------

    images = load_dataset_images(
        csv_path,
        image_col=ds_cfg["image_column"],
        has_header=ds_cfg["has_header"]
    )

    if not images:
        print(f"[WARN] No images found in CSV for dataset '{dataset_name}'. Skipping.")
        return None

    # Ограничиваем количество примеров
    if num_samples is not None and len(images) > num_samples:
        images = images[:num_samples]

    print(f"Testing on {len(images)} images...")

    recognize_fn = load_easyocr_model(model_key)

    # Замеры времени
    times = []
    predictions = []
    failed = 0

    print("\nProcessing images...")
    for i, img in enumerate(images, 1):
        img_path = os.path.join(images_dir, img)
        
        if not os.path.exists(img_path):
            failed += 1
            continue

        try:
            pred, elapsed = recognize_fn(img_path)
            times.append(elapsed)
            predictions.append((img, pred, elapsed))
        except Exception as e:
            print(f"[ERROR] Failed to process {img}: {e}")
            failed += 1

        if i % 50 == 0:
            print(f"  Processed {i}/{len(images)} images...")

    if not times:
        print("[ERROR] No valid measurements!")
        return None

    # Статистика
    total_time = sum(times)
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    samples_per_second = 1.0 / mean_time if mean_time > 0 else 0.0
    words_per_second = samples_per_second  # для одного слова на изображение

    # Вывод результатов
    print(f"\n{'='*60}")
    print(f"RESULTS: {display_name} on {dataset_name}")
    print(f"{'='*60}")
    print(f"Total images:        {len(images)}")
    print(f"Successful:          {len(times)}")
    print(f"Failed:              {failed}")
    print(f"\nTIME STATISTICS:")
    print(f"  Total time:        {total_time:.3f} sec")
    print(f"  Mean time:         {mean_time*1000:.2f} ms/image")
    print(f"  Median time:       {median_time*1000:.2f} ms/image")
    print(f"  Min time:          {min_time*1000:.2f} ms/image")
    print(f"  Max time:          {max_time*1000:.2f} ms/image")
    print(f"  Std deviation:     {std_time*1000:.2f} ms")
    print(f"\nTHROUGHPUT:")
    print(f"  Images/sec:        {samples_per_second:.2f}")
    print(f"  Words/sec:         {words_per_second:.2f}")
    print(f"{'='*60}\n")

    # Сохраняем детальные результаты
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV с детальными замерами
    detail_file = output_dir / f"{dataset_name}_{display_name}_detailed.csv"
    with open(detail_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction", "time_ms"])
        for img, pred, elapsed in predictions:
            writer.writerow([img, pred, f"{elapsed*1000:.2f}"])
    
    print(f"[OK] Detailed results saved: {detail_file}")

    # Возвращаем сводку для общей таблицы
    return {
        "model": display_name,
        "dataset": dataset_name,
        "total_images": len(images),
        "successful": len(times),
        "failed": failed,
        "total_time_sec": total_time,
        "mean_time_ms": mean_time * 1000,
        "median_time_ms": median_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "std_time_ms": std_time * 1000,
        "images_per_sec": samples_per_second,
        "words_per_sec": words_per_second,
    }


def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    output_dir = Path(OUTPUT_DIR)

    config = load_config(CONFIG_PATH)
    datasets = config["datasets"]

    print("="*60)
    print("EASYOCR SPEED BENCHMARK")
    print("="*60)
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Models:   {[m[1] for m in MODEL_LIST]}")
    print(f"Samples:  {NUM_SAMPLES if NUM_SAMPLES else 'ALL'}")
    print("="*60)

    all_results = []

    for model_key, model_name in MODEL_LIST:
        for dataset_name, ds_cfg in datasets.items():
            result = benchmark_speed(
                model_key, 
                model_name, 
                dataset_name, 
                ds_cfg, 
                DATA_ROOT, 
                output_dir,
                num_samples=NUM_SAMPLES
            )
            if result:
                all_results.append(result)

    # Сохраняем сводную таблицу
    if all_results:
        summary_file = output_dir / "speed_summary.csv"
        with open(summary_file, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n{'='*60}")
        print(f"[OK] Summary saved: {summary_file}")
        print(f"{'='*60}\n")

    print("\n✅ Speed benchmark completed!")


if __name__ == "__main__":
    main()
