# -*- coding: utf-8 -*-
# AzbukaRunner — EasyOCR Cyrillic models (g1 / g2)
import os
import csv
import yaml
from pathlib import Path
from easyocr import Reader
import cv2


# ----------------- USER SETTINGS -----------------

CONFIG_PATH = r"C:\Users\USER\AzbukaBoard\config2.yaml"       # config рядом с файлом
DATA_ROOT   = r"C:\shared\ocr_datasets"     # здесь лежат папки датасетов
OUTPUT_DIR  = ""       # куда сохранять результаты


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
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        horizontal_list = [[0, w, 0, h]]
        free_list = []

        out = reader.recognize(
            gray,
            horizontal_list=horizontal_list,
            free_list=free_list,
            detail=1
        )

        return out[0][1] if out else ""

    _MODEL_CACHE[gen] = recognize_fn
    return recognize_fn


# ----------------- RUNNER -----------------

def run_model_on_dataset(model_key: str, display_name: str, dataset_name: str, ds_cfg: dict, data_root: str, output_dir: Path):
    print(f"\nRunning model: {display_name}  |  dataset: {dataset_name}")

    csv_path = os.path.join(data_root, ds_cfg["csv"])
    images_dir = os.path.join(data_root, ds_cfg["images_dir"])

    # ---------- SAFETY CHECKS ----------
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found → skipping dataset '{dataset_name}'")
        print(f"       Path: {csv_path}")
        return

    if not os.path.isdir(images_dir):
        print(f"[WARN] images_dir not found → skipping dataset '{dataset_name}'")
        print(f"       Path: {images_dir}")
        return
    # -----------------------------------

    images = load_dataset_images(
        csv_path,
        image_col=ds_cfg["image_column"],
        has_header=ds_cfg["has_header"]
    )

    if not images:
        print(f"[WARN] No images found in CSV for dataset '{dataset_name}'. Skipping.")
        return

    recognize_fn = load_easyocr_model(model_key)

    # ---------- NEW FILENAME FORMAT ----------
    filename = f"{dataset_name}_{display_name}.csv"
    out_file = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------------------

    import csv
    with open(out_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction"])
        for img in images:
            img_path = os.path.join(images_dir, img)
            pred = recognize_fn(img_path)
            writer.writerow([img, pred])

    print(f"[OK] Saved submission: {out_file}")

def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    output_dir = Path(OUTPUT_DIR)

    config = load_config(CONFIG_PATH)
    datasets = config["datasets"]

    print("Datasets:", list(datasets.keys()))
    print("Models:", [m[1] for m in MODEL_LIST])

    for model_key, model_name in MODEL_LIST:
        for dataset_name, ds_cfg in datasets.items():
            run_model_on_dataset(model_key, model_name, dataset_name, ds_cfg, DATA_ROOT, output_dir)

    print("\nAll models completed!")


if __name__ == "__main__":
    main()
