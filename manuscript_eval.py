# -*- coding: utf-8 -*-
# AzbukaRunner — TRBA models runner (manuscript-ocr)

import os
import csv
import yaml
from pathlib import Path
import cv2
from manuscript.recognizers import TRBA


# ----------------- USER SETTINGS -----------------

CONFIG_PATH = r"C:\Users\USER\AzbukaBoard\config.yaml"
DATA_ROOT   = r"C:\shared\ocr_datasets"
OUTPUT_DIR  = ""    # куда сохранять результаты

# путь к модели TRBA
TRBA_MODEL_PATH = r"C:\Users\USER\manuscript-ocr\model.onnx"
TRBA_CONFIG_PATH = r"C:\Users\USER\manuscript-ocr\experiments\trba_exp_printed_lite256\config.json"


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


# ----------------- LOAD TRBA MODEL -----------------

_TRBA_INSTANCE = None

def load_trba():
    global _TRBA_INSTANCE
    if _TRBA_INSTANCE:
        return _TRBA_INSTANCE

    print("\n========== Initializing TRBA recognizer ==========")

    recognizer = TRBA(
        model_path=TRBA_MODEL_PATH,
        config_path=TRBA_CONFIG_PATH
    )

    _TRBA_INSTANCE = recognizer
    print("[OK] TRBA model loaded\n")
    return recognizer


# ----------------- RUNNER -----------------

def run_trba_on_dataset(dataset_name: str, ds_cfg: dict, data_root: str, output_dir: Path):
    print(f"\nRunning TRBA  |  dataset: {dataset_name}")

    csv_path = os.path.join(data_root, ds_cfg["csv"])
    images_dir = os.path.join(data_root, ds_cfg["images_dir"])

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV missing → skip dataset '{dataset_name}'")
        print(f"       {csv_path}")
        return

    if not os.path.isdir(images_dir):
        print(f"[WARN] images_dir missing → skip dataset '{dataset_name}'")
        print(f"       {images_dir}")
        return

    images = load_dataset_images(
        csv_path,
        image_col=ds_cfg["image_column"],
        has_header=ds_cfg["has_header"]
    )

    if not images:
        print(f"[WARN] CSV returned 0 images → skip dataset '{dataset_name}'")
        return

    recognizer = load_trba()

    # full paths list
    full_paths = [os.path.join(images_dir, img) for img in images]

    # run inference
    res = recognizer.predict(images=full_paths, batch_size=16)

    # output file
    filename = f"{dataset_name}_TRBA.csv"
    out_file = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)

    import csv
    with open(out_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction"])
        for img, r in zip(images, res):
            text = r.get("text", "")
            writer.writerow([img, text])

    print(f"[OK] Saved submission: {out_file}")


# ----------------- MAIN -----------------

def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    if not os.path.exists(TRBA_MODEL_PATH):
        raise FileNotFoundError(f"TRBA model file not found: {TRBA_MODEL_PATH}")

    if not os.path.exists(TRBA_CONFIG_PATH):
        raise FileNotFoundError(f"TRBA json config not found: {TRBA_CONFIG_PATH}")

    output_dir = Path(OUTPUT_DIR)
    config = load_config(CONFIG_PATH)
    datasets = config["datasets"]

    print("Datasets:", list(datasets.keys()))
    print("Model: TRBA")

    for dataset_name, ds_cfg in datasets.items():
        run_trba_on_dataset(dataset_name, ds_cfg, DATA_ROOT, output_dir)

    print("\nAll TRBA submissions completed!")


if __name__ == "__main__":
    main()
