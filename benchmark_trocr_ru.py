# -*- coding: utf-8 -*-
# AzbukaRunner — TrOCR Russian model (batch inference)

import os
import csv
import yaml
from pathlib import Path
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ----------------- USER SETTINGS -----------------

CONFIG_PATH = r"C:\Users\USER\AzbukaBoard\config2.yaml"
DATA_ROOT   = r"C:\shared\ocr_datasets"
OUTPUT_DIR  = ""

# Модель одна, но оставляем как список (аналог EasyOCR runner)
MODEL_LIST = [
    ("trocr_cyrillic_hw",  "cyrillic-trocr/trocr-handwritten-cyrillic"),
    ("trocr_ru_base_rxt",  "raxtemur/trocr-base-ru"),
    ("trocr_ru_1700s",     "taiga75/ru-trocr-1700s"),
    ("dialecticstackmixDomino",  "Daniil-Domino/trocr-base-ru-dialectic-stackmix"),
    ("dialecticDomino", "Daniil-Domino/trocr-base-ru-dialectic"),
    ("trocr_ru_kazars24", "kazars24/trocr-base-handwritten-ru"),
    #("trocr_ru_1933",  "taiga75/trocr-russian-print-1933")
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


# ----------------- TrOCR MODEL LOADER -----------------

_MODEL_CACHE = {}


def load_trocr_model(model_id: str, use_gpu: bool = True):
    """Создаёт и кэширует TrOCR модель (устойчивая версия)."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    print(f"\n========== INITIALIZING TrOCR MODEL: {model_id} ==========")

    try:
        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
    except Exception as e:
        print(f"[ERROR] FAILED TO LOAD MODEL '{model_id}': {e}")
        print("        → skipping this model entirely.\n")
        _MODEL_CACHE[model_id] = None
        return None

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    def recognize_fn(img_path: str):
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return ""

        try:
            with torch.no_grad():
                pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            print(f"[WARN] Inference error on {img_path}: {e}")
            return ""

    _MODEL_CACHE[model_id] = recognize_fn
    return recognize_fn


# ----------------- RUNNER -----------------

def run_model_on_dataset(model_key: str,
                         model_name: str,
                         dataset_name: str,
                         ds_cfg: dict,
                         data_root: str,
                         output_dir: Path):

    print(f"\nRunning model: {model_key}  |  dataset: {dataset_name}")

    recognize_fn = load_trocr_model(model_name)
    if recognize_fn is None:
        print(f"[SKIP] Model '{model_key}' unavailable → skipping dataset\n")
        return

    print(f"\nRunning model: {model_key}  |  dataset: {dataset_name}")

    csv_path     = os.path.join(data_root, ds_cfg["csv"])
    images_dir   = os.path.join(data_root, ds_cfg["images_dir"])

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

    recognize_fn = load_trocr_model(model_name)

    # ---------- OUTPUT CSV ----------
    filename = f"{dataset_name}_{model_key}.csv"
    out_file = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------

    import csv
    with open(out_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction"])
        for img in images:
            img_path = os.path.join(images_dir, img)
            try:
                pred = recognize_fn(img_path)
            except Exception as e:
                print(f"[WARN] Could not process {img_path}: {e}")
                pred = ""

            writer.writerow([img, pred])

    print(f"[OK] Saved submission: {out_file}")


# ----------------- MAIN -----------------

def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"CONFIG_PATH not found: {CONFIG_PATH}")

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT not found: {DATA_ROOT}")

    output_dir = Path(OUTPUT_DIR)

    config = load_config(CONFIG_PATH)
    datasets = config["datasets"]

    print("Datasets:", list(datasets.keys()))
    print("Models:", [m[0] for m in MODEL_LIST])

    for model_key, model_name in MODEL_LIST:
        for dataset_name, ds_cfg in datasets.items():
            run_model_on_dataset(model_key, model_name,
                                 dataset_name, ds_cfg,
                                 DATA_ROOT, output_dir)

    print("\nAll models completed!\n")


if __name__ == "__main__":
    main()
