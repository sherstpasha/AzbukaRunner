import os
import json
import glob
import pandas as pd

# Define the folders and model names
folders = [
    ("benchmark_results_craft", "craft"),
    ("benchmark_results_east", "east"),
    ("benchmark_results_yolo", "yolo"),
]


def find_accuracy_metrics(d: dict):
    # accuracy_metrics may be at root, or under gpu, or under cpu
    if not isinstance(d, dict):
        return {}
    for key in ("accuracy_metrics",):
        if key in d and isinstance(d[key], dict):
            return d[key]
    for parent in ("gpu", "cpu", "results"):
        p = d.get(parent)
        if isinstance(p, dict) and "accuracy_metrics" in p and isinstance(p["accuracy_metrics"], dict):
            return p["accuracy_metrics"]
    return {}


def get_f1_from_metrics(m: dict, primary_key_variants=("f1@0.50", "f1@0.5")):
    for k in primary_key_variants:
        if k in m:
            return m[k]
    # some files may include lower-case, or other variants
    for k in m:
        if k.lower().startswith("f1@0.5") and ":" not in k:
            return m[k]
    return None


def get_f1_aom(m: dict):
    # area under IoU range: prefer key with range, try variants
    for k in ("f1@0.5:0.95", "f1@0.50:0.95", "f1@0.5:0.95"):
        if k in m:
            return m[k]
    # fallback: try any key that looks like a range
    for k in m:
        if "0.5" in k and "0.95" in k:
            return m[k]
    return None


def get_throughput(d: dict):
    # Prefer GPU throughput, else CPU throughput
    if isinstance(d.get("gpu"), dict):
        t = d["gpu"].get("throughput_fps")
        if t is not None:
            return t
    if isinstance(d.get("cpu"), dict):
        t = d["cpu"].get("throughput_fps") or d["cpu"].get("throughput_fps")
        if t is not None:
            return t
    # some files use throughput_fps at root
    if "throughput_fps" in d:
        return d["throughput_fps"]
    return None


def get_mem_peak(d: dict):
    # Use max of ram_peak_mb and gpu_peak_mb when available
    ram = None
    gpu = None
    if isinstance(d.get("gpu"), dict):
        gpu = d["gpu"].get("gpu_peak_mb") or d["gpu"].get("gpu_peak_mb")
        # older files may name it gpu_peak_mb; keep attempt
        if gpu is None:
            gpu = d["gpu"].get("gpu_after_load_mb")
    if isinstance(d.get("cpu"), dict):
        ram = d["cpu"].get("ram_peak_mb")
    # top-level ram
    if ram is None and "ram_peak_mb" in d:
        ram = d.get("ram_peak_mb")
    # top-level gpu
    if gpu is None and "gpu_peak_mb" in d:
        gpu = d.get("gpu_peak_mb")
    # choose the maximum non-None
    vals = [v for v in (ram, gpu) if isinstance(v, (int, float))]
    return max(vals) if vals else None


data = []

for folder, model in folders:
    for file in glob.glob(os.path.join(folder, "*.json")):
        with open(file, encoding="utf-8") as f:
            d = json.load(f)
        dataset = d.get("dataset") or d.get("dataset_name") or os.path.splitext(os.path.basename(file))[0]
        row = {"Dataset": dataset, "Model": model}

        acc = find_accuracy_metrics(d)
        row["F1-0.5"] = get_f1_from_metrics(acc)
        row["F1-0.5:0.95"] = get_f1_aom(acc)

        row["Throughput (img/sec, GPU)"] = get_throughput(d)
        row["Mem. (max, MB)"] = get_mem_peak(d)

        data.append(row)


# Create DataFrame and save as table
results = pd.DataFrame(data)
results = results.sort_values(["Dataset", "Model"]).reset_index(drop=True)
results.to_csv("benchmark_aggregated_results.csv", index=False)
print(results.to_string(index=False))
