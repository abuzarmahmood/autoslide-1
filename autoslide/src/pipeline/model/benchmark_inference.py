"""
Standalone per-image inference timing benchmark.

Runs the trained Mask R-CNN model over a fixed list of images and records
wall-clock prediction time for each one, along with machine/GPU identifying
info, so results from different machines can be compared directly.

Usage:
    python -m autoslide.src.pipeline.model.benchmark_inference \\
        --image-dir /path/to/images \\
        --image-list /path/to/image_list.txt \\
        --model-path /path/to/best_val_mask_rcnn_model.pth \\
        --output-csv /path/to/results.csv \\
        --num-warmup 10
"""

import argparse
import csv
import os
import platform
import socket

import torch
from PIL import Image
from tqdm import tqdm

from autoslide.src.pipeline.model.prediction_utils import (
    load_model,
    predict_single_image,
    setup_device,
)


def get_gpu_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def load_image_list(image_dir, image_list_path=None):
    if image_list_path:
        with open(image_list_path) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = sorted(os.listdir(image_dir))
    return names


def run_benchmark(model, device, transform, image_dir, image_names, num_warmup):
    warmup_names = image_names[:num_warmup]
    for img_name in tqdm(warmup_names, desc="Warmup"):
        image = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        _ = predict_single_image(model, image, device,
                                 transform, return_time=False)

    rows = []
    for img_name in tqdm(image_names, desc="Benchmarking"):
        image_path = os.path.join(image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        pred_time, _ = predict_single_image(
            model, image, device, transform, return_time=True)
        rows.append({"image_name": img_name, "prediction_time_s": pred_time})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-image inference timing")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--image-list", type=str, default=None,
                        help="Optional text file with one image filename per line "
                        "(controls exact set/order used across machines)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved model (default: best_val_mask_rcnn_model.pth)")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to write per-image timing results as CSV")
    parser.add_argument("--num-warmup", type=int, default=10,
                        help="Number of warmup iterations before timing")
    args = parser.parse_args()

    device = setup_device()
    model, device, transform = load_model(args.model_path, device)

    image_names = load_image_list(args.image_dir, args.image_list)
    print(
        f"Running benchmark on {len(image_names)} images using device: {device}")

    rows = run_benchmark(model, device, transform, args.image_dir,
                         image_names, args.num_warmup)

    gpu_name = get_gpu_name(device)
    hostname = socket.gethostname()

    os.makedirs(os.path.dirname(
        os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_name", "prediction_time_s", "gpu_name", "hostname"])
        writer.writeheader()
        for row in rows:
            row["gpu_name"] = gpu_name
            row["hostname"] = hostname
            writer.writerow(row)

    times = [r["prediction_time_s"] for r in rows]
    mean_time = sum(times) / len(times)
    print(f"Device: {device} ({gpu_name}) on host {hostname}")
    print(f"Mean prediction time: {mean_time:.4f}s over {len(times)} images")
    print(f"Results written to: {args.output_csv}")


if __name__ == "__main__":
    main()
