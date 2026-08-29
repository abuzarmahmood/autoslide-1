"""
Batch-size sweep inference timing benchmark, with optional forced-CPU mode.

Times batched forward passes across a range of batch sizes, on either the
auto-detected device, a forced CUDA device, or a forced CPU device, and
records wall-clock timing per batch to CSV.

Usage:
    python -m autoslide.src.pipeline.model.benchmark_inference \\
        --image-dir /path/to/images \\
        --image-list /path/to/image_list.txt \\
        --model-path /path/to/best_val_mask_rcnn_model.pth \\
        --output-csv /path/to/results.csv \\
        --batch-sizes 1,2,4,8,16,32 \\
        --device auto \\
        --num-warmup-batches 3 \\
        --num-timed-batches 10
"""

import argparse
import csv
import os
import random
import socket

import torch
from PIL import Image
from tqdm import tqdm

from autoslide.src.pipeline.model.prediction_utils import (
    load_model,
    predict_batch,
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


def load_images(image_dir, names):
    return [Image.open(os.path.join(image_dir, name)).convert("RGB") for name in names]


def sample_batches(image_names, batch_size, num_batches, rng):
    """
    Return num_batches lists of length batch_size, sampled from image_names.
    Samples without replacement across the whole requested pool if it fits,
    otherwise falls back to per-draw sampling with replacement.
    """
    total_needed = batch_size * num_batches
    if total_needed <= len(image_names):
        pool = list(image_names)
        rng.shuffle(pool)
        selected = pool[:total_needed]
    else:
        selected = [rng.choice(image_names) for _ in range(total_needed)]
    return [selected[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def run_one_batch_size(model, device, transform, image_dir, image_names,
                       batch_size, num_warmup_batches, num_timed_batches, seed):
    """
    Run warmup + timed batches for a single batch size. Returns whatever
    timed rows were collected -- an OOM anywhere in this batch size's
    warmup or timed batches is caught here and does not propagate, so the
    caller can keep going with the next batch size.
    """
    rng = random.Random(seed * 10_000 + batch_size)
    rows = []
    try:
        warmup_batches = sample_batches(
            image_names, batch_size, num_warmup_batches, rng)
        for names in tqdm(warmup_batches, desc=f"Warmup bs={batch_size}"):
            images = load_images(image_dir, names)
            _ = predict_batch(model, images, device, transform)

        timed_batches = sample_batches(
            image_names, batch_size, num_timed_batches, rng)
        for names in tqdm(timed_batches, desc=f"Timed bs={batch_size}"):
            images = load_images(image_dir, names)
            batch_time = predict_batch(model, images, device, transform)
            rows.append({
                "batch_size": batch_size,
                "batch_time_s": batch_time,
                "per_image_time_s": batch_time / batch_size,
                "num_threads": torch.get_num_threads(),
                "image_names": ";".join(names),
            })
    except torch.cuda.OutOfMemoryError:
        print(
            f"OOM at batch_size={batch_size}; skipping remaining batches for this size")
        torch.cuda.empty_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-size sweep inference timing benchmark")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--image-list", type=str, default=None,
                        help="Optional text file with one image filename per line")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to saved model (default: best_val_mask_rcnn_model.pth)")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to write per-batch timing results as CSV")
    parser.add_argument("--batch-sizes", type=str, default="1",
                        help="Comma-separated batch sizes to sweep, e.g. '1,2,4,8,16,32'")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                        help="'auto' picks cuda if available else cpu; "
                        "'cpu' forces CPU even if a GPU is present; "
                        "'cuda' errors out if no CUDA device is available")
    parser.add_argument("--num-warmup-batches", type=int, default=3,
                        help="Untimed warmup batches run per batch size before timing")
    parser.add_argument("--num-timed-batches", type=int, default=10,
                        help="Timed batches recorded per batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible batch sampling")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: --device cuda requested but no CUDA device is available on this machine.")

    if args.device == "auto":
        device = setup_device()
    else:
        device = setup_device(force_cpu=(args.device == "cpu"))

    model, device, transform = load_model(args.model_path, device)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    image_names = load_image_list(args.image_dir, args.image_list)
    print(
        f"Running batch sweep {batch_sizes} on {len(image_names)} images using device: {device}")

    gpu_name = get_gpu_name(device)
    hostname = socket.gethostname()

    os.makedirs(os.path.dirname(
        os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = ["batch_size", "batch_time_s", "per_image_time_s",
                  "device_type", "gpu_name", "hostname", "num_threads", "image_names"]

    # Write incrementally (flush per batch size) so a crash partway through
    # the sweep doesn't discard already-collected batch sizes.
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch_size in batch_sizes:
            bs_rows = run_one_batch_size(
                model, device, transform, args.image_dir, image_names,
                batch_size, args.num_warmup_batches, args.num_timed_batches, args.seed)
            for row in bs_rows:
                row["device_type"] = device.type
                row["gpu_name"] = gpu_name
                row["hostname"] = hostname
                writer.writerow(row)
            f.flush()

            if not bs_rows:
                print(f"  batch_size={batch_size}: no timed batches (OOM?)")
                continue
            mean_per_image = sum(r["per_image_time_s"]
                                 for r in bs_rows) / len(bs_rows)
            print(f"  batch_size={batch_size}: mean {mean_per_image:.4f}s/image "
                  f"({1.0 / mean_per_image:.2f} img/s) over {len(bs_rows)} batches")

    print(f"Device: {device} ({gpu_name}) on host {hostname}")
    print(f"Results written to: {args.output_csv}")


if __name__ == "__main__":
    main()
