"""Short CUDA compute benchmark on synthetic tensors; never saves trained weights."""
import argparse
from pathlib import Path
import time

import torch

from .augmentation import mixup_batch, soft_target_cross_entropy
from .models import build_model, parameter_counts
from .train import amp_context, validate_config
from .utils import dependency_versions, device_for, file_hash, precision_policy, read_json, seed_all, sync, write_json


def benchmark(config_path, output, batch_sizes=(128,), warmup=10, steps=30):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Benchmark output exists; choose a new file")
    if warmup < 1 or steps < 1 or not batch_sizes or any(b < 1 for b in batch_sizes):
        raise ValueError("Positive warmup, steps and batch sizes required")
    config = read_json(config_path)
    validate_config(config)
    if config["training"].get("optimization") is not None:
        raise ValueError("Use a bounded src.train pilot for controlled optimizers; this legacy compute benchmark does not run SAM/ASAM")
    device = device_for("cuda")
    t = config["training"]
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if config["runtime"]["amp"] else None
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    report = {"kind": "synthetic_cuda_compute_only", "config_sha256": file_hash(config_path),
              "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "precision_policy": precision_policy(dtype), **dependency_versions(),
              "warmup_steps": warmup, "measured_steps": steps,
              "excludes": ["PNG decode", "data loading", "CPU crop/flip/Cutout", "host-to-device transfer",
                           "validation", "checkpoint/export writes"], "measurements": []}
    for batch_size in batch_sizes:
        seed_all(config["seed"])
        model = build_model(config["model"]).to(device).train()
        if t["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"], weight_decay=t["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        labels = torch.arange(batch_size, device=device) % 10
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
        skipped = 0

        def step():
            nonlocal skipped
            x, targets = mixup_batch(images, labels, t.get("mixup_alpha", 0.0), t.get("mixup_probability", 1.0))
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, dtype):
                logits = model(x)
                loss = (soft_target_cross_entropy(logits, targets, t["label_smoothing"])
                        if targets.ndim == 2 else criterion(logits, targets))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite benchmark loss")
            scaler.scale(loss).backward()
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
            # Match the trainer's scalar loss/accuracy synchronization.
            loss.item()
            logits.argmax(1).eq(labels).sum().item()

        for _ in range(warmup):
            step()
        sync(device)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        sync(device)
        elapsed = time.perf_counter() - start
        rate = batch_size * steps / elapsed
        report["measurements"].append({"batch_size": batch_size, "seconds": elapsed,
                                       "images_per_second": rate, "seconds_per_step": elapsed / steps,
                                       "peak_cuda_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                                       "compute_only_seconds_per_45000_images": 45000 / rate,
                                       "amp_skipped_updates_including_warmup": skipped,
                                       "parameter_counts": parameter_counts(model)})
        del model, optimizer, scaler, images, labels
        torch.cuda.empty_cache()
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    print(benchmark(args.config, args.output, args.batch_sizes, args.warmup, args.steps))
