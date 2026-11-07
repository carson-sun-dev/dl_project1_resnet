"""Scratch Shake-Shake + CIFAR-10 AutoAugment/Cutout on the saved team split.

No external data, pretrained initialization, EMA or hidden-image selection.
Epoch-boundary checkpoints preserve training/RNG state and the selected model.
The exact CPU continuation is testable; fast CUDA kernels may not be bitwise
deterministic. Validation uses FP32 without TF32 and a fixed mirror policy.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .data import CLASSES, image_path, load_split
from .models import build_model, parameter_counts
from .pixel_cache import load_training_pixels, training_rows
from .train import amp_context
from .utils import (dependency_versions, device_for, epoch_lr, file_hash,
                    precision_policy, read_json, restore_rng, rng_state,
                    save_checkpoint, seed_all, sync, validate_normalization,
                    validate_resume_precision, write_json)


class Cutout:
    """One square mask centered uniformly on the image, clipped at its border."""
    def __init__(self, length=16):
        if not isinstance(length, int) or not 0 <= length <= 32:
            raise ValueError("Cutout length must be an integer from 0 to 32")
        self.length = length

    def __call__(self, tensor):
        if self.length:
            height, width = tensor.shape[-2:]
            cy, cx = int(torch.randint(height, (1,))), int(torch.randint(width, (1,)))
            half = self.length // 2
            tensor[:, max(0, cy-half):min(height, cy+self.length-half),
                   max(0, cx-half):min(width, cx+self.length-half)] = 0
        return tensor


class CachedShakeDataset(Dataset):
    """Shared CPU pixels; augmentation only happens for permitted training IDs."""
    def __init__(self, pixels, labels, mean, std, padding=4, cutout=16):
        if (pixels.dtype != torch.uint8 or pixels.ndim != 4 or tuple(pixels.shape[1:]) != (32, 32, 3)
                or labels.dtype != torch.long or len(labels) != len(pixels) or not len(labels)):
            raise ValueError("Expected nonempty uint8 NHWC training pixels and hard labels")
        validate_normalization({"mean": mean, "std": std})
        self.pixels, self.labels = pixels, labels
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=padding, padding_mode="constant"),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10,
                                   interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(), transforms.Normalize(mean, std), Cutout(cutout),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.pixels[index].numpy())), self.labels[index]


def seed_shake_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def epoch_loader(dataset, batch_size, seed, epoch, num_workers=2, pin_memory=False):
    """Recreate worker streams and shuffle from the same explicit epoch seed."""
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                      num_workers=num_workers, worker_init_fn=seed_shake_worker,
                      persistent_workers=False, multiprocessing_context="spawn" if num_workers else None,
                      prefetch_factor=2 if num_workers else None, pin_memory=pin_memory, drop_last=False)


def permitted_pixels(root, rows):
    """Read/check the cache once in the parent, with a verified PIL fallback."""
    root = Path(root)
    if (root / "train_pixels.npy").is_file():
        return load_training_pixels(root, rows)
    permitted = {row["id"]: row for row in training_rows(root)}
    pixels = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    for index, row in enumerate(rows):
        if permitted.get(row["id"]) != row:
            raise ValueError("Requested image differs from permitted training data")
        with Image.open(image_path(root, "train_images", row["filename"])) as image:
            if image.mode != "RGB" or image.size != (32, 32):
                raise ValueError("Expected permitted RGB 32x32 training images")
            pixels[index] = np.asarray(image)
    return pixels


@torch.no_grad()
def evaluate_views(model, images, labels, batch_size=256):
    """Two raw forwards give raw and mirror metrics on the exact same IDs."""
    model.eval()
    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    raw_correct = mirror_correct = total = 0
    raw_loss = mirror_loss = 0.0
    try:
        for start in range(0, len(labels), batch_size):
            x, y = images[start:start+batch_size], labels[start:start+batch_size]
            raw = model.classify(x)
            flipped = model.classify(x.flip(-1))
            if raw.dtype != torch.float32 or flipped.dtype != torch.float32:
                raise ValueError("Validation must use FP32 logits without autocast")
            if raw.shape != (len(y), 10) or flipped.shape != raw.shape or not torch.isfinite(raw).all() or not torch.isfinite(flipped).all():
                raise RuntimeError("Non-finite or malformed validation logits")
            mirror = (raw + flipped) * .5
            raw_loss += float(nn.functional.cross_entropy(raw, y, reduction="sum"))
            mirror_loss += float(nn.functional.cross_entropy(mirror, y, reduction="sum"))
            raw_correct += int((raw.argmax(1) == y).sum())
            mirror_correct += int((mirror.argmax(1) == y).sum())
            total += len(y)
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = previous
    if not total:
        raise ValueError("Empty held-out validation split")
    return {"raw_val_accuracy": raw_correct / total, "raw_val_correct": raw_correct,
            "raw_val_loss": raw_loss / total, "val_accuracy": mirror_correct / total,
            "val_correct": mirror_correct, "val_loss": mirror_loss / total, "val_images": total}


def validate_config(c):
    t, a, r = c["training"], c["augmentation"], c["runtime"]
    if c["model"].get("architecture") != "shake_shake":
        raise ValueError("This trainer only supports scratch Shake-Shake")
    if (not all(math.isfinite(t[k]) for k in ("lr", "min_lr", "momentum", "weight_decay"))
            or t["epochs"] < 1 or t["batch_size"] < 1 or not 0 <= t["min_lr"] <= t["lr"]
            or t["lr"] <= 0 or not 0 <= t["warmup_epochs"] < t["epochs"]
            or not 0 <= t["momentum"] < 1 or t["weight_decay"] < 0
            or (t["nesterov"] and t["momentum"] <= 0)):
        raise ValueError("Invalid fixed SGD schedule")
    if (a["crop_padding"] != 4 or a["policy"] != "cifar10_autoaugment"
            or a["interpolation"] != "nearest" or a["padding_mode"] != "constant"
            or not 0 <= a["cutout_length"] <= 32):
        raise ValueError("Unsupported augmentation recipe")
    if r["num_workers"] < 0 or r["cpu_threads"] < 1 or r["validation_batch_size"] < 1:
        raise ValueError("Invalid worker, CPU thread or validation batch count")


def _jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def train(config, device_name="auto", resume=False, stop_after_epoch=None):
    root = Path(config["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(config, device_name, resume, stop_after_epoch)


def _train(config, device_name, resume, stop_after_epoch):
    c = copy.deepcopy(config)
    validate_config(c)
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    root, t, a, r = Path(c["run_dir"]), c["training"], c["augmentation"], c["runtime"]
    last_path = root / "last.pt"
    if not resume and any((root / name).exists() for name in ("config.json", "last.pt", "best.pt")):
        raise FileExistsError("Run exists; resume it or use a fresh directory")
    if resume and not last_path.is_file():
        raise FileNotFoundError("No committed epoch checkpoint to resume")
    split, train_rows, val_rows = load_split(c["data_dir"], c["split_file"])
    split_hash = file_hash(c["split_file"])
    if c.get("split_sha256") not in (None, split_hash):
        raise ValueError("The configured team split checksum differs")
    if split["classes"] != CLASSES:
        raise ValueError("Unexpected class mapping")
    pixels = permitted_pixels(c["data_dir"], train_rows + val_rows)
    pixel_hash = hashlib.sha256(memoryview(pixels).cast("B")).hexdigest()
    versions = dependency_versions()
    dependency_names = ("shake_train.py", "shake_shake.py", "models.py", "data.py", "pixel_cache.py", "train.py", "utils.py")
    identity = {"method": "scratch_shake_shake_autoaugment_cutout", "training_scope": "train_split",
                "seed": c["seed"], "model": c["model"], "training": t, "augmentation": a, "runtime": r,
                "split_sha256": split_hash, "train_csv_sha256": file_hash(Path(c["data_dir"]) / "train_labels.csv"),
                "ordered_split_pixels_sha256": pixel_hash,
                "source_hashes": {name: file_hash(Path(__file__).with_name(name)) for name in dependency_names},
                "software": {"torch": str(torch.__version__), **versions},
                "epoch_loader_seed": "seed + zero-based epoch; workers recreated each epoch",
                "selection_policy": "maximum FP32 mirror accuracy, ties lower mirror cross-entropy"}
    device = device_for(device_name)
    torch.set_num_threads(r["cpu_threads"])
    seed_all(c["seed"])
    torch.backends.cudnn.benchmark = r["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if device.type == "cuda" and r["amp"] and torch.cuda.is_bf16_supported() else None
    policy = precision_policy(dtype)
    model = build_model(c["model"]).to(device)
    if not hasattr(model, "classify") or not hasattr(model, "inference_description"):
        raise ValueError("Model must expose raw classify and intrinsic mirror inference")
    counts = parameter_counts(model)
    if counts["trainable"] > 5_000_000:
        raise ValueError("Final single-model parameter budget exceeded")
    train_pixels = torch.from_numpy(pixels[:len(train_rows)].copy())
    train_labels = torch.tensor([int(row["label"]) for row in train_rows], dtype=torch.long)
    if r["num_workers"]:
        train_pixels.share_memory_()
        train_labels.share_memory_()
    dataset = CachedShakeDataset(train_pixels, train_labels, split["mean"], split["std"],
                                 a["crop_padding"], a["cutout_length"])
    mean = torch.tensor(split["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(split["std"], device=device).view(1, 3, 1, 1)
    val_images = torch.from_numpy(pixels[len(train_rows):].copy()).to(device).permute(0, 3, 1, 2).float().div_(255)
    val_images = ((val_images - mean) / std).contiguous()
    val_labels = torch.tensor([int(row["label"]) for row in val_rows], device=device, dtype=torch.long)
    del pixels
    optimizer = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                                nesterov=t["nesterov"], weight_decay=t["weight_decay"])
    start, history, best_epoch, best_result, best_model = 0, [], 0, None, None
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Source, data, model, augmentation, schedule or runtime changed")
        if file_hash(root / "split.json") != split_hash:
            raise ValueError("The run's saved validation split changed")
        saved_config = read_json(root / "config.json")
        if any(saved_config.get(key) != c.get(key) for key in
               ("seed", "model", "training", "augmentation", "runtime", "split_sha256")):
            raise ValueError("The run's saved recipe changed")
        validate_resume_precision(saved, policy)
        if not 1 <= saved["epoch"] <= t["epochs"] or len(saved["history"]) != saved["epoch"]:
            raise ValueError("Invalid committed epoch/history")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, history = saved["epoch"], saved["history"]
        best_epoch, best_result, best_model = saved["best_epoch"], saved["best_result"], saved["best_model"]
        restore_rng(saved["rng"])
    else:
        write_json(root / "config.json", c)
        shutil.copyfile(c["split_file"], root / "split.json")
        write_json(root / "environment.json", {**versions, "torch": str(torch.__version__), "identity": identity,
                   "precision_policy": policy, "validation_precision": "FP32; TF32 disabled",
                   "cudnn_benchmark_training": r["cudnn_benchmark"], "cudnn_deterministic_training": False,
                   "parameter_counts": counts, "initialization": "from scratch; no pretrained state"})

    def sync_artifacts():
        # last.pt is authoritative. Rebuild exports after an interrupted write.
        selected = {"model": best_model, "model_config": c["model"], "epoch": best_epoch,
                    "parameter_counts": counts, "normalization": {k: split[k] for k in ("mean", "std")},
                    "classes": CLASSES, "identity": identity, "precision_policy": policy,
                    "training_scope": "train_split", "training_images": len(train_rows),
                    "validation": best_result, "smoke_test": False}
        save_checkpoint(root / "best.pt", selected)
        write_json(root / "metrics.json", history)
        _jsonl(root / "metrics.jsonl", history)
        summary = {"complete": len(history) == t["epochs"], "completed_epochs": len(history),
                   "planned_epochs": t["epochs"], "best_epoch": best_epoch, "validation": best_result,
                   "training_scope": "train_split", "training_images": len(train_rows),
                   "validation_images": len(val_rows), "parameter_counts": counts, "smoke_test": False,
                   "selection_policy": identity["selection_policy"],
                   "exceeds_97_5_on_fixed_5000": len(val_rows) == 5000 and best_result["val_correct"] >= 4876,
                   "target_note": "Strictly above97.5% needs at least4876/5000; no leaderboard claim."}
        write_json(root / "summary.json", summary)
        return summary

    if resume:
        sync_artifacts()
    print(f"Scratch Shake-Shake: {len(train_rows)} training/{len(val_rows)} validation; {counts['trainable']} parameters; {start}/{t['epochs']}", flush=True)
    end_epoch = t["epochs"] if stop_after_epoch is None else min(t["epochs"], stop_after_epoch)
    for epoch in range(start, end_epoch):
        model.train()
        lr = epoch_lr(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = epoch_loader(dataset, t["batch_size"], c["seed"], epoch, r["num_workers"], device.type == "cuda")
        sync(device)
        started = time.perf_counter()
        loss_sum, train_correct, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, dtype):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; previous committed epoch preserved")
            loss.backward()
            # Infinity performs no effective clipping; foreach fuses the finite
            # norm check instead of launching a separate kernel per parameter.
            nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            train_correct += int((logits.detach().argmax(1) == y).sum())
            seen += len(y)
        sync(device)
        if seen != len(train_rows):
            raise RuntimeError("Training epoch did not cover every permitted training ID")
        training_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        result = evaluate_views(model, val_images, val_labels, r["validation_batch_size"])
        sync(device)
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / seen,
               "train_accuracy": train_correct / seen, "train_images": seen,
               "training_seconds": training_seconds, "validation_seconds": time.perf_counter() - validation_started,
               **result}
        if best_result is None or (result["val_accuracy"], -result["val_loss"]) > (best_result["val_accuracy"], -best_result["val_loss"]):
            best_epoch, best_result = epoch + 1, result
            best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": {},
                        "epoch": epoch + 1, "history": history, "best_epoch": best_epoch,
                        "best_result": best_result, "best_model": best_model,
                        "identity": identity, "precision_policy": policy, "rng": rng_state()})
        report = sync_artifacts()
        print(row, flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            break
    return read_json(root / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    configuration = read_json(args.config)
    if args.run_dir:
        configuration["run_dir"] = args.run_dir
    print(train(configuration, args.device, args.resume, args.stop_after_epoch))
