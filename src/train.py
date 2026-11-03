"""Train one experiment; restart with --resume from the last completed epoch."""
import argparse
from contextlib import nullcontext
import copy
import csv
import fcntl
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageDataset, load_split, make_transform
from .models import build_model, parameter_counts
from .utils import (device_for, epoch_lr, file_hash, read_json, restore_rng, rng_state,
                    save_checkpoint, seed_all, seed_worker, sync, write_json)


def amp_context(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, dtype, limit=0):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for index, (x, y) in enumerate(loader):
        if limit and index >= limit:
            break
        x, y = x.to(device), y.to(device)
        with amp_context(device, dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        total += y.numel()
        total_loss += loss.item() * y.numel()
        correct += (logits.argmax(1) == y).sum().item()
    if total == 0:
        raise ValueError("Empty validation loader")
    return {"val_loss": total_loss/total, "val_accuracy": correct/total, "val_images": total}


def validate_config(c):
    t = c["training"]
    if t["epochs"] <= 0 or t["batch_size"] <= 0 or t["lr"] <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    if not 0 <= t["warmup_epochs"] < t["epochs"]:
        raise ValueError("warmup_epochs must be >= 0 and less than epochs")
    if not 0 <= t["min_lr"] <= t["lr"] or t["weight_decay"] < 0:
        raise ValueError("Invalid LR/weight decay")
    if not 0 <= t["label_smoothing"] < 1 or not 0 <= t["momentum"] < 1:
        raise ValueError("Invalid smoothing/momentum")
    if t["max_train_batches"] < 0 or t["max_val_batches"] < 0:
        raise ValueError("Batch limits must be nonnegative")
    if t["optimizer"] not in ("sgd", "adamw"):
        raise ValueError("optimizer must be sgd or adamw")
    if c["runtime"]["num_workers"] < 0 or c["runtime"]["cpu_threads"] < 1:
        raise ValueError("Invalid worker/thread count")


def train(config, resume=False, stop_after_epoch=None):
    c = copy.deepcopy(config)
    validate_config(c)
    run_dir = Path(c["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    # OS releases the advisory lock after process exit, including termination.
    with (run_dir/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process is using this run directory") from exc
        return _train_locked(c, run_dir, resume, stop_after_epoch)


def _train_locked(c, run_dir, resume, stop_after_epoch):
    t = c["training"]
    torch.set_num_threads(c["runtime"]["cpu_threads"])
    device = device_for(c["runtime"]["device"])
    seed_all(c["seed"])
    split, train_rows, val_rows = load_split(c["data_dir"], c["split_file"])
    identity = {"model": c["model"], "training": t, "seed": c["seed"],
                "split_sha256": file_hash(c["split_file"])}
    config_path = run_dir/"config.json"
    last_path = run_dir/"last.pt"
    if not resume and (config_path.exists() or last_path.exists()):
        raise FileExistsError("Run exists. Use --resume or choose a new --run-dir; no files were replaced.")
    if resume and not last_path.exists():
        raise FileNotFoundError("No last.pt to resume; choose a new run directory for a fresh run.")
    if not resume:
        write_json(config_path, c)
        write_json(run_dir/"split.json", split)
    dtype = None
    if device.type == "cuda" and c["runtime"]["amp"]:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model = build_model(c["model"]).to(device)
    counts = parameter_counts(model)
    train_ds = ImageDataset(c["data_dir"], train_rows, make_transform(split["mean"], split["std"], True))
    val_ds = ImageDataset(c["data_dir"], val_rows, make_transform(split["mean"], split["std"], False))
    train_generator = torch.Generator().manual_seed(c["seed"])
    val_generator = torch.Generator().manual_seed(c["seed"] + 1)
    loader_options = {"batch_size": t["batch_size"], "num_workers": c["runtime"]["num_workers"],
                      "pin_memory": device.type == "cuda", "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_ds, shuffle=True, generator=train_generator, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, generator=val_generator, **loader_options)
    if t["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                                    weight_decay=t["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    history, start, skipped, prior_peak = [], 0, 0, 0
    best_score, best_epoch = (-1.0, float("-inf")), 0
    training_seconds, run_seconds = 0.0, 0.0
    smoke = bool(t["max_train_batches"] or t["max_val_batches"])
    if resume:
        # Only load checkpoints generated by this project/team.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Resume config/split differs. Preserve settings or start a new run.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, history = state["epoch"], state["history"]
        best_score, best_epoch = tuple(state["best_score"]), state["best_epoch"]
        skipped, prior_peak = state["skipped_updates"], state["peak_cuda_bytes"]
        training_seconds, run_seconds = state["training_seconds"], state["run_seconds"]
        train_generator.set_state(state["train_generator"])
        val_generator.set_state(state["val_generator"])
        restore_rng(state["rng"])
        del state
    sources = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    runtime_record = {"device": str(device), "precision": str(dtype) if dtype else "float32",
                      "torch": str(torch.__version__), "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                      "parameter_counts": counts, "source_hashes": sources,
                      "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                      "smoke_test": smoke}
    if resume:
        runtime_path = run_dir/"resume_environments.json"
        environments = read_json(runtime_path) if runtime_path.exists() else []
        environments.append(runtime_record)
        write_json(runtime_path, environments)
    else:
        write_json(run_dir/"environment.json", runtime_record)
    print(json.dumps(runtime_record, indent=2), flush=True)
    print(f"Training {len(train_ds)} / validation {len(val_ds)}; resume epoch {start}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, t["epochs"]):
        sync(device)
        epoch_start = time.perf_counter()
        lr = epoch_lr(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum, seen, correct = 0.0, 0, 0
        for index, (x, y) in enumerate(train_loader):
            if t["max_train_batches"] and index >= t["max_train_batches"]:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, dtype):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; resume only after diagnosing the cause")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            loss_sum += loss.item()*y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.numel()
        sync(device)
        elapsed = time.perf_counter()-epoch_start
        training_seconds += elapsed
        result = evaluate(model, val_loader, device, dtype, t["max_val_batches"])
        row = {"epoch": epoch+1, "lr": lr, "train_loss": loss_sum/seen,
               "train_accuracy": correct/seen, "train_images": seen,
               "training_seconds": elapsed, "images_per_second": seen/elapsed, **result}
        history.append(row)
        score = (result["val_accuracy"], -result["val_loss"])
        if score > best_score:
            best_score, best_epoch = score, epoch+1
            save_checkpoint(run_dir/"best.pt", {
                "model": model.state_dict(), "model_config": c["model"], "epoch": best_epoch,
                "normalization": {"mean": split["mean"], "std": split["std"]},
                "parameter_counts": counts, "validation": result, "identity": identity,
                "smoke_test": smoke, "classes": split["classes"]})
        sync(device)
        run_seconds += time.perf_counter()-epoch_start
        prior_peak = max(prior_peak, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch+1, "identity": identity,
                 "rng": rng_state(), "train_generator": train_generator.get_state(),
                 "val_generator": val_generator.get_state(), "history": history,
                 "best_score": best_score, "best_epoch": best_epoch, "skipped_updates": skipped,
                 "peak_cuda_bytes": prior_peak, "training_seconds": training_seconds,
                 "run_seconds": run_seconds}
        save_checkpoint(last_path, state)
        write_json(run_dir/"metrics.json", history)
        print(f"Epoch {epoch+1}/{t['epochs']} | train {loss_sum/seen:.4f} | "
              f"val {result['val_loss']:.4f} | val acc {result['val_accuracy']:.2%} | "
              f"{elapsed:.1f}s train | best epoch {best_epoch}", flush=True)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            break
    if not history:
        raise RuntimeError("No completed epochs")
    # Rebuild exports from the checkpoint history, even if a previous export was interrupted.
    write_json(run_dir/"metrics.json", history)
    with (run_dir/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"completed_epochs": history[-1]["epoch"], "planned_epochs": t["epochs"],
               "complete": history[-1]["epoch"] == t["epochs"], "smoke_test": smoke,
               "best_val_accuracy": best_score[0], "best_val_loss_at_best_accuracy": -best_score[1],
               "best_epoch": best_epoch, "parameter_counts": counts,
               "training_seconds": training_seconds, "compute_seconds_with_validation": run_seconds,
               "images_per_second": sum(h["train_images"] for h in history)/training_seconds,
               "peak_cuda_allocated_GiB": prior_peak/2**30 if device.type == "cuda" else None,
               "amp_skipped_updates": skipped}
    write_json(run_dir/"summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir")
    p.add_argument("--split-file")
    p.add_argument("--run-dir")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-epoch", type=int, help="Stop cleanly after this absolute epoch; preserve full LR schedule")
    a = p.parse_args()
    config = read_json(a.config)
    for key in ["data_dir", "split_file", "run_dir"]:
        if getattr(a, key) is not None:
            config[key] = getattr(a, key)
    if a.device:
        config["runtime"]["device"] = a.device
    print(json.dumps(train(config, a.resume, a.stop_after_epoch), indent=2))


if __name__ == "__main__":
    main()
