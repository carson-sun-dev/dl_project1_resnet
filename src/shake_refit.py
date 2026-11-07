"""Fixed all-permitted-data Shake-Shake refits; no held-out score exists here.

Scratch600 repeats the selected architecture/recipe on all 50k images.
Warmstart120 continues the verified own 45k checkpoint with fresh SGD.
Both retain the original 45k normalization, so dataset size is the only
preprocessing change and warmstart normalization remains compatible.
The fixed final epoch is exported; no validation or hidden-data selection.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import shutil
import time

import torch
from filelock import FileLock
from torch import nn

from .data import CLASSES, load_split
from .models import build_model, parameter_counts
from .pixel_cache import training_rows
from .shake_train import CachedShakeDataset, epoch_loader, permitted_pixels
from .train import amp_context
from .utils import (dependency_versions, device_for, epoch_lr, file_hash, precision_policy,
                    read_json, restore_rng, rng_state, save_checkpoint, seed_all,
                    sync, validate_resume_precision, write_json)


SOURCE_SHA256 = "60aa42297459d9681e3732607e4de7d007d93ab3c4f67852c4a27520ca3505a3"
SPLIT_SHA256 = "2fc944cf610ada2c12bd22adb33e93f460ac0611ece0c3991f3d3c329581751d"
SOURCE_EPOCHS = 600
SOURCE_COUNTS = (45000, 5000)
SCHEDULES = {"scratch": {"epochs": 600, "lr": .2, "warmup_epochs": 5},
             "warmstart": {"epochs": 120, "lr": .02, "warmup_epochs": 3}}


def source_selection(source_run, data_dir):
    root = Path(source_run)
    path = root / "best.pt"
    if file_hash(path) != SOURCE_SHA256:
        raise ValueError("Unrecognized own Shake-Shake source checkpoint")
    source = torch.load(path, map_location="cpu", weights_only=True)
    config, report = read_json(root / "config.json"), read_json(root / "summary.json")
    verified = read_json(root / "validation" / "metrics.json")
    split, train_rows, val_rows = load_split(data_dir, root / "split.json")
    ident = source["identity"]
    split_hash = file_hash(root / "split.json")
    if (split_hash != SPLIT_SHA256 or ident["split_sha256"] != split_hash
            or (len(train_rows), len(val_rows)) != SOURCE_COUNTS
            or not report["complete"] or report["completed_epochs"] != SOURCE_EPOCHS
            or config["training"]["epochs"] != SOURCE_EPOCHS or ident["training"] != config["training"]
            or ident["seed"] != config["seed"] or ident["augmentation"] != config["augmentation"]
            or ident["runtime"] != config["runtime"]
            or source["epoch"] != report["best_epoch"] or not 1 <= source["epoch"] <= SOURCE_EPOCHS
            or source.get("training_scope") != "train_split" or ident.get("training_scope") != "train_split"
            or report.get("training_scope") != "train_split" or source.get("training_images") != len(train_rows)
            or source.get("smoke_test", True) or report.get("smoke_test", True)
            or source["classes"] != CLASSES or split["classes"] != CLASSES
            or source["normalization"] != {k: split[k] for k in ("mean", "std")}
            or source["model_config"] != config["model"] or ident["model"] != config["model"]
            or config["model"].get("architecture") != "shake_shake"
            or ident["train_csv_sha256"] != file_hash(Path(data_dir) / "train_labels.csv")
            or verified["checkpoint_sha256"] != SOURCE_SHA256 or verified["checkpoint_epoch"] != source["epoch"]
            or verified["split_sha256"] != split_hash or verified["images"] != len(val_rows)
            or verified["precision"] != "float32" or verified["smoke_test"]
            or verified["parameter_counts"] != source["parameter_counts"]):
        raise ValueError("Expected completed verified own Shake source on its original disjoint split")
    rows = training_rows(data_dir)
    if len(rows) != sum(SOURCE_COUNTS) or {r["id"]: r for r in rows} != {r["id"]: r for r in train_rows + val_rows}:
        raise ValueError("Full-data rows differ from the original permitted training/validation union")
    if file_hash(path) != SOURCE_SHA256:
        raise RuntimeError("Source changed while loading")
    return source, config, split, rows


def configuration(mode, run_dir, source_run="runs/shake_aa_s2026", data_dir="data"):
    if mode not in SCHEDULES:
        raise ValueError("Expected scratch or warmstart mode")
    source, parent, split, rows = source_selection(source_run, data_dir)
    t = copy.deepcopy(parent["training"])
    t.update(SCHEDULES[mode], min_lr=0.)
    c = {"mode": mode, "seed": parent["seed"], "model": parent["model"], "training": t,
         "augmentation": parent["augmentation"], "runtime": parent["runtime"],
         "run_dir": str(run_dir), "source_run": str(source_run), "data_dir": str(data_dir),
         "training_scope": "full_data", "normalization": source["normalization"]}
    return c, source, split, rows


def train(mode="scratch", run_dir=None, source_run="runs/shake_aa_s2026", data_dir="data",
          device_name="cuda", resume=False, stop_after_epoch=None, epochs=None):
    if stop_after_epoch is not None and stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")
    run_dir = run_dir or f"runs/shake_{mode}_full_s2026"
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".run.lock"), timeout=0):
        c, source, split, rows = configuration(mode, run_dir, source_run, data_dir)
        if epochs is not None:
            if mode != "scratch" or not 6 <= epochs <= 1000:
                raise ValueError("Measured scratch schedule must be 6..1000 epochs")
            c["training"]["epochs"] = epochs
        return _train(c, source, split, rows, device_name, resume, stop_after_epoch)


def _train(c, source, split, rows, device_name, resume, stop_after_epoch):
    root, t, a, r = Path(c["run_dir"]), c["training"], c["augmentation"], c["runtime"]
    last_path = root / "last.pt"
    if not resume and any((root / n).exists() for n in ("config.json", "last.pt", "final.pt", "summary.json")):
        raise FileExistsError("Existing refit artifacts; explicitly resume the same recipe")
    if resume and not last_path.exists():
        raise FileNotFoundError("No committed epoch available for resume")
    pixels = permitted_pixels(c["data_dir"], rows)
    pixel_hash = hashlib.sha256(memoryview(pixels).cast("B")).hexdigest()
    dependencies = ("shake_refit.py", "shake_train.py", "shake_shake.py", "models.py",
                    "data.py", "pixel_cache.py", "train.py", "utils.py")
    identity = {"method": "full_data_shake_shake", "mode": c["mode"], "training_scope": "full_data",
        "seed": c["seed"], "model": c["model"], "training": t, "augmentation": a, "runtime": r,
        "selection_checkpoint_sha256": SOURCE_SHA256,
        "initial_checkpoint_sha256": SOURCE_SHA256 if c["mode"] == "warmstart" else None,
        "selection_split_sha256": SPLIT_SHA256,
        "selection_validation_sha256": file_hash(Path(c["source_run"]) / "validation" / "metrics.json"),
        "train_csv_sha256": file_hash(Path(c["data_dir"]) / "train_labels.csv"),
        "ordered_full_training_pixels_sha256": pixel_hash, "training_images": len(rows),
        "normalization": c["normalization"], "normalization_policy": "retain original45k normalization; no recomputation",
        "selection_policy": "predeclared final epoch only; no held-out validation after full-data training",
        "source_hashes": {n: file_hash(Path(__file__).with_name(n)) for n in dependencies},
        "software": {"torch": str(torch.__version__), **dependency_versions()}}
    device = device_for(device_name)
    torch.set_num_threads(r["cpu_threads"])
    seed_all(c["seed"])
    torch.backends.cudnn.benchmark = r["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if device.type == "cuda" and r["amp"] and torch.cuda.is_bf16_supported() else None
    policy = precision_policy(dtype)
    model = build_model(c["model"]).to(device)
    if c["mode"] == "warmstart":
        model.load_state_dict(source["model"])
    counts = parameter_counts(model)
    if counts != source["parameter_counts"] or counts["trainable"] > 5_000_000:
        raise ValueError("Full-data model count differs from source or exceeds budget")
    initial_model_sha = hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    px = torch.from_numpy(pixels)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    if r["num_workers"]:
        px.share_memory_(); labels.share_memory_()
    dataset = CachedShakeDataset(px, labels, split["mean"], split["std"], a["crop_padding"], a["cutout_length"])
    del pixels
    # Fresh SGD for both modes; never inherit optimizer momentum from selection.
    optimizer = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                                nesterov=t["nesterov"], weight_decay=t["weight_decay"])
    history, start = [], 0
    if resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or read_json(root / "config.json") != c:
            raise ValueError("Refit source, data, configuration, schedule or runtime changed")
        if file_hash(root / "selection_split.json") != SPLIT_SHA256:
            raise ValueError("Saved selection split changed")
        validate_resume_precision(saved, policy)
        if not 1 <= saved["epoch"] <= t["epochs"] or saved["epoch"] != len(saved["history"]):
            raise ValueError("Invalid committed refit epoch/history")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start = saved["history"], saved["epoch"]
        initial_model_sha = saved["initial_model_state_sha256"]
        restore_rng(saved["rng"])
    else:
        write_json(root / "config.json", c)
        shutil.copyfile(Path(c["source_run"]) / "split.json", root / "selection_split.json")
        write_json(root / "environment.json", {"identity": identity, "device": str(device),
            "precision_policy": policy, "parameter_counts": counts, "optimizer_initialization": "fresh SGD",
            "initial_model_state_sha256": initial_model_sha, "validation": None})

    def artifacts():
        complete = len(history) == t["epochs"]
        report = {"complete": complete, "completed_epochs": len(history), "planned_epochs": t["epochs"],
            "training_scope": "full_data", "training_images": len(rows), "validation": None,
            "validation_images": 0, "parameter_counts": counts, "mode": c["mode"],
            "selection_checkpoint_sha256": SOURCE_SHA256, "identity": identity, "smoke_test": False,
            "initial_model_state_sha256": initial_model_sha}
        write_json(root / "metrics.json", history)
        if complete:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
                raise RuntimeError("Non-finite final weights; no final artifact exported")
            save_checkpoint(root / "final.pt", {"model": state, "model_config": c["model"],
                "epoch": t["epochs"], "parameter_counts": counts, "normalization": c["normalization"],
                "classes": CLASSES, "identity": identity, "precision_policy": policy,
                "training_scope": "full_data", "training_images": len(rows), "validation": None,
                "smoke_test": False, "complete": True})
            report["final_checkpoint_sha256"] = file_hash(root / "final.pt")
        elif (root / "final.pt").exists():
            raise ValueError("Unexpected final artifact before fixed schedule completion")
        write_json(root / "summary.json", report)
        return report

    if resume:
        artifacts()
    end = t["epochs"] if stop_after_epoch is None else min(t["epochs"], stop_after_epoch)
    for epoch in range(start, end):
        model.train()
        lr = epoch_lr(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = epoch_loader(dataset, t["batch_size"], c["seed"], epoch, r["num_workers"], device.type == "cuda")
        sync(device); started = time.perf_counter()
        loss_sum, correct, seen = 0., 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, dtype):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite full-data training loss; last committed epoch preserved")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True, foreach=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.detach().argmax(1) == y).sum()); seen += len(y)
        sync(device)
        if seen != len(rows):
            raise RuntimeError("Incomplete full-data epoch")
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / seen,
               "train_accuracy": correct / seen, "train_images": seen,
               "training_seconds": time.perf_counter() - started}
        history.append(row)
        save_checkpoint(last_path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "history": history, "identity": identity, "precision_policy": policy,
            "rng": rng_state(), "initial_model_state_sha256": initial_model_sha, "scaler": {}})
        artifacts()
        print(row, flush=True)
    return read_json(root / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["scratch", "warmstart"], default="scratch")
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--epochs", type=int, help="Fixed measured scratch schedule, set before training")
    args = parser.parse_args()
    print(train(args.mode, args.run_dir, device_name=args.device, resume=args.resume,
                stop_after_epoch=args.stop_after_epoch, epochs=args.epochs))
