"""Train one experiment; restart with --resume from the last completed epoch."""
import argparse
from contextlib import nullcontext
import copy
import csv
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentation import mixup_batch, soft_target_cross_entropy, validate_augmentation
from .augmix import AUGMIX_VERSION, AugMixTransform, augmix_loss, validate_augmix
from .data import ImageDataset, load_split, make_transform
from .models import build_model, parameter_counts
from .optimization import EMA, optimization_step, validate_optimization
from .utils import (dependency_versions, device_for, epoch_lr, file_hash, precision_policy,
                    read_json, restore_rng, rng_state, run_lock, save_checkpoint, seed_all,
                    seed_worker, sync, validate_resume_precision, write_json)


def amp_context(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if dtype is not None else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, dtype, limit=0, require_finite_logits=False):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    for index, (x, y) in enumerate(loader):
        if limit and index >= limit:
            break
        x, y = x.to(device), y.to(device)
        with amp_context(device, dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
        if require_finite_logits and not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite validation logits")
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
    validate_augmentation(t)
    validate_augmix(t)
    optimization = validate_optimization(t)
    if optimization is not None:
        if t["optimizer"] != "sgd" or t.get("augmix"):
            raise ValueError("Controlled optimization requires SGD and the single-view Color recipe")
        if not c["runtime"].get("strict_source_resume", False):
            raise ValueError("Controlled optimization requires strict_source_resume")
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
    if not isinstance(c["runtime"].get("strict_source_resume", False), bool):
        raise ValueError("strict_source_resume must be a boolean")


def optimization_scheduler_state(epoch, training):
    """Explicit identity for the existing stateless, epoch-boundary LR schedule."""
    return {"version": 1, "implementation": "epoch_lr_v1", "completed_epochs": epoch,
            "planned_epochs": training["epochs"],
            "last_lr": epoch_lr(epoch-1, training) if epoch else None}


def validate_optimization_resume(state, training, settings):
    epoch = state["epoch"]
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or not 1 <= epoch <= training["epochs"]
            or len(state["history"]) != epoch
            or [row["epoch"] for row in state["history"]] != list(range(1, epoch+1))):
        raise ValueError("Optimization checkpoint history/epoch integrity failure")
    if state.get("scheduler") != optimization_scheduler_state(epoch, training):
        raise ValueError("Optimization scheduler state differs from the frozen epoch schedule")
    for index, row in enumerate(state["history"]):
        if row["lr"] != epoch_lr(index, training):
            raise ValueError("Optimization history learning rates differ from the frozen schedule")
        if not all(math.isfinite(row[key]) for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy")):
            raise ValueError("Non-finite optimization checkpoint history")
        def finite_history(value):
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError("Non-finite optimization checkpoint history")
            if isinstance(value, dict):
                for child in value.values():
                    finite_history(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite_history(child)
        finite_history(row)
    steps = state.get("optimization_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps != sum(row["optimizer_steps"] for row in state["history"]):
        raise ValueError("Optimization successful-update counter differs from history")
    if any(group["lr"] != state["scheduler"]["last_lr"] for group in state["optimizer"]["param_groups"]):
        raise ValueError("Optimizer LR differs from checkpoint scheduler state")
    for group in state["optimizer"]["param_groups"]:
        if (group["momentum"] != training["momentum"] or group["weight_decay"] != training["weight_decay"]
                or group.get("dampening", 0) != 0 or group.get("nesterov", False)
                or group.get("maximize", False)):
            raise ValueError("Optimizer settings differ from the frozen SGD configuration")
    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite optimization checkpoint tensor")
        elif isinstance(value, dict):
            for child in value.values():
                finite_tensors(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                finite_tensors(child)
    for key in ("model", "optimizer", "ema"):
        finite_tensors(state.get(key))
    if settings["ema"]["enabled"]:
        if not isinstance(state.get("ema"), dict) or state["ema"].get("updates") != steps:
            raise ValueError("EMA update counter differs from successful optimizer updates")
    elif state.get("ema") is not None:
        raise ValueError("Unexpected EMA state for an EMA-disabled experiment")


def train(config, resume=False, stop_after_epoch=None):
    c = copy.deepcopy(config)
    validate_config(c)
    if stop_after_epoch is not None and stop_after_epoch <= 0:
        raise ValueError("stop_after_epoch must be positive")
    run_dir = Path(c["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    controlled = validate_optimization(c["training"]) is not None
    backend_before = ({"precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic} if controlled else None)
    try:
        # OS releases the advisory lock after process exit, including termination.
        with run_lock(run_dir/".run.lock"):
            return _train_locked(c, run_dir, resume, stop_after_epoch)
    finally:
        if backend_before is not None:
            torch.set_float32_matmul_precision(backend_before["precision"])
            torch.backends.cuda.matmul.allow_tf32 = backend_before["matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = backend_before["cudnn_tf32"]
            torch.backends.cudnn.benchmark = backend_before["benchmark"]
            torch.backends.cudnn.deterministic = backend_before["deterministic"]


def _train_locked(c, run_dir, resume, stop_after_epoch):
    t = c["training"]
    use_augmix = validate_augmix(t)
    optimization = validate_optimization(t)
    torch.set_num_threads(c["runtime"]["cpu_threads"])
    device = device_for(c["runtime"]["device"])
    seed_all(c["seed"])
    if optimization is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    split, train_rows, val_rows = load_split(c["data_dir"], c["split_file"])
    sources = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    identity = {"model": c["model"], "training": t, "seed": c["seed"],
                "split_sha256": file_hash(c["split_file"])}
    # Opt in for new experiments without changing legacy checkpoint identities.
    if c["runtime"].get("strict_source_resume", False):
        identity["source_hashes"] = sources
    if use_augmix:
        if not c["runtime"].get("strict_source_resume", False):
            raise ValueError("AugMix requires strict_source_resume for reproducible multi-view training")
        identity["augmix_version"] = AUGMIX_VERSION
        identity["loader_runtime"] = {key: c["runtime"][key]
                                      for key in ("num_workers", "cpu_threads", "amp")}
    if optimization is not None:
        identity["optimization_schema"] = 1
        identity["loader_runtime"] = {key: c["runtime"][key]
                                      for key in ("num_workers", "cpu_threads", "amp")}
        identity["backend_policy"] = {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}
        import numpy as np
        identity["software_policy"] = {"torch": str(torch.__version__),
            "numpy_import_version": np.__version__, **dependency_versions()}
    config_path = run_dir/"config.json"
    last_path = run_dir/"last.pt"
    if not resume and (config_path.exists() or last_path.exists()):
        raise FileExistsError("Run exists. Use --resume or choose a new --run-dir; no files were replaced.")
    if resume and not last_path.exists():
        raise FileNotFoundError("No last.pt to resume; choose a new run directory for a fresh run.")
    if not resume:
        write_json(config_path, c)
        shutil.copyfile(c["split_file"], run_dir/"split.json")
    dtype = None
    if device.type == "cuda" and c["runtime"]["amp"]:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if use_augmix and device.type == "cuda" and dtype != torch.bfloat16:
        raise ValueError("This AugMix CUDA experiment requires native BF16 AMP")
    if optimization is not None and device.type == "cuda" and dtype != torch.bfloat16:
        raise ValueError("Controlled optimization CUDA experiments require native BF16 AMP")
    policy = precision_policy(dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    model = build_model(c["model"]).to(device)
    counts = parameter_counts(model)
    train_transform = (AugMixTransform(split["mean"], split["std"], t["augmix"]) if use_augmix else
                       make_transform(split["mean"], split["std"], True, t.get("cutout_length", 0),
                                      t.get("cutout_probability", 1.0), color_jitter=t.get("color_jitter"),
                                      random_grayscale_probability=t.get("random_grayscale_probability", 0.0)))
    train_ds = ImageDataset(c["data_dir"], train_rows, train_transform)
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
    ema = (EMA(model, optimization["ema"]["decay"], optimization["ema"]["warmup"])
           if optimization is not None and optimization["ema"]["enabled"] else None)
    # deepcopy consumes no RNG and never feeds EMA weights back into online optimization.
    ema_model = copy.deepcopy(model).requires_grad_(False).eval() if ema is not None else None
    optimization_steps = 0
    ema_best_score, ema_best_epoch = (-1.0, float("-inf")), 0
    history, start, skipped, prior_peak = [], 0, 0, 0
    best_score, best_epoch = (-1.0, float("-inf")), 0
    training_seconds, run_seconds = 0.0, 0.0
    smoke = bool(t["max_train_batches"] or t["max_val_batches"])
    if resume:
        # Only load checkpoints generated by this project/team.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        saved_sources = state["identity"].get("source_hashes")
        if saved_sources is not None and saved_sources != sources:
            raise ValueError("Resume source hashes differ. Restore the recorded source or start a new run.")
        if state["identity"] != identity:
            raise ValueError("Resume config/split differs. Preserve settings or start a new run.")
        legacy_environment = None
        if "precision_policy" not in state:
            environments_path = run_dir/"resume_environments.json"
            environments = read_json(environments_path) if environments_path.exists() else []
            environment_path = run_dir/"environment.json"
            legacy_environment = (environments[-1] if environments else
                                  read_json(environment_path) if environment_path.exists() else None)
        validate_resume_precision(state, policy, legacy_environment)
        if optimization is not None:
            validate_optimization_resume(state, t, optimization)
            optimization_steps = state["optimization_steps"]
            if ema is not None:
                ema.load_state_dict(state["ema"])
                ema_best_score = tuple(state["ema_best_score"])
                ema_best_epoch = state["ema_best_epoch"]
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
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    runtime_record = {"device": str(device), "precision": str(dtype) if dtype else "float32",
                      "precision_policy": policy, **dependency_versions(),
                      "torch": str(torch.__version__), "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                      "parameter_counts": counts, "source_hashes": sources,
                      "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                      "smoke_test": smoke}
    if optimization is not None:
        import numpy as np
        runtime_record.update(optimization=optimization, backend_policy=identity["backend_policy"],
                              numpy_import_version=np.__version__,
                              ema_evaluation="separate weight variant; copied online BN buffers, not recalibrated")
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
    end = min(t["epochs"], stop_after_epoch) if stop_after_epoch is not None else t["epochs"]
    for epoch in range(start, end):
        sync(device)
        epoch_start = time.perf_counter()
        lr = epoch_lr(epoch, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum, seen, correct = 0.0, 0, 0
        ce_sum, js_sum, gradient_checks, epoch_steps = 0.0, 0.0, 0, 0
        base_loss_sum = perturbed_loss_sum = base_norm_sum = perturbed_norm_sum = 0.0
        perturbation_max = adaptive_perturbation_max = 0.0
        skipped_at_start = skipped
        for index, (x, y) in enumerate(train_loader):
            if t["max_train_batches"] and index >= t["max_train_batches"]:
                break
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if optimization is not None:
                x = x.to(device)
                # Augmentation/MixUp happens ONCE. Both optimizer passes reuse these tensors.
                x, targets = mixup_batch(x, y, t.get("mixup_alpha", 0.0),
                    t.get("mixup_probability", 1.0), c["model"].get("num_classes", 10))
                measured = optimization_step(model, optimizer, x, targets, optimization,
                    dtype=dtype, label_smoothing=t["label_smoothing"])
                logits = measured["logits"]
                loss = torch.tensor(measured["base_loss"], device=device)
                base_loss_sum += measured["base_loss"] * y.numel()
                base_norm_sum += measured["base_gradient_norm"]
                if measured["perturbed_loss"] is not None:
                    perturbed_loss_sum += measured["perturbed_loss"] * y.numel()
                    perturbed_norm_sum += measured["perturbed_gradient_norm"]
                perturbation_max = max(perturbation_max, measured["perturbation_norm"])
                adaptive_perturbation_max = max(adaptive_perturbation_max, measured["adaptive_perturbation_norm"])
                gradient_checks += measured["gradient_checks"]
                epoch_steps += measured["optimizer_steps"]
                optimization_steps += measured["optimizer_steps"]
                if ema is not None:
                    ema.update(model)
            elif use_augmix:
                # One shared-weight forward/BN batch and one update per batch of originals.
                views = torch.cat(x, dim=0).to(device)
                with amp_context(device, dtype):
                    all_logits = model(views)
                logits, aug1, aug2 = all_logits.split(y.numel())
                loss, ce, js = augmix_loss(logits, aug1, aug2, y, t["augmix"]["js_coefficient"])
                ce_sum += ce.item() * y.numel()
                js_sum += js.item() * y.numel()
                targets = y
            else:
                x = x.to(device)
                x, targets = mixup_batch(x, y, t.get("mixup_alpha", 0.0),
                                         t.get("mixup_probability", 1.0), c["model"].get("num_classes", 10))
                with amp_context(device, dtype):
                    logits = model(x)
                    loss = (soft_target_cross_entropy(logits, targets, t["label_smoothing"])
                            if targets.ndim == 2 else criterion(logits, targets))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; resume only after diagnosing the cause")
            if optimization is None:
                scaler.scale(loss).backward()
            if use_augmix:
                # Reduce on device and synchronize once; BF16 has no active loss scaler.
                finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                                if p.grad is not None]).all()
                if not finite_gradients.item():
                    raise RuntimeError("Non-finite training gradients; optimizer update aborted")
                gradient_checks += 1
            if optimization is None:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < old_scale)
                epoch_steps += 1
            loss_sum += loss.item()*y.numel()
            predicted = logits.argmax(1)
            correct += (targets.gather(1, predicted[:, None]).sum().item() if targets.ndim == 2
                        else (predicted == y).sum().item())
            seen += y.numel()
        sync(device)
        elapsed = time.perf_counter()-epoch_start
        training_seconds += elapsed
        validation_start = time.perf_counter()
        result = evaluate(model, val_loader, device, dtype, t["max_val_batches"],
                          require_finite_logits=use_augmix or optimization is not None)
        sync(device)
        validation_seconds = time.perf_counter()-validation_start
        ema_result = None
        ema_validation_seconds = 0.0
        if ema is not None:
            ema.copy_to(ema_model)
            ema_start = time.perf_counter()
            ema_result = evaluate(ema_model, val_loader, device, dtype, t["max_val_batches"], require_finite_logits=True)
            sync(device)
            ema_validation_seconds = time.perf_counter()-ema_start
        row = {"epoch": epoch+1, "lr": lr, "train_loss": loss_sum/seen,
               "train_accuracy": correct/seen, "train_images": seen,
               "training_seconds": elapsed, "validation_seconds": validation_seconds,
               "images_per_second": seen/elapsed, **result}
        if use_augmix:
            row.update(train_ce=ce_sum/seen, train_js=js_sum/seen,
                       train_model_views=3*seen, model_views_per_second=3*seen/elapsed,
                       finite_gradient_checks=gradient_checks, optimizer_steps=epoch_steps,
                       amp_skipped_updates=skipped-skipped_at_start,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0)
        if optimization is not None:
            two_pass = optimization["method"] != "sgd"
            row.update(train_base_loss=base_loss_sum/seen,
                train_perturbed_loss=perturbed_loss_sum/seen if two_pass else None,
                base_gradient_norm_mean=base_norm_sum/epoch_steps,
                perturbed_gradient_norm_mean=perturbed_norm_sum/epoch_steps if two_pass else None,
                perturbation_norm_max=perturbation_max, adaptive_perturbation_norm_max=adaptive_perturbation_max,
                finite_gradient_checks=gradient_checks, optimizer_steps=epoch_steps,
                amp_skipped_updates=skipped-skipped_at_start,
                train_model_views=(2 if two_pass else 1)*seen,
                model_views_per_second=(2 if two_pass else 1)*seen/elapsed,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0)
        if ema_result is not None:
            row.update(ema_val_loss_uncalibrated=ema_result["val_loss"],
                ema_val_accuracy_uncalibrated=ema_result["val_accuracy"],
                ema_validation_seconds=ema_validation_seconds, ema_updates=ema.updates)
        history.append(row)
        score = (result["val_accuracy"], -result["val_loss"])
        if score > best_score:
            best_score, best_epoch = score, epoch+1
            save_checkpoint(run_dir/"best.pt", {
                "model": model.state_dict(), "model_config": c["model"], "epoch": best_epoch,
                "normalization": {"mean": split["mean"], "std": split["std"]},
                "parameter_counts": counts, "validation": result, "identity": identity,
                "precision_policy": policy,
                "smoke_test": smoke, "classes": split["classes"]})
        if ema_result is not None:
            ema_score = (ema_result["val_accuracy"], -ema_result["val_loss"])
            improved_ema = ema_score > ema_best_score
            if improved_ema:
                ema_best_score, ema_best_epoch = ema_score, epoch+1
            ema_export = {"model": {k:v.detach().cpu() for k,v in ema_model.state_dict().items()},
                "model_config": c["model"], "epoch": epoch+1,
                "normalization": {"mean": split["mean"], "std": split["std"]},
                "parameter_counts": counts, "validation": ema_result, "identity": identity,
                "precision_policy": policy, "smoke_test": smoke, "classes": split["classes"],
                "weight_variant": "EMA; online BN buffers copied; no BN recalibration", "ema_updates": ema.updates}
            save_checkpoint(run_dir/"ema_last.pt", ema_export)
            if improved_ema:
                save_checkpoint(run_dir/"ema_best.pt", ema_export)
        sync(device)
        run_seconds += time.perf_counter()-epoch_start
        prior_peak = max(prior_peak, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "precision_policy": policy,
                 "scaler": scaler.state_dict(), "epoch": epoch+1, "identity": identity,
                 "rng": rng_state(), "train_generator": train_generator.get_state(),
                 "val_generator": val_generator.get_state(), "history": history,
                 "best_score": best_score, "best_epoch": best_epoch, "skipped_updates": skipped,
                 "peak_cuda_bytes": prior_peak, "training_seconds": training_seconds,
                 "run_seconds": run_seconds}
        if optimization is not None:
            state.update(scheduler=optimization_scheduler_state(epoch+1, t),
                optimization_steps=optimization_steps, ema=ema.state_dict() if ema is not None else None,
                ema_best_score=ema_best_score, ema_best_epoch=ema_best_epoch)
        save_checkpoint(last_path, state)
        write_json(run_dir/"metrics.json", history)
        print(f"Epoch {epoch+1}/{t['epochs']} | train {loss_sum/seen:.4f} | "
              f"val {result['val_loss']:.4f} | val acc {result['val_accuracy']:.2%} | "
              f"{elapsed:.1f}s train | {validation_seconds:.1f}s val | best epoch {best_epoch}", flush=True)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            break
    if not history:
        raise RuntimeError("No completed epochs")
    # Rebuild exports from the checkpoint history, even if a previous export was interrupted.
    write_json(run_dir/"metrics.json", history)
    with (run_dir/"metrics.csv").open("w", newline="") as f:
        # Older checkpoint rows have no separate validation timer.
        fields = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
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
    if use_augmix:
        summary.update(augmix_version=AUGMIX_VERSION,
                       model_views_per_second=sum(h["train_model_views"] for h in history)/training_seconds,
                       finite_gradient_checks=sum(h["finite_gradient_checks"] for h in history),
                       optimizer_steps=sum(h["optimizer_steps"] for h in history))
    if optimization is not None:
        summary.update(optimization=optimization, optimizer_steps=optimization_steps,
            finite_gradient_checks=sum(h["finite_gradient_checks"] for h in history),
            model_views_per_second=sum(h["train_model_views"] for h in history)/training_seconds,
            ema_updates=ema.updates if ema is not None else 0,
            ema_best_val_accuracy_uncalibrated=ema_best_score[0] if ema is not None else None,
            ema_best_epoch_uncalibrated=ema_best_epoch if ema is not None else None,
            ema_validation_seconds=sum(h.get("ema_validation_seconds", 0) for h in history))
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
