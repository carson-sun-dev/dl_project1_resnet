"""Conditional scratch refit on all permitted training images, with NO holdout.

Run only after architecture/recipe/inference selection is frozen using prior
legitimate experiments. The original 45k/5k manifest supplies normalization and
provenance, not evaluation: its former validation images now belong to training.
There is no validation loader, checkpoint ranking, hidden-data reader, download,
prediction export, or automatic experiment launch in this module.

Original project implementation; uses the existing SGD/SAM/ASAM/FP32-loss/EMA helpers.
Only trusted team-created resume checkpoints may be deserialized.
"""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
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
from torch.utils.data import DataLoader

from .augmentation import mixup_batch
from .data import CLASSES, ImageDataset, image_path, load_split, make_transform, read_csv
from .models import build_model, parameter_counts
from .optimization import EMA, optimization_step, validate_optimization
from .train import optimization_scheduler_state, validate_config
from .utils import (dependency_versions, device_for, epoch_lr, file_hash,
                    precision_policy, read_json, run_lock, save_checkpoint,
                    seed_worker, sync,
                    validate_resume_precision, write_json)


EXPECTED_IMAGES = 50_000
EXPECTED_ORIGINAL_TRAIN = 45_000
EXPECTED_ORIGINAL_VALIDATION = 5_000
REFIT_VERSION = 1


def _seed_for_device(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    # Seed CPU directly; unlike torch.manual_seed this does not schedule any
    # CUDA RNG work when running a genuinely CPU-only integrity test.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_refit_config(config):
    validate_config(config)
    settings = validate_optimization(config["training"])
    _require(settings is not None and settings["method"] in ("sgd", "sam", "asam"),
             "Final refit supports only frozen SGD, SAM or ASAM recipes")
    _require(settings["ema"]["enabled"], "Track raw and EMA variants separately")
    _require(config["runtime"].get("strict_source_resume") is True,
             "Final refit requires strict source identity")
    _require(config["runtime"]["device"] in ("cpu", "cuda"), "Use explicit CPU or CUDA")
    _require(config["training"]["epochs"] == 400, "Preserve the frozen 400-epoch horizon")
    _require(config["training"]["max_train_batches"] == config["training"]["max_val_batches"] == 0,
             "Full-data refit cannot use truncated batches or validation")
    _require(config["seed"] == 2026, "Preserve the selected random seed")
    _require(not config["training"].get("augmix"), "This refit uses the selected single-view recipe")
    refit = config.get("refit")
    keys = {"schema_version", "image_manifest", "image_manifest_sha256",
            "original_split_sha256", "train_csv_sha256", "checkpoint_interval",
            "inference_policy", "recipe_frozen"}
    optional = {"deploy_weight_variant"}
    _require(isinstance(refit, dict) and keys <= set(refit) and not set(refit) - keys - optional,
             "Invalid final-refit schema")
    _require(refit.get("deploy_weight_variant", "ema_fixed_epoch400") == "ema_fixed_epoch400",
             "Freeze fixed-endpoint EMA deployment before refit")
    _require(type(refit["schema_version"]) is int and refit["schema_version"] == REFIT_VERSION,
             "Unsupported final-refit schema")
    _require(refit["recipe_frozen"] is True, "Freeze recipe selection before final refit")
    _require(refit["inference_policy"] == "standard_fp32_single_model",
             "Final refit does not select TTA using former validation images")
    _require(type(refit["checkpoint_interval"]) is int and refit["checkpoint_interval"] > 0,
             "Positive checkpoint interval required")
    for name in ("image_manifest_sha256", "original_split_sha256", "train_csv_sha256"):
        digest = refit[name]
        _require(isinstance(digest, str) and len(digest) == 64
                 and set(digest) <= set("0123456789abcdef"), f"Invalid {name}")
    return settings


def verify_permitted_full_data(config):
    """Verify only training CSV, original split, and all permitted training PNGs."""
    root, refit = Path(config["data_dir"]), config["refit"]
    _require(file_hash(config["split_file"]) == refit["original_split_sha256"],
             "Original split SHA256 differs")
    _require(file_hash(root / "train_labels.csv") == refit["train_csv_sha256"],
             "Training CSV SHA256 differs")
    split, original_train, original_validation = load_split(root, config["split_file"])
    _require(split["seed"] == 2026 and split["classes"] == CLASSES,
             "Original split seed/class mapping differs")
    _require(len(original_train) == EXPECTED_ORIGINAL_TRAIN
             and len(original_validation) == EXPECTED_ORIGINAL_VALIDATION,
             "Original 45k/5k split sizes differ")
    rows = read_csv(root / "train_labels.csv", ["id", "filename", "label"])
    _require(len(rows) == EXPECTED_IMAGES, "Full refit requires all permitted training images")
    _require(Counter(row["label"] for row in rows) ==
             {str(label): EXPECTED_IMAGES // 10 for label in range(10)},
             "Unexpected full-training class balance/mapping")
    names = [row["filename"] for row in rows]
    _require(len(names) == len(set(names)), "Duplicate training filenames")
    manifest_path = Path(refit["image_manifest"])
    _require(file_hash(manifest_path) == refit["image_manifest_sha256"],
             "Permitted training-image manifest SHA256 differs")
    manifest = read_csv(manifest_path, ["id", "filename", "label", "partition", "decoded_rgb_sha256"])
    _require(len(manifest) == len(rows), "Image manifest size differs")
    lookup = {row["id"]: row for row in manifest}
    _require(set(lookup) == {row["id"] for row in rows}, "Image manifest IDs differ")
    train_ids = set(split["train_ids"])
    image_digest = hashlib.sha256()
    train_root = (root / "train_images").resolve()
    for row in rows:
        expected = lookup[row["id"]]
        _require(expected["filename"] == row["filename"] and expected["label"] == row["label"],
                 "Training image/label manifest differs")
        _require(expected["partition"] == ("train" if row["id"] in train_ids else "validation"),
                 "Original manifest partition differs")
        path = image_path(root, "train_images", row["filename"])
        _require(path.resolve().is_relative_to(train_root), "Training image escapes permitted directory")
        with Image.open(path) as image:
            _require(image.mode == "RGB" and image.size == (32, 32), "Invalid permitted training RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
        _require(digest == expected["decoded_rgb_sha256"], "Permitted training image pixels differ")
        image_digest.update((row["id"] + "\0" + row["filename"] + "\0" + row["label"]
                             + "\0" + file_hash(path) + "\n").encode())
    provenance = {"images": len(rows), "original_training_images": len(original_train),
                  "former_validation_images_included_in_training": len(original_validation),
                  "original_split_sha256": refit["original_split_sha256"],
                  "train_csv_sha256": refit["train_csv_sha256"],
                  "image_manifest_sha256": refit["image_manifest_sha256"],
                  "encoded_training_image_corpus_sha256": image_digest.hexdigest(),
                  "normalization": {key: split[key] for key in ("mean", "std")},
                  "classes": list(CLASSES), "hidden_data_accessed": False}
    return split, rows, provenance


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all().item(), "Non-finite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_tree(child)


def _validate_checkpoint(state, identity, config, policy):
    _require(state.get("purpose") == "full_data_final_refit" and state.get("schema_version") == REFIT_VERSION,
             "Not a final-refit resume checkpoint")
    _require(state.get("identity") == identity, "Resume source/config/split/data identity differs")
    validate_resume_precision(state, policy)
    epoch, history = state["epoch"], state["history"]
    _require(type(epoch) is int and 1 <= epoch <= 400 and len(history) == epoch,
             "Final-refit history/epoch mismatch")
    _require(state["scheduler"] == optimization_scheduler_state(epoch, config["training"]),
             "Frozen refit scheduler differs")
    batches = math.ceil(EXPECTED_IMAGES / config["training"]["batch_size"])
    settings = validate_optimization(config["training"])
    checks_per_step = 2 if settings["method"] in ("sam", "asam") else 1
    for index, row in enumerate(history):
        _require(row["epoch"] == index + 1 and row["lr"] == epoch_lr(index, config["training"]),
                 "Refit history learning rate differs")
        _require(row["train_images"] == EXPECTED_IMAGES and row["optimizer_steps"] == batches
                 and row["finite_gradient_checks"] == checks_per_step * batches and row["amp_skipped_updates"] == 0,
                 "Incomplete or skipped full-data refit epoch")
        _require(row["train_model_views"] == checks_per_step * EXPECTED_IMAGES,
                 "Refit model-view accounting differs")
        for name in ("train_perturbed_loss", "perturbed_gradient_norm_mean"):
            value = row[name]
            _require(isinstance(value, (float, int)) and math.isfinite(value)
                     if checks_per_step == 2 else value is None,
                     "Refit perturbed-pass accounting differs")
        _require(not any(key.startswith("val_") or "validation_accuracy" in key for key in row),
                 "Full-data refit must not report held-out validation")
        _require(all(not isinstance(value, (float, int)) or math.isfinite(value)
                     for value in row.values()), "Non-finite refit history")
    _require(state["optimization_steps"] == batches * epoch
             and state["ema"]["updates"] == state["optimization_steps"]
             and state["skipped_updates"] == 0, "Refit update/EMA counters differ")
    for group in state["optimizer"]["param_groups"]:
        _require(group["lr"] == state["scheduler"]["last_lr"]
                 and group["momentum"] == config["training"]["momentum"]
                 and group["weight_decay"] == config["training"]["weight_decay"]
                 and group.get("dampening", 0) == 0 and not group.get("nesterov", False)
                 and not group.get("maximize", False), "Refit optimizer settings differ")
    for key in ("model", "optimizer", "ema", "rng", "train_generator"):
        _finite_tree(state[key])
    expected_bn_updates = batches * epoch
    for key, value in state["model"].items():
        if key.endswith("num_batches_tracked"):
            _require(value.item() == expected_bn_updates, "Refit BN update counter differs")


def verify_checkpoint_cpu(path, identity, config, policy):
    """Strict CPU load and state checks; preserve all caller RNG state."""
    before = _rng_state(torch.device("cpu"))
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        _validate_checkpoint(state, identity, config, policy)
        model = build_model(config["model"])
        model.load_state_dict(state["model"], strict=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=config["training"]["lr"],
                                    momentum=config["training"]["momentum"],
                                    weight_decay=config["training"]["weight_decay"])
        optimizer.load_state_dict(state["optimizer"])
        if config["training"]["momentum"] > 0:
            _require(len(optimizer.state) == len(list(model.parameters()))
                     and all(parameter in optimizer.state for parameter in model.parameters()),
                     "Checkpoint optimizer momentum state is incomplete")
        for parameter, buffers in optimizer.state.items():
            momentum = buffers.get("momentum_buffer")
            _require(isinstance(momentum, torch.Tensor) and momentum.shape == parameter.shape
                     and momentum.dtype == torch.float32,
                     "Checkpoint optimizer momentum shape/precision differs")
        settings = validate_optimization(config["training"])
        ema = EMA(model, settings["ema"]["decay"], settings["ema"]["warmup"])
        ema.load_state_dict(state["ema"])
        parameters = set(name for name, _ in model.named_parameters())
        _require(all(name in parameters or torch.equal(value, ema.shadow[name])
                     for name, value in model.state_dict().items()),
                 "EMA nonparameter buffers differ from the online model")
        return {"epoch": state["epoch"], "sha256": file_hash(path),
                "bytes": Path(path).stat().st_size, "strict_cpu_load": True,
                "optimizer_steps": state["optimization_steps"], "ema_updates": ema.updates,
                "parameter_counts": parameter_counts(model), "all_checked_tensors_finite": True}
    finally:
        _restore_rng(before)


def _immutable_backup(path, record):
    destination = Path(path).parent / "backups" / f"epoch_{record['epoch']:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    backup = destination / "last.pt"
    if backup.exists():
        _require(file_hash(backup) == record["sha256"], "Immutable backup already has different bytes")
    else:
        temporary = destination / "last.pt.tmp"
        shutil.copyfile(path, temporary)
        _require(file_hash(temporary) == record["sha256"], "Backup copy integrity failure")
        os.replace(temporary, backup)
    manifest = destination / "verification.json"
    if manifest.exists():
        _require(read_json(manifest) == record, "Immutable backup verification differs")
    else:
        write_json(manifest, record)


def _same_value(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right))
    return left == right


def _export_fixed_endpoint(run_dir, state, config, split, counts):
    _require(state["epoch"] == config["training"]["epochs"] == 400,
             "Final exports require the fixed completed 400-epoch endpoint")
    common = {"model_config": config["model"], "epoch": 400,
              "normalization": {key: split[key] for key in ("mean", "std")},
              "classes": list(CLASSES), "parameter_counts": counts,
              "identity": state["identity"], "precision_policy": state["precision_policy"],
              "smoke_test": EXPECTED_IMAGES != 50_000, "training_images": EXPECTED_IMAGES,
              "no_heldout_validation": True, "checkpoint_selection": "fixed_final_epoch_only",
              "deployment_weight_variant_frozen": config["refit"].get("deploy_weight_variant", "ema_fixed_epoch400"),
              "inference_policy_frozen_before_refit": config["refit"]["inference_policy"]}
    exports = {}
    for name, weights, variant in (("final_raw.pt", state["model"], "raw_online"),
                                    ("final_ema.pt", state["ema"]["shadow"], "ema_online_bn_buffers")):
        export = {**common, "model": {key: value.detach().cpu().clone() for key, value in weights.items()},
                  "weight_variant": variant, "ema_updates": state["ema"]["updates"] if name == "final_ema.pt" else 0}
        path = run_dir / name
        if path.exists():
            _require(_same_value(torch.load(path, map_location="cpu", weights_only=True), export),
                     "Existing immutable final export differs")
        else:
            save_checkpoint(path, export)
        check = torch.load(path, map_location="cpu", weights_only=True)
        _finite_tree(check["model"])
        _require(_same_value(check, export), "Final export verification failed")
        exports[name] = {"sha256": file_hash(path), "bytes": path.stat().st_size,
                         "epoch": 400, "weight_variant": variant}
    return exports


def train_full_data(config, resume=False, stop_after_epoch=None):
    """Execute one explicitly requested scratch refit or epoch-boundary resume."""
    config = copy.deepcopy(config)
    settings = validate_refit_config(config)
    _require(stop_after_epoch is None or type(stop_after_epoch) is int and 1 <= stop_after_epoch <= 400,
             "stop_after_epoch must be an absolute epoch in [1,400]")
    before = {"precision": torch.get_float32_matmul_precision(),
              "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
              "cudnn_tf32": torch.backends.cudnn.allow_tf32,
              "deterministic": torch.backends.cudnn.deterministic,
              "benchmark": torch.backends.cudnn.benchmark}
    run_dir = Path(config["run_dir"])
    if resume:
        _require((run_dir / "last.pt").is_file(), "No completed refit checkpoint to resume")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Choose a new empty isolated refit run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with run_lock(run_dir / ".run.lock"):
            return _train_locked(config, settings, run_dir, resume, stop_after_epoch)
    finally:
        torch.set_float32_matmul_precision(before["precision"])
        torch.backends.cuda.matmul.allow_tf32 = before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = before["cudnn_tf32"]
        torch.backends.cudnn.deterministic = before["deterministic"]
        torch.backends.cudnn.benchmark = before["benchmark"]


def _train_locked(config, settings, run_dir, resume, stop_after_epoch):
    training, runtime = config["training"], config["runtime"]
    checks_per_step = 2 if settings["method"] in ("sam", "asam") else 1
    torch.set_num_threads(runtime["cpu_threads"])
    device = device_for(runtime["device"])
    _seed_for_device(config["seed"], device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dtype = None
    if device.type == "cuda":
        _require(runtime["amp"] and torch.cuda.is_bf16_supported(), "CUDA refit requires BF16 AMP")
        dtype = torch.bfloat16
    policy = precision_policy(dtype)
    split, rows, data_provenance = verify_permitted_full_data(config)
    sources = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    frozen = {key: value for key, value in config.items() if key != "run_dir"}
    identity = {"purpose": "full_data_final_refit", "schema_version": REFIT_VERSION,
                "frozen_config": frozen, "canonical_config_sha256": _json_hash(frozen),
                "source_hashes": sources, "data": data_provenance,
                "software": {"torch": str(torch.__version__), "numpy_import_version": np.__version__,
                             **dependency_versions()},
                "precision_policy": policy, "backend_policy": {
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision()}}
    last = run_dir / "last.pt"
    state = None
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=False)
        _validate_checkpoint(state, identity, config, policy)
    model = build_model(config["model"]).to(device)
    counts = parameter_counts(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=training["lr"],
                                momentum=training["momentum"], weight_decay=training["weight_decay"])
    ema = EMA(model, settings["ema"]["decay"], settings["ema"]["warmup"])
    transform = make_transform(split["mean"], split["std"], True,
        training.get("cutout_length", 0), training.get("cutout_probability", 1),
        color_jitter=training.get("color_jitter"),
        random_grayscale_probability=training.get("random_grayscale_probability", 0),
        sampling_augmentation=training.get("sampling_augmentation"))
    dataset = ImageDataset(config["data_dir"], rows, transform, labeled=True)
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(dataset, batch_size=training["batch_size"], shuffle=True,
                        num_workers=runtime["num_workers"], pin_memory=device.type == "cuda",
                        generator=generator, worker_init_fn=seed_worker)
    history, start, steps, train_seconds, wall_seconds, peak = [], 0, 0, 0., 0., 0
    if resume:
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        generator.set_state(state["train_generator"])
        history, start, steps = state["history"], state["epoch"], state["optimization_steps"]
        train_seconds, wall_seconds, peak = state["training_seconds"], state["run_seconds"], state["peak_cuda_bytes"]
        _restore_rng(state["rng"])
        _immutable_backup(last, verify_checkpoint_cpu(last, identity, config, policy))
    else:
        write_json(run_dir / "config.json", config)
        shutil.copyfile(config["split_file"], run_dir / "original_split_provenance.json")
        write_json(run_dir / "data_provenance.json", data_provenance)
        write_json(run_dir / "environment.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "parameter_counts": counts, "identity": identity,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "no_heldout_validation": True, "training_scope": "all_permitted_training_images"})
    end = min(400, stop_after_epoch) if stop_after_epoch is not None else 400
    print(f"Full-data scratch refit: {len(rows)} training images; NO held-out validation; resume epoch {start}", flush=True)
    for epoch in range(start, end):
        sync(device)
        epoch_start = time.perf_counter()
        lr = epoch_lr(epoch, training)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum, correct, seen, gradient_sum, epoch_steps = 0., 0., 0, 0., 0
        gradient_checks, perturbed_loss_sum, perturbed_gradient_sum, max_perturbation = 0, 0., 0., 0.
        for images, labels in loader:
            if os.environ.get("SPRINT_STOP_FILE") and Path(os.environ["SPRINT_STOP_FILE"]).exists():
                raise RuntimeError("SPRINT_STOP_REQUESTED; previous completed refit epoch preserved")
            images, labels = images.to(device), labels.to(device)
            images, targets = mixup_batch(images, labels, training.get("mixup_alpha", 0),
                training.get("mixup_probability", 1), config["model"].get("num_classes", 10))
            result = optimization_step(model, optimizer, images, targets, settings,
                                       dtype=dtype, label_smoothing=training["label_smoothing"])
            ema.update(model)
            predicted = result["logits"].argmax(1)
            correct += ((targets.gather(1, predicted[:, None]).sum().item())
                        if targets.ndim == 2 else (predicted == targets).sum().item())
            seen += len(labels)
            loss_sum += result["base_loss"] * len(labels)
            gradient_sum += result["base_gradient_norm"]
            gradient_checks += result["gradient_checks"]
            if result["perturbed_loss"] is not None:
                perturbed_loss_sum += result["perturbed_loss"] * len(labels)
                perturbed_gradient_sum += result["perturbed_gradient_norm"]
                max_perturbation = max(max_perturbation, result["perturbation_norm"])
            epoch_steps += result["optimizer_steps"]
        sync(device)
        elapsed = time.perf_counter() - epoch_start
        _require(seen == EXPECTED_IMAGES, "Refit epoch did not cover all permitted training images")
        steps += epoch_steps
        train_seconds += elapsed
        peak = max(peak, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": loss_sum / seen,
               "train_accuracy": correct / seen, "train_images": seen,
               "training_seconds": elapsed, "images_per_second": seen / elapsed,
               "base_gradient_norm_mean": gradient_sum / epoch_steps,
               "finite_gradient_checks": gradient_checks, "optimizer_steps": epoch_steps,
               "train_perturbed_loss": perturbed_loss_sum / seen if settings["method"] in ("sam", "asam") else None,
               "perturbed_gradient_norm_mean": perturbed_gradient_sum / epoch_steps if settings["method"] in ("sam", "asam") else None,
               "perturbation_norm_max": max_perturbation,
               "train_model_views": seen * checks_per_step,
               "model_views_per_second": seen * checks_per_step / elapsed,
               "amp_skipped_updates": 0, "ema_updates": ema.updates,
               "peak_cuda_allocated_bytes": peak}
        history.append(row)
        wall_seconds += time.perf_counter() - epoch_start
        state = {"purpose": "full_data_final_refit", "schema_version": REFIT_VERSION,
                 "epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "ema": ema.state_dict(), "identity": identity, "history": history,
                 "precision_policy": policy, "scaler": {}, "rng": _rng_state(device),
                 "train_generator": generator.get_state(), "optimization_steps": steps,
                 "scheduler": optimization_scheduler_state(epoch + 1, training),
                 "skipped_updates": 0, "training_seconds": train_seconds,
                 "run_seconds": wall_seconds, "peak_cuda_bytes": peak}
        save_checkpoint(last, state)
        verification = verify_checkpoint_cpu(last, identity, config, policy)
        write_json(run_dir / "last_verification.json", verification)
        if (epoch + 1) % config["refit"]["checkpoint_interval"] == 0 or epoch + 1 == end:
            _immutable_backup(last, verification)
        write_json(run_dir / "metrics.json", history)
        print(json.dumps(row, allow_nan=False), flush=True)
    _require(bool(history), "No completed refit epoch")
    exports = _export_fixed_endpoint(run_dir, state, config, split, counts) if history[-1]["epoch"] == 400 else {}
    summary = {"completed_epochs": history[-1]["epoch"], "planned_epochs": 400,
               "complete": history[-1]["epoch"] == 400, "parameter_counts": counts,
               "training_images": EXPECTED_IMAGES, "no_heldout_validation": True,
               "former_validation_images_included_in_training": EXPECTED_ORIGINAL_VALIDATION,
               "training_accuracy_definition": "MixUp-weighted augmented-training accuracy; not held-out accuracy",
               "training_seconds": train_seconds, "run_seconds_excluding_checkpoint_io": wall_seconds,
               "optimizer_steps": steps, "ema_updates": ema.updates, "amp_skipped_updates": 0,
               "checkpoint_selection": "fixed_final_epoch_only", "exports": exports,
               "hidden_data_accessed": False, "inference_performed": False}
    write_json(run_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.device:
        config["runtime"]["device"] = args.device
    print(json.dumps(train_full_data(config, args.resume, args.stop_after_epoch), indent=2))


if __name__ == "__main__":
    main()
