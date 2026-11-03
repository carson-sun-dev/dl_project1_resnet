import hashlib
import json
import math
import os
import random
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path

import numpy as np
import torch


def validate_normalization(normalization):
    """Accept only three finite channel means and positive finite stds."""
    try:
        mean, std = normalization["mean"], normalization["std"]
        valid = (len(mean) == len(std) == 3
                 and all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)
                         for x in [*mean, *std])
                 and all(x > 0 for x in std))
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("Invalid normalization: expected three finite means and three positive finite standard deviations")


def precision_policy(dtype):
    """Describe effective training precision; None means FP32 without autocast."""
    if dtype not in (None, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported autocast dtype: {dtype}")
    return {"compute_dtype": str(dtype).split(".")[-1] if dtype is not None else "float32",
            "autocast_enabled": dtype is not None, "grad_scaler_enabled": dtype == torch.float16}


def validate_resume_precision(state, current_policy, legacy_environment=None):
    """Reject precision changes before restoring any training state."""
    policies = {p["compute_dtype"]: p for p in
                (precision_policy(None), precision_policy(torch.float16), precision_policy(torch.bfloat16))}
    saved = state.get("precision_policy")
    if "precision_policy" not in state:
        recorded = (legacy_environment or {}).get("precision")
        if recorded is not None:
            saved = policies.get(str(recorded).replace("torch.", ""))
        elif state.get("scaler"):
            # Only FP16 used an enabled scaler in the original trainer.
            saved = policies["float16"]
        else:
            raise ValueError("Cannot determine legacy checkpoint precision (FP32 or BF16). "
                             "Restore its environment.json/resume_environments.json before resuming.")
    if not isinstance(saved, dict) or saved not in policies.values():
        raise ValueError("Invalid or unknown checkpoint precision policy")
    if saved != current_policy:
        raise ValueError(f"Unsupported resume precision transition: {saved['compute_dtype']} -> "
                         f"{current_policy['compute_dtype']}. Use the original effective precision "
                         "or start a new run directory.")
    if "scaler" not in state or bool(state["scaler"]) != saved["grad_scaler_enabled"]:
        raise ValueError("Checkpoint scaler state does not match its precision policy")


def dependency_versions():
    result = {}
    for name in ("torchvision", "numpy", "Pillow"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return {**result, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def epoch_lr(epoch, training):
    """One LR per epoch; epoch is zero-based. Reproducible without scheduler state."""
    peak, floor = training["lr"], training["min_lr"]
    warmup, total = training["warmup_epochs"], training["epochs"]
    if warmup and epoch < warmup:
        return peak * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
