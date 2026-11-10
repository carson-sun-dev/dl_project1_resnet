"""Training-only BatchNorm recalibration with transactional model/RNG safety.

The caller should pass a copied model and its checkpoint's split SHA256. No
arbitrary loader is accepted: this module loads the verified split itself and
constructs a deterministic dataset from its training IDs only. The estimator is
the sample-weighted average of per-batch BN statistics, including the unbiased
batch variance used by PyTorch. It is not a pooled global-variance estimator.
"""
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASSES, ImageDataset, load_split, make_transform
from .utils import file_hash, seed_worker


def _rng_snapshot():
    # Do not initialize CUDA merely to recalibrate a CPU model.
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _positive_integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def recalibrate_batch_norm(model, data_dir, split_file, device="cpu", batch_size=125,
                          num_workers=0, seed=2026, expected_split_sha256=None):
    """Update only BN buffers using verified, unaugmented training images.

    Model parameters must already reside in FP32 on ``device``. All module modes,
    BN momenta, RNG streams and precision-backend flags are restored. On any
    failure, original parameters and buffers are restored as well. Validation
    images are never passed to the model, and training labels are not used.
    """
    _positive_integer(batch_size, "batch_size", 1)
    _positive_integer(num_workers, "num_workers", 0)
    _positive_integer(seed, "seed", 0)
    if seed > 2**63 - 1:
        raise ValueError("seed must not exceed 2**63 - 1")
    try:
        target = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must identify a CPU or CUDA device") from exc
    if target.type not in ("cpu", "cuda"):
        raise ValueError("BN recalibration supports CPU and CUDA devices")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
    if expected_split_sha256 is not None and (
            not isinstance(expected_split_sha256, str) or len(expected_split_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in expected_split_sha256)):
        raise ValueError("expected_split_sha256 must be a SHA256 hex string")
    split_file, data_dir = Path(split_file), Path(data_dir)
    split_sha = file_hash(split_file)
    if expected_split_sha256 is not None and split_sha != expected_split_sha256.lower():
        raise ValueError("BN recalibration split SHA256 differs from the checkpoint")
    # load_split verifies CSV hash, unique IDs, disjointness and full coverage.
    split, train_rows, val_rows = load_split(data_dir, split_file)
    if split.get("classes") != CLASSES:
        raise ValueError("Unexpected CIFAR-10 class mapping in calibration split")
    if any(row["label"] not in {str(i) for i in range(10)} for row in train_rows + val_rows):
        raise ValueError("Calibration labels must be CIFAR-10 integer strings 0..9")
    names = [row["filename"] for row in train_rows + val_rows]
    if len(names) != len(set(names)) or any(
            not name or Path(name).name != name or "\\" in name for name in names):
        raise ValueError("Calibration split has duplicate or unsafe image filenames")
    modules = list(model.modules())
    bns = [(name, module) for name, module in model.named_modules()
           if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        raise ValueError("Model has no BatchNorm layers to recalibrate")
    if any(not bn.track_running_stats or bn.running_mean is None or bn.running_var is None
           or bn.num_batches_tracked is None for _, bn in bns):
        raise ValueError("Every BatchNorm layer must track running statistics")
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    for name, tensor in [*parameters.items(), *buffers.items()]:
        if tensor.device != target:
            raise ValueError(f"Model tensor {name} is not on calibration device {target}; move a copy explicitly")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"Model tensor {name} must be FP32 for recalibration")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite model tensor before BN recalibration: {name}")
    transform = make_transform(split["mean"], split["std"], training=False)
    dataset = ImageDataset(data_dir, train_rows, transform, labeled=True)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, drop_last=False,
                        num_workers=int(num_workers), worker_init_fn=seed_worker, generator=generator)
    modes = [module.training for module in modules]
    momenta = [bn.momentum for _, bn in bns]
    parameter_before = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    buffer_before = {name: value.detach().cpu().clone() for name, value in buffers.items()}
    allowed_buffers = {f"{name}.{key}" if name else key for name, _ in bns
                       for key in ("running_mean", "running_var", "num_batches_tracked")}
    rng_before = _rng_snapshot()
    backend_before = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
    }
    images = batches = 0
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % 2**32)
        torch.set_rng_state(torch.Generator().manual_seed(int(seed)).get_state())
        if rng_before["cuda"] is not None:
            torch.cuda.manual_seed_all(int(seed))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # eval disables dropout and stochastic training-only modules. Enable only
        # individual BN layers, never the surrounding network.
        model.eval()
        for _, bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
            bn.train()
        with torch.no_grad(), torch.autocast(device_type=target.type, enabled=False):
            for inputs, _ in loader:
                inputs = inputs.to(target)
                if inputs.dtype != torch.float32 or not torch.isfinite(inputs).all():
                    raise RuntimeError("Non-finite or non-FP32 calibration input")
                count = len(inputs)
                if not count:
                    raise RuntimeError("Empty calibration batch")
                images += count
                # Equivalent to cumulative averaging for equal batches, while a
                # partial final batch contributes only its sample-count weight.
                for _, bn in bns:
                    bn.momentum = count / images
                logits = model(inputs)
                if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
                        or not torch.isfinite(logits).all()):
                    raise RuntimeError("Non-finite or non-FP32 calibration output")
                batches += 1
        if images != len(train_rows):
            raise RuntimeError("BN recalibration did not cover all training IDs")
        for name, value in parameters.items():
            if not torch.equal(value.detach().cpu(), parameter_before[name]):
                raise RuntimeError(f"Model parameter changed during BN recalibration: {name}")
        changed = []
        for name, value in buffers.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite BN recalibration buffer: {name}")
            if not torch.equal(value.detach().cpu(), buffer_before[name]):
                if name not in allowed_buffers:
                    raise RuntimeError(f"Non-BatchNorm buffer changed during calibration: {name}")
                changed.append(name)
        metadata = {
            "method": "training_only_batchnorm_recalibration",
            "split_sha256": split_sha, "training_csv_sha256": split["train_csv_sha256"],
            "training_ids_sha256": hashlib.sha256(json.dumps(
                split["train_ids"], separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "training_images": images, "validation_images_used": 0, "hidden_images_used": 0,
            "batch_size": int(batch_size), "batches": batches,
            "num_workers": int(num_workers), "seed": int(seed),
            "normalization": {key: split[key] for key in ("mean", "std")}, "classes": CLASSES.copy(),
            "transform": "RGB, ToTensor float32/255, frozen Normalize; no training augmentation",
            "statistic_estimator": "sample-weighted mean of per-batch means and unbiased variances",
            "momentum_policy": "batch_images / cumulative_images; handles partial final batch",
            "precision": "FP32; autocast and TF32 disabled", "device": str(target),
            "batchnorm_layers": len(bns), "changed_buffers": changed,
            "parameters_bitwise_unchanged": True, "external_rng_preserved": True,
            "module_modes_and_momenta_restored": True,
        }
    except BaseException:
        with torch.no_grad():
            for name, value in parameters.items():
                value.copy_(parameter_before[name])
            for name, value in buffers.items():
                value.copy_(buffer_before[name])
        raise
    finally:
        for (_, bn), momentum in zip(bns, momenta):
            bn.momentum = momentum
        # Direct assignment restores heterogeneous original child modes without
        # recursive train() calls overriding one another.
        for module, training in zip(modules, modes):
            module.training = training
        _restore_rng(rng_before)
        torch.backends.cuda.matmul.allow_tf32 = backend_before["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = backend_before["cudnn_tf32"]
        torch.set_float32_matmul_precision(backend_before["matmul_precision"])
        torch.backends.cudnn.benchmark = backend_before["benchmark"]
        torch.backends.cudnn.deterministic = backend_before["deterministic"]
    return metadata
