"""Read-only appearance diagnostics on a checkpoint's original validation IDs."""
import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from .data import CLASSES, ImageDataset, load_split, make_transform
from .models import build_model, parameter_counts
from .tta import checked_logits
from .utils import dependency_versions, device_for, file_hash, read_json, run_lock, seed_all, sync, write_json


def validate_protocol(protocol):
    if protocol.get("schema_version") != 1 or not isinstance(protocol.get("name"), str) or not protocol["name"]:
        raise ValueError("Expected a named schema_version 1 robustness protocol")
    conditions = protocol.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 2:
        raise ValueError("Protocol needs clean and at least one transformed condition")
    names = set()
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict) or set(condition) != {"name", "kind", "factor"}:
            raise ValueError("Each condition needs exactly name, kind, factor")
        name, kind, factor = condition["name"], condition["kind"], condition["factor"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Condition names must be nonempty and unique")
        names.add(name)
        if (isinstance(factor, bool) or not isinstance(factor, (int, float))
                or not math.isfinite(factor) or factor <= 0):
            raise ValueError("Condition factors must be positive and finite")
        if kind not in ("identity", "brightness", "contrast", "grayscale"):
            raise ValueError("Unsupported appearance transformation")
        if kind in ("identity", "grayscale") and factor != 1:
            raise ValueError("Identity and grayscale require factor 1")
        if (index == 0 and (name != "clean" or kind != "identity")) or (index > 0 and kind == "identity"):
            raise ValueError("Exactly one clean identity condition must come first")


class AppearanceTransform:
    """A deterministic, label-preserving PIL operation before tensor normalization."""
    def __init__(self, condition):
        self.kind, self.factor = condition["kind"], condition["factor"]

    def __call__(self, image):
        if self.kind == "identity":
            return image
        if self.kind == "brightness":
            return TF.adjust_brightness(image, self.factor)
        if self.kind == "contrast":
            return TF.adjust_contrast(image, self.factor)
        if self.kind == "grayscale":
            return TF.rgb_to_grayscale(image, num_output_channels=3)
        raise ValueError("Unsupported appearance transformation")


def make_robustness_transform(mean, std, condition):
    return transforms.Compose([AppearanceTransform(condition), make_transform(mean, std, False)])


def aggregate_conditions(results):
    """Equal weight for each transformed condition; clean is explicitly excluded."""
    summary = {}
    for method in ("standard", "horizontal_flip"):
        shifted = results[1:]
        worst = min(shifted, key=lambda row: row[method]["accuracy"])
        families = sorted({row["kind"] for row in shifted})
        summary[method] = {
            "clean_accuracy": results[0][method]["accuracy"],
            "mean_transformed_accuracy": sum(row[method]["accuracy"] for row in shifted) / len(shifted),
            "worst_transformed_accuracy": worst[method]["accuracy"], "worst_condition": worst["name"],
            "family_mean_accuracy": {kind: sum(row[method]["accuracy"] for row in shifted if row["kind"] == kind)
                                     / sum(row["kind"] == kind for row in shifted) for kind in families}}
    return summary


@torch.inference_mode()
def _measure_conditions(model, data_dir, rows, split, protocol, device, batch_size, num_workers, seed):
    results, clean_correctness = [], {}
    for condition in protocol["conditions"]:
        dataset = ImageDataset(data_dir, rows, make_robustness_transform(split["mean"], split["std"], condition))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")
        totals = {method: {"correct": 0, "loss": 0.0, "correctness": [], "per_class_correct": [0]*10}
                  for method in ("standard", "horizontal_flip")}
        supports, seen = [0]*10, 0
        sync(device)
        started = time.perf_counter()
        for images, labels in loader:
            original_labels = labels.tolist()
            images, labels = images.to(device), labels.to(device)
            if not torch.isfinite(images).all():
                raise RuntimeError("Non-finite validation input")
            plain = checked_logits(model, images).log_softmax(1)
            flipped = checked_logits(model, images.flip(-1)).log_softmax(1)
            combined = torch.logaddexp(plain, flipped) - math.log(2)
            for method, log_probs in (("standard", plain), ("horizontal_flip", combined)):
                if not torch.isfinite(log_probs).all() or not torch.isfinite(log_probs.exp()).all():
                    raise RuntimeError("Non-finite validation inference output")
                correct = log_probs.argmax(1).eq(labels).cpu().tolist()
                totals[method]["correct"] += sum(correct)
                totals[method]["loss"] += F.nll_loss(log_probs, labels, reduction="sum").item()
                totals[method]["correctness"].extend(correct)
                for label, is_correct in zip(original_labels, correct):
                    totals[method]["per_class_correct"][label] += int(is_correct)
            for label in original_labels:
                supports[label] += 1
            seen += len(labels)
        sync(device)
        if seen != len(rows):
            raise RuntimeError("Incomplete original validation split")
        result = {**condition, "images": seen, "seconds": time.perf_counter()-started}
        for method, total in totals.items():
            correctness = total["correctness"]
            if condition["kind"] == "identity":
                clean_correctness[method] = correctness
            base = clean_correctness[method]
            result[method] = {
                "correct": total["correct"], "accuracy": total["correct"]/seen, "cross_entropy": total["loss"]/seen,
                "delta_from_clean": total["correct"]/seen - sum(base)/seen,
                "clean_correct_lost": sum(old and not new for old, new in zip(base, correctness)),
                "clean_incorrect_recovered": sum(not old and new for old, new in zip(base, correctness)),
                "per_class_accuracy": {name: total["per_class_correct"][index]/supports[index] if supports[index] else None
                                       for index, name in enumerate(CLASSES)}}
            if not math.isfinite(result[method]["cross_entropy"]):
                raise RuntimeError("Non-finite validation cross-entropy")
        results.append(result)
        print(f"{condition['name']}: standard {result['standard']['accuracy']:.2%}, "
              f"flip {result['horizontal_flip']['accuracy']:.2%} ({result['seconds']:.1f}s)", flush=True)
    return results


def benchmark(run_dir, data_dir, protocol_path, output, device="auto", batch_size=128, num_workers=0,
              cpu_threads=4, allow_smoke=False):
    root, output = Path(run_dir), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Save robustness reports outside the existing model run directory")
    csv_path = output.with_suffix(".csv")
    if output.suffix != ".json":
        raise ValueError("Robustness output must be a .json path")
    if batch_size < 1 or num_workers < 0 or cpu_threads < 1:
        raise ValueError("Invalid batch size, workers or CPU threads")
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(output.with_suffix(".lock")):
        if output.exists() or csv_path.exists():
            raise FileExistsError("Robustness report exists; choose a new output path")
        checkpoint, split_path = root/"best.pt", root/"split.json"
        checkpoint_hash, split_hash = file_hash(checkpoint), file_hash(split_path)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["smoke_test"] and not allow_smoke:
            raise ValueError("Smoke checkpoint is not a formal robustness result")
        if split_hash != state["identity"]["split_sha256"]:
            raise ValueError("Validation split differs from checkpoint training split")
        split, training, rows = load_split(data_dir, split_path)
        if state["normalization"] != {key: split[key] for key in ("mean", "std")}:
            raise ValueError("Checkpoint normalization differs from its split")
        if state["classes"] != split["classes"] or split["classes"] != CLASSES:
            raise ValueError("Unexpected class mapping")
        seed = state["identity"]["seed"]
        seed_all(seed)
        torch.set_num_threads(cpu_threads)
        device = device_for(device)
        # Match final submission validation: full FP32, no TF32 or autocast.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model = build_model(state["model_config"]).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
        counts = parameter_counts(model)
        if counts != state["parameter_counts"]:
            raise ValueError("Checkpoint parameter counts differ from the actual model")
        with torch.autocast(device_type=device.type, enabled=False):
            results = _measure_conditions(model, data_dir, rows, split, protocol, device, batch_size, num_workers, seed)
        if file_hash(checkpoint) != checkpoint_hash or file_hash(split_path) != split_hash:
            raise RuntimeError("Checkpoint or split changed during read-only evaluation")
        report = {"schema_version": 1, "measured_utc": datetime.now(timezone.utc).isoformat(),
                  "checkpoint_sha256": checkpoint_hash, "checkpoint_epoch": state["epoch"],
                  "parameter_counts": counts, "split_sha256": split_hash,
                  "train_csv_sha256": split["train_csv_sha256"], "training_ids": len(training),
                  "validation_images_per_condition": len(rows), "full_original_validation_split": True,
                  "classes": CLASSES, "normalization": state["normalization"],
                  "protocol": protocol, "protocol_sha256": file_hash(protocol_path),
                  "results": results, "aggregate": aggregate_conditions(results),
                  "precision": "float32", "autocast": False, "matmul_tf32": False, "cudnn_tf32": False,
                  "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "torch": str(torch.__version__), **dependency_versions(), "seed": seed,
                  "batch_size": batch_size, "num_workers": num_workers, "cpu_threads": cpu_threads,
                  "source_hashes": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")},
                  "smoke_test": state["smoke_test"], "all_outputs_finite": True,
                  "hidden_evaluation_images_or_labels_accessed": False, "training_performed": False,
                  "note": "Synthetic appearance diagnostics only; not CIFAR-10-C, mCE, or an estimate of private leaderboard accuracy. "
                          "Transformed mean weights each non-clean condition equally. No transformed normalization fitting."}
        write_json(output, report)
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            columns = ["condition", "kind", "factor", "method", "images", "correct", "accuracy", "cross_entropy",
                       "delta_from_clean", "clean_correct_lost", "clean_incorrect_recovered"]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in results:
                for method in ("standard", "horizontal_flip"):
                    writer.writerow({"condition": row["name"], "kind": row["kind"], "factor": row["factor"],
                                     "method": method, "images": row["images"],
                                     **{key: row[method][key] for key in columns[5:]}})
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--protocol", default="configs/appearance_robustness_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-smoke", action="store_true", help="Only for synthetic pipeline tests")
    args = parser.parse_args()
    result = benchmark(args.run_dir, args.data_dir, args.protocol, args.output, args.device, args.batch_size,
                       args.num_workers, args.cpu_threads, args.allow_smoke)
    print(result["aggregate"], flush=True)
