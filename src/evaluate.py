"""Evaluate best.pt on its original validation split, never on hidden labels."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, ImageDataset, load_split, make_transform
from .models import build_model, parameter_counts
from .utils import device_for, file_hash, write_json


@torch.no_grad()
def evaluate_checkpoint(run_dir, data_dir, device="auto", batch_size=256, num_workers=0):
    root = Path(run_dir)
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid batch size/worker count")
    checkpoint = root / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    split_path = root / "split.json"
    if file_hash(split_path) != state["identity"]["split_sha256"]:
        raise ValueError("Validation split differs from the checkpoint's training split")
    split, _, rows = load_split(data_dir, split_path)
    if state["normalization"] != {"mean": split["mean"], "std": split["std"]}:
        raise ValueError("Checkpoint normalization differs from its split")
    if state["classes"] != CLASSES or split["classes"] != CLASSES:
        raise ValueError("Unexpected class mapping")
    device = device_for(device)
    model = build_model(state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    ds = ImageDataset(data_dir, rows, make_transform(split["mean"], split["std"], False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    matrix = np.zeros((10, 10), dtype=np.int64)
    predictions, loss_sum, offset = [], 0.0, 0
    for x, y in loader:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite validation logits")
        loss_sum += torch.nn.functional.cross_entropy(
            logits, y.to(device), reduction="sum").item()
        predicted = logits.argmax(1).cpu().tolist()
        for actual, guess in zip(y.tolist(), predicted):
            matrix[actual, guess] += 1
            row = rows[offset]
            predictions.append({"id": row["id"], "filename": row["filename"],
                                "true_label": actual, "predicted_label": guess,
                                "correct": int(actual == guess)})
            offset += 1
    if offset != len(rows):
        raise RuntimeError("Validation prediction count mismatch")
    output = root / "validation"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (output / "confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class / predicted_class"] + CLASSES)
        writer.writerows([name] + matrix[i].tolist() for i, name in enumerate(CLASSES))
    per_class = []
    for i, name in enumerate(CLASSES):
        support, predicted_count, correct = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        per_class.append({"label": i, "class": name, "images": support,
                          "correct": correct, "accuracy": recall, "precision": precision,
                          "recall": recall, "f1": 2*precision*recall/(precision+recall)
                          if precision+recall else 0.0})
    with (output / "per_class.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    fig, ax = plt.subplots(figsize=(9, 8))
    heat = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(heat, ax=ax, label="Images")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class",
           title="Validation confusion matrix (best checkpoint, FP32)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    metrics = {"checkpoint_epoch": state["epoch"], "checkpoint_sha256": file_hash(checkpoint),
               "split_sha256": file_hash(split_path), "images": offset,
               "accuracy": float(matrix.trace()/offset), "cross_entropy": loss_sum/offset,
               "macro_f1": sum(row["f1"] for row in per_class)/10,
               "parameter_counts": parameter_counts(model), "device": str(device),
               "precision": "float32", "smoke_test": state["smoke_test"],
               "note": "Full original validation split; no hidden evaluation labels. "
                       "FP32 reevaluation may differ slightly from training-time AMP metrics."}
    write_json(output / "metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    print(evaluate_checkpoint(args.run_dir, args.data_dir, args.device, args.batch_size, args.num_workers))
