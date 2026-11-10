"""Compare no TTA with horizontal-flip TTA on the original labeled validation set."""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CLASSES, ImageDataset, load_split, make_transform
from .models import build_model
from .utils import device_for, file_hash, read_json, write_json


def checked_logits(model, images):
    logits = model(images)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite prediction logits; no submission was written")
    return logits.float()


def probabilities(model, images, horizontal_flip=False):
    result = checked_logits(model, images).softmax(1)
    if horizontal_flip:
        result = (result + checked_logits(model, images.flip(-1)).softmax(1)) / 2
    return result


def require_tta_benefit(report_path, checkpoint, state):
    if report_path is None:
        raise ValueError("Horizontal-flip TTA requires --tta-validation from src.tta")
    report = read_json(report_path)
    if (report.get("method") != "horizontal_flip_probability_mean"
            or report.get("checkpoint_sha256") != file_hash(checkpoint)
            or report.get("split_sha256") != state["identity"]["split_sha256"]
            or report.get("smoke_test") is not False
            or report.get("full_validation") is not True
            or report.get("beneficial") is not True
            or not report["horizontal_flip"]["accuracy"] > report["no_tta"]["accuracy"]):
        raise ValueError("TTA validation must show an accuracy benefit on this checkpoint's full split")
    return report


@torch.no_grad()
def compare(run_dir, data_dir, output, device="auto", batch_size=256, num_workers=0):
    output, root = Path(output), Path(run_dir)
    if output.exists():
        raise FileExistsError("TTA report exists; choose a new output file")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid batch size/worker count")
    checkpoint, split_path = root / "best.pt", root / "split.json"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
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
    dataset = ImageDataset(data_dir, rows, make_transform(split["mean"], split["std"], False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    totals = {name: {"correct": 0, "loss": 0.0} for name in ("no_tta", "horizontal_flip")}
    helped = harmed = seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = checked_logits(model, images)
        plain_log_probs = logits.log_softmax(1)
        # logaddexp retains stable CE even if softmax probabilities underflow.
        flipped_log_probs = checked_logits(model, images.flip(-1)).log_softmax(1)
        mixed_log_probs = torch.logaddexp(plain_log_probs, flipped_log_probs) - torch.log(torch.tensor(2.0, device=device))
        correctness = []
        for name, log_probs in (("no_tta", plain_log_probs), ("horizontal_flip", mixed_log_probs)):
            correct = log_probs.argmax(1).eq(labels)
            totals[name]["correct"] += correct.sum().item()
            totals[name]["loss"] += F.nll_loss(log_probs, labels, reduction="sum").item()
            correctness.append(correct)
        helped += (~correctness[0] & correctness[1]).sum().item()
        harmed += (correctness[0] & ~correctness[1]).sum().item()
        seen += len(labels)
    report = {"method": "horizontal_flip_probability_mean", "checkpoint_sha256": file_hash(checkpoint),
              "checkpoint_epoch": state["epoch"], "split_sha256": file_hash(split_path),
              "smoke_test": state["smoke_test"], "full_validation": seen == len(rows),
              "images": seen, "precision": "float32", "device": str(device),
              "helped_images": helped, "harmed_images": harmed}
    for name, total in totals.items():
        report[name] = {"accuracy": total["correct"] / seen, "cross_entropy": total["loss"] / seen}
    report["accuracy_gain"] = report["horizontal_flip"]["accuracy"] - report["no_tta"]["accuracy"]
    report["beneficial"] = not state["smoke_test"] and report["accuracy_gain"] > 0
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    print(compare(args.run_dir, args.data_dir, args.output, args.device, args.batch_size, args.num_workers))
