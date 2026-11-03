"""Export predictions in sample-submission order from one trained ResNet."""
import argparse
import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, audit, make_transform, read_csv
from .models import build_model, parameter_counts
from .submission import validate_submission
from .utils import device_for, file_hash, write_json


@torch.no_grad()
def predict(checkpoint, data_dir, output, device="auto", batch_size=256, num_workers=0, allow_smoke=False):
    root, output = Path(data_dir), Path(output)
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError("Submission exists. Choose another output directory to preserve provenance.")
    if output.name != "submission.csv":
        raise ValueError("The output filename must be submission.csv (use a distinct parent directory per run)")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid batch size/worker count")
    audit(root)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["smoke_test"] and not allow_smoke:
        raise ValueError("Smoke checkpoint: do not submit it. --allow-smoke is only for pipeline testing.")
    device = device_for(device)
    model = build_model(state["model_config"]).to(device).eval()
    model.load_state_dict(state["model"])
    test = read_csv(root/"test.csv", ["id", "filename"])
    sample = read_csv(root/"sample_submission.csv", ["id", "label"])
    norm = state["normalization"]
    ds = ImageDataset(root, test, make_transform(norm["mean"], norm["std"], False), labeled=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = {}
    for x, ids in loader:
        labels = model(x.to(device)).argmax(1).cpu().tolist()
        predictions.update(zip(ids, labels))
    if set(predictions) != {r["id"] for r in sample}:
        raise ValueError("Missing or unexpected prediction IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows({"id": r["id"], "label": predictions[r["id"]]} for r in sample)
    validate_submission(temp, root/"sample_submission.csv")
    os.replace(temp, output)
    metadata = {"checkpoint_sha256": file_hash(checkpoint), "checkpoint_epoch": state["epoch"],
                "parameter_counts": parameter_counts(model), "model_config": state["model_config"],
                "test_csv_sha256": file_hash(root/"test.csv"), "submission_sha256": file_hash(output),
                "smoke_test": state["smoke_test"], "rows": len(sample), "inference": "single model, FP32, no TTA"}
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="submissions/baseline/submission.csv")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--allow-smoke", action="store_true")
    a = p.parse_args()
    print(predict(a.checkpoint, a.data_dir, a.output, a.device, a.batch_size, a.num_workers, a.allow_smoke))
