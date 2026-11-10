"""Only the supplied images/CSVs are used; no external dataset downloads."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentation import Cutout, validate_color_augmentation
from .utils import file_hash, read_json, validate_normalization, write_json

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def read_csv(path, columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: expected columns {columns}, got {reader.fieldnames}")
        rows = list(reader)
    if any(set(row) != set(columns) or any(row[c] is None for c in columns) for row in rows):
        raise ValueError(f"{path}: malformed CSV row or wrong number of fields")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"{path}: empty or duplicate IDs")
    if any(not row["id"] for row in rows):
        raise ValueError(f"{path}: empty ID")
    return rows


def image_path(root, folder, filename):
    if not filename or Path(filename).name != filename or "\\" in filename:
        raise ValueError(f"Unsafe image filename: {filename!r}")
    return Path(root) / folder / filename


def audit(root, verify_images=False):
    root = Path(root)
    train = read_csv(root / "train_labels.csv", ["id", "filename", "label"])
    test = read_csv(root / "test.csv", ["id", "filename"])
    sample = read_csv(root / "sample_submission.csv", ["id", "label"])
    if {r["id"] for r in test} != {r["id"] for r in sample}:
        raise ValueError("Test and sample-submission IDs differ")
    if any(r["label"] not in {str(i) for i in range(10)} for r in train):
        raise ValueError("Training labels must be integer strings 0..9")
    for rows, folder in [(train, "train_images"), (test, "test_images")]:
        names = [r["filename"] for r in rows]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate filenames in {folder}")
        for row in rows:
            path = image_path(root, folder, row["filename"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_images:
                with Image.open(path) as image:
                    if image.size != (32, 32) or image.mode != "RGB":
                        raise ValueError(f"Unexpected image format: {path}")
                    image.load()
    return {"train_images": len(train), "test_images": len(test),
            "class_counts": dict(sorted(Counter(r["label"] for r in train).items())),
            "test_matches_sample_order": [r["id"] for r in test] == [r["id"] for r in sample],
            "all_image_pixels_verified": verify_images,
            "csv_sha256": {n: file_hash(root/n) for n in ["train_labels.csv", "test.csv", "sample_submission.csv"]}}


def prepare_split(root, output, seed=2026, val_fraction=0.1):
    """Create immutable, class-stratified split and train-only normalization stats."""
    root, output = Path(root), Path(output)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    if output.exists():
        existing = read_json(output)
        load_split(root, output)
        if existing["seed"] != seed or existing["val_fraction"] != val_fraction:
            raise ValueError("Existing split settings differ; choose a new output filename")
        return existing
    audit(root)
    rows = read_csv(root/"train_labels.csv", ["id", "filename", "label"])
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for label in range(10):
        ids = sorted([r["id"] for r in rows if int(r["label"]) == label], key=int)
        if len(ids) < 2:
            raise ValueError(f"Class {label} needs at least two images")
        rng.shuffle(ids)
        n_val = max(1, min(len(ids)-1, round(len(ids)*val_fraction)))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    train_ids.sort(key=int)
    val_ids.sort(key=int)
    by_id = {row["id"]: row for row in rows}
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for i, ident in enumerate(train_ids):
        path = image_path(root, "train_images", by_id[ident]["filename"])
        with Image.open(path) as image:
            if image.size != (32, 32) or image.mode != "RGB":
                raise ValueError(f"Unexpected image format: {path}")
            pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)/255.0
        channel_sum += pixels.sum(0)
        channel_sq += np.square(pixels).sum(0)
        count += len(pixels)
        if (i+1) % 10000 == 0:
            print(f"Train-only normalization: {i+1}/{len(train_ids)}", flush=True)
    mean = channel_sum/count
    std = np.sqrt(np.maximum(channel_sq/count - mean**2, 0))
    if np.any(std <= 0):
        raise ValueError("Zero-variance image channel")
    result = {"version": 1, "seed": seed, "val_fraction": val_fraction,
              "train_csv_sha256": file_hash(root/"train_labels.csv"),
              "train_ids": train_ids, "val_ids": val_ids,
              "mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES}
    write_json(output, result)
    return result


def load_split(root, path):
    split = read_json(path)
    if split["train_csv_sha256"] != file_hash(Path(root)/"train_labels.csv"):
        raise ValueError("Training CSV differs from the saved split")
    rows = read_csv(Path(root)/"train_labels.csv", ["id", "filename", "label"])
    ids = {r["id"] for r in rows}
    a, b = split["train_ids"], split["val_ids"]
    if not a or not b or len(a) != len(set(a)) or len(b) != len(set(b)):
        raise ValueError("Invalid split sizes or duplicate IDs")
    if set(a) & set(b) or set(a) | set(b) != ids:
        raise ValueError("Split overlaps or does not cover all training IDs")
    validate_normalization(split)
    lookup = {r["id"]: r for r in rows}
    return split, [lookup[i] for i in a], [lookup[i] for i in b]


def make_transform(mean, std, training, cutout_length=0, cutout_probability=1.0,
                   color_jitter=None, random_grayscale_probability=0.0):
    validate_normalization({"mean": mean, "std": std})
    validate_color_augmentation({"color_jitter": color_jitter,
                                 "random_grayscale_probability": random_grayscale_probability})
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
    if training:
        jitter = color_jitter or {}
        strengths = {key: jitter.get(key, 0.0) for key in ("brightness", "contrast", "saturation", "hue")}
        probability = jitter.get("probability", 1.0)
        if probability and any(strengths.values()):
            operation = transforms.ColorJitter(**strengths)
            steps.append(transforms.RandomApply([operation], p=probability) if probability < 1 else operation)
        if random_grayscale_probability:
            steps.append(transforms.RandomGrayscale(p=random_grayscale_probability))
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if training and cutout_length:
        steps.append(Cutout(cutout_length, cutout_probability))
    return transforms.Compose(steps)


class ImageDataset(Dataset):
    def __init__(self, root, rows, transform, labeled=True):
        self.root, self.rows, self.transform, self.labeled = Path(root), rows, transform, labeled

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        folder = "train_images" if self.labeled else "test_images"
        with Image.open(image_path(self.root, folder, row["filename"])) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, int(row["label"]) if self.labeled else row["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--data-dir", default="data")
    audit_parser.add_argument("--verify-images", action="store_true")
    prep_parser = sub.add_parser("prepare")
    prep_parser.add_argument("--data-dir", default="data")
    prep_parser.add_argument("--output", default="splits/seed2026.json")
    prep_parser.add_argument("--seed", type=int, default=2026)
    prep_parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "audit":
        print(audit(args.data_dir, args.verify_images))
    else:
        s = prepare_split(args.data_dir, args.output, args.seed, args.val_fraction)
        print({"train": len(s["train_ids"]), "validation": len(s["val_ids"]), "mean": s["mean"], "std": s["std"]})


if __name__ == "__main__":
    main()
