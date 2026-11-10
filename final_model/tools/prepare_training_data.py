"""Reproduce the frozen split and Phase0 decoded-RGB manifest, training data only.

Only train_labels.csv and named files under train_images/ are read. The output
uses the original CSV row order, integer labels, RGB bytes and CRLF records.
The production entry point requires every frozen digest before writing. Existing
files are reusable only when their bytes match; mismatches are never overwritten.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
CONFIG_FILE = PACKAGE_ROOT / "configs/final_refit_preact_asam400_s2026_cloud.json"
COLUMNS = ["id", "filename", "label", "partition", "decoded_rgb_sha256"]
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_rows(data_dir):
    """Reject malformed IDs, labels, duplicate names and nonportable paths first."""
    with (Path(data_dir) / "train_labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "filename", "label"]:
            raise ValueError("Expected train_labels.csv columns id,filename,label")
        rows = list(reader)
    if not rows:
        raise ValueError("Training CSV is empty")
    ids, names = set(), set()
    for row in rows:
        if set(row) != {"id", "filename", "label"} or any(value is None for value in row.values()):
            raise ValueError("Malformed training CSV row")
        ident, filename, label = row["id"], row["filename"], row["label"]
        if not re.fullmatch(r"0|[1-9][0-9]*", ident):
            raise ValueError("Training IDs must be canonical nonnegative decimal integers")
        # Restrict names to the supplied train_<ID>.png convention. This rejects
        # separators, absolute paths, drive names, ADS and ambiguous Windows names.
        if filename != f"train_{int(ident):05d}.png":
            raise ValueError(f"Unsafe or unexpected training image filename: {filename!r}")
        if label not in {str(index) for index in range(10)}:
            raise ValueError("Training labels must be integer strings 0..9")
        if ident in ids or filename in names:
            raise ValueError("Duplicate training ID or filename")
        ids.add(ident)
        names.add(filename)
    return rows


def training_image_path(data_dir, filename):
    if not re.fullmatch(r"train_[0-9]{5,}\.png", filename):
        raise ValueError("Unsafe training image filename")
    root = Path(data_dir).resolve() / "train_images"
    if root.resolve() != root:
        raise ValueError("train_images directory resolves away from the permitted training directory")
    path = root / filename
    if not path.resolve().is_relative_to(root):
        raise ValueError("Training image escapes the permitted train_images directory")
    return path


def read_rgb(data_dir, filename):
    path = training_image_path(data_dir, filename)
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (32, 32) or image.format != "PNG":
            raise ValueError(f"Expected a 32x32 RGB PNG: {filename}")
        return np.asarray(image).copy()


def validate_split(split, rows, train_csv_sha256):
    if (split.get("version") != 1 or split.get("seed") != 2026
            or split.get("val_fraction") != 0.1 or split.get("classes") != CLASSES
            or split.get("train_csv_sha256") != train_csv_sha256):
        raise ValueError("Training split settings, class map or CSV identity differ")
    train, validation = split.get("train_ids"), split.get("val_ids")
    if not isinstance(train, list) or not isinstance(validation, list) or not train or not validation:
        raise ValueError("Invalid training split partitions")
    if any(not isinstance(ident, str) for ident in train + validation):
        raise ValueError("Split IDs must be strings")
    if (len(set(train)) != len(train) or len(set(validation)) != len(validation)
            or set(train) & set(validation)
            or set(train) | set(validation) != {row["id"] for row in rows}):
        raise ValueError("Training split overlaps, duplicates or does not cover the CSV")
    for key in ("mean", "std"):
        values = split.get(key)
        if (not isinstance(values, list) or len(values) != 3
                or any(type(value) not in (float, int) or not np.isfinite(value) for value in values)
                or key == "std" and any(value <= 0 for value in values)):
            raise ValueError("Invalid saved normalization")
    return split


def create_split(data_dir, rows, train_csv_sha256):
    """Mirror frozen src.data.prepare_split without its hidden-data audit."""
    rng = np.random.default_rng(2026)
    train_ids, val_ids = [], []
    for label in range(10):
        ids = sorted([row["id"] for row in rows if int(row["label"]) == label], key=int)
        if len(ids) < 2:
            raise ValueError(f"Class {label} needs at least two training images")
        rng.shuffle(ids)
        count = max(1, min(len(ids) - 1, round(len(ids) * 0.1)))
        val_ids.extend(ids[:count])
        train_ids.extend(ids[count:])
    train_ids.sort(key=int)
    val_ids.sort(key=int)
    lookup = {row["id"]: row for row in rows}
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for ident in train_ids:
        pixels = read_rgb(data_dir, lookup[ident]["filename"]).astype(np.float64).reshape(-1, 3) / 255.0
        channel_sum += pixels.sum(0)
        channel_sq += np.square(pixels).sum(0)
        count += len(pixels)
    mean = channel_sum / count
    std = np.sqrt(np.maximum(channel_sq / count - mean ** 2, 0))
    if np.any(std <= 0):
        raise ValueError("Zero-variance training channel")
    return {"version": 1, "seed": 2026, "val_fraction": 0.1,
            "train_csv_sha256": train_csv_sha256, "train_ids": train_ids,
            "val_ids": val_ids, "mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES}


def split_bytes(split):
    # The preserved split has LF records. Use bytes to avoid platform newline
    # translation and retain the same digest on Windows and Linux.
    return (json.dumps(split, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def build_manifest_bytes(data_dir, rows, split, progress=False):
    train_ids = set(split["train_ids"])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for index, row in enumerate(rows):
        array = read_rgb(data_dir, row["filename"])
        writer.writerow({"id": row["id"], "filename": row["filename"], "label": int(row["label"]),
                         "partition": "train" if row["id"] in train_ids else "validation",
                         "decoded_rgb_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
        if progress and (index + 1) % 10000 == 0:
            print(f"Hashed permitted training images {index + 1}/{len(rows)}", flush=True)
    return stream.getvalue().encode("utf-8")


def write_or_verify(path, payload, verify_only=False):
    """Exclusive creation; preserve any existing bytes, including mismatches."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path}")
        return "verified_existing"
    if verify_only:
        raise FileNotFoundError(f"Required existing file is absent: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return "created"


def prepare_training_data(data_dir="data", split_file="splits/seed2026.json",
                          output="provenance/training_image_hashes.csv", *,
                          verify_only=False, prepare_split=False):
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    frozen = config["refit"]
    csv_path = Path(data_dir) / "train_labels.csv"
    if sha256(csv_path) != frozen["train_csv_sha256"]:
        raise ValueError("Training CSV SHA256 differs from the frozen final recipe")
    rows = load_training_rows(data_dir)
    if (len(rows) != 50000 or {row["id"] for row in rows} != {str(index) for index in range(50000)}
            or Counter(row["label"] for row in rows) != {str(index): 5000 for index in range(10)}):
        raise ValueError("Expected the exact permitted balanced 50,000-image training corpus")
    path = Path(split_file)
    if verify_only and not Path(output).is_file():
        raise FileNotFoundError(f"Required existing manifest is absent: {output}")
    pending_split = None
    if path.exists():
        if sha256(path) != frozen["original_split_sha256"]:
            raise ValueError("Existing split SHA256 differs; refusing to overwrite")
        split = json.loads(path.read_text(encoding="utf-8"))
    elif verify_only or not prepare_split:
        raise FileNotFoundError("Original split absent; use --prepare-split or run.py data prepare")
    else:
        split = create_split(data_dir, rows, frozen["train_csv_sha256"])
        pending_split = split_bytes(split)
        if hashlib.sha256(pending_split).hexdigest() != frozen["original_split_sha256"]:
            raise ValueError("Recreated split digest differs; no files written")
    validate_split(split, rows, frozen["train_csv_sha256"])
    if len(split["train_ids"]) != 45000 or len(split["val_ids"]) != 5000:
        raise ValueError("Original split must contain 45,000/5,000 permitted images")
    payload = build_manifest_bytes(data_dir, rows, split, progress=True)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != frozen["image_manifest_sha256"]:
        raise ValueError("Recreated training-image manifest digest differs; no files written")
    # Validate existing output before creating even a missing split.
    if Path(output).exists() and Path(output).read_bytes() != payload:
        raise ValueError("Existing manifest differs; refusing to overwrite")
    split_action = write_or_verify(path, pending_split, verify_only) if pending_split is not None else "verified_existing"
    action = write_or_verify(output, payload, verify_only)
    return {"images": len(rows), "manifest_sha256": digest, "manifest_action": action,
            "split_sha256": frozen["original_split_sha256"], "split_action": split_action,
            "hidden_evaluation_data_accessed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-file", default="splits/seed2026.json")
    parser.add_argument("--output", default="provenance/training_image_hashes.csv")
    parser.add_argument("--prepare-split", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Require matching existing split and manifest; never write")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise ValueError("Run this tool from the repository root")
    # Share the lightweight identity checker without ever importing a src package.
    sys.path.insert(0, str(PACKAGE_ROOT))
    from run import verify_frozen_files
    verify_frozen_files()
    result = prepare_training_data(args.data_dir, args.split_file, args.output,
                                   verify_only=args.verify, prepare_split=args.prepare_split)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
