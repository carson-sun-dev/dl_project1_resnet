"""Packed RGB pixels from the permitted training archive, in training CSV order."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_rows(root):
    with (Path(root) / "train_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "filename", "label"]:
            raise ValueError("Unexpected training CSV columns")
        rows = list(reader)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Training IDs must be nonempty and unique")
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Training filenames must be unique")
    for row in rows:
        name = row["filename"]
        if (not row["id"] or not name or PurePosixPath(name).name != name
                or "\\" in name or row["label"] not in {str(i) for i in range(10)}):
            raise ValueError("Invalid training row")
    return rows


def load_training_pixels(root, rows):
    """Return uint8 NHWC pixels in the requested training-row order, with strict IDs."""
    root = Path(root)
    source_rows = training_rows(root)
    metadata = json.loads((root / "train_pixels.json").read_text(encoding="utf-8"))
    if metadata["train_csv_sha256"] != sha256_file(root / "train_labels.csv"):
        raise ValueError("Packed pixels belong to a different training CSV")
    path = root / "train_pixels.npy"
    if metadata["npy_sha256"] != sha256_file(path):
        raise ValueError("Packed training pixels checksum mismatch")
    pixels = np.load(path, mmap_mode="r", allow_pickle=False)
    if pixels.shape != (len(source_rows), 32, 32, 3) or pixels.dtype != np.uint8:
        raise ValueError("Packed training pixels have an invalid shape or dtype")
    lookup = {row["id"]: (index, row) for index, row in enumerate(source_rows)}
    indices = []
    for row in rows:
        if row["id"] not in lookup:
            raise ValueError("Requested ID is absent from the training data")
        index, expected = lookup[row["id"]]
        if row["filename"] != expected["filename"] or row.get("label") != expected["label"]:
            raise ValueError("Requested training row disagrees with its ID")
        indices.append(index)
    return np.asarray(pixels[indices])


def build(root, archive):
    root, archive = Path(root), Path(archive)
    rows = training_rows(root)
    output, manifest = root / "train_pixels.npy", root / "train_pixels.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("Training cache already exists; preserve its provenance")
    csv_hash = sha256_file(root / "train_labels.csv")
    pixels = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    with zipfile.ZipFile(archive) as zipped:
        csv_names = [n for n in zipped.namelist() if PurePosixPath(n).name == "train_labels.csv"]
        if len(csv_names) != 1:
            raise ValueError("Expected one training CSV in the archive")
        csv_name = csv_names[0]
        if hashlib.sha256(zipped.read(csv_name)).hexdigest() != csv_hash:
            raise ValueError("Archive CSV differs from the permitted workspace training CSV")
        prefix = csv_name[:-len("train_labels.csv")]
        for index, row in enumerate(rows):
            entry = prefix + "train_images/" + row["filename"]
            info = zipped.getinfo(entry)
            if info.file_size > 1024 * 1024:
                raise ValueError("Unexpected training image size")
            with Image.open(io.BytesIO(zipped.read(info))) as image:
                if image.mode != "RGB" or image.size != (32, 32) or image.format != "PNG":
                    raise ValueError("Expected a 32x32 RGB PNG training image")
                pixels[index] = np.asarray(image)
    metadata = {"version": 1, "rows": len(rows), "shape": list(pixels.shape),
                "dtype": "uint8", "layout": "NHWC", "order": "train_labels.csv row order",
                "train_csv_sha256": csv_hash, "source_archive_name": archive.name,
                "source_archive_sha256": sha256_file(archive),
                "pixels_sha256": hashlib.sha256(memoryview(pixels).cast("B")).hexdigest()}
    temporary = output.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, pixels, allow_pickle=False)
    metadata["npy_sha256"] = sha256_file(temporary)
    os.replace(temporary, output)
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.archive), indent=2))
