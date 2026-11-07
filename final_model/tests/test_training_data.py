"""Synthetic train-only preparation, exact CSV encoding and rejection checks."""
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_training_prep", PACKAGE_ROOT / "tools/prepare_training_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="final-train-only-")
        self.root = Path(self.temporary.name)
        (self.root / "train_images").mkdir()
        self.rows = []
        rng = np.random.default_rng(19)
        for index in range(20):
            row = {"id": str(index), "filename": f"train_{index:05d}.png", "label": str(index % 10)}
            self.rows.append(row)
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(self.root / "train_images" / row["filename"])
        self.write_rows(self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def write_rows(self, rows):
        with (self.root / "train_labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "filename", "label"])
            writer.writeheader()
            writer.writerows(rows)

    def split(self):
        rows = prep.load_training_rows(self.root)
        digest = prep.sha256(self.root / "train_labels.csv")
        return prep.create_split(self.root, rows, digest)

    def test_split_and_manifest_are_deterministic_with_only_training_files_present(self):
        # There is deliberately no test.csv, sample_submission.csv or test_images.
        rows = prep.load_training_rows(self.root)
        split = self.split()
        again = self.split()
        self.assertEqual(prep.split_bytes(split), prep.split_bytes(again))
        self.assertNotIn(b"\r", prep.split_bytes(split))
        prep.validate_split(split, rows, prep.sha256(self.root / "train_labels.csv"))
        payload = prep.build_manifest_bytes(self.root, rows, split)
        self.assertEqual(payload, prep.build_manifest_bytes(self.root, rows, again))
        self.assertEqual(payload.count(b"\r\n"), len(rows) + 1)
        parsed = list(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual([row["id"] for row in parsed], [row["id"] for row in rows])
        self.assertEqual({row["partition"] for row in parsed}, {"train", "validation"})
        first = prep.read_rgb(self.root, rows[0]["filename"])
        self.assertEqual(parsed[0]["decoded_rgb_sha256"], hashlib.sha256(first.tobytes()).hexdigest())

    def test_manifest_uses_csv_row_order_and_checks_only_train_image_paths(self):
        split = self.split()
        original_open = Image.open
        accessed = []
        def checked_open(path, *args, **kwargs):
            self.assertEqual(Path(path).resolve().parent, (self.root / "train_images").resolve())
            accessed.append(Path(path).name)
            return original_open(path, *args, **kwargs)
        with patch.object(prep.Image, "open", side_effect=checked_open):
            payload = prep.build_manifest_bytes(self.root, list(reversed(self.rows)), split)
        parsed = list(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual([row["id"] for row in parsed], [row["id"] for row in reversed(self.rows)])
        self.assertEqual(accessed, [row["filename"] for row in reversed(self.rows)])

    def test_malformed_ids_paths_labels_and_duplicates_are_rejected_before_image_decode(self):
        cases = [("id", "../0"), ("id", "00"), ("id", "-1"), ("id", ""),
                 ("filename", "../train_00000.png"), ("filename", "..\\train_00000.png"),
                 ("filename", "C:\\train_00000.png"), ("filename", "/train_00000.png"),
                 ("filename", "train_00000.png:stream"), ("filename", "train_00001.png"),
                 ("label", "10"), ("label", "1.0")]
        for key, value in cases:
            rows = [dict(row) for row in self.rows]
            rows[0][key] = value
            self.write_rows(rows)
            with self.subTest(key=key, value=value), patch.object(prep.Image, "open") as image_open:
                with self.assertRaises(ValueError):
                    prep.load_training_rows(self.root)
                image_open.assert_not_called()
        self.write_rows(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prep.load_training_rows(self.root)

    def test_symlink_escape_is_rejected_before_decode(self):
        filename = self.root / "train_images/train_00000.png"
        original_resolve = Path.resolve
        def resolved(path, *args, **kwargs):
            if path == filename:
                return self.root.parent / "outside.png"
            return original_resolve(path, *args, **kwargs)
        with patch.object(prep.Path, "resolve", resolved), self.assertRaisesRegex(ValueError, "escapes"):
            prep.training_image_path(self.root, filename.name)

    def test_redirected_train_images_directory_is_rejected(self):
        original_resolve = Path.resolve
        def resolved(path, *args, **kwargs):
            if path == self.root / "train_images":
                return self.root / "test_images"
            return original_resolve(path, *args, **kwargs)
        with patch.object(prep.Path, "resolve", resolved), self.assertRaisesRegex(ValueError, "resolves away"):
            prep.training_image_path(self.root, "train_00000.png")

    def test_rgb_format_and_split_overlap_rejected(self):
        Image.new("L", (32, 32)).save(self.root / "train_images/train_00000.png")
        with self.assertRaisesRegex(ValueError, "RGB PNG"):
            prep.read_rgb(self.root, "train_00000.png")
        split = {"version": 1, "seed": 2026, "val_fraction": 0.1, "classes": prep.CLASSES,
                 "train_csv_sha256": "digest", "train_ids": ["0"], "val_ids": ["0"],
                 "mean": [0.5] * 3, "std": [0.2] * 3}
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prep.validate_split(split, self.rows, "digest")

    def test_exclusive_outputs_reuse_matching_bytes_and_preserve_mismatches(self):
        output = self.root / "provenance/manifest.csv"
        payload = b"id,filename,label,partition,decoded_rgb_sha256\r\n"
        with self.assertRaises(FileNotFoundError):
            prep.write_or_verify(output, payload, verify_only=True)
        self.assertFalse(output.exists())
        self.assertEqual(prep.write_or_verify(output, payload), "created")
        self.assertEqual(prep.write_or_verify(output, payload), "verified_existing")
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            prep.write_or_verify(output, b"different")
        self.assertEqual(output.read_bytes(), payload)

    def test_production_gate_rejects_synthetic_corpus_before_any_image_decode_or_output(self):
        with patch.object(prep.Image, "open") as image_open:
            with self.assertRaisesRegex(ValueError, "CSV SHA256"):
                prep.prepare_training_data(self.root, self.root / "split.json", self.root / "manifest.csv")
            image_open.assert_not_called()
        self.assertFalse((self.root / "manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
