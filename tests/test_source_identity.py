"""Synthetic fixtures exercise opt-in source identity without touching run artifacts."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from src.data import prepare_split
from src.train import train, validate_config
from src.utils import file_hash, read_json


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="resnet-source-identity-")
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        (cls.data / "train_images").mkdir(parents=True)
        rng = np.random.default_rng(19)
        rows = []
        for index in range(40):
            name = f"train_{index}.png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data / "train_images" / name)
            rows.append([str(index), name, str(index % 10)])
        with (cls.data / "train_labels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "filename", "label"])
            writer.writerows(rows)
        # prepare_split audits the full manifest even though these tests never infer test images.
        (cls.data / "test_images").mkdir()
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data / "test_images/test.png")
        (cls.data / "test.csv").write_text("id,filename\n0,test.png\n", encoding="utf-8")
        (cls.data / "sample_submission.csv").write_text("id,label\n0,0\n", encoding="utf-8")
        cls.split = cls.root / "split.json"
        prepare_split(cls.data, cls.split, seed=2026, val_fraction=0.25)
        cls.template = read_json(Path(__file__).resolve().parents[1] / "configs/color_robustness_wide.json")
        cls.template.update(data_dir=str(cls.data), split_file=str(cls.split))
        cls.template["model"].update(channels=[4, 8, 16], blocks=[1, 1, 1])
        cls.template["training"].update(batch_size=8, max_train_batches=1, max_val_batches=1)
        cls.template["runtime"].update(device="cpu", num_workers=0, cpu_threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configuration(self, suffix, strict=True):
        config = copy.deepcopy(self.template)
        config["run_dir"] = str(self.root / suffix)
        if strict:
            config["runtime"]["strict_source_resume"] = True
        return config

    def quiet_train(self, config, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(config, **kwargs)

    @staticmethod
    def run_hashes(config):
        return {path.name: file_hash(path) for path in Path(config["run_dir"]).iterdir() if path.is_file()}

    def test_strict_source_identity_is_saved_and_resumes(self):
        config = self.configuration("strict_roundtrip")
        self.quiet_train(config, stop_after_epoch=1)
        partial = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)
        expected = {p.name: file_hash(p) for p in (Path(__file__).resolve().parents[1] / "src").glob("*.py")}
        self.assertEqual(partial["identity"]["source_hashes"], expected)
        best = torch.load(Path(config["run_dir"]) / "best.pt", weights_only=True)
        self.assertEqual(best["identity"]["source_hashes"], expected)
        self.assertEqual(read_json(Path(config["run_dir"]) / "environment.json")["source_hashes"], expected)
        self.quiet_train(config, resume=True, stop_after_epoch=2)
        resumed = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)
        self.assertEqual(resumed["identity"], partial["identity"])
        self.assertEqual([row["epoch"] for row in resumed["history"]], [1, 2])

    def test_source_drift_rejected_before_state_loading_or_metadata_writes(self):
        config = self.configuration("source_rejected")
        self.quiet_train(config, stop_after_epoch=1)
        before = self.run_hashes(config)

        def drift(path):
            path = Path(path)
            return "0" * 64 if path.name == "models.py" and path.parent.name == "src" else file_hash(path)

        with patch("src.train.file_hash", side_effect=drift), patch.object(torch.nn.Module, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "Resume source hashes differ"):
                self.quiet_train(config, resume=True, stop_after_epoch=2)
            load.assert_not_called()
        self.assertEqual(before, self.run_hashes(config))

    def test_strict_setting_cannot_be_disabled_to_bypass_identity(self):
        config = self.configuration("strict_disabled")
        self.quiet_train(config, stop_after_epoch=1)
        before = self.run_hashes(config)
        for remove in (False, True):
            modified = copy.deepcopy(config)
            if remove:
                modified["runtime"].pop("strict_source_resume")
            else:
                modified["runtime"]["strict_source_resume"] = False
            with self.subTest(remove=remove), self.assertRaisesRegex(ValueError, "Resume config/split differs"):
                self.quiet_train(modified, resume=True, stop_after_epoch=2)
            self.assertEqual(before, self.run_hashes(config))

    def test_config_and_model_mismatches_preserve_completed_files(self):
        config = self.configuration("config_rejected")
        self.quiet_train(config, stop_after_epoch=1)
        before = self.run_hashes(config)
        for field in ("optimizer", "architecture"):
            modified = copy.deepcopy(config)
            if field == "optimizer":
                modified["training"]["momentum"] = 0.8
            else:
                modified["model"]["channels"] = [4, 8, 12]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Resume config/split differs"):
                self.quiet_train(modified, resume=True, stop_after_epoch=2)
            self.assertEqual(before, self.run_hashes(config))

    def test_legacy_identity_remains_unchanged_and_cannot_silently_opt_in(self):
        config = self.configuration("legacy", strict=False)
        self.quiet_train(config, stop_after_epoch=1)
        state = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)
        self.assertEqual(set(state["identity"]), {"model", "training", "seed", "split_sha256"})
        self.quiet_train(config, resume=True, stop_after_epoch=2)
        before = self.run_hashes(config)
        modified = copy.deepcopy(config)
        modified["runtime"]["strict_source_resume"] = True
        with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
            self.quiet_train(modified, resume=True, stop_after_epoch=3)
        self.assertEqual(before, self.run_hashes(config))

    def test_source_resume_flag_requires_boolean(self):
        for value in (None, 0, 1, "true", [], {}):
            config = self.configuration("invalid")
            config["runtime"]["strict_source_resume"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
