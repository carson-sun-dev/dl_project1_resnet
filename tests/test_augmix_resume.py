"""Exercise AugMix through real synthetic loaders and epoch-boundary checkpoints."""
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

from src.augmix import AugMixTransform
from src.data import CLASSES, ImageDataset, load_split, make_transform
from src.models import build_model
from src.train import train
from src.utils import epoch_lr, file_hash, precision_policy, read_json, write_json


PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = {"width": 3, "depth": -1, "severity": 3,
            "mixture_concentration": 1.0, "js_coefficient": 12.0, "all_ops": False}


class AugMixResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="resnet-augmix-resume-")
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        (cls.data / "train_images").mkdir(parents=True)
        rng = np.random.default_rng(81)
        rows, training_pixels = [], []
        for index in range(40):
            name = f"train_{index}.png"
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(cls.data / "train_images" / name)
            rows.append([str(index), name, str(index % 10)])
            if index < 30:
                training_pixels.append(pixels.reshape(-1, 3).astype(np.float64) / 255)
        with (cls.data / "train_labels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "filename", "label"])
            writer.writerows(rows)
        pixels = np.concatenate(training_pixels)
        cls.split = cls.root / "split.json"
        write_json(cls.split, {"version": 1, "seed": 2026, "val_fraction": 0.25,
                              "train_csv_sha256": file_hash(cls.data / "train_labels.csv"),
                              "train_ids": [str(i) for i in range(30)],
                              "val_ids": [str(i) for i in range(30, 40)],
                              "mean": pixels.mean(0).tolist(), "std": pixels.std(0).tolist(),
                              "classes": CLASSES})
        cls.template = read_json(PROJECT / "configs/color_robustness_wide.json")
        cls.template.update(data_dir=str(cls.data), split_file=str(cls.split))
        cls.template["model"].update(channels=[2, 4, 8], blocks=[1, 1, 1])
        cls.template["training"].update(
            batch_size=8, max_train_batches=2, max_val_batches=0, label_smoothing=0.0,
            mixup_alpha=0.0, mixup_probability=0.0, cutout_length=0,
            cutout_probability=0.0, color_jitter=None, random_grayscale_probability=0.0,
            augmix=copy.deepcopy(SETTINGS))
        cls.template["runtime"].update(device="cpu", num_workers=0, cpu_threads=2,
                                       strict_source_resume=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configuration(self, suffix, workers=0):
        config = copy.deepcopy(self.template)
        config["run_dir"] = str(self.root / suffix)
        config["runtime"]["num_workers"] = workers
        return config

    def quiet_train(self, config, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(config, **kwargs)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def run_hashes(config):
        return {path.name: file_hash(path) for path in Path(config["run_dir"]).iterdir() if path.is_file()}

    def test_three_epoch_resume_matches_uninterrupted_model_optimizer_rng_and_losses(self):
        # Both paths exercise Torch geometric RNG and NumPy operator/mixture RNG.
        for workers in (0, 1):
            with self.subTest(workers=workers):
                config = self.configuration(f"uninterrupted_{workers}", workers)
                self.quiet_train(config, stop_after_epoch=3)
                resumed = self.configuration(f"resumed_{workers}", workers)
                first = self.quiet_train(resumed, stop_after_epoch=1)
                self.assertEqual(first["completed_epochs"], 1)
                self.assertEqual(first["planned_epochs"], 200)
                self.assertFalse(first["complete"])
                first_state = torch.load(Path(resumed["run_dir"]) / "last.pt", weights_only=False)
                self.assertEqual(first_state["epoch"], 1)
                result = self.quiet_train(resumed, resume=True, stop_after_epoch=3)
                self.assertEqual(result["completed_epochs"], 3)
                self.assertFalse(result["complete"])
                a = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)
                b = torch.load(Path(resumed["run_dir"]) / "last.pt", weights_only=False)
                for key in ("model", "optimizer", "rng", "train_generator", "val_generator",
                            "identity", "precision_policy", "scaler", "best_score", "best_epoch",
                            "skipped_updates"):
                    self.assert_nested_equal(a[key], b[key])
                self.assertEqual(b["identity"]["training"]["augmix"], SETTINGS)
                self.assertIn("augmix.py", b["identity"]["source_hashes"])
                self.assertEqual(b["precision_policy"], precision_policy(None))
                self.assertEqual(b["epoch"], 3)
                self.assertEqual([row["epoch"] for row in b["history"]], [1, 2, 3])
                self.assertEqual([row["lr"] for row in b["history"]],
                                 [epoch_lr(i, config["training"]) for i in range(3)])
                self.assertAlmostEqual(epoch_lr(3, config["training"]), 0.08)
                for uninterrupted, continued in zip(a["history"], b["history"]):
                    for key in ("epoch", "lr", "train_loss", "train_ce", "train_js",
                                "train_accuracy", "train_images", "train_model_views", "finite_gradient_checks",
                                "optimizer_steps", "amp_skipped_updates", "val_loss", "val_accuracy", "val_images"):
                        self.assertEqual(uninterrupted[key], continued[key], key)
                    self.assertEqual(continued["train_images"], 16)
                    self.assertEqual(continued["train_model_views"], 48)
                    self.assertEqual(continued["finite_gradient_checks"], 2)
                    self.assertEqual(continued["optimizer_steps"], 2)
                    self.assertEqual(continued["amp_skipped_updates"], 0)
                    self.assertEqual(continued["val_images"], 10)
                    self.assertAlmostEqual(continued["model_views_per_second"],
                                           3 * continued["images_per_second"])
                    self.assertTrue(np.isfinite(continued["train_ce"]))
                    self.assertTrue(np.isfinite(continued["train_js"]))
                before = self.run_hashes(resumed)
                with self.assertRaises(FileExistsError):
                    self.quiet_train(resumed, stop_after_epoch=3)
                self.assertEqual(before, self.run_hashes(resumed))

    def test_dataset_keeps_original_label_and_validation_is_single_view(self):
        split, train_rows, val_rows = load_split(self.data, self.split)
        training = ImageDataset(self.data, train_rows,
                                AugMixTransform(split["mean"], split["std"], SETTINGS))
        views, label = training[7]
        self.assertEqual(label, int(train_rows[7]["label"]))
        self.assertEqual(len(views), 3)
        lower = -torch.tensor(split["mean"]) / torch.tensor(split["std"])
        upper = (1 - torch.tensor(split["mean"])) / torch.tensor(split["std"])
        for view in views:
            self.assertEqual(view.shape, (3, 32, 32))
            self.assertEqual(view.dtype, torch.float32)
            self.assertTrue(torch.isfinite(view).all())
            self.assertTrue((view >= lower[:, None, None] - 1e-6).all())
            self.assertTrue((view <= upper[:, None, None] + 1e-6).all())
        validation = ImageDataset(self.data, val_rows,
                                  make_transform(split["mean"], split["std"], False))
        before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        first, first_label = validation[0]
        second, second_label = validation[0]
        self.assertIsInstance(first, torch.Tensor)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first_label, second_label)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assert_nested_equal(np.random.get_state(), numpy_before)
        with Image.open(self.data / "train_images" / val_rows[0]["filename"]) as image:
            raw = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        expected = (raw - torch.tensor(split["mean"])[:, None, None]) / torch.tensor(split["std"])[:, None, None]
        torch.testing.assert_close(first, expected, rtol=0, atol=0)

    def test_augmix_resume_rejects_source_config_and_precision_drift_without_writes(self):
        config = self.configuration("reject_drift")
        self.quiet_train(config, stop_after_epoch=1)
        before = self.run_hashes(config)

        def drift(path):
            path = Path(path)
            return "0" * 64 if path.name == "augmix.py" and path.parent.name == "src" else file_hash(path)

        with patch("src.train.file_hash", side_effect=drift), patch.object(torch.nn.Module, "load_state_dict") as load:
            with self.assertRaisesRegex(ValueError, "Resume source hashes differ"):
                self.quiet_train(config, resume=True, stop_after_epoch=2)
            load.assert_not_called()
        self.assertEqual(before, self.run_hashes(config))
        changed = copy.deepcopy(config)
        changed["training"]["augmix"]["js_coefficient"] = 6.0
        with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
            self.quiet_train(changed, resume=True, stop_after_epoch=2)
        self.assertEqual(before, self.run_hashes(config))
        changed_workers = copy.deepcopy(config)
        changed_workers["runtime"]["num_workers"] = 1
        with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
            self.quiet_train(changed_workers, resume=True, stop_after_epoch=2)
        self.assertEqual(before, self.run_hashes(config))
        checkpoint = Path(config["run_dir"]) / "last.pt"
        original_bytes = checkpoint.read_bytes()
        state = torch.load(checkpoint, weights_only=False)
        state["precision_policy"] = precision_policy(torch.bfloat16)
        torch.save(state, checkpoint)
        incompatible = self.run_hashes(config)
        with self.assertRaisesRegex(ValueError, "Unsupported resume precision transition"):
            self.quiet_train(config, resume=True, stop_after_epoch=2)
        self.assertEqual(incompatible, self.run_hashes(config))
        checkpoint.write_bytes(original_bytes)
        self.assertEqual(before, self.run_hashes(config))

    def test_nonfinite_gradient_aborts_before_optimizer_step_or_checkpoint(self):
        config = self.configuration("nonfinite_gradient")

        def poisoned_model(model_config):
            model = build_model(model_config)
            next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return model

        with patch("src.train.build_model", side_effect=poisoned_model), \
                patch.object(torch.optim.SGD, "step", autospec=True) as step:
            with self.assertRaisesRegex(RuntimeError, "Non-finite training gradients"):
                self.quiet_train(config, stop_after_epoch=1)
            step.assert_not_called()
        run = Path(config["run_dir"])
        self.assertFalse((run / "best.pt").exists())
        self.assertFalse((run / "last.pt").exists())
        self.assertFalse((run / "metrics.json").exists())

    def test_augmix_requires_strict_source_identity_before_writing_config(self):
        config = self.configuration("strict_required")
        config["runtime"]["strict_source_resume"] = False
        with self.assertRaisesRegex(ValueError, "AugMix requires strict_source_resume"):
            self.quiet_train(config, stop_after_epoch=1)
        run = Path(config["run_dir"])
        self.assertFalse((run / "config.json").exists())
        self.assertFalse((run / "best.pt").exists())
        self.assertFalse((run / "last.pt").exists())


if __name__ == "__main__":
    unittest.main()
