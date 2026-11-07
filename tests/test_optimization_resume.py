"""Real synthetic Color loaders verify controlled optimizer/EMA resume boundaries."""
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

from src.data import CLASSES
from src.models import build_model
from src.train import train
from src.utils import dependency_versions, epoch_lr, file_hash, precision_policy, read_json, write_json


PROJECT = Path(__file__).resolve().parents[1]
METHODS = {"sgd": (0.0, 0.0), "sam": (0.05, 0.0), "asam": (0.5, 0.01)}


class OptimizationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="resnet-optimization-resume-")
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        (cls.data / "train_images").mkdir(parents=True)
        rng = np.random.default_rng(93)
        rows, train_pixels = [], []
        for index in range(40):
            name = f"train_{index}.png"
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(cls.data / "train_images" / name)
            rows.append([str(index), name, str(index % 10)])
            if index < 30:
                train_pixels.append(pixels.reshape(-1, 3).astype(np.float64) / 255)
        with (cls.data / "train_labels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "filename", "label"])
            writer.writerows(rows)
        pixels = np.concatenate(train_pixels)
        cls.split = cls.root / "split.json"
        write_json(cls.split, {"version": 1, "seed": 2026, "val_fraction": 0.25,
            "train_csv_sha256": file_hash(cls.data / "train_labels.csv"),
            "train_ids": [str(i) for i in range(30)], "val_ids": [str(i) for i in range(30, 40)],
            "mean": pixels.mean(0).tolist(), "std": pixels.std(0).tolist(), "classes": CLASSES})
        cls.template = read_json(PROJECT / "configs/preactivation_wrn22_4_color.json")
        cls.template.update(data_dir=str(cls.data), split_file=str(cls.split))
        cls.template["model"].update(channels=[2, 4, 8], blocks=[1, 1, 1], stem_channels=2)
        # Retain actual crop/flip/jitter/grayscale/Cutout/MixUp strengths, exercising
        # Torch, NumPy and worker RNG. Only fixture size/model/batch limits shrink.
        cls.template["training"].update(epochs=400, batch_size=8, max_train_batches=2, max_val_batches=0)
        cls.template["runtime"].update(device="cpu", num_workers=0, cpu_threads=2,
                                       strict_source_resume=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configuration(self, suffix, method="sgd", workers=0, ema=True):
        config = copy.deepcopy(self.template)
        config["run_dir"] = str(self.root / suffix)
        config["runtime"]["num_workers"] = workers
        rho, eta = METHODS[method]
        config["training"]["optimization"] = {
            "schema_version": 1, "method": method, "rho": rho, "eta": eta,
            "ema": {"enabled": ema, "decay": .999, "warmup": True}}
        return config

    @staticmethod
    def quiet_train(config, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return train(config, **kwargs)

    @staticmethod
    def checkpoint(config):
        return torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)

    @staticmethod
    def run_hashes(config):
        return {p.name: file_hash(p) for p in Path(config["run_dir"]).iterdir() if p.is_file()}

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

    def test_three_epoch_resume_matches_uninterrupted_with_ema_and_workers(self):
        for method in METHODS:
            for workers in (0, 1):
                with self.subTest(method=method, workers=workers):
                    whole = self.configuration(f"whole_{method}_{workers}", method, workers)
                    resumed = self.configuration(f"resumed_{method}_{workers}", method, workers)
                    self.quiet_train(whole, stop_after_epoch=3)
                    first = self.quiet_train(resumed, stop_after_epoch=1)
                    self.assertEqual(first["completed_epochs"], 1)
                    self.assertEqual(first["planned_epochs"], 400)
                    self.assertFalse(first["complete"])
                    first_state = self.checkpoint(resumed)
                    self.assertEqual(first_state["epoch"], 1)
                    self.assertEqual(first_state["scheduler"]["planned_epochs"], 400)
                    result = self.quiet_train(resumed, resume=True, stop_after_epoch=3)
                    self.assertEqual(result["completed_epochs"], 3)
                    self.assertFalse(result["complete"])
                    a, b = self.checkpoint(whole), self.checkpoint(resumed)
                    for key in ("model", "optimizer", "scheduler", "ema", "optimization_steps",
                                "rng", "train_generator", "val_generator", "identity", "precision_policy",
                                "scaler", "best_score", "best_epoch", "ema_best_score", "ema_best_epoch",
                                "skipped_updates"):
                        self.assert_nested_equal(a[key], b[key])
                    self.assertEqual(b["identity"]["training"]["optimization"],
                                     resumed["training"]["optimization"])
                    self.assertIn("optimization.py", b["identity"]["source_hashes"])
                    self.assertEqual(b["identity"]["optimization_schema"], 1)
                    self.assertEqual(b["precision_policy"], precision_policy(None))
                    self.assertEqual(b["optimization_steps"], 6)
                    self.assertEqual(b["ema"]["updates"], 6)
                    self.assertEqual(b["scheduler"], {
                        "version": 1, "implementation": "epoch_lr_v1", "completed_epochs": 3,
                        "planned_epochs": 400, "last_lr": epoch_lr(2, resumed["training"])})
                    self.assertEqual([h["epoch"] for h in b["history"]], [1, 2, 3])
                    for row, expected_lr in zip(b["history"], (.02, .04, .06)):
                        self.assertAlmostEqual(row["lr"], expected_lr, places=12)
                    self.assertAlmostEqual(epoch_lr(3, resumed["training"]), .08)
                    self.assertAlmostEqual(epoch_lr(399, resumed["training"]), .001)
                    # Each BN sees two original batches per completed epoch,
                    # regardless of whether optimization requires a second pass.
                    for key, value in b["model"].items():
                        if key.endswith("num_batches_tracked"):
                            self.assertEqual(value.item(), 6, key)
                    for left, right in zip(a["history"], b["history"]):
                        deterministic_fields = ["epoch", "lr", "train_loss", "train_accuracy", "train_images",
                            "val_loss", "val_accuracy", "val_images", "train_base_loss", "train_perturbed_loss",
                            "base_gradient_norm_mean", "perturbed_gradient_norm_mean", "perturbation_norm_max",
                            "adaptive_perturbation_norm_max", "finite_gradient_checks", "optimizer_steps",
                            "amp_skipped_updates", "train_model_views", "ema_val_loss_uncalibrated",
                            "ema_val_accuracy_uncalibrated", "ema_updates"]
                        for key in deterministic_fields:
                            self.assertEqual(left[key], right[key], key)
                            if right[key] is not None:
                                self.assertTrue(np.isfinite(right[key]), key)
                        self.assertEqual(right["train_images"], 16)
                        self.assertEqual(right["val_images"], 10)
                        self.assertEqual(right["optimizer_steps"], 2)
                        self.assertEqual(right["amp_skipped_updates"], 0)
                        self.assertEqual(right["finite_gradient_checks"], 2 if method == "sgd" else 4)
                        self.assertEqual(right["train_model_views"], 16 if method == "sgd" else 32)
                    for filename in ("best.pt", "ema_last.pt", "ema_best.pt"):
                        left = torch.load(Path(whole["run_dir"]) / filename, weights_only=True)
                        right = torch.load(Path(resumed["run_dir"]) / filename, weights_only=True)
                        self.assert_nested_equal(left, right)
                        loaded_model = build_model(right["model_config"])
                        loaded_model.load_state_dict(right["model"])
                    before = self.run_hashes(resumed)
                    with self.assertRaises(FileExistsError):
                        self.quiet_train(resumed, stop_after_epoch=3)
                    self.assertEqual(before, self.run_hashes(resumed))

    def test_ema_tracking_does_not_change_online_sgd_training_trajectory(self):
        enabled = self.configuration("ema_enabled", ema=True)
        disabled = self.configuration("ema_disabled", ema=False)
        self.quiet_train(enabled, stop_after_epoch=3)
        self.quiet_train(disabled, stop_after_epoch=3)
        a, b = self.checkpoint(enabled), self.checkpoint(disabled)
        for key in ("model", "optimizer", "rng", "train_generator", "scheduler", "optimization_steps"):
            self.assert_nested_equal(a[key], b[key])
        for left, right in zip(a["history"], b["history"]):
            for key in ("train_loss", "train_accuracy", "train_base_loss", "base_gradient_norm_mean",
                        "val_loss", "val_accuracy", "optimizer_steps"):
                self.assertEqual(left[key], right[key], key)
        self.assertIsNone(b["ema"])
        self.assertNotIn("ema_val_loss_uncalibrated", b["history"][-1])
        self.assertFalse((Path(disabled["run_dir"]) / "ema_best.pt").exists())

    def test_resume_rejects_config_runtime_and_source_drift_without_writes(self):
        config = self.configuration("reject_identity", "sam")
        self.quiet_train(config, stop_after_epoch=1)
        before = self.run_hashes(config)
        changes = []
        for key, value in (("num_workers", 1), ("cpu_threads", 1), ("amp", False)):
            changed = copy.deepcopy(config)
            changed["runtime"][key] = value
            changes.append(changed)
        for field, value in (("epochs", 401), ("mixup_alpha", .3)):
            changed = copy.deepcopy(config)
            changed["training"][field] = value
            changes.append(changed)
        for field, value in (("rho", .06), ("method", "asam")):
            changed = copy.deepcopy(config)
            changed["training"]["optimization"][field] = value
            changes.append(changed)
        changed = copy.deepcopy(config)
        changed["training"]["optimization"]["ema"]["decay"] = .998
        changes.append(changed)
        for changed in changes:
            with patch.object(torch.nn.Module, "load_state_dict") as loading:
                with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
                    self.quiet_train(changed, resume=True, stop_after_epoch=2)
                loading.assert_not_called()
            self.assertEqual(before, self.run_hashes(config))

        def drift(path):
            path = Path(path)
            return "0" * 64 if path.parent.name == "src" and path.name == "optimization.py" else file_hash(path)

        with patch("src.train.file_hash", side_effect=drift), patch.object(torch.nn.Module, "load_state_dict") as loading:
            with self.assertRaisesRegex(ValueError, "Resume source hashes differ"):
                self.quiet_train(config, resume=True, stop_after_epoch=2)
            loading.assert_not_called()
        self.assertEqual(before, self.run_hashes(config))
        software = dependency_versions()
        software["numpy"] = "0.0.test-drift"
        with patch("src.train.dependency_versions", return_value=software), \
                patch.object(torch.nn.Module, "load_state_dict") as loading:
            with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
                self.quiet_train(config, resume=True, stop_after_epoch=2)
            loading.assert_not_called()
        self.assertEqual(before, self.run_hashes(config))

    def test_resume_rejects_scheduler_ema_update_and_precision_corruption_before_loading(self):
        config = self.configuration("reject_state", "asam")
        self.quiet_train(config, stop_after_epoch=1)
        path = Path(config["run_dir"]) / "last.pt"
        original = path.read_bytes()
        original_hashes = self.run_hashes(config)
        base = self.checkpoint(config)
        mutations = []
        changed = copy.deepcopy(base)
        changed["scheduler"]["planned_epochs"] = 401
        mutations.append((changed, ValueError, "(?i)scheduler"))
        changed = copy.deepcopy(base)
        changed["ema"]["updates"] += 1
        mutations.append((changed, ValueError, "EMA update counter"))
        changed = copy.deepcopy(base)
        changed["optimization_steps"] += 1
        mutations.append((changed, ValueError, "successful-update counter"))
        changed = copy.deepcopy(base)
        changed["ema"]["shadow"][next(iter(changed["ema"]["shadow"]))].fill_(float("nan"))
        mutations.append((changed, ValueError, "Non-finite optimization checkpoint tensor"))
        changed = copy.deepcopy(base)
        changed["model"][next(iter(changed["model"]))].fill_(float("nan"))
        mutations.append((changed, ValueError, "Non-finite optimization checkpoint tensor"))
        changed = copy.deepcopy(base)
        first_optimizer_state = next(iter(changed["optimizer"]["state"].values()))
        first_optimizer_state["momentum_buffer"].fill_(float("nan"))
        mutations.append((changed, ValueError, "Non-finite optimization checkpoint tensor"))
        for key, value in (("momentum", .8), ("weight_decay", .0006), ("dampening", .1),
                           ("nesterov", True), ("maximize", True)):
            changed = copy.deepcopy(base)
            changed["optimizer"]["param_groups"][0][key] = value
            mutations.append((changed, ValueError, "Optimizer settings differ"))
        changed = copy.deepcopy(base)
        changed["precision_policy"] = precision_policy(torch.bfloat16)
        mutations.append((changed, ValueError, "Unsupported resume precision transition"))
        try:
            for changed, exception, message in mutations:
                torch.save(changed, path)
                before_rejection = self.run_hashes(config)
                with patch.object(torch.nn.Module, "load_state_dict") as loading:
                    with self.assertRaisesRegex(exception, message):
                        self.quiet_train(config, resume=True, stop_after_epoch=2)
                    loading.assert_not_called()
                self.assertEqual(before_rejection, self.run_hashes(config))
        finally:
            path.write_bytes(original)
        self.assertEqual(original_hashes, self.run_hashes(config))

    def test_optimization_requires_strict_identity_before_writing_run_config(self):
        config = self.configuration("strict_identity_required")
        config["runtime"]["strict_source_resume"] = False
        with self.assertRaisesRegex(ValueError, "strict_source_resume"):
            self.quiet_train(config, stop_after_epoch=1)
        self.assertFalse((Path(config["run_dir"]) / "config.json").exists())

    def test_resume_rejects_nonfinite_telemetry_before_loading(self):
        config = self.configuration("reject_history_telemetry", "sam")
        self.quiet_train(config, stop_after_epoch=1)
        path = Path(config["run_dir"]) / "last.pt"
        original = path.read_bytes()
        state = self.checkpoint(config)
        state["history"][0]["perturbation_norm_max"] = float("nan")
        torch.save(state, path)
        before = self.run_hashes(config)
        try:
            with patch.object(torch.nn.Module, "load_state_dict") as loading:
                with self.assertRaisesRegex(ValueError, "Non-finite optimization checkpoint history"):
                    self.quiet_train(config, resume=True, stop_after_epoch=2)
                loading.assert_not_called()
            self.assertEqual(before, self.run_hashes(config))
        finally:
            path.write_bytes(original)

    def test_controlled_training_restores_global_backend_policy(self):
        def policy():
            return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic)
        original = policy()
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            before = policy()
            self.quiet_train(self.configuration("restore_backend"), stop_after_epoch=1)
            self.assertEqual(before, policy())
        finally:
            torch.set_float32_matmul_precision(original[0])
            torch.backends.cuda.matmul.allow_tf32 = original[1]
            torch.backends.cudnn.allow_tf32 = original[2]
            torch.backends.cudnn.benchmark = original[3]
            torch.backends.cudnn.deterministic = original[4]

    def test_legacy_compute_benchmark_cannot_silently_benchmark_sam_as_sgd(self):
        from src.benchmark import benchmark
        config = self.configuration("benchmark_rejected", "sam")
        path = self.root / "sam_benchmark_config.json"
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "bounded src.train pilot"):
            benchmark(path, self.root / "not_written.json")
        self.assertFalse((self.root / "not_written.json").exists())

    def test_legacy_checkpoint_without_new_optimizer_scheduler_or_ema_schema_resumes(self):
        whole = self.configuration("legacy_whole")
        del whole["training"]["optimization"]
        whole["runtime"]["strict_source_resume"] = False
        whole["training"]["epochs"] = 200
        resumed = copy.deepcopy(whole)
        resumed["run_dir"] = str(self.root / "legacy_resumed")
        self.quiet_train(whole, stop_after_epoch=3)
        self.quiet_train(resumed, stop_after_epoch=1)
        first = self.checkpoint(resumed)
        self.assertNotIn("optimization_schema", first["identity"])
        self.assertNotIn("source_hashes", first["identity"])
        for key in ("scheduler", "ema", "optimization_steps", "ema_best_score", "ema_best_epoch"):
            first.pop(key, None)
        torch.save(first, Path(resumed["run_dir"]) / "last.pt")
        result = self.quiet_train(resumed, resume=True, stop_after_epoch=3)
        self.assertEqual(result["completed_epochs"], 3)
        self.assertEqual(result["planned_epochs"], 200)
        a, b = self.checkpoint(whole), self.checkpoint(resumed)
        for key in ("model", "optimizer", "rng", "train_generator", "val_generator", "identity"):
            self.assert_nested_equal(a[key], b[key])
        self.assertNotIn("scheduler", b)
        self.assertNotIn("ema", b)
        self.assertNotIn("optimization_steps", b)


if __name__ == "__main__":
    unittest.main()
