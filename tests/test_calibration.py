"""Training-only BN membership, precision, rollback and state-isolation checks."""
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

import src.data as data
from src.calibration import recalibrate_batch_norm
from src.utils import file_hash


class CalibrationModel(nn.Module):
    def __init__(self, fail_second=False, mutate_buffer=False, draw_random=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(3, momentum=0.23)
        self.dropout = nn.Dropout(0.8)
        self.head = nn.Linear(3, 10)
        self.register_buffer("auxiliary", torch.tensor(7.0))
        self.fail_second, self.mutate_buffer, self.draw_random = fail_second, mutate_buffer, draw_random
        self.calls, self.seen = 0, []

    def forward(self, images):
        self.calls += 1
        self.seen.append({"input": images.detach().clone(), "model_training": self.training,
                          "bn_training": self.bn.training, "dropout_training": self.dropout.training})
        if self.draw_random:
            random.random()
            np.random.random()
            torch.rand(1)
        if self.mutate_buffer:
            self.auxiliary.add_(1)
        logits = self.head(self.dropout(self.bn(images)).mean((2, 3)))
        return logits * float("nan") if self.fail_second and self.calls == 2 else logits


def snapshot(model):
    return {"state": {name: value.clone() for name, value in model.state_dict().items()},
            "modes": [m.training for m in model.modules()],
            "momenta": [m.momentum for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]}


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="bn-calibration-tests-")
        cls.root = Path(cls.temp.name)
        cls.data_dir = cls.root / "data"
        (cls.data_dir / "train_images").mkdir(parents=True)
        # Intentionally no hidden-test files, folders or manifests exist here.
        cls.rows = []
        for index in range(7):
            name = f"train_{index}.png"
            Image.fromarray(np.full((32, 32, 3), index * 40, dtype=np.uint8)).save(cls.data_dir / "train_images" / name)
            cls.rows.append({"id": str(index), "filename": name, "label": str(index)})
        with (cls.data_dir / "train_labels.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "filename", "label"])
            writer.writeheader()
            writer.writerows(cls.rows)
        cls.split = cls.root / "split.json"
        cls.split_info = {"train_ids": [str(i) for i in range(5)], "val_ids": ["5", "6"],
                          "classes": data.CLASSES, "mean": [0.0] * 3, "std": [1.0] * 3,
                          "train_csv_sha256": file_hash(cls.data_dir / "train_labels.csv")}
        cls.split.write_text(json.dumps(cls.split_info), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assert_model_equal(self, model, before):
        self.assertEqual(list(model.state_dict()), list(before["state"]))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before["state"][name]), name)
        self.assertEqual([m.training for m in model.modules()], before["modes"])
        self.assertEqual([m.momentum for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)],
                         before["momenta"])

    def call(self, model, **kwargs):
        options = dict(device="cpu", batch_size=2, seed=2026,
                       expected_split_sha256=file_hash(self.split))
        options.update(kwargs)
        return recalibrate_batch_norm(model, self.data_dir, self.split, **options)

    def test_only_training_images_weighted_partial_batch_parameters_and_modes(self):
        model = CalibrationModel().train()
        model.bn.eval()  # Restore heterogeneous module modes, not just root mode.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        parameters = {name: value.clone() for name, value in model.named_parameters()}
        gradients = {name: value.grad.clone() for name, value in model.named_parameters()}
        modes = [m.training for m in model.modules()]
        hashes = {str(path): file_hash(path) for path in self.data_dir.rglob("*") if path.is_file()}
        original_path = data.image_path
        accesses = []

        def training_only_path(root, folder, filename):
            self.assertEqual(Path(root), self.data_dir)
            self.assertEqual(folder, "train_images")
            self.assertIn(filename, {row["filename"] for row in self.rows[:5]})
            accesses.append(filename)
            return original_path(root, folder, filename)

        with patch.object(data, "image_path", side_effect=training_only_path):
            report = self.call(model)
        self.assertEqual(accesses, [row["filename"] for row in self.rows[:5]])
        self.assertEqual((report["training_images"], report["batches"], report["validation_images_used"],
                          report["hidden_images_used"]), (5, 3, 0, 0))
        self.assertTrue(report["parameters_bitwise_unchanged"])
        self.assertEqual(report["split_sha256"], file_hash(self.split))
        self.assertEqual(model.bn.num_batches_tracked.item(), 3)
        torch.testing.assert_close(model.bn.running_mean, torch.full((3,), 80 / 255), rtol=0, atol=1e-7)
        # Batch variances use PyTorch's unbiased per-batch estimator; sample-count
        # weighting gives the one-image final batch 1/5 rather than 1/3 weight.
        pair = torch.cat((torch.zeros(1024), torch.full((1024,), 40 / 255)))
        expected_variance = pair.var(unbiased=True) * 4 / 5
        torch.testing.assert_close(model.bn.running_var, expected_variance.expand(3), rtol=1e-6, atol=1e-8)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, parameters[name]), name)
            self.assertTrue(torch.equal(value.grad, gradients[name]), name)
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual(model.bn.momentum, 0.23)
        self.assertEqual(model.auxiliary.item(), 7)
        for record in model.seen:
            self.assertFalse(record["model_training"])
            self.assertTrue(record["bn_training"])
            self.assertFalse(record["dropout_training"])
            self.assertEqual(record["input"].dtype, torch.float32)
        self.assertEqual(hashes, {str(path): file_hash(path) for path in self.data_dir.rglob("*") if path.is_file()})

    def test_rng_isolation_and_deterministic_repeated_copies(self):
        original = CalibrationModel(draw_random=True)
        original_state = snapshot(original)
        first, second = copy.deepcopy(original), copy.deepcopy(original)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.call(first)
        self.assertEqual(random.getstate(), python_state)
        current_np = np.random.get_state()
        self.assertEqual(current_np[0], numpy_state[0])
        np.testing.assert_array_equal(current_np[1], numpy_state[1])
        self.assertEqual(current_np[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.call(second)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assert_model_equal(original, original_state)

    def test_failure_restores_all_buffers_modes_rng_and_momenta(self):
        model = CalibrationModel(fail_second=True, draw_random=True).eval()
        model.bn.train()
        model.bn.running_mean.fill_(0.4)
        model.bn.running_var.fill_(2)
        model.bn.num_batches_tracked.fill_(11)
        before = snapshot(model)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "Non-finite.*output"):
            self.call(model)
        self.assert_model_equal(model, before)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_non_batchnorm_buffer_mutation_is_detected_and_rolled_back(self):
        model = CalibrationModel(mutate_buffer=True)
        before = snapshot(model)
        with self.assertRaisesRegex(RuntimeError, "Non-BatchNorm buffer"):
            self.call(model)
        self.assert_model_equal(model, before)

    def test_nonfinite_input_is_rejected_and_rolled_back(self):
        model = CalibrationModel()
        before = snapshot(model)
        with patch("src.calibration.make_transform", return_value=lambda image: torch.full((3, 32, 32), float("nan"))):
            with self.assertRaisesRegex(RuntimeError, "calibration input"):
                self.call(model)
        self.assert_model_equal(model, before)
        self.assertEqual(model.calls, 0)

    def test_invalid_arguments_and_identity_are_rejected_before_reset(self):
        invalid = [dict(batch_size=0), dict(batch_size=True), dict(batch_size=1.5),
                   dict(num_workers=-1), dict(num_workers=False), dict(seed=-1),
                   dict(seed=True), dict(seed=2**63), dict(device="invalid"),
                   dict(expected_split_sha256="invalid"), dict(expected_split_sha256="0" * 64)]
        for options in invalid:
            with self.subTest(options=options):
                model = CalibrationModel()
                before = snapshot(model)
                with self.assertRaises(ValueError):
                    self.call(model, **options)
                self.assert_model_equal(model, before)
                self.assertEqual(model.calls, 0)

    def test_bad_split_class_mapping_and_csv_identity_rejected(self):
        mutations = [dict(classes=data.CLASSES[::-1]), dict(val_ids=["0", "6"]),
                     dict(train_csv_sha256="0" * 64)]
        for index, update in enumerate(mutations):
            with self.subTest(update=update):
                bad = self.root / f"invalid_calibration_split_{index}.json"
                bad.write_text(json.dumps({**self.split_info, **update}), encoding="utf-8")
                model = CalibrationModel()
                before = snapshot(model)
                with self.assertRaises(ValueError):
                    recalibrate_batch_norm(model, self.data_dir, bad, batch_size=2)
                self.assert_model_equal(model, before)

    def test_fp32_rejects_nonfinite_or_wrong_dtype_parameters_before_reset(self):
        for bad_type in ("infinite", "double"):
            with self.subTest(bad_type=bad_type):
                model = CalibrationModel()
                if bad_type == "infinite":
                    with torch.no_grad():
                        model.head.bias[0] = float("inf")
                else:
                    model.double()
                before = snapshot(model)
                with self.assertRaisesRegex(ValueError, "Non-finite|FP32"):
                    self.call(model)
                self.assert_model_equal(model, before)

    def test_cpu_autocast_and_backend_flags_are_restored(self):
        model = CalibrationModel()
        flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            report = self.call(model)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
            self.assertEqual(report["precision"], "FP32; autocast and TF32 disabled")
        self.assertEqual(flags, (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
                                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                                torch.backends.cudnn.deterministic))

    def test_missing_or_untracked_batchnorm_rejected(self):
        for model in (nn.Identity(), nn.BatchNorm2d(3, track_running_stats=False)):
            with self.subTest(model=type(model).__name__):
                before = snapshot(model)
                with self.assertRaisesRegex(ValueError, "BatchNorm"):
                    self.call(model)
                self.assert_model_equal(model, before)

    def test_independent_worker_loader_has_same_training_order_and_statistics(self):
        first, second = CalibrationModel(), CalibrationModel()
        second.load_state_dict(first.state_dict())
        self.call(first, num_workers=0)
        before_rng = torch.get_rng_state()
        self.call(second, num_workers=1)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
        self.assertEqual([record["input"].shape[0] for record in second.seen], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
