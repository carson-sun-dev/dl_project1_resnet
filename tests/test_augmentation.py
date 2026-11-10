import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.augmentation import Cutout, mixup_batch, soft_target_cross_entropy, validate_augmentation
from src.data import make_transform
from src.models import build_model, parameter_counts
from src.tta import probabilities, require_tta_benefit
from src.utils import file_hash, read_json, run_lock, write_json


class AugmentationTests(unittest.TestCase):
    def test_soft_ce_matches_weighted_hard_ce_and_gradients(self):
        torch.manual_seed(1)
        labels = torch.tensor([0, 2, 1, 3])
        other = labels.flip(0)
        lam = 0.3
        targets = lam * F.one_hot(labels, 4) + (1 - lam) * F.one_hot(other, 4)
        for smoothing in (0.0, 0.1):
            logits = torch.randn(4, 4, requires_grad=True)
            loss = soft_target_cross_entropy(logits, targets, smoothing)
            expected = (lam * F.cross_entropy(logits, labels, label_smoothing=smoothing)
                        + (1 - lam) * F.cross_entropy(logits, other, label_smoothing=smoothing))
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(torch.autograd.grad(loss, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits)[0])
        # Explicit FP32 accumulation for AMP logits, including very large magnitudes.
        logits = torch.tensor([[1000., -1000., 0., 2.]] * 4, dtype=torch.float16, requires_grad=True)
        loss = soft_target_cross_entropy(logits, targets)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mixup_pairs_images_and_targets(self):
        images = torch.arange(4.).reshape(4, 1, 1, 1).expand(4, 3, 32, 32)
        labels = torch.arange(4)
        perm = torch.tensor([2, 0, 3, 1])
        with patch("torch.distributions.Beta.sample", return_value=torch.tensor(0.25)), patch("torch.randperm", return_value=perm):
            mixed, targets = mixup_batch(images, labels, alpha=0.2, num_classes=4)
        torch.testing.assert_close(mixed, 0.25 * images + 0.75 * images[perm])
        torch.testing.assert_close(targets, 0.25 * F.one_hot(labels, 4) + 0.75 * F.one_hot(labels[perm], 4))
        torch.testing.assert_close(targets.sum(1), torch.ones(4))
        self.assertTrue((targets >= 0).all())

    def test_disabled_augmentation_is_identity_and_consumes_no_rng(self):
        images, labels = torch.randn(4, 3, 32, 32), torch.arange(4)
        before = torch.get_rng_state().clone()
        for alpha, probability in ((0, 1), (0.2, 0)):
            x, y = mixup_batch(images, labels, alpha, probability)
            self.assertIs(x, images)
            self.assertIs(y, labels)
        image = images[0]
        self.assertIs(Cutout(0)(image), image)
        self.assertIs(Cutout(16, 0)(image), image)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with patch("torch.rand", return_value=torch.tensor(0.9)):
            self.assertIs(Cutout(16, 0.5)(image), image)
            self.assertIs(mixup_batch(images, labels, 0.2, 0.5)[0], images)

    def test_cutout_clips_square_and_does_not_mutate_input(self):
        image = torch.ones(3, 32, 32)
        for length, center, expected_area in ((16, 16, 256), (16, 0, 64), (1, 16, 1)):
            with patch("torch.randint", return_value=torch.tensor(center)):
                result = Cutout(length)(image)
            self.assertEqual(int((result[0] == 0).sum()), expected_area)
            torch.testing.assert_close(result[0], result[1])
            self.assertTrue((image == 1).all())

    def test_validation_transform_ignores_training_regularization(self):
        image = Image.fromarray(np.arange(32*32*3, dtype=np.uint8).reshape(32, 32, 3))
        transform = make_transform([0.5]*3, [0.25]*3, False, cutout_length=32)
        before = torch.get_rng_state().clone()
        expected = (torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255 - 0.5) / 0.25
        torch.testing.assert_close(transform(image), expected)
        torch.testing.assert_close(transform(image), expected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIsInstance(make_transform([0.5]*3, [0.25]*3, True, 16).transforms[-1], Cutout)

    def test_invalid_augmentation_settings(self):
        for key, values in {"mixup_alpha": [-1, float("nan"), float("inf"), True, "0.2"],
                            "cutout_length": [-1, 33, 1.5, True],
                            "mixup_probability": [-0.1, 1.1, float("nan"), True],
                            "cutout_probability": [-1, 2, float("inf"), None]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_augmentation({key: value})

    def test_regularized_config_preserves_wide_architecture(self):
        root = Path(__file__).resolve().parents[1]
        wide = read_json(root / "configs/wide.json")
        regularized = read_json(root / "configs/regularized_wide.json")
        self.assertEqual(wide["model"], regularized["model"])
        self.assertEqual(parameter_counts(build_model(regularized["model"]))["trainable"], 4_327_754)
        self.assertNotEqual(wide["run_dir"], regularized["run_dir"])
        for name in ("baseline", "wide", "deep"):
            config = read_json(root / f"configs/{name}.json")
            self.assertFalse({"mixup_alpha", "cutout_length"} & config["training"].keys())

    def test_run_lock_excludes_second_writer_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock"
            with self.assertRaisesRegex(ValueError, "test exit"):
                with run_lock(path):
                    with self.assertRaisesRegex(RuntimeError, "Another training process"):
                        with run_lock(path):
                            self.fail("Second writer acquired lock")
                    raise ValueError("test exit")
            with run_lock(path):
                pass

    def test_tta_averages_probabilities_and_checks_both_views(self):
        class Asymmetric(torch.nn.Module):
            def forward(self, x):
                return torch.stack((x[:, 0, 0, 0], x[:, 0, 0, -1]), 1)
        model = Asymmetric().eval()
        image = torch.zeros(2, 3, 32, 32)
        image[:, 0, 0, 0] = 3
        plain = model(image).softmax(1)
        torch.testing.assert_close(probabilities(model, image), plain)
        torch.testing.assert_close(probabilities(model, image, True),
                                   (plain + model(image.flip(-1)).softmax(1)) / 2)
        image[:, 0, 0, -1] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite prediction logits"):
            probabilities(model, image, True)

    def test_tta_requires_matching_beneficial_formal_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, report_path = Path(directory)/"best.pt", Path(directory)/"tta.json"
            checkpoint.write_bytes(b"synthetic checkpoint")
            state = {"identity": {"split_sha256": "split"}}
            report = {"method": "horizontal_flip_probability_mean", "checkpoint_sha256": file_hash(checkpoint),
                      "split_sha256": "split", "smoke_test": False, "full_validation": True,
                      "beneficial": True, "no_tta": {"accuracy": 0.90}, "horizontal_flip": {"accuracy": 0.91}}
            with self.assertRaisesRegex(ValueError, "requires"):
                require_tta_benefit(None, checkpoint, state)
            write_json(report_path, report)
            self.assertEqual(require_tta_benefit(report_path, checkpoint, state), report)
            for key, value in (("checkpoint_sha256", "different"), ("split_sha256", "different"),
                               ("smoke_test", True), ("full_validation", False), ("beneficial", False),
                               ("horizontal_flip", {"accuracy": 0.89}), ("horizontal_flip", {"accuracy": 0.90})):
                invalid = copy.deepcopy(report)
                invalid[key] = value
                write_json(report_path, invalid)
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "accuracy benefit"):
                    require_tta_benefit(report_path, checkpoint, state)


if __name__ == "__main__":
    unittest.main()
