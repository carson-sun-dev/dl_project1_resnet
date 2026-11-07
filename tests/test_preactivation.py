"""Architecture and synthetic-resume checks for the isolated preactivation model."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from src.data import prepare_split
from src.models import (CifarResNet, PreActivationBlock, PreActivationWideResNet,
                        build_model, parameter_counts)
from src.train import train
from src.utils import epoch_lr, read_json


PROJECT = Path(__file__).resolve().parents[1]


class PreActivationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_isolated_configuration_preserves_color_training_protocol(self):
        original = read_json(PROJECT / "configs/color_robustness_wide.json")
        candidate = read_json(PROJECT / "configs/preactivation_wrn22_4_color.json")
        self.assertEqual(candidate["training"], original["training"])
        self.assertEqual(candidate["seed"], original["seed"])
        self.assertEqual(candidate["data_dir"], original["data_dir"])
        self.assertEqual(candidate["split_file"], original["split_file"])
        runtime = candidate["runtime"].copy()
        self.assertIs(runtime.pop("strict_source_resume"), True)
        self.assertEqual(runtime, original["runtime"])
        self.assertEqual(candidate["run_dir"], "runs/preactivation_wrn22_4_color_s2026_local")
        self.assertNotEqual(candidate["run_dir"], original["run_dir"])
        self.assertEqual(candidate["model"], {
            "architecture": "preactivation_wrn", "channels": [64, 128, 256],
            "blocks": [3, 3, 3], "num_classes": 10, "stem_channels": 16, "dropout": 0.0})

    def test_exact_parameter_count_shapes_initialization_and_finite_gradients(self):
        config = read_json(PROJECT / "configs/preactivation_wrn22_4_color.json")["model"]
        model = build_model(config)
        self.assertIsInstance(model, PreActivationWideResNet)
        self.assertEqual(parameter_counts(model), {"trainable": 4_298_970, "total": 4_298_970})
        self.assertLess(parameter_counts(model)["trainable"], 5_000_000)
        self.assertIsInstance(model.stem, nn.Conv2d)
        self.assertEqual((model.stem.in_channels, model.stem.out_channels), (3, 16))
        self.assertEqual(model.stem.kernel_size, (3, 3))
        self.assertIsNone(model.stem.bias)
        self.assertIsInstance(model.final_bn, nn.BatchNorm2d)
        self.assertEqual(model.final_bn.num_features, 256)
        self.assertEqual((model.head.in_features, model.head.out_features), (256, 10))
        self.assertIsNotNone(model.head.bias)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in model.modules()))
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.assertIsNone(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                self.assertTrue(torch.equal(module.weight, torch.ones_like(module.weight)))
                self.assertTrue(torch.equal(module.bias, torch.zeros_like(module.bias)))
        shapes = []
        handles = [module.register_forward_hook(lambda _m, _i, result: shapes.append(tuple(result.shape)))
                   for module in (model.stem, model.blocks[2], model.blocks[5], model.blocks[8])]
        try:
            logits = model(torch.randn(2, 3, 32, 32))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(shapes, [(2, 16, 32, 32), (2, 64, 32, 32),
                                  (2, 128, 16, 16), (2, 256, 8, 8)])
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits, torch.tensor([2, 7])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_equal_shape_identity_preserves_negative_values_and_gradients(self):
        for training in (False, True):
            with self.subTest(training=training):
                block = PreActivationBlock(4, 4)
                block.train(training)
                self.assertIsInstance(block.skip, nn.Identity)
                with torch.no_grad():
                    block.conv1.weight.zero_()
                    block.conv2.weight.zero_()
                images = torch.linspace(-3, 2, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8).requires_grad_()
                original = images.detach().clone()
                output = block(images)
                torch.testing.assert_close(output, original, rtol=0, atol=0)
                torch.testing.assert_close(images.detach(), original, rtol=0, atol=0)
                self.assertTrue((output < 0).any())
                output.sum().backward()
                torch.testing.assert_close(images.grad, torch.ones_like(images), rtol=0, atol=0)

    def test_projection_uses_preactivated_input_without_shortcut_batchnorm(self):
        for out_channels, stride in ((8, 1), (8, 2), (4, 2)):
            with self.subTest(out_channels=out_channels, stride=stride):
                block = PreActivationBlock(4, out_channels, stride).eval()
                self.assertIsInstance(block.skip, nn.Conv2d)
                self.assertEqual(block.skip.kernel_size, (1, 1))
                self.assertEqual(block.skip.stride, (stride, stride))
                self.assertIsNone(block.skip.bias)
                self.assertEqual(len([m for m in block.modules() if isinstance(m, nn.BatchNorm2d)]), 2)
                self.assertEqual((block.bn1.num_features, block.bn2.num_features), (4, out_channels))
                with torch.no_grad():
                    block.bn1.running_mean.fill_(1)
                    block.bn1.running_var.fill_(4)
                    block.bn1.weight.fill_(2)
                    block.bn1.bias.fill_(-0.5)
                    block.conv1.weight.zero_()
                    block.conv2.weight.zero_()
                    block.skip.weight.fill_(0.25)
                images = torch.linspace(-2, 3, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8)
                original = images.clone()
                received = {}

                def capture(name):
                    def hook(_module, inputs):
                        received[name] = inputs[0].detach().clone()
                    return hook

                handles = [block.conv1.register_forward_pre_hook(capture("residual")),
                           block.skip.register_forward_pre_hook(capture("shortcut"))]
                try:
                    with torch.no_grad():
                        expected_input = F.relu(block.bn1(images))
                        expected = block.skip(expected_input)
                        raw_projection = block.skip(images)
                        output = block(images)
                finally:
                    for handle in handles:
                        handle.remove()
                torch.testing.assert_close(received["residual"], expected_input, rtol=0, atol=0)
                torch.testing.assert_close(received["shortcut"], expected_input, rtol=0, atol=0)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertFalse(torch.equal(output, raw_projection))
                torch.testing.assert_close(images, original, rtol=0, atol=0)

    def test_factory_preserves_all_legacy_model_initialization_and_logits(self):
        seen = set()
        for path in sorted((PROJECT / "configs").glob("*.json")):
            config = read_json(path)
            if "model" not in config or config["model"].get("architecture") == "preactivation_wrn":
                continue
            model_config = config["model"]
            self.assertNotIn("architecture", model_config)
            signature = repr(model_config)
            if signature in seen:
                continue
            seen.add(signature)
            with self.subTest(config=path.name):
                original = copy.deepcopy(model_config)
                torch.manual_seed(37)
                direct = CifarResNet(**model_config).eval()
                torch.manual_seed(37)
                dispatched = build_model(model_config).eval()
                self.assertIs(type(dispatched), CifarResNet)
                self.assertEqual(model_config, original)
                self.assertEqual(list(direct.state_dict()), list(dispatched.state_dict()))
                for key, value in direct.state_dict().items():
                    torch.testing.assert_close(dispatched.state_dict()[key], value, rtol=0, atol=0)
                images = torch.randn(1, 3, 32, 32)
                with torch.no_grad():
                    torch.testing.assert_close(dispatched(images), direct(images), rtol=0, atol=0)
                dispatched.load_state_dict(direct.state_dict(), strict=True)
        self.assertGreaterEqual(len(seen), 4)

    def test_factory_rejects_unknown_and_invalid_architecture_without_mutation(self):
        config = {"architecture": "preactivation_wrn", "channels": [4, 8, 16],
                  "blocks": [1, 1, 1], "stem_channels": 4, "dropout": 0.0}
        original = copy.deepcopy(config)
        self.assertIsInstance(build_model(config), PreActivationWideResNet)
        self.assertEqual(config, original)
        for override in ({"architecture": "unknown"}, {"channels": [0, 8, 16]},
                         {"channels": [True, 8, 16]}, {"blocks": [1, 1.5, 1]},
                         {"stem_channels": 0}, {"dropout": 0.1}, {"dropout": True},
                         {"num_classes": 9}):
            changed = {**config, **override}
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_model(changed)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            build_model({"architecture": "preactivation_wrn", "channels": [64, 128, 256],
                         "blocks": [4, 4, 4]})

    def test_preactivation_checkpoint_roundtrip_preserves_logits_and_model_type(self):
        config = {"architecture": "preactivation_wrn", "channels": [4, 8, 16],
                  "blocks": [1, 1, 1], "stem_channels": 4}
        original = build_model(config).eval()
        images = torch.randn(2, 3, 32, 32)
        with tempfile.TemporaryDirectory(prefix="preact-roundtrip-") as temp:
            path = Path(temp) / "checkpoint.pt"
            torch.save({"model_config": config, "model": original.state_dict()}, path)
            state = torch.load(path, weights_only=True)
            restored = build_model(state["model_config"]).eval()
            restored.load_state_dict(state["model"], strict=True)
            self.assertIsInstance(restored, PreActivationWideResNet)
            with torch.no_grad():
                torch.testing.assert_close(restored(images), original(images), rtol=0, atol=0)
            with self.assertRaises(RuntimeError):
                CifarResNet(channels=[4, 8, 16], blocks=[1, 1, 1]).load_state_dict(state["model"], strict=True)


class PreActivationResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="preact-resume-")
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        for folder in ("train_images", "test_images"):
            (cls.data / folder).mkdir(parents=True)
        rng = np.random.default_rng(19)
        rows = []
        for index in range(40):
            filename = f"train_{index}.png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data / "train_images" / filename)
            rows.append([str(index), filename, str(index % 10)])
        with (cls.data / "train_labels.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "filename", "label"])
            writer.writerows(rows)
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data / "test_images/test.png")
        (cls.data / "test.csv").write_text("id,filename\n0,test.png\n", encoding="utf-8")
        (cls.data / "sample_submission.csv").write_text("id,label\n0,0\n", encoding="utf-8")
        cls.split = cls.root / "split.json"
        prepare_split(cls.data, cls.split, seed=2026, val_fraction=0.25)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

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

    def test_color_preactivation_resume_preserves_model_optimizer_rng_and_schedule(self):
        base = read_json(PROJECT / "configs/preactivation_wrn22_4_color.json")
        base.update(data_dir=str(self.data), split_file=str(self.split))
        base["model"].update(channels=[4, 8, 16], blocks=[1, 1, 1], stem_channels=4)
        base["training"].update(batch_size=8, max_train_batches=2)
        for workers in (0, 1):
            with self.subTest(workers=workers), contextlib.redirect_stdout(io.StringIO()):
                config = copy.deepcopy(base)
                config["runtime"].update(device="cpu", num_workers=workers, cpu_threads=2)
                config["run_dir"] = str(self.root / f"full_{workers}")
                train(config, stop_after_epoch=2)
                resumed = copy.deepcopy(config)
                resumed["run_dir"] = str(self.root / f"resumed_{workers}")
                first = train(resumed, stop_after_epoch=1)
                self.assertFalse(first["complete"])
                result = train(resumed, resume=True, stop_after_epoch=2)
                self.assertFalse(result["complete"])
                self.assertEqual(result["planned_epochs"], 200)
                self.assertEqual(result["completed_epochs"], 2)
                a = torch.load(Path(config["run_dir"]) / "last.pt", weights_only=False)
                b = torch.load(Path(resumed["run_dir"]) / "last.pt", weights_only=False)
                for key in ("model", "optimizer", "rng", "train_generator", "val_generator",
                            "identity", "precision_policy", "scaler", "best_score", "best_epoch", "skipped_updates"):
                    self.assert_nested_equal(a[key], b[key])
                self.assertIn("source_hashes", b["identity"])
                self.assertEqual(b["identity"]["model"]["architecture"], "preactivation_wrn")
                self.assertEqual([row["lr"] for row in b["history"]],
                                 [epoch_lr(i, config["training"]) for i in range(2)])
                for uninterrupted, continued in zip(a["history"], b["history"]):
                    for key in ("epoch", "lr", "train_loss", "train_accuracy", "train_images",
                                "val_loss", "val_accuracy", "val_images"):
                        self.assertEqual(uninterrupted[key], continued[key], key)
                for modify in (lambda c: c["model"].update(stem_channels=5),
                               lambda c: c["training"].update(mixup_alpha=0.4)):
                    changed = copy.deepcopy(resumed)
                    modify(changed)
                    with self.assertRaisesRegex(ValueError, "Resume config/split differs"):
                        train(changed, resume=True, stop_after_epoch=3)


if __name__ == "__main__":
    unittest.main()
