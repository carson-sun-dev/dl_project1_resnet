"""Sharpness-update semantics, independent EMA state and deterministic recovery."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import importlib.util
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("final_optimization_runner", PACKAGE_ROOT / "run.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
_runner.activate_frozen_source()
from src.optimization import EMA, optimization_step, validate_optimization


def settings(method="sam", **changes):
    result = {"schema_version": 1, "method": method,
              "rho": {"sgd": 0.0, "sam": 0.05, "asam": 0.5}.get(method, 0.05),
              "eta": 0.01 if method == "asam" else 0.0,
              "ema": {"enabled": True, "decay": 0.999, "warmup": True}}
    result.update(changes)
    return result


class RecordedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6)
        self.bn = nn.BatchNorm1d(6, momentum=0.2)
        self.head = nn.Linear(6, 3)
        self.inputs = []

    def forward(self, value):
        self.inputs.append((value.data_ptr(), value.detach().clone()))
        return self.head(torch.relu(self.bn(self.linear(value))))


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.random.default_generator.manual_seed(113)
        self.images = torch.randn(8, 4)
        self.targets = torch.arange(8) % 3

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_opt_in_validation_preserves_absent_legacy_configuration(self):
        self.assertIsNone(validate_optimization({"optimizer": "sgd"}))
        for method in ("sgd", "sam", "asam"):
            config = {"optimizer": "sgd", "optimization": settings(method)}
            original = copy.deepcopy(config)
            self.assertEqual(validate_optimization(config), settings(method))
            self.assertEqual(config, original)

    def test_invalid_schema_methods_ranges_and_boolean_settings_are_rejected(self):
        invalid = [None, {}, settings(schema_version=True), settings(schema_version=2),
                   settings(method="adam"), settings(extra=1), settings(rho=float("nan")),
                   settings(rho=float("inf")), settings(rho=-1), settings(rho=True),
                   settings(eta=0.01), settings("sgd", rho=0.1), settings("sgd", eta=0.1),
                   settings(ema=None), settings(ema={"enabled": 1}),
                   settings(ema={"warmup": "true"}), settings(ema={"decay": 1}),
                   settings(ema={"decay": float("nan")}), settings(ema={"unknown": 1})]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_optimization({"optimization": value})
        with self.assertRaisesRegex(ValueError, "SGD base"):
            validate_optimization({"optimizer": "adamw", "optimization": settings()})

    def test_sgd_matches_native_update_and_does_not_consume_rng(self):
        first = nn.Linear(4, 3)
        second = copy.deepcopy(first)
        first_opt = torch.optim.SGD(first.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
        second_opt = torch.optim.SGD(second.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
        before_rng = torch.get_rng_state().clone()
        result = optimization_step(first, first_opt, self.images, self.targets, settings("sgd"))
        self.assert_nested_equal(before_rng, torch.get_rng_state())
        second_opt.zero_grad(set_to_none=True)
        F.cross_entropy(second(self.images), self.targets).backward()
        second_opt.step()
        self.assert_nested_equal(first.state_dict(), second.state_dict())
        self.assert_nested_equal(first_opt.state_dict(), second_opt.state_dict())
        self.assertEqual(result["gradient_checks"], 1)
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertIsNone(result["perturbed_loss"])

    def test_sam_and_asam_match_independent_two_pass_equations(self):
        for method in ("sam", "asam"):
            with self.subTest(method=method):
                model = nn.Linear(4, 3)
                manual = copy.deepcopy(model)
                config = settings(method)
                original = {name: value.detach().clone() for name, value in manual.named_parameters()}
                F.cross_entropy(manual(self.images), self.targets).backward()
                scales = {name: (value.detach().abs() + config["eta"]
                                 if method == "asam" and name == "weight" else torch.ones_like(value))
                          for name, value in manual.named_parameters()}
                denominator = sum(((value.grad * scales[name]) ** 2).sum()
                                  for name, value in manual.named_parameters()).sqrt()
                with torch.no_grad():
                    for name, value in manual.named_parameters():
                        value.add_(config["rho"] * scales[name] ** 2 * value.grad / denominator)
                manual.zero_grad(set_to_none=True)
                second_loss = F.cross_entropy(manual(self.images), self.targets)
                second_loss.backward()
                expected = {name: original[name] - 0.03 * value.grad
                            for name, value in manual.named_parameters()}
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
                result = optimization_step(model, optimizer, self.images, self.targets, config)
                for name, value in model.named_parameters():
                    torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)
                self.assertAlmostEqual(result["perturbed_loss"], second_loss.item(), places=6)
                self.assertAlmostEqual(result["adaptive_perturbation_norm"], config["rho"], places=6)
                self.assertEqual(result["gradient_checks"], 2)

    def test_same_mixed_batch_is_reused_and_bn_updates_once(self):
        for method in ("sam", "asam"):
            with self.subTest(method=method):
                model = RecordedNet()
                first_pass = copy.deepcopy(model)
                permutation = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
                mixed = 0.3 * self.images + 0.7 * self.images[permutation]
                labels = F.one_hot(self.targets, 3).float()
                soft = 0.3 * labels + 0.7 * labels[permutation]
                first_pass(mixed)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                result = optimization_step(model, optimizer, mixed, soft, settings(method))
                self.assertEqual(len(model.inputs), 2)
                self.assertEqual(model.inputs[0][0], mixed.data_ptr())
                self.assertEqual(model.inputs[1][0], mixed.data_ptr())
                self.assert_nested_equal(model.inputs[0][1], model.inputs[1][1])
                for name in ("running_mean", "running_var", "num_batches_tracked"):
                    self.assert_nested_equal(getattr(model.bn, name), getattr(first_pass.bn, name))
                self.assertEqual(model.bn.num_batches_tracked.item(), 1)
                self.assertTrue(model.bn.track_running_stats)
                self.assertTrue(model.bn.training)
                self.assertEqual(model.bn.momentum, 0.2)
                self.assertEqual(result["logits"].dtype, torch.float32)
                self.assertFalse(result["logits"].requires_grad)

    def test_base_optimizer_sees_bitwise_restored_parameters_and_steps_once(self):
        model = RecordedNet()
        original = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
        native_step = optimizer.step
        def checked_step(*args, **kwargs):
            for name, value in model.named_parameters():
                self.assert_nested_equal(value, original[name])
            return native_step(*args, **kwargs)
        with patch.object(optimizer, "step", side_effect=checked_step) as step:
            optimization_step(model, optimizer, self.images, self.targets, settings("asam"))
            step.assert_called_once()
        self.assertTrue(all("momentum_buffer" in optimizer.state[parameter]
                            for parameter in model.parameters()))

    def test_second_pass_failure_restores_exact_parameters_and_bn_flags(self):
        for failure in ("forward", "gradient"):
            model = RecordedNet()
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            calls = [0]
            if failure == "forward":
                def poison_forward(module, inputs, output):
                    calls[0] += 1
                    return output if calls[0] == 1 else output * float("nan")
                model.register_forward_hook(poison_forward)
            else:
                def poison_gradient(gradient):
                    calls[0] += 1
                    return gradient if calls[0] == 1 else torch.full_like(gradient, float("nan"))
                model.head.weight.register_hook(poison_gradient)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
            with self.subTest(failure=failure), patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                    optimization_step(model, optimizer, self.images, self.targets, settings("asam"))
                step.assert_not_called()
            for name, value in model.named_parameters():
                self.assert_nested_equal(value, before[name])
                self.assertIsNone(value.grad)
            self.assertTrue(model.bn.track_running_stats)
            self.assertEqual(model.bn.num_batches_tracked.item(), 1)
            self.assertFalse(optimizer.state)

    def test_second_forward_exception_restores_parameters(self):
        model = RecordedNet()
        original = {name: value.detach().clone() for name, value in model.named_parameters()}
        def fail(module, inputs):
            if len(module.inputs) == 1:
                raise RuntimeError("deliberate second forward failure")
        model.register_forward_pre_hook(fail)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            optimization_step(model, optimizer, self.images, self.targets, settings())
        self.assertTrue(model.bn.track_running_stats)
        for name, parameter in model.named_parameters():
            self.assert_nested_equal(parameter, original[name])

    def test_zero_gradients_and_zero_scale_are_finite(self):
        model = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        soft = torch.full((8, 3), 1 / 3)
        result = optimization_step(model, optimizer, torch.zeros_like(self.images), soft,
                                   settings("asam", eta=0.0))
        self.assertEqual(result["base_gradient_norm"], 0)
        self.assertEqual(result["perturbation_norm"], 0)
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_invalid_precision_inputs_targets_and_perturbations_abort(self):
        for invalid_dtype in (torch.float16, torch.float64):
            model = nn.Linear(4, 3)
            with self.subTest(dtype=invalid_dtype), self.assertRaisesRegex(ValueError, "not FP16"):
                optimization_step(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  self.images, self.targets, settings(), dtype=invalid_dtype)
        model = nn.Linear(4, 3).double()
        with self.assertRaisesRegex(ValueError, "FP32 master"):
            optimization_step(model, torch.optim.SGD(model.parameters(), lr=0.01),
                              self.images.double(), self.targets, settings())
        for value in (float("nan"), float("inf")):
            model = nn.Linear(4, 3)
            bad = self.images.clone()
            bad[0, 0] = value
            with self.assertRaisesRegex(RuntimeError, "training inputs"):
                optimization_step(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  bad, self.targets, settings())
        model = nn.Linear(4, 3)
        original = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "perturbation"):
            optimization_step(model, torch.optim.SGD(model.parameters(), lr=0.01),
                              self.images, self.targets, settings(rho=1e39))
        self.assert_nested_equal(model.state_dict(), original)
        with self.assertRaisesRegex(ValueError, "probability distributions"):
            optimization_step(model, torch.optim.SGD(model.parameters(), lr=0.01),
                              self.images, torch.ones(8, 3), settings())

    def test_nonfinite_first_gradient_and_post_update_optimizer_state_are_detected(self):
        model = nn.Linear(4, 3)
        model.weight.register_hook(lambda gradient: gradient * float("nan"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(RuntimeError, "training gradients"):
                optimization_step(model, optimizer, self.images, self.targets, settings())
            step.assert_not_called()
        model = nn.Linear(4, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for parameter in model.parameters():
            optimizer.state[parameter]["momentum_buffer"] = torch.full_like(parameter, float("inf"))
        with self.assertRaisesRegex(RuntimeError, "updated model parameters"):
            optimization_step(model, optimizer, self.images, self.targets, settings("sgd"))

    def test_ema_update_formula_copies_buffers_and_consumes_no_rng(self):
        model = RecordedNet()
        ema = EMA(model, decay=0.999, warmup=True)
        original = ema.model_state_dict()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
            model.bn.running_mean.add_(4)
            model.bn.num_batches_tracked.add_(7)
        before = torch.get_rng_state().clone()
        ema.update(model)
        self.assert_nested_equal(before, torch.get_rng_state())
        for name, value in model.named_parameters():
            expected = 0.1 * original[name] + 0.9 * value
            torch.testing.assert_close(ema.shadow[name], expected, rtol=1e-7, atol=1e-7)
        self.assert_nested_equal(ema.shadow["bn.running_mean"], model.bn.running_mean)
        self.assert_nested_equal(ema.shadow["bn.num_batches_tracked"], model.bn.num_batches_tracked)
        self.assertEqual(ema.shadow["bn.num_batches_tracked"].dtype, torch.int64)
        self.assertEqual(ema.updates, 1)
        copied = copy.deepcopy(model)
        ema.copy_to(copied)
        self.assert_nested_equal(copied.state_dict(), ema.model_state_dict())
        export = ema.model_state_dict()
        export["head.weight"].zero_()
        self.assertFalse(torch.equal(export["head.weight"], ema.shadow["head.weight"]))

    def test_ema_cannot_change_optimization_trajectory(self):
        first = RecordedNet()
        second = copy.deepcopy(first)
        tracked = EMA(second)
        first_opt = torch.optim.SGD(first.parameters(), lr=0.01, momentum=0.9)
        second_opt = torch.optim.SGD(second.parameters(), lr=0.01, momentum=0.9)
        for _ in range(3):
            optimization_step(first, first_opt, self.images, self.targets, settings())
            optimization_step(second, second_opt, self.images, self.targets, settings())
            tracked.update(second)
        self.assert_nested_equal(first.state_dict(), second.state_dict())
        self.assert_nested_equal(first_opt.state_dict(), second_opt.state_dict())

    def test_ema_rejects_identity_and_nonfinite_state_without_mutation(self):
        model = RecordedNet()
        ema = EMA(model)
        valid = ema.state_dict()
        changes = [{"schema_version": 2}, {"decay": 0.99}, {"warmup": False},
                   {"updates": -1}, {"updates": True}, {"parameter_names": ()}]
        for change in changes:
            invalid = copy.deepcopy(valid)
            invalid.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ema.load_state_dict(invalid)
            self.assert_nested_equal(ema.state_dict(), valid)
        invalid = copy.deepcopy(valid)
        invalid["shadow"]["head.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "EMA checkpoint tensors"):
            ema.load_state_dict(invalid)
        self.assert_nested_equal(ema.state_dict(), valid)
        with self.assertRaisesRegex(ValueError, "parameter identity"):
            ema.update(nn.Linear(4, 3))

    def test_optimizer_ema_and_rng_recover_exactly_after_checkpoint_reload(self):
        for method in ("sgd", "sam", "asam"):
            def make():
                torch.random.default_generator.manual_seed(71)
                model = RecordedNet()
                return model, torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                              weight_decay=0.0005), EMA(model)
            def updates(model, optimizer, ema, count):
                metrics = []
                for _ in range(count):
                    images = torch.randn(8, 4)
                    result = optimization_step(model, optimizer, images, self.targets, settings(method))
                    ema.update(model)
                    metrics.append({key: value for key, value in result.items() if key != "logits"})
                return metrics
            full, full_opt, full_ema = make()
            full_metrics = updates(full, full_opt, full_ema, 5)
            full_rng = torch.get_rng_state().clone()
            partial, partial_opt, partial_ema = make()
            partial_metrics = updates(partial, partial_opt, partial_ema, 2)
            with tempfile.TemporaryDirectory(prefix="optimization-recovery-") as directory:
                path = Path(directory) / "last.pt"
                torch.save({"model": partial.state_dict(), "optimizer": partial_opt.state_dict(),
                            "ema": partial_ema.state_dict(), "rng": torch.get_rng_state()}, path)
                recovered, recovered_opt, recovered_ema = make()
                state = torch.load(path, weights_only=False)
                recovered.load_state_dict(state["model"])
                recovered_opt.load_state_dict(state["optimizer"])
                recovered_ema.load_state_dict(state["ema"])
                torch.set_rng_state(state["rng"])
                recovered_metrics = updates(recovered, recovered_opt, recovered_ema, 3)
            with self.subTest(method=method):
                self.assert_nested_equal(full.state_dict(), recovered.state_dict())
                self.assert_nested_equal(full_opt.state_dict(), recovered_opt.state_dict())
                self.assert_nested_equal(full_ema.state_dict(), recovered_ema.state_dict())
                self.assert_nested_equal(full_rng, torch.get_rng_state())
                self.assertEqual(full_metrics, partial_metrics + recovered_metrics)


if __name__ == "__main__":
    unittest.main()
