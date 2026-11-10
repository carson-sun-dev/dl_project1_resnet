import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from src.augmix import (OPERATOR_NAMES, AugMixTransform, apply_operator,
                        augmix_loss, validate_augmix)


SETTINGS = {"width": 3, "depth": -1, "severity": 3, "mixture_concentration": 1.0,
            "js_coefficient": 12.0, "all_ops": False}


class AugMixTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = Image.fromarray(np.random.default_rng(2026).integers(
            0, 256, size=(32, 32, 3), dtype=np.uint8))

    def test_default_pool_and_operator_rgb_range(self):
        self.assertEqual(OPERATOR_NAMES, ("autocontrast", "equalize", "posterize", "rotate", "solarize",
                                         "shear_x", "shear_y", "translate_x", "translate_y"))
        before = np.asarray(self.image).copy()
        for severity in (0.1, 3, 10):
            for operator in OPERATOR_NAMES:
                result = apply_operator(self.image, operator, severity)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, (32, 32))
                self.assertEqual(np.asarray(result).dtype, np.uint8)
                self.assertGreaterEqual(int(np.asarray(result).min()), 0)
                self.assertLessEqual(int(np.asarray(result).max()), 255)
        np.testing.assert_array_equal(before, np.asarray(self.image))
        for image in (self.image.convert("L"), Image.new("RGB", (28, 28))):
            with self.assertRaises(ValueError):
                apply_operator(image, "rotate", 3)
        with self.assertRaises(ValueError):
            apply_operator(self.image, "brightness", 3)

    def test_operator_parameterization_matches_original_ranges(self):
        with patch("numpy.random.uniform", return_value=3):
            # severity3: int(3*4/10)=1 -> three posterization bits.
            result = apply_operator(self.image, "posterize", 3)
            self.assertTrue(((np.asarray(result) % 32) == 0).all())
            from PIL import ImageOps
            np.testing.assert_array_equal(apply_operator(self.image, "solarize", 3),
                                          ImageOps.solarize(self.image, 180))
        with patch("numpy.random.uniform", side_effect=[3, 0.1]), \
                patch.object(Image.Image, "rotate", autospec=True, return_value=self.image) as rotate:
            apply_operator(self.image, "rotate", 3)
            self.assertEqual(rotate.call_args.args[1], 9)
        with patch("numpy.random.uniform", side_effect=[3, 0.9]), \
                patch.object(Image.Image, "transform", autospec=True, return_value=self.image) as transform:
            apply_operator(self.image, "translate_x", 3)
            self.assertEqual(transform.call_args.args[3], (1, 0, -3, 0, 1, 0))

    def test_shared_geometric_base_two_independent_calls_normalized_once(self):
        settings = copy.deepcopy(SETTINGS)
        transform = AugMixTransform([0.1, 0.2, 0.3], [0.5, 0.25, 0.1], settings)
        base = Image.new("RGB", (32, 32), (100, 150, 200))
        with patch.object(transform, "base_transform", return_value=base) as geometry, \
                patch.object(transform, "augment", side_effect=[torch.full((3, 32, 32), 0.25),
                                                                torch.full((3, 32, 32), 0.75)]) as augment:
            views = transform(self.image)
        geometry.assert_called_once_with(self.image)
        self.assertEqual(augment.call_count, 2)
        self.assertTrue(all(call.args[0] is base for call in augment.call_args_list))
        mean, std = torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.5, 0.25, 0.1])
        torch.testing.assert_close(views[0][:, 0, 0], (torch.tensor([100, 150, 200])/255 - mean)/std)
        torch.testing.assert_close(views[1][:, 0, 0], (0.25 - mean)/std)
        torch.testing.assert_close(views[2][:, 0, 0], (0.75 - mean)/std)
        self.assertEqual(settings, SETTINGS)

    def test_convex_mixture_weights_and_original_skip(self):
        settings = {**SETTINGS, "depth": 1}
        transform = AugMixTransform([0]*3, [1]*3, settings)
        base = Image.new("RGB", (32, 32), (255, 255, 255))
        chains = [Image.new("RGB", (32, 32), (value,)*3) for value in (0, 100, 200)]
        with patch("numpy.random.dirichlet", return_value=np.array([0.2, 0.3, 0.5])), \
                patch("numpy.random.beta", return_value=0.25), \
                patch("src.augmix.apply_operator", side_effect=chains):
            result = transform.augment(base)
        expected = 0.75 + 0.25 * (0.2*0 + 0.3*100/255 + 0.5*200/255)
        torch.testing.assert_close(result, torch.full((3, 32, 32), expected))
        with patch("numpy.random.beta", return_value=0):
            torch.testing.assert_close(transform.augment(base), torch.ones(3, 32, 32))

    def test_seeded_three_view_shape_range_and_independence(self):
        transform = AugMixTransform([0.5]*3, [0.25]*3, SETTINGS)
        torch.manual_seed(1)
        np.random.seed(1)
        first = transform(self.image)
        torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
        torch.manual_seed(1)
        np.random.seed(1)
        second = transform(self.image)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            self.assertEqual(tuple(a.shape), (3, 32, 32))
            self.assertEqual(a.dtype, torch.float32)
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreaterEqual(a.min().item(), -2-1e-6)
            self.assertLessEqual(a.max().item(), 2+1e-6)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertFalse(torch.equal(first[1], first[2]))

    def test_random_depth_is_1_to_3(self):
        transform = AugMixTransform([0]*3, [1]*3, SETTINGS)
        with patch("numpy.random.randint", side_effect=[1, 2, 3]) as randint, \
                patch("src.augmix.apply_operator", return_value=self.image) as operation:
            transform.augment(self.image)
        self.assertEqual(randint.call_count, 3)
        self.assertTrue(all(call.args == (1, 4) for call in randint.call_args_list))
        self.assertEqual(operation.call_count, 6)

    def test_validation_and_conflicting_configuration(self):
        self.assertFalse(validate_augmix({}))
        self.assertTrue(validate_augmix({"augmix": SETTINGS}))
        for key, values in {"width": [0, 11, 1.5, True], "depth": [0, -2, 4, True],
                            "severity": [0, 11, float("nan"), True],
                            "mixture_concentration": [0, -1, float("inf"), "1"],
                            "js_coefficient": [-1, float("nan"), True],
                            "all_ops": [True, 0, "false"]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_augmix({"augmix": {**SETTINGS, key: value}})
        for settings in ({}, {**SETTINGS, "extra": 1}, [], False):
            with self.assertRaises(ValueError):
                validate_augmix({"augmix": settings})
        for conflict in ({"mixup_alpha": 0.2}, {"cutout_length": 16},
                         {"color_jitter": {"brightness": 0.2}},
                         {"random_grayscale_probability": 0.1}, {"label_smoothing": 0.1}):
            with self.subTest(conflict=conflict), self.assertRaises(ValueError):
                validate_augmix({"augmix": SETTINGS, **conflict})


class AugMixLossTests(unittest.TestCase):
    def test_manual_js_and_gradients_match(self):
        torch.manual_seed(3)
        views = [torch.randn(5, 10, requires_grad=True) for _ in range(3)]
        labels = torch.arange(5)
        loss, ce, js = augmix_loss(*views, labels)
        ps = [view.softmax(1) for view in views]
        mean = sum(ps)/3
        expected_js = sum((p*(p.log()-mean.log())).sum(1).mean() for p in ps)/3
        expected = F.cross_entropy(views[0], labels) + 12*expected_js
        torch.testing.assert_close(js, expected_js)
        torch.testing.assert_close(ce, F.cross_entropy(views[0], labels))
        torch.testing.assert_close(loss, expected)
        actual_gradients = torch.autograd.grad(loss, views, retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, views)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertGreater(actual.abs().sum().item(), 0)

    def test_identical_views_have_zero_js_and_symmetry(self):
        logits = torch.randn(4, 10)
        labels = torch.arange(4)
        _, ce, js = augmix_loss(logits, logits, logits, labels)
        self.assertAlmostEqual(js.item(), 0, delta=2e-7)
        views = [torch.randn(4, 10) for _ in range(3)]
        expected_js = augmix_loss(*views, labels)[2]
        for permutation in ((1, 0, 2), (2, 1, 0)):
            torch.testing.assert_close(augmix_loss(*(views[i] for i in permutation), labels)[2], expected_js)

    def test_extreme_bfloat_logits_are_finite_fp32_with_bounded_js(self):
        for dtype in (torch.float32, torch.bfloat16):
            views = [torch.tensor([[10000 if j == i else -10000 for j in range(10)]],
                                  dtype=dtype, requires_grad=True) for i in range(3)]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss, ce, js = augmix_loss(*views, torch.tensor([9]))
            for value in (loss, ce, js):
                self.assertEqual(value.dtype, torch.float32)
                self.assertTrue(torch.isfinite(value))
            self.assertAlmostEqual(js.item(), math.log(3), delta=2e-6)
            loss.backward()
            self.assertTrue(all(torch.isfinite(view.grad).all() for view in views))

    def test_invalid_logits_labels_and_nonfinite_inputs_rejected(self):
        logits = torch.randn(3, 10)
        labels = torch.arange(3)
        for bad in (torch.randn(3, 9), torch.randn(3, 10, 1)):
            with self.assertRaises(ValueError):
                augmix_loss(logits, bad, logits, labels)
        for bad in (labels.float(), labels[:, None], torch.tensor([0])):
            with self.assertRaises(ValueError):
                augmix_loss(logits, logits, logits, bad)
        for value in (float("nan"), float("inf")):
            bad = logits.clone()
            bad[0, 0] = value
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                augmix_loss(logits, bad, logits, labels)


if __name__ == "__main__":
    unittest.main()
