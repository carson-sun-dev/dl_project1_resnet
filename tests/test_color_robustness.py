import copy
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.augmentation import Cutout, validate_augmentation
from src.data import make_transform
from src.robustness import aggregate_conditions, make_robustness_transform, validate_protocol
from src.utils import read_json


class ColorAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(Path(__file__).resolve().parents[1] / "configs/color_robustness_wide.json")
        cls.color = cls.config["training"]["color_jitter"]
        cls.image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8))

    def test_config_changes_only_color_package_and_run_directory(self):
        original = read_json(Path(__file__).resolve().parents[1] / "configs/local_regularized_wide.json")
        candidate = copy.deepcopy(self.config)
        self.assertNotEqual(candidate.pop("run_dir"), original.pop("run_dir"))
        self.assertEqual(candidate["training"].pop("color_jitter"),
                         {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02, "probability": 0.8})
        self.assertEqual(candidate["training"].pop("random_grayscale_probability"), 0.1)
        self.assertEqual(candidate, original)
        validate_augmentation(self.config["training"])

    def test_color_order_and_config_not_mutated(self):
        color = copy.deepcopy(self.color)
        transform = make_transform([0.5]*3, [0.2]*3, True, 16, color_jitter=color,
                                   random_grayscale_probability=0.1)
        self.assertEqual([type(operation) for operation in transform.transforms],
                         [transforms.RandomCrop, transforms.RandomHorizontalFlip, transforms.RandomApply,
                          transforms.RandomGrayscale, transforms.ToTensor, transforms.Normalize, Cutout])
        self.assertIsInstance(transform.transforms[2].transforms[0], transforms.ColorJitter)
        self.assertEqual(color, self.color)

    def test_disabled_color_matches_legacy_output_and_rng(self):
        legacy = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.2]*3), Cutout(16)])
        for color in (None, {}, {"brightness": 0, "probability": 1}, {"brightness": 0.2, "probability": 0}):
            with self.subTest(color=color):
                torch.manual_seed(11)
                expected = legacy(self.image)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(11)
                actual = make_transform([0.5]*3, [0.2]*3, True, 16, color_jitter=color,
                                        random_grayscale_probability=0)(self.image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_color_seed_reproducibility_and_finite_shape(self):
        transform = make_transform([0.5]*3, [0.2]*3, True, 16, color_jitter=self.color,
                                   random_grayscale_probability=0.1)
        torch.manual_seed(17)
        first = transform(self.image)
        first_rng = torch.get_rng_state().clone()
        torch.manual_seed(17)
        second = transform(self.image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(first_rng, torch.get_rng_state()))
        self.assertEqual(tuple(first.shape), (3, 32, 32))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.isfinite(first).all())
        torch.manual_seed(23)
        self.assertFalse(torch.equal(first, transform(self.image)))

    def test_grayscale_preserves_rgb_before_original_normalization(self):
        transform = make_transform([0.1, 0.2, 0.3], [0.5, 0.25, 0.1], True,
                                   random_grayscale_probability=1)
        # Exclude the two geometric operations to isolate the colour/normalization contract.
        result = transforms.Compose(transform.transforms[2:])(Image.new("RGB", (32, 32), (255, 0, 0)))
        expected = (torch.tensor([76/255]*3) - torch.tensor([0.1, 0.2, 0.3])) / torch.tensor([0.5, 0.25, 0.1])
        self.assertEqual(tuple(result.shape), (3, 32, 32))
        torch.testing.assert_close(result[:, 16, 16], expected)

    def test_clean_validation_ignores_all_training_color_operations(self):
        reference = make_transform([0.1, 0.2, 0.3], [0.5, 0.25, 0.1], False)
        transform = make_transform([0.1, 0.2, 0.3], [0.5, 0.25, 0.1], False, 32,
                                   color_jitter=self.color, random_grayscale_probability=1)
        before = torch.get_rng_state().clone()
        torch.testing.assert_close(transform(self.image), reference(self.image), rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_color_configuration(self):
        invalid = [{"color_jitter": value} for value in ([], "0.2", {"typo": 0.2})]
        for key in ("brightness", "contrast", "saturation", "hue"):
            invalid.extend({"color_jitter": {key: value}} for value in (-0.1, float("nan"), float("inf"), True, "0.2"))
        invalid.extend({"color_jitter": {"probability": value}} for value in (-0.1, 1.1, float("nan"), None, True))
        invalid.extend({"random_grayscale_probability": value} for value in (-0.1, 1.1, float("inf"), None, True))
        invalid.append({"color_jitter": {"hue": 0.51}})
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_augmentation(config)


class RobustnessProtocolTests(unittest.TestCase):
    def test_appearance_operates_on_raw_pixels_before_normalization(self):
        image = Image.new("RGB", (32, 32), (200, 100, 50))
        condition = {"name": "brightness_half", "kind": "brightness", "factor": 0.5}
        transform = make_robustness_transform([0.1, 0.2, 0.3], [0.5, 0.25, 0.1], condition)
        before = torch.get_rng_state().clone()
        result = transform(image)
        expected = (torch.tensor([100, 50, 25])/255 - torch.tensor([0.1, 0.2, 0.3])) / torch.tensor([0.5, 0.25, 0.1])
        torch.testing.assert_close(result[:, 0, 0], expected)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(image.getpixel((0, 0)), (200, 100, 50))

    def test_contrast_and_grayscale_semantics(self):
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[:, 16:] = 255
        result = make_robustness_transform([0]*3, [1]*3,
                   {"name": "contrast_half", "kind": "contrast", "factor": 0.5})(Image.fromarray(pixels))
        torch.testing.assert_close(result[:, 0, 0], torch.tensor([64/255]*3))
        torch.testing.assert_close(result[:, 0, 31], torch.tensor([191/255]*3))
        gray = make_robustness_transform([0]*3, [1]*3,
                   {"name": "gray", "kind": "grayscale", "factor": 1})(Image.new("RGB", (32, 32), (255, 0, 0)))
        torch.testing.assert_close(gray, torch.full((3, 32, 32), 76/255))

    def test_aggregation_excludes_clean_and_reports_families(self):
        rows = [{"name": name, "kind": kind, "standard": {"accuracy": standard},
                 "horizontal_flip": {"accuracy": flip}} for name, kind, standard, flip in
                (("clean", "identity", 1, 1), ("dark", "brightness", 0.2, 0.4),
                 ("light", "brightness", 0.8, 0.8), ("gray", "grayscale", 0.4, 0.6))]
        report = aggregate_conditions(rows)
        self.assertAlmostEqual(report["standard"]["mean_transformed_accuracy"], (0.2+0.8+0.4)/3)
        self.assertEqual(report["standard"]["worst_condition"], "dark")
        self.assertEqual(report["standard"]["family_mean_accuracy"], {"brightness": 0.5, "grayscale": 0.4})
        self.assertAlmostEqual(report["horizontal_flip"]["mean_transformed_accuracy"], 0.6)

    def test_protocol_rejects_invalid_or_random_conditions(self):
        valid = read_json(Path(__file__).resolve().parents[1] / "configs/appearance_robustness_v1.json")
        validate_protocol(valid)
        invalid = []
        missing_clean = copy.deepcopy(valid)
        missing_clean["conditions"] = missing_clean["conditions"][1:]
        invalid.append(missing_clean)
        for key, value in (("name", "clean"), ("kind", "random_crop"), ("factor", 0),
                           ("factor", float("nan")), ("factor", True), ("factor", -1)):
            item = copy.deepcopy(valid)
            item["conditions"][1][key] = value
            invalid.append(item)
        item = copy.deepcopy(valid)
        item["conditions"][-1]["factor"] = 0.5
        invalid.append(item)
        for item in invalid:
            with self.subTest(protocol=item), self.assertRaises(ValueError):
                validate_protocol(item)


if __name__ == "__main__":
    unittest.main()
