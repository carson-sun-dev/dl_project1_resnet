"""CPU integration checks for coexisting historical, ASAM and Shake-Shake routes."""
import copy
import json
from pathlib import Path
import unittest

import torch

from src.models import build_model, parameter_counts


class ModelRouteTests(unittest.TestCase):
    def test_configured_routes_keep_parameter_budgets(self):
        for filename, expected in (
            ("baseline.json", 1_084_586),
            ("wide.json", 4_327_754),
            ("deep.json", 4_180_858),
            ("preactivation_wrn22_4_color.json", 4_298_970),
            ("shake_aa_s2026.json", 4_561_354),
            ("shake_full1000_s2026.json", 4_561_354),
        ):
            with self.subTest(config=filename):
                config = json.loads((Path("configs") / filename).read_text(encoding="utf-8-sig"))
                counts = parameter_counts(build_model(config["model"]))
                self.assertEqual(counts["trainable"], expected)
                self.assertLessEqual(counts["trainable"], 5_000_000)

    def test_factory_keeps_checkpoint_shapes_and_eval_protocols(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)
        images = torch.randn(2, 3, 32, 32)
        for config in (
            {"channels": [4, 8, 16], "blocks": [1, 1, 1]},
            {"architecture": "preactivation_wrn", "stem_channels": 4,
             "channels": [4, 8, 16], "blocks": [1, 1, 1]},
            {"architecture": "shake_shake", "width": 4, "blocks": [1, 1, 1]},
        ):
            with self.subTest(config=config), torch.no_grad():
                model = build_model(config).eval()
                restored = build_model(config).eval()
                restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                expected = model(images)
                self.assertEqual(tuple(expected.shape), (2, 10))
                self.assertTrue(torch.isfinite(expected).all())
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
                if config.get("architecture") == "shake_shake":
                    torch.testing.assert_close(expected,
                        (model.classify(images) + model.classify(images.flip(-1))) / 2,
                        rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
