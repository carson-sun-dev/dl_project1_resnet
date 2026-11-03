"""Precision and metadata checks runnable without image/plotting dependencies."""
import copy
from importlib.metadata import PackageNotFoundError, version
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.utils import (dependency_versions, precision_policy, validate_normalization,
                       validate_resume_precision)


class SafetyTests(unittest.TestCase):
    def test_normalization_accepts_finite_values(self):
        validate_normalization({"mean": [0, 0.5, -0.1], "std": [1, 0.2, 0.3]})

    def test_normalization_rejects_invalid_values(self):
        valid = {"mean": [0.5]*3, "std": [0.2]*3}
        for field, values in {"mean": [float("nan"), float("inf"), -float("inf"), "0.5", None, True],
                              "std": [float("nan"), float("inf"), -float("inf"), 0, -0.1, "0.2", None, True]}.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    norm = copy.deepcopy(valid)
                    norm[field][1] = value
                    with self.assertRaisesRegex(ValueError, "Invalid normalization"):
                        validate_normalization(norm)
        for norm in (None, {}, {"mean": [0]*3}, {"mean": [0]*2, "std": [1]*3},
                     {"mean": [0]*3, "std": None}, {"mean": [0]*3, "std": [1]*4}):
            with self.subTest(norm=norm), self.assertRaisesRegex(ValueError, "Invalid normalization"):
                validate_normalization(norm)

    def test_precision_policy_describes_existing_modes(self):
        for dtype, name in ((None, "float32"), (torch.float16, "float16"), (torch.bfloat16, "bfloat16")):
            with self.subTest(dtype=name):
                policy = precision_policy(dtype)
                self.assertEqual(policy, {"compute_dtype": name, "autocast_enabled": dtype is not None,
                                          "grad_scaler_enabled": dtype == torch.float16})
        with self.assertRaises(ValueError):
            precision_policy(torch.float64)

    def test_same_precision_scaler_restore_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="resnet-precision-tests-") as directory:
            checkpoint = Path(directory)/"last.pt"
            for dtype in (None, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype):
                    policy = precision_policy(dtype)
                    # Scaler state format is device independent; no GPU update is needed.
                    scaler = torch.amp.GradScaler("cpu", enabled=policy["grad_scaler_enabled"])
                    state = {"precision_policy": policy, "scaler": scaler.state_dict()}
                    torch.save(state, checkpoint)
                    loaded = torch.load(checkpoint, weights_only=True)
                    validate_resume_precision(loaded, policy)
                    restored = torch.amp.GradScaler("cpu", enabled=policy["grad_scaler_enabled"])
                    restored.load_state_dict(loaded["scaler"])
                    self.assertEqual(restored.state_dict(), scaler.state_dict())

    def test_all_precision_transitions_are_rejected(self):
        for saved_dtype in (None, torch.float16, torch.bfloat16):
            for current_dtype in (None, torch.float16, torch.bfloat16):
                if saved_dtype == current_dtype:
                    continue
                with self.subTest(saved=saved_dtype, current=current_dtype):
                    saved = precision_policy(saved_dtype)
                    state = {"precision_policy": saved,
                             "scaler": {"scale": 65536.0} if saved["grad_scaler_enabled"] else {}}
                    with self.assertRaisesRegex(ValueError, "Unsupported resume precision transition"):
                        validate_resume_precision(state, precision_policy(current_dtype))

    def test_legacy_checkpoints_use_recorded_effective_precision(self):
        for dtype in (None, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                policy = precision_policy(dtype)
                state = {"scaler": {"scale": 65536.0} if policy["grad_scaler_enabled"] else {}}
                environment = {"precision": str(dtype) if dtype is not None else "float32"}
                before = copy.deepcopy(state)
                validate_resume_precision(state, policy, environment)
                self.assertEqual(state, before)
                other = precision_policy(torch.bfloat16 if dtype != torch.bfloat16 else None)
                with self.assertRaisesRegex(ValueError, "Unsupported resume precision transition"):
                    validate_resume_precision(state, other, environment)

    def test_legacy_fp16_can_be_inferred_from_scaler(self):
        validate_resume_precision({"scaler": {"scale": 65536.0}}, precision_policy(torch.float16))

    def test_ambiguous_or_invalid_precision_metadata_is_rejected(self):
        for dtype in (None, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "Cannot determine legacy"):
                validate_resume_precision({"scaler": {}}, precision_policy(dtype))
        for bad in (None, {}, {"compute_dtype": "float64"}):
            with self.subTest(policy=bad), self.assertRaisesRegex(ValueError, "Invalid or unknown"):
                validate_resume_precision({"precision_policy": bad, "scaler": {}}, precision_policy(None))
        with self.assertRaisesRegex(ValueError, "Invalid or unknown"):
            validate_resume_precision({"scaler": {}}, precision_policy(None), {"precision": "unknown"})

    def test_inconsistent_scaler_state_is_rejected(self):
        for dtype in (None, torch.float16, torch.bfloat16):
            policy = precision_policy(dtype)
            wrong_scaler = {} if policy["grad_scaler_enabled"] else {"scale": 65536.0}
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "scaler state"):
                validate_resume_precision({"precision_policy": policy, "scaler": wrong_scaler}, policy)

    def test_provenance_records_dependency_and_cuda_versions(self):
        result = dependency_versions()
        for name in ("torchvision", "numpy", "Pillow"):
            try:
                expected = version(name)
            except PackageNotFoundError:
                expected = None
            self.assertEqual(result[name], expected)
        self.assertEqual(result["cuda"], torch.version.cuda)
        self.assertEqual(result["cudnn"], torch.backends.cudnn.version())
        with patch("src.utils.version", side_effect=PackageNotFoundError):
            absent = dependency_versions()
        self.assertTrue(all(absent[name] is None for name in ("torchvision", "numpy", "Pillow")))


if __name__ == "__main__":
    unittest.main()
