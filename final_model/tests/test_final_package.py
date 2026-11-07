"""Cheap CPU checks of the exact frozen architecture and wrapper policy."""
import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_package_run", PACKAGE_ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.activate_frozen_source()
from src.final_refit import validate_refit_config
from src.models import build_model, parameter_counts


class FinalPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_hashes_and_configuration_match_recorded_final_candidate(self):
        identity = runner.verify_frozen_files()
        self.assertEqual(len(identity["source_hashes"]), 20)
        self.assertEqual(identity["config_sha256"], "01e8130d84df0608979e31b6dbeadad1ea81727cf98fff7a0237c4ae2d57f881")
        self.assertEqual(identity["source_hashes"]["final_refit.py"], "661f09a1079ac018e8e10b9524566808ca2568d8f654954c4fb70056ac343d9b")
        self.assertEqual(identity["source_hashes"]["models.py"], "831e84200857561e3b11b11864bae56ac9cccadbb4627b8f958ef04f56f12cbc")
        config = runner.frozen_config()
        self.assertEqual(validate_refit_config(config)["method"], "asam")
        self.assertEqual(config["refit"]["image_manifest_sha256"], "c7bfa316130fec20850170ff5257e36318c4bf4f1b9be7cd9e09c62114ff6bde")
        self.assertEqual(config["refit"]["inference_policy"], "standard_fp32_single_model")

    def test_imports_resolve_to_final_package_and_relative_paths_stay_at_repository_root(self):
        import src.models
        self.assertEqual(Path(src.models.__file__).resolve(), PACKAGE_ROOT / "src/models.py")
        config = runner.frozen_config()
        self.assertEqual(config["data_dir"], "data")
        self.assertEqual(config["split_file"], "splits/seed2026.json")
        self.assertEqual(config["refit"]["image_manifest"], "provenance/training_image_hashes.csv")
        self.assertEqual(config["run_dir"], "runs/final_refit_preact_asam400_s2026_cloud")

    def test_exact_parameter_count_and_tiny_finite_cpu_forward_and_gradient(self):
        torch.random.default_generator.manual_seed(2026)
        model = build_model(runner.frozen_config()["model"]).cpu()
        self.assertEqual(parameter_counts(model), {"trainable": 4298970, "total": 4298970})
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        logits = model(images)
        self.assertEqual(list(logits.shape), [2, 10])
        self.assertTrue(torch.isfinite(logits).all())
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_mutated_or_extra_source_and_mutated_config_fail_closed(self):
        import shutil
        with tempfile.TemporaryDirectory(prefix="final-source-check-") as directory:
            target = Path(directory)
            shutil.copytree(PACKAGE_ROOT / "src", target / "src")
            shutil.copytree(PACKAGE_ROOT / "configs", target / "configs")
            shutil.copyfile(PACKAGE_ROOT / "source_identity.json", target / "source_identity.json")
            runner.verify_frozen_files(target)
            changed = target / "src/models.py"
            original = changed.read_bytes()
            changed.write_bytes(original + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                runner.verify_frozen_files(target)
            changed.write_bytes(original)
            extra = target / "src/extra.py"
            extra.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "module set"):
                runner.verify_frozen_files(target)
            extra.unlink()
            config = target / json.loads((target / "source_identity.json").read_text())["config_file"]
            config.write_bytes(config.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "configuration"):
                runner.verify_frozen_files(target)

    def test_prediction_policy_rejects_tta_gpu_and_wrong_checkpoint_without_loading_weights(self):
        for kwargs in ({"device": "cuda"}, {"horizontal_flip": True}, {"tta_validation": "report.json"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "no TTA"):
                runner.validate_prediction_arguments("nonexistent.pt", **kwargs)
        with tempfile.TemporaryDirectory(prefix="final-checkpoint-hash-") as directory:
            path = Path(directory) / "wrong.pt"
            path.write_bytes(b"synthetic, not a checkpoint")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
                runner.validate_prediction_arguments(path)
        parser = runner._parser()
        arguments = parser.parse_args(["predict", "--checkpoint", "team.pt", "--output", "out/submission.csv"])
        self.assertEqual((arguments.device, arguments.batch_size, arguments.num_workers), ("cpu", 32, 0))
        self.assertFalse(arguments.reproduction_checkpoint)

    def reproduction_state(self, device="cuda"):
        identity = runner.verify_frozen_files()
        config = runner.frozen_config()
        trained_config = {key: value for key, value in config.items() if key != "run_dir"}
        trained_config["runtime"]["device"] = device
        policy = {"compute_dtype": "bfloat16" if device == "cuda" else "float32",
                  "autocast_enabled": device == "cuda", "grad_scaler_enabled": False}
        return {"model_config": config["model"], "epoch": 400,
                "normalization": identity["training_data"]["normalization"],
                "classes": identity["training_data"]["classes"],
                "parameter_counts": {"trainable": 4298970, "total": 4298970},
                "identity": {"purpose": "full_data_final_refit", "schema_version": 1,
                             "frozen_config": trained_config,
                             "canonical_config_sha256": runner._canonical_hash(trained_config),
                             "source_hashes": identity["source_hashes"],
                             "data": identity["training_data"],
                             "software": {"torch": "2.8.0+cu128", "numpy": "2.0.2"},
                             "precision_policy": policy,
                             "backend_policy": {"cudnn_deterministic": True, "cudnn_benchmark": False,
                                                "matmul_tf32": False, "cudnn_tf32": False,
                                                "float32_matmul_precision": "highest"}},
                "precision_policy": policy, "smoke_test": False, "training_images": 50000,
                "no_heldout_validation": True, "checkpoint_selection": "fixed_final_epoch_only",
                "deployment_weight_variant_frozen": "ema_fixed_epoch400",
                "inference_policy_frozen_before_refit": "standard_fp32_single_model",
                # Deliberately tiny fake weights: metadata validation does not
                # build or run a model. The actual predictor strictly loads the
                # complete state dict against the exact budget-checked model.
                "model": {"synthetic.weight": torch.tensor([0.123])},
                "weight_variant": "ema_online_bn_buffers", "ema_updates": 156400}

    def test_reproduction_metadata_accepts_only_explicit_cpu_or_cuda_frozen_recipe_variants(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                state = self.reproduction_state(device)
                result = runner.validate_reproduction_checkpoint_metadata(state)
                self.assertEqual(result["training_device"], device)
                self.assertFalse(result["incumbent_score_guaranteed"])
                state["model"]["synthetic.weight"].add_(1)
                runner.validate_reproduction_checkpoint_metadata(state)

    def test_reproduction_opt_in_uses_safe_cpu_load_and_preserves_source_verification(self):
        state = self.reproduction_state()
        with patch.object(torch, "load", return_value=state) as load, \
                patch.object(torch.cuda, "is_initialized", return_value=False):
            runner.validate_prediction_arguments("new_scratch_final_ema.pt", reproduction_checkpoint=True)
            load.assert_called_once_with("new_scratch_final_ema.pt", map_location="cpu", weights_only=True)
        with patch.object(torch, "load") as load, \
                patch.object(runner, "verify_frozen_files", side_effect=ValueError("source differs")):
            with self.assertRaisesRegex(ValueError, "source differs"):
                runner.validate_prediction_arguments("new_scratch_final_ema.pt", reproduction_checkpoint=True)
            load.assert_not_called()
        arguments = runner._parser().parse_args(["predict", "--checkpoint", "new.pt", "--output",
                                                "new/submission.csv", "--reproduction-checkpoint"])
        self.assertTrue(arguments.reproduction_checkpoint)

    def test_reproduction_metadata_rejects_incomplete_different_or_nonfinal_exports(self):
        changes = [
            lambda value: value.pop("identity"),
            lambda value: value.update(epoch=399),
            lambda value: value.update(weight_variant="raw_online"),
            lambda value: value.update(checkpoint_selection="best_validation"),
            lambda value: value.update(smoke_test=True),
            lambda value: value.update(no_heldout_validation=False),
            lambda value: value.update(training_images=45000),
            lambda value: value.update(ema_updates=156399),
            lambda value: value.update(model={}),
            lambda value: value.update(inference_policy_frozen_before_refit="horizontal_flip"),
            lambda value: value["model_config"].update(channels=[32, 64, 128]),
            lambda value: value["normalization"].update(std=[1.0, 1.0, 1.0]),
            lambda value: value.update(classes=list(reversed(value["classes"]))),
            lambda value: value.update(parameter_counts={"trainable": 5000001, "total": 5000001}),
            lambda value: value["identity"].update(purpose="distillation"),
            lambda value: value["identity"].update(schema_version=True),
            lambda value: value["identity"]["source_hashes"].update({"models.py": "0" * 64}),
            lambda value: value["identity"]["data"].update(image_manifest_sha256="0" * 64),
            lambda value: value["identity"]["frozen_config"]["training"].update(epochs=399),
            lambda value: value["identity"]["frozen_config"]["training"]["optimization"].update(method="sam"),
            lambda value: value["identity"]["frozen_config"].update(seed=42),
            lambda value: value["identity"].update(canonical_config_sha256="0" * 64),
            lambda value: value["identity"]["precision_policy"].update(compute_dtype="float16"),
            lambda value: value["identity"]["backend_policy"].update(matmul_tf32=True),
            lambda value: value["identity"].update(software={}),
        ]
        for index, change in enumerate(changes):
            state = copy.deepcopy(self.reproduction_state())
            change(state)
            with self.subTest(case=index), self.assertRaisesRegex(ValueError, "reproduction checkpoint"):
                runner.validate_reproduction_checkpoint_metadata(state)

    def test_native_bf16_training_preflight_rejects_unavailable_or_emulated_cuda(self):
        with patch.object(torch.cuda, "is_available") as available:
            runner.validate_training_device("cpu")
            available.assert_not_called()
        with patch.object(torch.cuda, "is_available", return_value=False), \
                self.assertRaisesRegex(ValueError, "available CUDA"):
            runner.validate_training_device("cuda")
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=False) as supported:
            with self.assertRaisesRegex(ValueError, "native BF16"):
                runner.validate_training_device("cuda")
            supported.assert_called_once_with(including_emulation=False)
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "is_bf16_supported", return_value=True) as supported:
            runner.validate_training_device("cuda")
            supported.assert_called_once_with(including_emulation=False)


if __name__ == "__main__":
    unittest.main()
