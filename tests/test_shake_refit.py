"""CPU-only real training, full-scope export, and strict resume tests."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np
import torch

from shake_fixture import ShakeFixture
from src import shake_refit as refit
from src.evaluate import evaluate_checkpoint
from src.predict import predict
from src.shake_train import train as split_train
from src.submission import validate_submission
from src.utils import file_hash, read_json, write_json


class ShakeRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ShakeFixture.setUpClass.__func__(cls)
        c = read_json("configs/shake_aa_s2026.json")
        cls.source = cls.root / "source"
        c.update(data_dir=str(cls.data), split_file=str(cls.split), run_dir=str(cls.source), split_sha256=file_hash(cls.split))
        c["model"].update(width=4, blocks=[1, 1, 1])
        c["training"].update(epochs=3, batch_size=10, lr=.03, warmup_epochs=0)
        c["runtime"].update(amp=False, cpu_threads=2, num_workers=0, cudnn_benchmark=False)
        with contextlib.redirect_stdout(io.StringIO()):
            split_train(c, "cpu")
            evaluate_checkpoint(cls.source, cls.data, "cpu", batch_size=10)
        cls.patches = [mock.patch.object(refit, "SOURCE_SHA256", file_hash(cls.source / "best.pt")),
                       mock.patch.object(refit, "SPLIT_SHA256", file_hash(cls.split)),
                       mock.patch.object(refit, "SOURCE_EPOCHS", 3),
                       mock.patch.object(refit, "SOURCE_COUNTS", (30, 10)),
                       mock.patch.object(refit, "SCHEDULES", {
                           "scratch": {"epochs": 3, "lr": .03, "warmup_epochs": 0},
                           "warmstart": {"epochs": 3, "lr": .01, "warmup_epochs": 0}})]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):
            p.stop()
        cls.temp.cleanup()

    def equal_tree(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.equal_tree(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal_tree(x, y)
        else:
            self.assertEqual(a, b)

    def run_refit(self, mode, root, **kwargs):
        return refit.train(mode, root, self.source, self.data, "cpu", **kwargs)

    def test_both_modes_exact_resume_all_rows_and_final_only(self):
        source = torch.load(self.source / "best.pt", weights_only=True)
        source_tensor_sha = hashlib.sha256(b"".join(t.contiguous().numpy().tobytes() for t in source["model"].values())).hexdigest()
        for mode in ("scratch", "warmstart"):
            with self.subTest(mode=mode), contextlib.redirect_stdout(io.StringIO()):
                whole, resumed = self.root / f"{mode}_whole", self.root / f"{mode}_resumed"
                result = self.run_refit(mode, whole)
                partial = self.run_refit(mode, resumed, stop_after_epoch=1)
                self.assertFalse(partial["complete"])
                self.assertFalse((resumed / "final.pt").exists())
                before = file_hash(resumed / "last.pt")
                self.run_refit(mode, resumed, resume=True, stop_after_epoch=1)
                self.assertEqual(before, file_hash(resumed / "last.pt"))
                completed = self.run_refit(mode, resumed, resume=True)
                self.assertTrue(result["complete"] and completed["complete"])
                self.assertIsNone(completed["validation"])
                self.assertEqual(completed["training_images"], 40)
                self.assertEqual(completed["validation_images"], 0)
                self.assertFalse((resumed / "best.pt").exists())
                a, b = [torch.load(r / "last.pt", weights_only=False) for r in (whole, resumed)]
                for key in ("model", "optimizer", "rng", "epoch", "initial_model_state_sha256"):
                    self.equal_tree(a[key], b[key])
                for x, y in zip(a["history"], b["history"]):
                    self.equal_tree({k: v for k, v in x.items() if k != "training_seconds"},
                                    {k: v for k, v in y.items() if k != "training_seconds"})
                    self.assertFalse(any(k.startswith("val") for k in x))
                    self.assertEqual(x["train_images"], 40)
                final = torch.load(resumed / "final.pt", weights_only=True)
                self.equal_tree(final["model"], b["model"])
                self.assertEqual(final["normalization"], source["normalization"])
                self.assertEqual(final["identity"]["selection_checkpoint_sha256"], refit.SOURCE_SHA256)
                self.assertEqual(final["identity"]["initial_checkpoint_sha256"], refit.SOURCE_SHA256 if mode == "warmstart" else None)
                self.assertEqual(final["epoch"], 3)
                self.assertIsNone(final["validation"])
                if mode == "warmstart":
                    self.assertEqual(completed["initial_model_state_sha256"], source_tensor_sha)
                else:
                    self.assertNotEqual(completed["initial_model_state_sha256"], source_tensor_sha)

    def test_full_export_predict_and_evaluation_guard(self):
        root = self.root / "export"
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_refit("warmstart", root)
        output = self.root / "submission" / "submission.csv"
        metadata = predict(root / "final.pt", self.data, output, "cpu", batch_size=2)
        self.assertTrue(validate_submission(output, self.data / "sample_submission.csv")["valid"])
        self.assertEqual(metadata["training_scope"], "full_data")
        self.assertEqual(metadata["training_images"], 40)
        self.assertIsNone(metadata["validation_accuracy"])
        self.assertEqual(metadata["selection_checkpoint_sha256"], refit.SOURCE_SHA256)
        # Even if a caller renames final.pt to the evaluator's best.pt convention,
        # full-data scope must block scoring on now-trained former validation IDs.
        shutil.copyfile(root / "final.pt", root / "best.pt")
        with self.assertRaisesRegex(ValueError, "no held-out"):
            evaluate_checkpoint(root, self.data, "cpu")

    def test_source_schedule_config_and_saved_split_guards(self):
        root = self.root / "guards"
        with contextlib.redirect_stdout(io.StringIO()):
            self.run_refit("warmstart", root, stop_after_epoch=1)
        with self.assertRaises(FileExistsError):
            self.run_refit("warmstart", root)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_refit("scratch", root, resume=True)
        cfg = read_json(root / "config.json")
        changed = copy.deepcopy(cfg); changed["training"]["lr"] = 1.
        write_json(root / "config.json", changed)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_refit("warmstart", root, resume=True)
        write_json(root / "config.json", cfg)
        original = (root / "selection_split.json").read_bytes()
        (root / "selection_split.json").write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "selection split changed"):
            self.run_refit("warmstart", root, resume=True)
        (root / "selection_split.json").write_bytes(original)
        with mock.patch.object(refit, "SOURCE_SHA256", "bad"):
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                self.run_refit("warmstart", root, resume=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.run_refit("scratch", self.root / "invalid_stop", stop_after_epoch=0)


if __name__ == "__main__":
    unittest.main()
