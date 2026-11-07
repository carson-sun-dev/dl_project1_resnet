"""Independent CPU training/resume/export checks for the Shake-Shake route."""
import contextlib
import copy
import io
from pathlib import Path
import unittest

import numpy as np
import torch

from shake_fixture import ShakeFixture
from src.data import load_split
from src.evaluate import evaluate_checkpoint
from src.models import build_model
from src.predict import predict
from src.shake_train import CachedShakeDataset, epoch_loader, permitted_pixels, train
from src.submission import validate_submission
from src.utils import file_hash, read_json


class ShakePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ShakeFixture.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, name):
        c = read_json("configs/shake_aa_s2026.json")
        c.update(data_dir=str(self.data), split_file=str(self.split), run_dir=str(self.root / name),
                 split_sha256=file_hash(self.split))
        c["model"].update(width=4, blocks=[1, 1, 1])
        c["training"].update(epochs=3, batch_size=10, lr=.05, min_lr=.005, warmup_epochs=0)
        c["runtime"].update(amp=False, cpu_threads=2, num_workers=0, cudnn_benchmark=False)
        return c

    def assert_tree_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            self.assertTrue(np.array_equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_real_cpu_resume_split_and_prediction_export(self):
        c = self.config("shake_reference")
        with contextlib.redirect_stdout(io.StringIO()):
            report = train(c, "cpu")
            a = torch.load(Path(c["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
            best_a = torch.load(Path(c["run_dir"]) / "best.pt", map_location="cpu", weights_only=True)
            resumed = self.config("shake_resumed")
            partial = train(resumed, "cpu", stop_after_epoch=1)
            self.assertFalse(partial["complete"])
            completed = train(resumed, "cpu", resume=True)
            b = torch.load(Path(resumed["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
            best_b = torch.load(Path(resumed["run_dir"]) / "best.pt", map_location="cpu", weights_only=True)
        self.assertTrue(report["complete"] and completed["complete"])
        for key in ("model", "optimizer", "epoch", "rng"):
            self.assert_tree_equal(a[key], b[key])
        self.assert_tree_equal(best_a["model"], best_b["model"])
        self.assertEqual(best_a["epoch"], best_b["epoch"])
        self.assertEqual(best_a["identity"]["split_sha256"], file_hash(self.split))
        self.assertFalse(best_a["smoke_test"])
        self.assertEqual(len(a["history"]), 3)
        for left, right in zip(a["history"], b["history"]):
            for key in ("epoch", "lr", "train_loss", "raw_val_accuracy", "raw_val_loss", "val_accuracy", "val_loss", "train_images"):
                if key in left:
                    self.assertEqual(left[key], right[key])
            self.assertEqual(left["train_images"], 30)
        split, train_rows, val_rows = load_split(self.data, self.split)
        self.assertFalse(set(r["id"] for r in train_rows) & set(r["id"] for r in val_rows))
        evaluated = evaluate_checkpoint(c["run_dir"], self.data, "cpu", batch_size=5)
        self.assertEqual(evaluated["images"], 10)
        selected_row = a["history"][best_a["epoch"] - 1]
        self.assertEqual(evaluated["accuracy"], selected_row["val_accuracy"])
        self.assertAlmostEqual(evaluated["cross_entropy"], selected_row["val_loss"], places=5)
        model = build_model(best_a["model_config"]).eval()
        model.load_state_dict(best_a["model"])
        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)
            torch.testing.assert_close(model(x), (model.classify(x) + model.classify(x.flip(-1))) / 2)
        output = self.root / "shake_submission" / "submission.csv"
        metadata = predict(Path(c["run_dir"]) / "best.pt", self.data, output, "cpu", batch_size=2)
        self.assertTrue(validate_submission(output, self.data / "sample_submission.csv")["valid"])
        self.assertEqual(metadata["rows"], 4)
        self.assertEqual(metadata["checkpoint_sha256"], file_hash(Path(c["run_dir"]) / "best.pt"))
        # A committed last.pt can rebuild interrupted secondary exports.
        resumed_root = Path(resumed["run_dir"])
        (resumed_root / "best.pt").write_bytes(b"interrupted artifact write")
        (resumed_root / "metrics.jsonl").write_text("incomplete", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            repaired = train(resumed, "cpu", resume=True)
        self.assertTrue(repaired["complete"])
        restored_best = torch.load(resumed_root / "best.pt", weights_only=True)
        self.assert_tree_equal(restored_best["model"], best_b["model"])
        self.assertEqual(len((resumed_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()), 3)

    def test_reject_changed_recipe_or_split_on_resume(self):
        c = self.config("shake_guards")
        with contextlib.redirect_stdout(io.StringIO()):
            train(c, "cpu", stop_after_epoch=1)
        changed = copy.deepcopy(c)
        changed["training"]["lr"] = .07
        with self.assertRaises(ValueError):
            train(changed, "cpu", resume=True)
        changed = copy.deepcopy(c)
        changed["split_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            train(changed, "cpu", resume=True)
        saved_split = Path(c["run_dir"]) / "split.json"
        original_split = saved_split.read_bytes()
        try:
            saved_split.write_bytes(original_split + b"\n")
            with self.assertRaisesRegex(ValueError, "saved validation split"):
                train(c, "cpu", resume=True)
        finally:
            saved_split.write_bytes(original_split)
        saved_config = Path(c["run_dir"]) / "config.json"
        original_config = saved_config.read_bytes()
        try:
            saved_config.write_text(original_config.decode("utf-8").replace('"seed": 2026', '"seed": 2027'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "saved recipe"):
                train(c, "cpu", resume=True)
        finally:
            saved_config.write_bytes(original_config)
        with self.assertRaises(FileExistsError):
            train(c, "cpu")

    def test_two_spawn_workers_replay_the_same_epoch_only_on_train_ids(self):
        split, rows, validation = load_split(self.data, self.split)
        pixels = torch.from_numpy(permitted_pixels(self.data, rows))
        labels = torch.tensor([int(r["label"]) for r in rows])
        dataset = CachedShakeDataset(pixels, labels, split["mean"], split["std"])
        self.assertEqual(len(dataset), 30)
        self.assertFalse(set(r["id"] for r in rows) & set(r["id"] for r in validation))
        first = list(epoch_loader(dataset, 10, 2026, 2, num_workers=2))
        # Main-process stochastic Shake-Shake draws must not disturb worker replay.
        torch.rand(123)
        resumed = list(epoch_loader(dataset, 10, 2026, 2, num_workers=2))
        self.assertEqual(sum(len(y) for _, y in first), 30)
        for (x, y), (xx, yy) in zip(first, resumed):
            self.assertTrue(torch.equal(x, xx))
            self.assertTrue(torch.equal(y, yy))

    def test_resume_at_stop_limit_does_not_advance(self):
        c = self.config("shake_already_at_stop")
        with contextlib.redirect_stdout(io.StringIO()):
            first = train(c, "cpu", stop_after_epoch=1)
        checkpoint = Path(c["run_dir"]) / "last.pt"
        before_hash = file_hash(checkpoint)
        before = torch.load(checkpoint, map_location="cpu", weights_only=False)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = train(c, "cpu", resume=True, stop_after_epoch=1)
        after = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(first["completed_epochs"], 1)
        self.assertEqual(resumed["completed_epochs"], 1)
        self.assertFalse(resumed["complete"])
        self.assertEqual(before_hash, file_hash(checkpoint))
        self.assert_tree_equal(before, after)


if __name__ == "__main__":
    unittest.main()
