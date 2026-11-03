import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.data import CLASSES, audit, load_split, make_transform, prepare_split, read_csv
from src.models import build_model, parameter_counts
from src.evaluate import evaluate_checkpoint
from src.workflow import run as run_workflow
from src.predict import predict
from src.submission import validate_submission
from src.train import train
from src.utils import precision_policy, read_json, write_json


def csv_write(path, columns, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="resnet-tests-")
        cls.root = Path(cls.temp.name)
        cls.data = cls.root/"data"
        for folder in ["train_images", "test_images"]:
            (cls.data/folder).mkdir(parents=True)
        rng = np.random.default_rng(7)
        train_rows = []
        for i in range(40):
            filename = f"train_{i:05d}.png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data/"train_images"/filename)
            train_rows.append([str(i), filename, str(i%10)])
        csv_write(cls.data/"train_labels.csv", ["id", "filename", "label"], train_rows)
        for i in range(4):
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(cls.data/"test_images"/f"test_{i}.png")
        # Deliberately different test and sample ordering exercises ID alignment.
        csv_write(cls.data/"test.csv", ["id", "filename"], [[str(i), f"test_{i}.png"] for i in range(4)])
        csv_write(cls.data/"sample_submission.csv", ["id", "label"], [[str(i), 0] for i in [3, 1, 0, 2]])
        cls.split = cls.root/"split.json"
        prepare_split(cls.data, cls.split, seed=2026, val_fraction=0.25)
        cls.split.write_text(json.dumps(read_json(cls.split), separators=(",", ":")))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_data_split_is_reproducible_disjoint_and_stratified(self):
        info = audit(self.data, verify_images=True)
        self.assertEqual(info["train_images"], 40)
        s, train_rows, val_rows = load_split(self.data, self.split)
        other = prepare_split(self.data, self.root/"split_again.json", seed=2026, val_fraction=0.25)
        self.assertEqual(s, other)
        self.assertEqual(len(train_rows), 30)
        self.assertEqual(len(val_rows), 10)
        self.assertFalse(set(s["train_ids"]) & set(s["val_ids"]))
        self.assertEqual(sorted(int(r["label"]) for r in val_rows), list(range(10)))
        self.assertEqual(s, prepare_split(self.data, self.split, seed=2026, val_fraction=0.25))

    def test_architectures_and_parameter_budget(self):
        root = Path(__file__).resolve().parents[1]
        for name in ["baseline", "wide", "deep", "smoke"]:
            m = build_model(read_json(root/"configs"/f"{name}.json")["model"]).eval()
            self.assertLessEqual(parameter_counts(m)["trainable"], 5_000_000)
            logits = m(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(logits.shape), (2, 10))
            logits.sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            build_model({"channels": [128, 256, 512], "blocks": [3, 3, 3]})

    def test_submission_rejects_bad_labels_and_id_order(self):
        out = self.root/"invalid.csv"
        sample = self.data/"sample_submission.csv"
        csv_write(out, ["id", "label"], [[i, 0] for i in range(4)])
        with self.assertRaises(ValueError):
            validate_submission(out, sample)
        out.write_text("id,label\n3,0,extra\n1,0\n0,0\n2,0\n")
        with self.assertRaises(ValueError):
            validate_submission(out, sample)
        csv_write(out, ["id", "label"], [[i, "1.0"] for i in [3, 1, 0, 2]])
        with self.assertRaises(ValueError):
            validate_submission(out, sample)

    def test_split_and_transform_reject_nonfinite_normalization(self):
        path = self.root/"invalid_split.json"
        for field, value in (("mean", float("nan")), ("mean", float("inf")),
                             ("std", float("nan")), ("std", float("inf")),
                             ("std", 0), ("std", -1)):
            with self.subTest(field=field, value=value):
                split = read_json(self.split)
                split[field][0] = value
                # Deliberately bypass write_json's finite-value guard.
                path.write_text(json.dumps(split), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid normalization"):
                    load_split(self.data, path)
                for training in (False, True):
                    with self.assertRaisesRegex(ValueError, "Invalid normalization"):
                        make_transform(split["mean"], split["std"], training)

    def test_prediction_rejects_invalid_checkpoint_metadata(self):
        checkpoint = self.root/"invalid_metadata.pt"
        output = self.root/"invalid_metadata"/"submission.csv"
        valid = {"smoke_test": False, "classes": CLASSES,
                 "normalization": {"mean": [0.5]*3, "std": [0.2]*3}}
        for classes in (None, CLASSES[::-1], CLASSES[:-1], list(range(10))):
            with self.subTest(classes=classes):
                state = copy.deepcopy(valid)
                state["classes"] = classes
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(ValueError, "class mapping"):
                    predict(checkpoint, self.data, output, device="cpu")
        for normalization in (None, {}, {"mean": [float("nan")]*3, "std": [1]*3},
                              {"mean": [0]*3, "std": [float("inf")]*3},
                              {"mean": [0]*3, "std": [0]*3}):
            with self.subTest(normalization=normalization):
                state = copy.deepcopy(valid)
                state["normalization"] = normalization
                torch.save(state, checkpoint)
                with self.assertRaisesRegex(ValueError, "Invalid normalization"):
                    predict(checkpoint, self.data, output, device="cpu")
        self.assertFalse(output.parent.exists())

    def test_prediction_rejects_nonfinite_logits_before_export(self):
        model_config = {"channels": [4, 8, 16], "blocks": [1, 1, 1]}
        model = build_model(model_config)
        checkpoint = self.root/"nonfinite.pt"
        output = self.root/"nonfinite"/"submission.csv"
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with torch.no_grad():
                    model.head.bias.fill_(value)
                torch.save({"smoke_test": False, "classes": CLASSES,
                            "normalization": {"mean": [0.5]*3, "std": [0.2]*3},
                            "model_config": model_config, "model": model.state_dict()}, checkpoint)
                with self.assertRaisesRegex(RuntimeError, "Non-finite prediction logits"):
                    predict(checkpoint, self.data, output, device="cpu")
                self.assertFalse(output.parent.exists())

    def test_training_resume_and_prediction_roundtrip(self):
        c = read_json(Path(__file__).resolve().parents[1]/"configs"/"smoke.json")
        c.update(data_dir=str(self.data), split_file=str(self.split))
        c["model"].update(channels=[4, 8, 16], blocks=[1, 1, 1])
        c["runtime"].update(device="cpu", num_workers=0, cpu_threads=2)
        c["training"].update(batch_size=8, max_train_batches=2, max_val_batches=0)
        c["run_dir"] = str(self.root/"uninterrupted")
        with contextlib.redirect_stdout(io.StringIO()):
            train(c)
            resumed = copy.deepcopy(c)
            resumed["run_dir"] = str(self.root/"resumed")
            partial = train(resumed, stop_after_epoch=1)
            self.assertFalse(partial["complete"])
            complete = train(resumed, resume=True)
        self.assertTrue(complete["complete"])
        self.assertTrue(complete["smoke_test"])
        a = torch.load(Path(c["run_dir"])/"last.pt", weights_only=False)
        b = torch.load(Path(resumed["run_dir"])/"last.pt", weights_only=False)
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        self.assertEqual(a["best_score"], b["best_score"])
        self.assertEqual(a["train_generator"].tolist(), b["train_generator"].tolist())
        self.assertEqual(b["precision_policy"], precision_policy(None))
        environment = read_json(Path(resumed["run_dir"])/"environment.json")
        self.assertEqual(environment["precision_policy"], precision_policy(None))
        self.assertTrue({"torchvision", "numpy", "Pillow", "cuda", "cudnn"} <= environment.keys())
        incompatible = copy.deepcopy(b)
        incompatible["precision_policy"] = precision_policy(torch.bfloat16)
        torch.save(incompatible, Path(resumed["run_dir"])/"last.pt")
        with self.assertRaisesRegex(ValueError, "Unsupported resume precision transition"):
            train(resumed, resume=True)
        # Simulate a legacy checkpoint/environment, preserving its recorded FP32 mode.
        legacy = copy.deepcopy(b)
        legacy.pop("precision_policy")
        torch.save(legacy, Path(resumed["run_dir"])/"last.pt")
        for name in ("environment.json", "resume_environments.json"):
            path = Path(resumed["run_dir"])/name
            record = read_json(path)
            for item in record if isinstance(record, list) else [record]:
                item.pop("precision_policy", None)
            write_json(path, record)
        best = Path(resumed["run_dir"])/"best.pt"
        legacy_best = torch.load(best, weights_only=True)
        self.assertEqual(legacy_best.pop("precision_policy"), precision_policy(None))
        torch.save(legacy_best, best)
        output = self.root/"predictions"/"submission.csv"
        with self.assertRaisesRegex(ValueError, "Smoke checkpoint"):
            predict(best, self.data, output, device="cpu")
        result = predict(best, self.data, output, device="cpu", allow_smoke=True)
        self.assertEqual(result["rows"], 4)
        self.assertTrue(validate_submission(output, self.data/"sample_submission.csv")["valid"])
        self.assertEqual([r["id"] for r in read_csv(output, ["id", "label"])], ["3", "1", "0", "2"])
        # Completed-run wrapper regenerates analysis without another training epoch.
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = run_workflow(resumed, resume=True)
        self.assertTrue(manifest["complete"])
        self.assertTrue(manifest["smoke_test"])
        self.assertIsNone(manifest["submission_path"])
        run_root = Path(resumed["run_dir"])
        metrics = read_json(run_root/"validation"/"metrics.json")
        self.assertEqual(metrics["images"], 10)
        self.assertTrue(metrics["smoke_test"])
        self.assertEqual(metrics["checkpoint_epoch"], complete["best_epoch"])
        preds = read_csv(run_root/"validation"/"predictions.csv",
                         ["id", "filename", "true_label", "predicted_label", "correct"])
        self.assertEqual([r["id"] for r in preds], read_json(self.split)["val_ids"])
        self.assertAlmostEqual(metrics["accuracy"], sum(int(r["correct"]) for r in preds)/10)
        with (run_root/"validation"/"confusion_matrix.csv").open(newline="") as f:
            matrix = list(csv.reader(f))[1:]
        self.assertEqual(sum(sum(map(int, r[1:])) for r in matrix), 10)
        self.assertTrue((run_root/"curves.png").is_file())
        self.assertTrue((run_root/"validation"/"confusion_matrix.png").is_file())
        # Never export a smoke run from the convenience wrapper.
        with self.assertRaisesRegex(ValueError, "smoke submissions"):
            run_workflow(resumed, resume=True, submission_output=self.root/"blocked"/"submission.csv")
        saved_split = read_json(run_root/"split.json")
        saved_split["seed"] += 1
        write_json(run_root/"split.json", saved_split)
        with self.assertRaisesRegex(ValueError, "split differs"):
            evaluate_checkpoint(run_root, self.data, device="cpu")
        with self.assertRaises(FileExistsError):
            train(resumed)
        with self.assertRaises(FileExistsError):
            predict(best, self.data, output, device="cpu", allow_smoke=True)


if __name__ == "__main__":
    unittest.main()
