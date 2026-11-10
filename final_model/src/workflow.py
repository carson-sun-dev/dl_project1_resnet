"""Train one configured model and export its measured validation artifacts."""
import argparse
from pathlib import Path

from .evaluate import evaluate_checkpoint
from .plot import plot
from .predict import predict
from .train import train
from .utils import file_hash, read_json, write_json


def run(config, resume=False, stop_after_epoch=None, submission_output=None):
    if submission_output:
        if config["training"]["max_train_batches"] or config["training"]["max_val_batches"]:
            raise ValueError("Workflow does not export smoke submissions")
        if stop_after_epoch is not None and stop_after_epoch < config["training"]["epochs"]:
            raise ValueError("A partial run cannot export a workflow submission")
        output = Path(submission_output)
        if output.name != "submission.csv":
            raise ValueError("Submission filename must be submission.csv")
        if output.exists() or output.with_suffix(".metadata.json").exists():
            raise FileExistsError("Choose a new submission directory; existing exports are preserved")
    root = Path(config["run_dir"])
    summary = train(config, resume=resume, stop_after_epoch=stop_after_epoch)
    plot(root)
    generated = ["config.json", "split.json", "environment.json", "best.pt", "last.pt",
                 "metrics.json", "metrics.csv", "summary.json", "curves.png"]
    if summary["complete"]:
        evaluate_checkpoint(root, config["data_dir"], config["runtime"]["device"],
                            config["training"]["batch_size"], config["runtime"]["num_workers"])
        generated += ["validation/metrics.json", "validation/predictions.csv",
                      "validation/per_class.csv", "validation/confusion_matrix.csv",
                      "validation/confusion_matrix.png"]
    else:
        print("Partial run saved. Resume to finish validation artifact export.", flush=True)
    submission = None
    if submission_output:
        submission = predict(root / "best.pt", config["data_dir"], submission_output,
                             device=config["runtime"]["device"],
                             batch_size=config["training"]["batch_size"],
                             num_workers=config["runtime"]["num_workers"])
    manifest = {"complete": summary["complete"], "smoke_test": summary["smoke_test"],
                "completed_epochs": summary["completed_epochs"],
                "files": {name: {"bytes": (root/name).stat().st_size,
                                 "sha256": file_hash(root/name)} for name in generated},
                "submission_path": str(Path(submission_output).resolve()) if submission_output else None,
                "submission_metadata": submission}
    write_json(root / "artifacts.json", manifest)
    print(f"Artifacts saved: {root.resolve()}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--run-dir")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--submission-output", help="Optional explicit export; never uploads to Kaggle")
    args = parser.parse_args()
    config = read_json(args.config)
    for key in ("data_dir", "split_file", "run_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.device:
        config["runtime"]["device"] = args.device
    run(config, args.resume, args.stop_after_epoch, args.submission_output)
