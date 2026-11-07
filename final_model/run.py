"""Repository-root entry points for the immutable final ASAM implementation.

The copied src/ and config are historical bytes. Safety checks and portable
training-only preparation live outside them. No command starts automatically.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
IDENTITY_FILE = PACKAGE_ROOT / "source_identity.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_files(package_root=PACKAGE_ROOT):
    """Check the complete module set and config without importing torch."""
    package_root = Path(package_root)
    identity = json.loads((package_root / "source_identity.json").read_text(encoding="utf-8"))
    expected = identity["source_hashes"]
    actual = {path.name for path in (package_root / "src").glob("*.py")}
    if len(expected) != 20 or actual != set(expected):
        raise ValueError("Frozen source module set differs from the 20-module identity")
    for name, digest in expected.items():
        if Path(name).name != name or sha256(package_root / "src" / name) != digest:
            raise ValueError(f"Frozen source SHA256 differs: {name}")
    config_path = package_root / identity["config_file"]
    if not config_path.resolve().is_relative_to(package_root.resolve()):
        raise ValueError("Config identity escapes final_model")
    if sha256(config_path) != identity["config_sha256"]:
        raise ValueError("Frozen configuration SHA256 differs")
    return identity


def activate_frozen_source():
    """Prepend the frozen project; never reuse a previously imported root src."""
    verify_frozen_files()
    for name, module in tuple(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(PACKAGE_ROOT / "src"):
                raise RuntimeError("Another src package is already imported; start a fresh Python process")
    sys.path.insert(0, str(PACKAGE_ROOT))
    module = importlib.import_module("src")
    if Path(module.__file__).resolve().parent != PACKAGE_ROOT / "src":
        raise RuntimeError("Could not activate the frozen final_model/src package")


def frozen_config():
    identity = verify_frozen_files()
    return json.loads((PACKAGE_ROOT / identity["config_file"]).read_text(encoding="utf-8"))


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _same_json(first, second):
    try:
        return _canonical_hash(first) == _canonical_hash(second)
    except (TypeError, ValueError):
        return False


def validate_reproduction_checkpoint_metadata(state, identity=None):
    """Accept only the frozen trainer's full-data, fixed-epoch EMA export schema.

    These checks bind a new export to the recipe and recorded input identities;
    they cannot attest to a new run's training history or leaderboard accuracy.
    The frozen predictor subsequently builds the exact model, enforces its
    parameter budget and strictly loads all state-dict keys and tensor shapes.
    """
    identity = identity or verify_frozen_files()
    config = frozen_config()
    expected_data = identity["training_data"]
    def require(condition, description):
        if not condition:
            raise ValueError(f"Invalid reproduction checkpoint: {description}")
    require(isinstance(state, dict), "expected an export dictionary")
    require(type(state.get("epoch")) is int and state["epoch"] == 400,
            "only completed epoch 400 is deployable")
    require(state.get("weight_variant") == "ema_online_bn_buffers"
            and state.get("deployment_weight_variant_frozen") == "ema_fixed_epoch400"
            and state.get("checkpoint_selection") == "fixed_final_epoch_only",
            "requires the fixed final EMA400 export, without raw or best weights")
    require(state.get("smoke_test") is False and state.get("no_heldout_validation") is True,
            "requires a full-data refit export")
    require(type(state.get("training_images")) is int and state["training_images"] == 50000,
            "expected all 50,000 permitted training images")
    require(type(state.get("ema_updates")) is int and state["ema_updates"] == 156400,
            "expected 156,400 EMA updates")
    require(state.get("inference_policy_frozen_before_refit") == "standard_fp32_single_model",
            "the standard single-model inference policy must be frozen")
    require(_same_json(state.get("model_config"), config["model"]), "model configuration differs")
    counts = {"trainable": identity["trainable_parameters"], "total": identity["trainable_parameters"]}
    require(_same_json(state.get("parameter_counts"), counts), "parameter counts differ")
    require(_same_json(state.get("normalization"), expected_data["normalization"]),
            "normalization differs from the original train-only split")
    require(_same_json(state.get("classes"), expected_data["classes"]), "class mapping differs")
    require(isinstance(state.get("model"), dict) and bool(state["model"]), "missing model state dictionary")
    trained = state.get("identity")
    identity_keys = {"purpose", "schema_version", "frozen_config", "canonical_config_sha256",
                     "source_hashes", "data", "software", "precision_policy", "backend_policy"}
    require(isinstance(trained, dict) and set(trained) == identity_keys,
            "missing or unexpected training identity fields")
    require(trained.get("purpose") == "full_data_final_refit"
            and type(trained.get("schema_version")) is int and trained["schema_version"] == 1,
            "training purpose or schema differs")
    require(_same_json(trained.get("source_hashes"), identity["source_hashes"]),
            "the complete 20-module frozen source identity differs")
    require(_same_json(trained.get("data"), expected_data), "permitted training-data identity differs")
    actual_config = trained.get("frozen_config")
    require(isinstance(actual_config, dict) and isinstance(actual_config.get("runtime"), dict),
            "missing trained configuration")
    device = actual_config["runtime"].get("device")
    require(device in ("cpu", "cuda"), "unexpected training device")
    # run.py train supports this one explicit device override. Every other
    # configuration value, including recipe/data/400 epochs, remains frozen.
    expected_config = {key: value for key, value in config.items() if key != "run_dir"}
    expected_config["runtime"] = {**expected_config["runtime"], "device": device}
    require(_same_json(actual_config, expected_config), "the frozen trained configuration differs")
    require(trained.get("canonical_config_sha256") == _canonical_hash(expected_config),
            "canonical trained-config SHA256 differs")
    policy = {"compute_dtype": "bfloat16" if device == "cuda" else "float32",
              "autocast_enabled": device == "cuda", "grad_scaler_enabled": False}
    require(_same_json(trained.get("precision_policy"), policy)
            and _same_json(state.get("precision_policy"), policy), "training precision policy differs")
    backend = {"cudnn_deterministic": True, "cudnn_benchmark": False,
               "matmul_tf32": False, "cudnn_tf32": False, "float32_matmul_precision": "highest"}
    require(_same_json(trained.get("backend_policy"), backend), "training backend policy differs")
    software = trained.get("software")
    require(isinstance(software, dict) and bool(software), "missing recorded training software")
    return {"checkpoint_kind": "reproduction_final_ema400", "training_device": device,
            "canonical_config_sha256": trained["canonical_config_sha256"],
            "incumbent_score_guaranteed": False}


def validate_prediction_arguments(checkpoint, device="cpu", horizontal_flip=False,
                                  tta_validation=None, reproduction_checkpoint=False):
    """Default to the exact incumbent; explicitly opt into a validated new export."""
    identity = verify_frozen_files()
    if device != "cpu" or horizontal_flip or tta_validation is not None:
        raise ValueError("Final deployment requires CPU, FP32, one standard view and no TTA")
    if reproduction_checkpoint:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("Start a fresh CPU-only process for reproduction checkpoint validation")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_reproduction_checkpoint_metadata(state, identity)
    elif sha256(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the final ASAM EMA400 export")
    return identity


def validate_training_device(device):
    """Reject unsupported CUDA hardware before creating a training run directory."""
    if device == "cpu":
        return
    if device != "cuda":
        raise ValueError("Choose explicit CPU or CUDA training")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA training requires an available CUDA device")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("CUDA training requires native BF16 support; the frozen recipe has no FP16 fallback")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify all 20 source hashes and frozen config; no model or data access")
    sub.add_parser("model", help="Count the frozen model and run its tiny synthetic CPU forward")
    data = sub.add_parser("data", help="Prepare or verify permitted training data only")
    data.add_argument("action", choices=("prepare", "audit"))
    data.add_argument("--data-dir", default="data")
    data.add_argument("--split-file", default="splits/seed2026.json")
    data.add_argument("--output", default="provenance/training_image_hashes.csv")
    data.add_argument("--verify", action="store_true", help="Verify existing files without writing")
    train = sub.add_parser("train", help="Explicit future 400-epoch full-data scratch refit or resume")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int)
    predict = sub.add_parser("predict", help="Explicit future CPU standard-view CSV export")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--data-dir", default="data")
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", choices=("cpu",), default="cpu")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--num-workers", type=int, default=0)
    predict.add_argument("--reproduction-checkpoint", action="store_true",
                         help="Explicitly accept a new scratch final EMA400 export after strict recipe/source/data checks; no score guarantee")
    validate = sub.add_parser("validate", help="Check CSV schema/order against the sample; no image access")
    validate.add_argument("--file", required=True)
    validate.add_argument("--sample", default="data/sample_submission.csv")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if Path.cwd().resolve() != REPOSITORY_ROOT:
        raise ValueError("Run final_model/run.py from the repository root so frozen relative paths retain their meaning")
    identity = verify_frozen_files()
    if args.command == "verify":
        result = {"verified_source_modules": 20, "config_sha256": identity["config_sha256"],
                  "candidate_id": identity["candidate_id"], "models_loaded": False,
                  "images_accessed": False}
    elif args.command == "data":
        sys.path.insert(0, str(PACKAGE_ROOT))
        from tools.prepare_training_data import prepare_training_data
        result = prepare_training_data(args.data_dir, args.split_file, args.output,
                                       verify_only=args.verify or args.action == "audit",
                                       prepare_split=args.action == "prepare")
    else:
        if args.command in ("model", "predict", "validate"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if args.command == "predict":
            validate_prediction_arguments(args.checkpoint, args.device,
                                          reproduction_checkpoint=args.reproduction_checkpoint)
        activate_frozen_source()
        if args.command == "model":
            import torch
            from src.models import build_model, parameter_counts
            torch.set_num_threads(1)
            torch.set_default_dtype(torch.float32)
            model = build_model(frozen_config()["model"]).cpu().eval()
            with torch.no_grad():
                logits = model(torch.zeros(2, 3, 32, 32))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite synthetic model output")
            result = {**parameter_counts(model), "output_shape": list(logits.shape),
                      "device": "cpu", "synthetic_input_only": True}
        elif args.command == "train":
            validate_training_device(args.device)
            from src.final_refit import train_full_data
            config = frozen_config()
            config["runtime"]["device"] = args.device
            result = train_full_data(config, resume=args.resume, stop_after_epoch=args.stop_after_epoch)
        elif args.command == "predict":
            import torch
            from src.predict import predict
            if torch.cuda.is_initialized():
                raise RuntimeError("Start a fresh CPU-only process for final deployment")
            torch.set_num_threads(1)
            torch.set_default_dtype(torch.float32)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            result = predict(args.checkpoint, args.data_dir, args.output, device="cpu",
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             allow_smoke=False, horizontal_flip=False, tta_validation=None)
            if torch.cuda.is_initialized():
                raise RuntimeError("Final deployment unexpectedly initialized CUDA")
        else:
            from src.submission import validate_submission
            result = validate_submission(args.file, args.sample)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
