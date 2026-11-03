# Project 1 — CIFAR-10 ResNet under 5M parameters

NYU Tandon Deep Learning F26 competition scaffold. It includes data validation,
a reproducible stratified split, explicit residual networks, training/resumption,
metric plots, and checked Kaggle CSV export. **No formal competition training has
been performed by this scaffold.** There is no accuracy claim or submission yet.

The repository is public at the owner's request. Competition data, checkpoints,
predictions, environments and credentials are excluded by `.gitignore`.

For the complete Chinese team walkthrough, see [TEAM_GUIDE.md](TEAM_GUIDE.md).
The `src.workflow` entry point runs training, plots and best-checkpoint validation
analysis. Hidden-evaluation predictions require an explicit export command.

## 1. Local setup

Python 3.9–3.12, macOS or Linux. Use an isolated environment. Run commands from
the project root; paths in configurations are relative to the working directory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements-installed.txt
```

Dependencies are pinned to PyTorch 2.8.0 / torchvision 0.23.0. For a GPU machine,
verify its PyTorch CUDA build using the official installer instructions; do not
replace an existing course environment blindly. Before a GPU run:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`device: auto` chooses CUDA, then Apple MPS, then CPU. AMP applies only to CUDA
(BF16 when supported, otherwise scaled FP16); MPS/CPU use FP32. A CPU fallback can
be slow; specify `--device cuda` on HPC to fail clearly if CUDA is unavailable.
The original development environment is outside this repository; a project-local
environment has not been installed automatically.

## 2. Supplied dataset

Place the course files here (already present on the owner's machine):

```text
data/
  train_images/             50,000 PNGs
  test_images/               2,000 PNGs
  train_labels.csv          id,filename,label
  test.csv                  id,filename
  sample_submission.csv     id,label
```

```bash
.venv/bin/python -m src.data audit --data-dir data --verify-images
.venv/bin/python -m src.data prepare --data-dir data --output splits/seed2026.json
```

The second command fixes 45,000 training / 5,000 validation IDs, with 500 validation
images per class, seed 2026. Mean/std are computed from the 45,000 training images
only. The split records the exact training CSV hash, all IDs and normalization.
Existing matching splits are reused; differing settings are not silently replaced.

All teammates must use the same dataset and exact split artifact. The split is
initially ignored by Git; transfer it within the registered team, or recreate it
using the pinned environment and compare the split SHA-256. Image audit checks
decoding/shape but does not establish the absence of content duplicates. Never
use hidden test images for training, normalization fitting or manual labeling.

## 3. Models and training configurations

| Config | Channels | Blocks per stage | Trainable parameters | Purpose |
|---|---|---|---|---|
| `configs/baseline.json` | 32 / 64 / 128 | 3 / 3 / 3 | 1,084,586 | Baseline |
| `configs/wide.json` | 64 / 128 / 256 | 3 / 3 / 3 | 4,327,754 | Width experiment |
| `configs/deep.json` | 48 / 96 / 192 | 5 / 5 / 5 | 4,180,858 | Depth experiment at a similar budget |
| `configs/smoke.json` | 16 / 32 / 64 | 1 / 1 / 1 | 78,042 | Two epochs, just a few batches; correctness only |

The stem is a 3×3 stride-1 convolution; stages downsample 32→16→8, followed by global
average pooling and a ten-class linear head. Residual branches contain two 3×3
convolutions and BatchNorm; transitions use 1×1 projection skips. All weights
start from random initialization. The model outputs logits for cross-entropy.

```bash
.venv/bin/python -m src.models --config configs/baseline.json
.venv/bin/python -m src.models --config configs/wide.json
.venv/bin/python -m src.models --config configs/deep.json
```

Every build enforces ≤5,000,000 trainable parameters, including BatchNorm and the
classification head. Width/depth candidates are custom architectures motivated
by residual-network literature, not exact reproductions of a named pretrained model.

Formal initial settings (editable before starting a new experiment): 200 epochs,
batch size 128, SGD momentum 0.9, LR 0.1 with five-epoch warmup then cosine decay
to 0.001, weight decay 0.0005 applied to all parameters, no label smoothing.
Training augmentation: padded random crop and horizontal flip. Validation:
tensor conversion and train-derived normalization only. No MixUp/CutMix is
enabled yet. Training/validation accuracy is stored as a fraction, not a percent.

Start with the small pipeline check, then launch a formal run deliberately:

```bash
.venv/bin/python -m src.train --config configs/smoke.json
.venv/bin/python -m src.train --config configs/baseline.json
```

Examples with explicit overrides:

```bash
.venv/bin/python -m src.train --config configs/baseline.json --device mps
.venv/bin/python -m src.train --config configs/wide.json --device cuda --data-dir /path/to/course/data --run-dir /path/to/scratch/wide_s2026
```

Default `num_workers: 0` is portable for initial debugging. Tune workers/batch size
after measuring on the actual GPU, using a new configuration/run when changing
training hyperparameters. Configurations are starting points, not tuned winners.

## 4. Checkpoints and interruption recovery

Each run saves:

- `config.json`, `split.json`, `environment.json`: setup, IDs, package/device info,
  parameter count, source hashes and Git revision.
- `best.pt`: weights selected by highest validation accuracy, with loss as tie-breaker.
- `last.pt`: last completed epoch, optimizer/scaler, random states and loader RNG states.
- `metrics.json`, `metrics.csv`, `summary.json`: measurements, not predictions.

```bash
.venv/bin/python -m src.train --config configs/baseline.json --resume
.venv/bin/python -m src.plot --run-dir runs/baseline_s2026
```

Use the same overrides and settings as the original run. Dataset location, device
and worker count can change, but hardware/software changes may change numerics.
Restore the original source revision for reproducible continuation; source hashes
are logged but code changes are not automatically reverted. RNG restoration is
tested for uninterrupted vs resumed CPU runs; CUDA/MPS behavior needs validation
on the actual machine and cross-device bitwise equality is not promised.

Checkpoints now store the effective FP32/BF16/FP16 precision policy. Resume requires
the same policy and checks it before restoring training state. For older checkpoints,
keep `environment.json` and `resume_environments.json` with the run: an empty scaler
alone cannot distinguish FP32 from BF16. Unsupported or ambiguous precision changes
fail with an explanation instead of a scaler-loading error. Existing checkpoint files
are not migrated in place. New environment records also include torchvision, NumPy,
Pillow, CUDA build and cuDNN versions plus the effective precision policy.

An interrupt during an epoch loses that unfinished epoch. Completed checkpoints
are written through a temporary file and atomic replacement. Concurrent writers
to the same run directory are blocked with a process lock. A clean timed pilot
can preserve the full schedule without pretending that a short run is final:

```bash
.venv/bin/python -m src.train --config configs/baseline.json --stop-after-epoch 3
.venv/bin/python -m src.train --config configs/baseline.json --resume
```

Training seconds include training data loading/updates but exclude validation.
`compute_seconds_with_validation` includes validation and best-model saving,
but excludes initial setup and some export/checkpoint-writing overhead. Peak GPU
memory is PyTorch allocated CUDA memory; MPS/CPU report `null`.

On NYU Burst, prepare/upload files before allocating long GPU sessions where
possible, then run the script from the allocated Jupyter terminal. Closing the
browser is not a guarantee that a cloud session survives resource reclamation.
Keep checkpoints on persistent storage and retain a separate backup. Do not run
training on a shared login node. Queue/resource provisioning is not automated here.

## 5. Generate and validate a submission

```bash
.venv/bin/python -m src.predict --checkpoint runs/baseline_s2026/best.pt --output submissions/baseline_s2026/submission.csv
.venv/bin/python -m src.submission --file submissions/baseline_s2026/submission.csv --sample data/sample_submission.csv
```

Export uses one model in FP32, no TTA. Predictions are keyed by ID and written in
the exact sample-submission order, even if `test.csv` has another order. The
validator checks unique IDs, row count/order, exactly `id,label`, and integer
labels 0–9. With the supplied dataset the CSV has 2,000 data rows plus its header.
Checkpoint and submission hashes are stored in `submission.metadata.json`.

Each candidate gets a separate submission directory; files are never silently
overwritten. Smoke-test checkpoints are rejected unless `--allow-smoke` is
explicitly used for local pipeline testing. **Do not upload smoke predictions.**
Upload only the final CSV in Kaggle's File Upload dialog; this code does not use
submission quota or log into Kaggle. Full-data retraining and stronger augmentations
can be added once the baseline is validated; they are not implemented here yet.

## 6. Tests and collaboration

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests use small synthetic images in a temporary directory, not the hidden test
set. They cover all candidate parameter budgets, forward/backward passes, fixed
stratified splits, submission validation/order, checkpoint prediction and exact
CPU resumption across a completed epoch. They do not measure CIFAR-10 accuracy.

See `CONTRIBUTING.md` for branch/PR workflow and `REFERENCES.md` for attribution.
Record measured experiments in `experiments/results.csv`. Keep data and large
artifacts outside Git. Project report drafting is deferred; log retention starts now.

## 7. Complete artifact workflow

After preparing the dataset and split, use one config at a time:

```bash
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps
.venv/bin/python -m src.workflow --config configs/baseline.json --device mps --resume
```

Use `--device cuda` on an allocated NVIDIA GPU. The wrapper preserves the original
trainer and its resume rules. It adds `curves.png`, full best-checkpoint FP32
validation metrics, per-class statistics, validation predictions, confusion-matrix
CSV/PNG and `artifacts.json` with file sizes/hashes. Smoke results are marked as
such; partial runs export curves but defer full validation analysis until completion.
Validation data come only from the saved split, whose hash must match the checkpoint.
FP32 reevaluation may differ slightly from training-time AMP metrics.

To regenerate validation analysis without training:

```bash
.venv/bin/python -m src.evaluate --run-dir runs/baseline_s2026 --device mps
```

The workflow defaults to no hidden-evaluation prediction. To explicitly export
a CSV after a complete formal training run, pass
`--submission-output submissions/baseline_s2026/submission.csv`. Existing files
are protected. For a previously completed run, use `src.predict` directly.

The new workflow and evaluation code were statically checked only at the user's
request; they have not been executed. See `VALIDATION.md` for verification scope.
