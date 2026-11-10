# Reproducing the final ASAM model

The reproducible target is **Full50k ASAM400, seed 2026, fixed final EMA400**:
PreAct WRN-22-4, 4,298,970 trainable parameters, verified public accuracy 89.80%,
submission 58063369. This guide describes executable commands; it does not claim
that a fresh 400-epoch retraining or prediction run was performed while preparing
the documentation.

## Source and environment

Use [final_model/](../final_model/) as a complete package. Its frozen `src/`
modules and configuration preserve the final trained implementation. The portable
launcher and data-manifest utility are packaging additions. Root `src/` contains Shake-Shake and earlier study implementations; do not mix
its modules with the final ASAM package. Shake-Shake uses its separate
[reproduction guide](../experiments/SHAKE899_REPRODUCTION.md) and environment.

Run from the repository root in an isolated environment:

```bash
python -m pip install -r final_model/requirements.txt
python final_model/run.py verify
```

Verification checks the 20 frozen source modules and canonical configuration
against [source_identity.json](../final_model/source_identity.json). The launcher
uses that canonical configuration automatically for model construction and
training; no `--config` override is supported.

Pinned packages are torch 2.8.0, torchvision 0.23.0, NumPy 2.0.2, Pillow 11.3.0
and matplotlib 3.9.4. The recorded training environment used torch
2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, cuDNN 91002, and an NVIDIA
RTX PRO 4500 Blackwell Server Edition. NumPy metadata and import versions both
recorded 2.0.2. Training used BF16 autocast with FP32 master state and loss,
without a GradScaler. The training record did not specify a Python version.
The CPU export used Python 3.11.9 on Windows with torch 2.8.0+cu128 and CUDA
uninitialized. A CUDA-enabled wheel can execute CPU inference.

Select a compatible CUDA PyTorch build for training. The saved backend settings
are documented in [TRAINING.md](TRAINING.md). Cross-hardware/version bitwise
retraining equality is not established; matching a recipe does not guarantee the
same checkpoint bytes or public score.

## Official inputs and their identities

Acquire the course files through the authorized registered-team/course channel;
they are not distributed here. Required files are `data/train_labels.csv`
(`id,filename,label`), `data/test.csv` (`id,filename`),
`data/sample_submission.csv` (`id,label`), and the corresponding
`data/train_images/` and `data/test_images/` directories.

```bash
python final_model/run.py data prepare --data-dir data --split-file splits/seed2026.json --output provenance/training_image_hashes.csv
python final_model/run.py data audit --data-dir data --split-file splits/seed2026.json --output provenance/training_image_hashes.csv --verify
```

The split is stratified 45k/5k, seed 2026, with 500 validation images per class.
Its training subset supplies fixed normalization; the final trainer uses all
50,000 images. Data preparation creates the split and training-only image
identity file required by the unchanged final configuration. Keep both generated
files private. Before training or resuming, compare these identities:

| Input | Expected SHA-256 |
| --- | --- |
| `splits/seed2026.json` | `2fc944cf610ada2c12bd22adb33e93f460ac0611ece0c3991f3d3c329581751d` |
| `data/train_labels.csv` | `b596f4cbb12107ecac987e3a61123af40408a6bb83719a1dbb5cc3ce70968e93` |
| `provenance/training_image_hashes.csv` | `c7bfa316130fec20850170ff5257e36318c4bf4f1b9be7cd9e09c62114ff6bde` |

These expected digests come from saved training records. A mismatch requires
investigation of input bytes, CSV ordering/format, decoding and dependency
versions; it is not a reason to bypass identity checks or replace the frozen
configuration's expected hash. The manifest utility can also be invoked directly:

```bash
python final_model/tools/prepare_training_data.py --data-dir data --output provenance/training_image_hashes.csv
```

## Scratch training and recovery

The frozen configuration retains `data/`, `splits/seed2026.json`,
`provenance/training_image_hashes.csv`, and
`runs/final_refit_preact_asam400_s2026_cloud` relative to the repository root.
Start in a fresh reproduction checkout with a new, empty configured run directory:

```bash
python final_model/run.py model
python final_model/run.py train --device cuda
```

The model command counts parameters and performs a tiny synthetic forward. The
train command starts the complete scratch refit, including its 400-epoch cosine
horizon. The launcher deliberately binds training to the canonical recipe and
does not offer generic data/run-directory or configuration overrides. Preserve
the frozen files and keep separate reproductions in separate working copies.

Recover the same run after a completed epoch:

```bash
python final_model/run.py train --device cuda --resume
```

Resume checks source, canonical configuration excluding `run_dir`, data/split
identity, software/precision/backend policy, optimizer/scheduler/EMA counters,
and RNG/loader state. Changing source or environment may intentionally prevent
resume. The trusted training recovery file `last.pt` contains optimizer and RNG
objects and is loaded with `weights_only=False`; use only your own or trusted
authorized-team recovery files. Deployment checkpoints use `weights_only=True`.

Completion writes `final_raw.pt` and `final_ema.pt` to the configured run
directory. The deployment target is **final_ema.pt at epoch 400**, not a best
validation checkpoint. Full-data training has no independent validation score.

## CPU prediction and CSV validation

Use an authorized deployment checkpoint. For exact re-export, the saved final
EMA checkpoint SHA-256 is:

```text
07925cdd0689e468ad01f009d26879c724b78a9c0abf5ec9753e1be753611276
```

The unchanged frozen config SHA-256 is
`01e8130d84df0608979e31b6dbeadad1ea81727cf98fff7a0237c4ae2d57f881`.
The verified submitted CSV SHA-256 is
`770dcf2eb84386767c3c204007fba587c9328e3040cdb4a0af5f79db7d7c29a3`.
Checkpoint bytes and predictions are not included in public source delivery.

For an authorized checkpoint saved locally as `private.pt`, use a new output
directory:

```bash
python final_model/run.py predict --checkpoint private.pt --data-dir data --output submissions/reproduction_NEW/submission.csv --device cpu --batch-size 32 --num-workers 0
python final_model/run.py validate --file submissions/reproduction_NEW/submission.csv --sample data/sample_submission.csv
```

The default command requires the original checkpoint hash. To export predictions
from your own completed scratch refit, explicitly opt into reproduction-checkpoint
validation and use a different new output directory:

```bash
python final_model/run.py predict --checkpoint runs/final_refit_preact_asam400_s2026_cloud/final_ema.pt --reproduction-checkpoint --data-dir data --output submissions/scratch_reproduction_NEW/submission.csv --device cpu --batch-size 32 --num-workers 0
python final_model/run.py validate --file submissions/scratch_reproduction_NEW/submission.csv --sample data/sample_submission.csv
```

This option checks the completed 400-epoch EMA endpoint and its source,
configuration, data identity, class map and normalization metadata. It does not
certify the original checkpoint bytes, submitted CSV identity or 89.80% public accuracy for
the newly trained weights.

The predictor loads the deployment state on CPU with `weights_only=True`, builds
from its model configuration, strictly loads weights and enters evaluation mode.
It uses the checkpoint normalization and standard FP32 single-view inference,
checks finite logits, and chooses argmax after softmax. It writes 2,000 unique
official IDs in sample order, integer labels 0-9, and `.metadata.json`; existing
outputs are protected. The final package supports no TTA or ensemble workflow.

Retain checkpoint, source/config and output hashes with the new metadata.
Cross-platform byte-identical re-export is not promised. CSV validation checks
format and ordering, not hidden-label accuracy, and neither command uploads or
selects a competition submission.
