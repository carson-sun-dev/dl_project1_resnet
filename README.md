# CIFAR-10 residual networks under 5 million parameters

Team HYS, NYU Tandon Deep Learning F26. This repository provides two
scratch-trained residual CNN routes using only the permitted CIFAR-10 data.
Both final full-data runs use all 50,000 labeled images and have **no independent
local validation score**. Private scores and final submission selections are unknown.

| Route | Implementation / configuration | Trainable parameters | Reported Public accuracy / inference |
| --- | --- | ---: | --- |
| PreAct WRN-22-4 + ASAM | [final_model/](final_model/), [frozen ASAM400 config](final_model/configs/final_refit_preact_asam400_s2026_cloud.json) | 4,298,970 | **89.80%**, receipt/CSV byte match, submission **58063369**; fixed final EMA400, FP32 CPU, single view |
| Shake-Shake-26 2x40d | [src/shake_shake.py](src/shake_shake.py), [1000-epoch config snapshot](configs/shake_full1000_s2026.json) | 4,561,354 | **89.90%**, teammate-confirmed; fixed final epoch1000, FP32 with built-in original/flip logit mean |

Both instantiated models are below 5,000,000 trainable parameters, including the
classifier and BatchNorm affine parameters. The Shake-Shake score is documented
in the merged teammate report; this integration does not independently reconcile
its Kaggle upload or establish permission for flip inference. Its inference
policy differs from ASAM's; do not present the scores as a controlled optimizer
ablation. No new training or competition-data inference was performed for integration.

Start with [ASAM reproduction](docs/REPRODUCIBILITY.md) or
[Shake-Shake reproduction](experiments/SHAKE899_REPRODUCTION.md).

## Code and documentation

| Location | Purpose |
| --- | --- |
| [Shake-Shake reproduction](experiments/SHAKE899_REPRODUCTION.md) | Scratch training, required private source-run materials, recovery and flip-aware prediction |
| [Shake-Shake optimization report](experiments/SHAOHAN_OPTIMIZATION_REPORT.md) | Teammate-authored results and limitations |
| [final_model/](final_model/) | Exact final training source and configuration, with a portable command launcher and source verification |
| [Architecture](docs/ARCHITECTURE.md) | Blocks, dimensions, initialization and parameter accounting |
| [Training](docs/TRAINING.md) | Data, augmentation, optimizer, EMA and precision policy |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Input identities, setup, scratch training, recovery and CPU prediction |
| [Experiments](docs/EXPERIMENTS.md) | Twenty completed trajectories, partial runs, controls and unrun proposals |
| [Results](docs/RESULTS.md) | Public results, evidence strength and interpretation limits |
| [References](docs/REFERENCES.md) | Methods, implementation attribution and assistance disclosure |
| [Integration review](docs/INTEGRATION_REVIEW.md) | Merge scope, preserved source identities, test limits and outstanding team materials |

Root `src/` provides the Shake-Shake route and earlier study implementations. Use `final_model/` for the final ASAM result; root
modules with the same names are not interchangeable with the frozen final source.
The experiment inventory includes historical variants whose complete frozen
implementations are not all distributed in this public package.

## Reproduce the ASAM pipeline

Run commands from the repository root in an isolated Python environment. Install
the pinned dependencies in [final_model/requirements.txt](final_model/requirements.txt), with the matching
CUDA build of PyTorch when training. See the reproduction guide for the recorded
environment and limits on exact numerical reproduction.

```bash
python -m pip install -r final_model/requirements.txt
python final_model/run.py verify
```

Obtain the official course data through the authorized competition/course
channel and place it under `data/`:

```text
data/
  train_images/           # 50,000 supplied RGB images
  test_images/            # 2,000 supplied evaluation images
  train_labels.csv        # id,filename,label
  test.csv                # id,filename
  sample_submission.csv   # id,label
```

Prepare and verify the original split and the training-image manifest locally:

```bash
python final_model/run.py data prepare --data-dir data --split-file splits/seed2026.json --output provenance/training_image_hashes.csv
python final_model/run.py data audit --data-dir data --split-file splits/seed2026.json --output provenance/training_image_hashes.csv
```

The [reproduction guide](docs/REPRODUCIBILITY.md) lists the required hashes. The
split fixes normalization and historical provenance; final training uses all
50,000 labeled images. Start with a new, empty configured run directory:

```bash
python final_model/run.py model
python final_model/run.py train --device cuda
# Recover the same run at an epoch boundary:
python final_model/run.py train --device cuda --resume
```

For CPU prediction, `private.pt` means an authorized copy of the original final
EMA deployment checkpoint. Use a new output path:

```bash
python final_model/run.py predict --checkpoint private.pt --data-dir data --output submissions/reproduction_NEW/submission.csv --device cpu --batch-size 32 --num-workers 0
python final_model/run.py validate --file submissions/reproduction_NEW/submission.csv --sample data/sample_submission.csv
```

For your own completed scratch run, use its `final_ema.pt` path and add
`--reproduction-checkpoint` to the prediction command. This explicitly validates
a new full-horizon EMA reproduction rather than requiring the original
checkpoint hash; it does not establish the original 89.80% score or byte identity.

The exporter writes `id,label` in official sample order, with 2,000 unique IDs and
integer labels 0-9, plus inference metadata. These commands generate and validate
a local CSV; they do not upload or select a competition submission.

## Reproduce the Shake-Shake pipeline

Use a **separate environment** with [requirements-shake899.txt](requirements-shake899.txt)
(NumPy/Pillow differ from the ASAM environment). Run from the repository root:

```bash
python -m pip install -r requirements-shake899.txt
python -m src.models --config configs/shake_aa_s2026.json
python -m src.shake_refit --mode scratch --epochs 1000 --run-dir runs/shake_full1000_reproduction_NEW --device cuda
# Recover the same run: append --resume to the identical training command.
python -m src.predict --checkpoint runs/shake_full1000_reproduction_NEW/final.pt --output submissions/shake_full1000_reproduction_NEW/submission.csv --device cpu
python -m src.submission --file submissions/shake_full1000_reproduction_NEW/submission.csv --sample data/sample_submission.csv
```

Before training, obtain the matching private `runs/shake_aa_s2026/` source-run
materials listed in the [Shake-Shake guide](experiments/SHAKE899_REPRODUCTION.md).
Scratch mode initializes new weights but still requires those selection/provenance
records. `configs/shake_full1000_s2026.json` is a recorded snapshot, **not** a
`src.train` input; `shake_refit` does not accept `--config`. Prediction calls
`ShakeShakeNet.forward`, which already averages original and flipped logits.
Do not add `--horizontal-flip`; `classify` alone is single-view inference.

## Private materials and verification limits

Neither route includes weights, datasets or predictions in Git. Exact ASAM
re-export requires the authorized final EMA400 checkpoint; its expected digest
and input digests are in [ASAM reproduction](docs/REPRODUCIBILITY.md).
Shake-Shake needs its final1000 checkpoint for exact re-export, and its matching
45k source-run `best.pt`, `config.json`, `summary.json`, `split.json` and
`validation/metrics.json` for the guarded scratch refit. Request those materials
through authorized team/course channels; do not bypass identity checks.

Synthetic CPU tests check imports, parameter budgets, recovery and export logic.
They do not establish that fresh training reproduces either Public score or
checkpoint bytes. See [VALIDATION.md](VALIDATION.md) for the measured test scope.
The original frozen ASAM source/configuration and Shake-Shake core/configuration
are preserved. Existing strict-source training runs must use their recorded
source workspace to resume, rather than this merged root tree.

## Data use, sharing and attribution

The supplied course rules require a residual CNN with at most 5,000,000 trainable
parameters, trained from scratch using only the authorized training data.
Augmentation is allowed; distillation is allowed with scratch-trained teachers
using permitted data. The final model uses no teacher, external data or pretrained
weights. Hidden evaluation images are never used for training, normalization,
BatchNorm fitting or manual labeling.

The user-provided competition rules specify accuracy, a 50%/50% public/private
split, at most five submissions per registered team per day, and two final
submission slots. Registered-team sharing must follow the course rules. Public
competition code must also be linked through the competition forum or a notebook
available to participants, with appropriate acknowledgment. Publication of this
repository does not establish that this separate sharing step has occurred.

Data, label files, split ID payloads, image-hash manifests, checkpoints,
predictions, credentials and private execution records are excluded from public
source delivery. Share private inputs and weights only through authorized
team/course channels. Report deadlines, format and upload instructions must be
confirmed from the actual course rubric.

GPT/Codex assisted implementation, debugging, experiment work and documentation;
the team is responsible for reviewing and understanding the work. Preserve the
adapted AugMix component's [Apache-2.0 license](third_party/augmix/LICENSE) and
[notice](third_party/augmix/NOTICE), including embedded source headers. This does
not assign that license to the repository as a whole.
