# CIFAR-10 residual networks under 5 million parameters

Course competition project studying scratch-trained residual CNNs on the supplied
CIFAR-10 data. The final implementation in this repository is **PreAct WRN-22-4
with ASAM, trained on all 50,000 permitted images for 400 epochs, seed 2026, using
the fixed final EMA weights**. It has **4,298,970 trainable parameters**.

This model achieved **89.80% public accuracy** (Kaggle submission **58063369**).
The saved receipt and submitted CSV were reconciled byte for byte. Prediction
used one model, standard FP32 CPU inference, one view, and no TTA or ensemble.
The full-data model has **no independent local validation score**. Private scores
and current final submission selections are unknown.

A teammate separately reported **89.90%** for a full-data Shake-Shake model. That
implementation and result were not independently verified here and are not
provided by this repository. The 89.80% result identifies this repository's
verified final model; it is not a claim to the team's highest reported score.

## Code and documentation

| Location | Purpose |
| --- | --- |
| [final_model/](final_model/) | Exact final training source and configuration, with a portable command launcher and source verification |
| [Architecture](docs/ARCHITECTURE.md) | Blocks, dimensions, initialization and parameter accounting |
| [Training](docs/TRAINING.md) | Data, augmentation, optimizer, EMA and precision policy |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Input identities, setup, scratch training, recovery and CPU prediction |
| [Experiments](docs/EXPERIMENTS.md) | Twenty completed trajectories, partial runs, controls and unrun proposals |
| [Results](docs/RESULTS.md) | Public results, evidence strength and interpretation limits |
| [References](docs/REFERENCES.md) | Methods, implementation attribution and assistance disclosure |

Root `src/`, configurations and older experiment documents are retained as
historical study material. Use `final_model/` for the final ASAM result; root
modules with the same names are not interchangeable with the frozen final source.
The experiment inventory includes historical variants whose complete frozen
implementations are not all distributed in this public package.

## Reproduce the final pipeline

Run commands from the repository root in an isolated Python environment. Install
the pinned dependencies in [requirements.txt](requirements.txt), with the matching
CUDA build of PyTorch when training. See the reproduction guide for the recorded
environment and limits on exact numerical reproduction.

```bash
python -m pip install -r requirements.txt
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
