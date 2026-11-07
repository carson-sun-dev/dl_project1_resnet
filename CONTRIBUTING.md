# Team workflow

The repository owner has chosen a public GitHub repository. Invite only registered
teammates as collaborators. Do not publish competition data, credentials, weights
or prediction files. The supplied Kaggle foundational rules require publicly
shared competition code to also be shared through the competition discussion
forum/notebooks for all competitors. Coordinate that public link with course
staff if necessary; this scaffold does not post messages on the team's behalf.

1. Pull `main` before starting a change. Create a descriptive branch (Codex-created
   branches use `codex/`, e.g. `codex/mixup-experiment`).
2. Use one branch/PR per coherent model or training change. Keep data reading,
   class mapping and validation IDs consistent across experiments.
3. New experiments get a new JSON config and unique `run_dir`. Do not silently
   edit completed experiment configs, overwrite predictions or share a live run
   directory between training processes.
4. Run `python -m unittest discover -s tests -v` and the parameter-count check.
   Ask a teammate to review the PR before merging.
   Changes involving the final model also require
   `python final_model/run.py verify`, `python final_model/run.py model`, and
   `python -m unittest discover -s final_model/tests -v` on CPU. Preserve the
   frozen `final_model/src/` and configuration byte identities; add portable
   preparation/entry-point tooling outside that frozen source tree.
5. Add a measured row to `experiments/results.csv`, including the Git commit and
   seed. Accuracy values are fractions (0 to 1); leave unknown fields empty.
6. Retain private checkpoints and logs outside Git. Copy artifacts between
   registered teammates using an agreed private channel. Every checkpoint must
   remain associated with its configuration, source revision and split hash.

## Suggested responsibilities

- Architecture: residual blocks, stage widths/depths and parameter budget.
- Training: optimizer, schedules, augmentation and runtime measurements.
- Integration (third member if available): data/prediction checks and experiment
  bookkeeping. With two members, share this work.

## Kaggle coordination

The team shares five submissions per day. Designate one person to upload each
candidate, record its artifact hash and description, and communicate before using
an attempt. The provided screenshot says up to two submissions can be selected
for final scoring; verify the current competition selection setting before close.
Never manually label or look up hidden evaluation images. This repository does
not submit to Kaggle automatically.
