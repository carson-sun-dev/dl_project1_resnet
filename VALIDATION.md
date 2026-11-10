# Validation and verification

See [docs/RESULTS.md](docs/RESULTS.md) and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for measured results.
Per-epoch BF16 validation, separately evaluated FP32 checkpoints, optional
historical flip TTA, and Kaggle Public Accuracy are different protocols.
The full-50k final refits have **no independent validation score**.

The exact final implementation and source/configuration byte hashes are in
[`final_model/`](final_model/). Root `src/` and `tests/` cover Shake-Shake and the
earlier local research implementation. Frozen final-model source/configuration files are
protected from Git line-ending conversion.

Cheap CPU-only verification, from the repository root:

```bash
python final_model/run.py verify
python final_model/run.py model
python -m unittest discover -s final_model/tests -v
python -m unittest discover -s tests -v
```

These checks use synthetic inputs and local source identities. The training-data
preparation tool does not require hidden evaluation images. It must reproduce the
expected split and image-manifest hashes, rather than bypassing mismatches. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Publication checks on November 6, 2026:

- Earlier local-study suite: **106 tests passed**, CPU-only.
- Final-package focused suite: **32 tests passed**, CPU-only, including SAM/ASAM
  equations, BatchNorm handling, exact parameter restoration, EMA, synthetic
  checkpoint recovery, source/config drift and safe export metadata checks.
- Instantiated final model: **4,298,970 trainable/total parameters**; a tiny
  synthetic CPU batch produced finite logits with shape `[2, 10]`.
- All **20 frozen source modules** and the original configuration matched their
  recorded SHA256 identities.
- The portable generator reproduced the exact image-manifest SHA256 from all
  **50,000 permitted training images**. Independent split/normalization recreation
  reproduced the exact original 45k/5k split SHA256. No hidden images were read.

Retraining, full inference and cloud access are not publication checks.
Checkpoints, raw metrics, historical logs and private verification receipts stay
outside Git. No cross-hardware bitwise retraining guarantee is made.

## Local integration checks: November 9, 2026

Latest fetched main `ae526c8` (Shake-Shake PR #2) was combined with ASAM commit
`d1e3bb9` in an isolated review worktree. See
[integration review](docs/INTEGRATION_REVIEW.md) for scope and pending publication.

- Root suite: **123 tests, 122 passed, 1 skipped**, 344.333 seconds.
- Final ASAM suite: **32 passed**, 2.233 seconds.
- The skipped native CPU BF16 backward check reports the actual oneDNN error:
  `DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`.
  The test only skips this specific unsupported backend; other errors fail.
  FP32 Shake-Shake backward and coefficient-dtype checks passed. CUDA AMP was
  not tested, and no GPU was used.
- The first combined run exposed a stale legacy-model test assumption that
  Shake-Shake configs were implicit original ResNets. Its scope now excludes
  named PreAct/Shake-Shake routes; separate integration tests exercise those
  routes, parameter budgets and strict synthetic checkpoint round trips.
- ASAM: **4,298,970** parameters; Shake-Shake: **4,561,354**. CPU model commands
  returned finite `[2, 10]` outputs. Both counts include trainable classifier/BN.
- All **20 frozen ASAM modules and its configuration** matched their recorded
  SHA256 values. The entire final package matches the ASAM branch's Git blobs.
  Protected Shake-Shake core/trainers/configs and the full-data evaluation guard
  match latest main. Document links and staged whitespace checks passed.
- Shake-Shake training/refit/prediction CLI imports passed. Synthetic export
  tests preserve sample ordering, full-data metadata and built-in flip-logit
  inference, and reject an additional probability-mean flip request.

The existing environment has FileLock **4.2.4**, rather than the declared
**3.20.0** pin. It also uses the ASAM NumPy/Pillow versions; a fresh pinned
Shake-Shake environment was not installed or tested. These are integration
checks, not fresh 400/1000-epoch reproduction or competition-score verification.
No course datasets or previously trained experiment checkpoints were loaded,
and no large files were downloaded. Checkpoint tests used synthetic states only.
