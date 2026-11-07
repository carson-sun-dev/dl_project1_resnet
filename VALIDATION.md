# Validation and verification

See [docs/RESULTS.md](docs/RESULTS.md) and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for measured results.
Per-epoch BF16 validation, separately evaluated FP32 checkpoints, optional
historical flip TTA, and Kaggle Public Accuracy are different protocols.
The full-50k final refits have **no independent validation score**.

The exact final implementation and source/configuration byte hashes are in
[`final_model/`](final_model/). Root `src/` and `tests/` cover the earlier local
research implementation. Frozen final-model source/configuration files are
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
