# Exact final ASAM implementation

This directory contains the exact 20 source modules and configuration used for
the verified 89.80% Public submission: PreAct WRN-22-4, 4,298,970 parameters,
50,000 permitted images, ASAM400, seed2026, fixed final EMA400, CPU FP32 single
view. This is not a private-leaderboard result or the team's highest reported
model. See the [main README](../README.md) and [reproduction guide](../docs/REPRODUCIBILITY.md).

Run from the **repository root**:

```bash
python final_model/run.py verify
python final_model/run.py model
python final_model/run.py data prepare
```

The data command uses only permitted training files. It deterministically creates
or verifies the original split and decoded-RGB image manifest, requiring their
recorded SHA256 values. It never replaces mismatching outputs.

Training and prediction are explicit commands, never automatic startup actions.
`run.py` activates this frozen source tree, checks its hashes and uses its fixed
configuration. Default prediction requires the exact private EMA400 checkpoint.
A separately requested scratch-reproduction checkpoint must pass the documented
strict identity/endpoint checks; its bytes and score are not promised to match.

`src/` and `configs/` remain unchanged historical bytes; portable safety and data
preparation additions live in `run.py`, `tools/` and `tests/`. Git attributes
preserve original source/configuration line endings. Retain `third_party/augmix/`
licenses and notices because the complete source identity includes that module.
No weights, datasets, hidden labels, predictions or cloud controls are bundled.
