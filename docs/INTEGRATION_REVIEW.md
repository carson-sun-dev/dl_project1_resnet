# ASAM / Shake-Shake integration review

Integration reviewed on November 9, 2026. The preserved local merge candidate
was subsequently authorized for publication by the repository user. Remote
publication must be verified against GitHub's main ref, not inferred from this note.

## Branches and scope

- Latest fetched main: `ae526c8e5dbbb4be3672fa20b2905e461e2e1312`, including
  the teammate's Shake-Shake PR #2.
- ASAM source branch: `codex/final-asam-project1`, commit
  `d1e3bb93b02686bc87ce639ac882f7e44a3e81e2`.
- Local review branch: `codex/integrate-asam-shake-main`, based on latest main.
- Method: preserve the prepared three-way merge (`--no-commit --no-ff`) and
  finalize it as a merge commit retaining both parents. Publication uses a normal
  fast-forward push to main; a protected branch requires a review branch/PR
  instead. No force-push, history rewrite or course submission is authorized.
  The original workspace and untracked report files remain separate and unchanged.

The ASAM branch adds its frozen `final_model/` package, earlier research modules,
configurations, tests and evidence-grounded documentation. Main contributes the
Shake-Shake core, trainers, full-data refit, pixel cache, configuration snapshots,
tests and teammate report. Both routes are retained.

## Conflict resolutions

1. **README:** replace obsolete scaffold-only and absent-Shake-Shake statements
   with both routes, parameter counts, actual entry points and private materials.
2. **Root model factory:** retain legacy ResNet and PreAct dispatch and add the
   existing Shake-Shake dispatch; retain the shared five-million parameter guard.
3. **Root trainer:** retain the ASAM branch's optimization, EMA, strict recovery,
   portable Windows/POSIX `run_lock`, and backend-state restoration. Shake-Shake
   continues to use its dedicated trainers; its refit still uses FileLock.
4. **Root predictor:** retain optional historical validated flip TTA and the
   main branch's full-data metadata and model-specific inference description.
   Reject additional probability-mean TTA for Shake-Shake because its eval
   forward already averages original/flip logits.

No changes were needed to the frozen ASAM package, Shake-Shake core/trainers,
pixel cache, either Shake-Shake configuration, its optimization report, or the
main branch's full-data validation prohibition. Historical strict-source
checkpoints must resume in their recorded source environment; the merged root
workspace is not silently substituted for that identity.

## Verification scope

Both instantiated deployment architectures remain under the parameter limit:
ASAM PreAct WRN-22-4 has **4,298,970**, and Shake-Shake-26 2x40d has
**4,561,354** trainable parameters. Counting includes the classifier and BN
affine parameters. Synthetic CPU forwards produce finite `[2, 10]` logits.

The ASAM source verifier checks all 20 frozen modules and the configuration
against their recorded SHA256 values. Those files are byte-identical to the
ASAM branch. Protected Shake-Shake files match main's Git blobs; normal Git
CRLF conversion for the requirements text is not a training-source change.

Tests cover synthetic checkpoint round trips, uninterrupted/resumed CPU runs,
split and precision/source rejection, SAM/ASAM restoration, EMA, Shake-Shake
branches, full-data validation rejection, prediction ID ordering and the
additional-flip guard. See [VALIDATION.md](../VALIDATION.md) for results.

The final root run completed **123 tests: 122 passed, 1 skipped**; the frozen
ASAM package completed **32 tests, all passed**. The root skip is the explicit
native CPU oneDNN BF16-backward limitation (`avx2_vnni_2`), not a skipped model
shape/gradient failure. FP32 model backward checks passed. A legacy test that
enumerated every config as an original ResNet was scoped to the appropriate
architecture family; the separate integration tests cover all new routes.

No real training, hidden-data inference, large downloads, GPU use or cloud work
was performed. Tests ran in the existing Python 3.11 environment, with torch
2.8.0+cu128, torchvision 0.23.0+cpu, NumPy 2.0.2, Pillow 11.3.0, matplotlib
3.9.4 and FileLock 4.2.4. The installed FileLock differs from the declared
3.20.0 pin; a fresh pinned Shake-Shake environment and CUDA execution were not
retested. Neither Public score nor cross-hardware bitwise reproduction is
established by synthetic tests.

## Outstanding team/course materials

- Provide authorized ASAM final EMA400 weights for exact re-export and official
  data/split materials; expected hashes are in the ASAM reproduction guide.
- Provide Shake-Shake final1000 weights and its guarded 45k source-run
  `best.pt`, `config.json`, `summary.json`, `split.json` and
  `validation/metrics.json`. Scratch mode requires provenance records even
  though it does not initialize from the old weights.
- Confirm the Shake-Shake submission ID and receipt; its 89.90% remains
  teammate-confirmed. ASAM 89.80% has the saved reconciled receipt, ID 58063369.
- Confirm course treatment of flip inference, final selections/private results,
  report deadline/format/upload location, and the required public competition
  code link. Do not infer those facts from local tests.
- Commit/publication of this reviewed integration is user-authorized. Keep
  data, checkpoints, credentials, local reports and historical execution records
  outside the commit. Confirm the published main commit before directing teachers
  to the integrated tree; branch-protection requirements must not be bypassed.
