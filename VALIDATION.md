# Scaffold validation — 2026-11-03

- Local environment: macOS, Python 3.9, PyTorch 2.8.0, torchvision 0.23.0.
- Four unittest cases passed. They cover all candidate models (forward and finite
  backward), rejection of an over-budget architecture, deterministic disjoint
  stratified splitting, rejection of malformed/incorrectly ordered submissions,
  training/checkpoint/prediction round-trip, and exact CPU model equality after
  a completed-epoch interruption and resumption. Synthetic images only.
- All 52,000 supplied PNGs were decoded and checked as 32×32 RGB; 50,000 training
  rows (5,000 per class) and 2,000 evaluation rows were found. Referenced files
  exist, filenames/IDs are unique, and sample/test ID ordering agrees.
- Parameter counts: baseline 1,084,586; wide 4,327,754; deep 4,180,858;
  smoke 78,042. All are below 5,000,000.
- A fixed 45,000/5,000 split and training-only normalization are prepared locally;
  the generated split is excluded from Git.

Not yet validated: formal CIFAR-10 training accuracy, CUDA/MPS training behavior,
HPC scheduling, GPU throughput or a Kaggle leaderboard submission. No results
from the synthetic tests should be reported as competition results.

## Team workflow extension — static checks only

Added `src.workflow`, `src.evaluate` and `TEAM_GUIDE.md`. Training now copies the
split JSON verbatim so its checkpoint identity remains valid even if the input
JSON uses different whitespace. Validation analysis checks the saved split hash
and exports full-split metrics, per-class metrics, predictions and confusion
matrices. The wrapper exports curves and an artifact manifest, with optional
explicit submission export.

The extension was checked by Python AST parsing, JSON parsing and `git diff
--check`. Per the user's request, no training, inference, data processing or
unittest execution was performed for this extension. The four historical test
passes above apply to the original scaffold, not to these new changes. New
regression assertions are authored but unexecuted; run the test suite and then
the smoke workflow before a formal experiment.
