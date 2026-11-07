# Results and limitations

The repository's strongest verified public result is **89.80%**: one scratch-trained
PreAct WRN-22-4, 4,298,970 trainable parameters, ASAM400 seed 2026, trained on all
50,000 permitted images and deployed using fixed final EMA400. Submission
**58063369** has a saved official score receipt and byte-for-byte reconciliation
of the submitted CSV.

## Full-data candidates

All four candidates below use the same architecture and parameter count. All
deploy fixed final EMA through standard FP32 CPU single-view inference, with no
TTA, ensemble or adaptation. None has an independent local validation score.

| Candidate | Seed / endpoint | Public accuracy | Evidence |
| --- | --- | ---: | --- |
| Full50k SAM400 | 2026 / EMA400 | 89.30% | Official saved receipt and submitted-byte match; ID 58046105 |
| **Full50k ASAM400** | **2026 / EMA400** | **89.80%** | **Official saved receipt and submitted-byte match; ID 58063369** |
| Full50k ASAM400 | 2027 / EMA400 | 89.00% | User-reported score; no independently matched submission ID |
| Full50k ASAM600 | 2026 / EMA600 | 88.30% | User-reported score; no independently matched submission ID |

Checkpoint/configuration and export identities were retained for these four
models. That provenance verifies which local candidate was exported; only the
first two rows also have independently reconciled upload receipts in the
inspected evidence. Private scores, final rank and current final submission
selections remain unknown.

A teammate separately reported **89.90%** for full50k Shake-Shake. That code and
score were not independently verified here and are not part of this repository's
implementation. The distinction matters when describing the team's overall
results: 89.80% is this repository's verified incumbent, not the team's highest
reported public score.

## Interpretation

The earlier 45k SAM result was reported as 88.40% public accuracy; the full-data
SAM refit reached 89.30%. Full-data ASAM400 seed 2026 reached 89.80%; the second
ASAM seed and separate 600-epoch trajectory did not improve that public outcome.
No supplied result reached 90%. These observations do not isolate the effect of
one method.

The 45k experiments often used best checkpoints selected on the reused holdout,
and some historical exports used horizontal-flip inference. The 50k refits use
fixed final EMA and standard inference. Training data size, endpoint policy,
initialization, hardware, augmentation randomness and recipe can differ.
The 600-epoch cosine schedule differs throughout training from the 400-epoch
schedule; it is not 200 extra epochs appended to the same trajectory. Two ASAM
seeds do not reliably estimate seed variance.

The complete [experiment inventory](EXPERIMENTS.md) preserves negative outcomes:
the frozen AugMix recipe, multiscale and representation controls, sampling
qualification failures, a rejected averaging/BN probe, and the completed KD
student. Config-only proposals and the paused 322/400 AA run are kept separate
from completed experiments.

## Validation and public-score gap

Selected original-split models reached approximately 97% local clean validation
accuracy, while public results were approximately 88-90%. The cause remains
unexplained. Saved permitted-data audits did not establish a supported inference
mismatch: they checked normalization, split membership, image decoding, and
consistency across evaluation/export paths. Distribution shift and optimism from
repeated holdout selection remain hypotheses, not demonstrated explanations.

The full-data refits include the former 5,000 validation images in training.
Their old validation scores cannot be inherited from the earlier weights or
presented as independent measurements. The public subset is also repeatedly
observed, and scores on it are correlated. Public differences alone establish
neither significance nor private-leaderboard ordering.

Local stress evaluations apply controlled transformations to the original
validation images. They are not external CIFAR-10-C data, official mCE results,
new independent examples, or evidence about the composition of hidden test
images. Paired image-level intervals condition on the fitted model and do not
capture training-seed uncertainty. Hidden labels were not inspected or inferred.
