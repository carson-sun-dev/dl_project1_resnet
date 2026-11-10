# Experiment inventory

This inventory summarizes saved results through the final project handoff.
The ASAM contributor inventory covers twenty formal training trajectories that
completed their planned horizons; the separate teammate Shake-Shake route is
listed below and is not counted among those twenty. Partial
pilots, repeated evaluations, recovery checkpoints and copied configurations are
not additional completed experiments. Complete per-epoch records were checked
against the comparison inventory; they are not all public release contents.

All accuracy values below are percentages. **S** means standard single-view FP32
evaluation; **F** means separate horizontal-flip FP32 evaluation. **Raw** is the
online model and **EMA** is its moving average. `best` denotes a checkpoint
selected using the reused validation set; `final` denotes the fixed endpoint.
Architecture codes and parameter counts are defined in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Completed trajectories

Unless stated otherwise, the seed is 2026 and the split is 45k training / 5k
validation. A dash means the result is unavailable or the model was not submitted;
it never means zero accuracy. Historical public scores are reported associations
unless an exact submission-byte receipt is identified below.

| # | Experiment / architecture | Epochs | Endpoint and clean validation | Public accuracy / status |
| ---: | --- | ---: | --- | --- |
| 1 | Regularized Wide / CW | 200 | best Raw188: 96.64 S / 96.68 F | 86.80 reported; upload-byte binding unavailable |
| 2 | Color Robustness Wide / CW | 200 | best Raw188: 96.64 S / 97.02 F | 87.80 reported; upload-byte binding unavailable |
| 3 | PreAct WRN-22-4 / P22 | 200 | best Raw196: 96.88 S / 97.14 F | 88.10 reported; upload-byte binding unavailable |
| 4 | AugMix + JS / CW | 200 | best Raw193: 95.44 S / 95.90 F | No verified assigned submission |
| 5 | SGD400 / P22 | 400 | best Raw398: 96.80 S / 97.22 F | 88.40; historically assigned ID 57996263 |
| 6 | SAM400 / P22 | 400 | best EMA384: 97.50 S / 97.50 F | 88.40; historically assigned ID 58005472 |
| 7 | ASAM400 / P22 | 400 | best Raw388: 97.46 S / 97.58 F | 88.20 reported, standard inference; exact ID binding unavailable |
| 8 | RegMixup400 / P22 | 400 | best EMA394: 97.34 S / 97.54 F | 88.30 reported, standard inference; exact ID binding unavailable |
| 9 | ResNeXt-29-4x24d / RX | 400 | best EMA396: 96.92 S / 97.16 F | No established submission |
| 10 | Multiscale no-AA control / MS | 400 | final Raw: 96.30 S; EMA: 96.54 S | Failed clean export screen |
| 11 | Two-view CE + SupCon / MS + head | 400 | final Raw: 96.44 S; EMA: 96.58 S | Failed clean export screen |
| 12 | Matched two-view CE / MS + head | 400 | final Raw: 96.60 S; EMA: 96.60 S | Failed clean export screen |
| 13 | AA + mild sampling, fresh initialization / MS | 400 | final Raw: 96.10 S; EMA: 96.40 S; best EMA396: 96.70 S | Robustness gains with clean deficit; not submitted |
| 14 | Full50k SAM400 / P22 | 400 | 50k / no holdout; fixed final EMA400 | **89.30**, verified bytes; ID 58046105 |
| 15 | WRN-22 SAM + mild sampling / P22 | 400 | fixed final EMA400: 97.10 S | Failed qualification for full-data refit |
| 16 | WRN-16-5 SAM / P16 | 400 | fixed final EMA400: 97.18 S | Failed qualification for full-data refit |
| 17 | Full50k ASAM400 / P22 | 400 | 50k / no holdout; fixed final EMA400 | **89.80**, verified bytes; ID 58063369 |
| 18 | Full50k ASAM400, seed 2027 / P22 | 400 | 50k / no holdout; fixed final EMA400 | **89.00**, user-reported; ID unverified |
| 19 | Full50k ASAM600 / P22 | 600 | 50k / no holdout; fixed final EMA600 | **88.30**, user-reported; ID unverified |
| 20 | Scratch student KD400 / P22 | 400 | final Raw: 97.10 S; EMA: 96.98 S | Failed qualification; no full-data refit or submission |

The early historical SAM export used flip inference after a documented tie
override, although standard and flip validation accuracy tied. This is distinct
from the full-data SAM/ASAM submissions, which all used standard FP32 CPU
inference without TTA or ensembles. The public-score column does not assign every
historical result to the table's standard or flip metric without evidence.

Per-epoch validation used BF16 AMP for most historical trajectories. The two
representation studies and KD used strict FP32 validation. These logged curves
are distinct from the selected-checkpoint FP32 values above: for example,
SGD Raw398 logged 96.88% BF16 but measured 96.80% standard FP32; SAM EMA384 logged
97.48% BF16 and measured 97.50% standard FP32. Full50k histories contain no
validation series. Missing full-data validation fields must not be backfilled
from earlier 45k-trained models.

## Partial trajectories and probes

| Study | Actual / planned | Outcome |
| --- | ---: | --- |
| Local ASAM pilot / P22 | 3 / 400 | Epoch 3 BF16 Raw/EMA 64.28/66.28; preserved pilot, not resumed into the fresh completed ASAM trajectory |
| SGD and SAM throughput/recovery benchmarks / P22 | 10 / 400 each | Separate pilots; no final competition claim |
| Local AA multiscale + SE / MS | **322 / 400** | Paused; last BF16 Raw/EMA 94.64/95.92; no standardized final FP32 report or verified cross-platform continuation |
| ASAM duration pilot / P22 | 3 / 600 | Epoch 3 Raw/EMA diagnostics 58.40/68.14; stopped after duration forecast; its conditional 50k branch unrun |
| Same-trajectory averaging + BN recalibration / P22 | No new optimizer epochs | Averaged raw epochs 300/325/350/375/400: 96.22 S versus native EMA400 97.34 S; rejected before full-data transfer |
| Earlier BN recalibration controls | No new optimizer epochs | Flip accuracy fell to 95.62 for Color, 95.94 for PreAct and 94.48 for AugMix; original weights preserved |

The completed full50k ASAM600 run is a separate scratch trajectory from the
three-epoch 45k duration pilot. Restarts and endpoint backups within a trajectory
do not create new seeds or models.

## External baseline and unexecuted preparations

A saved official record labeled **Baseline** scored 83.50% (ID 57962336). Its
actual architecture, checkpoint, uploaded bytes and local training history are
not reconstructable here. The root baseline configuration has 1,084,586
parameters; that fact does not bind it to the external submission.

| Preparation | Established status |
| --- | --- |
| Original baseline / unregularized Wide / Deep | Configurations only; no reconstructable formal training records; configured counts 1,084,586 / 4,327,754 / 4,180,858 |
| Smoke | Correctness template, 78,042 parameters, two limited-batch epochs; no saved formal competition run |
| CutMix package | Prepared/tested package; completed training not established |
| AutoAugment branch | Scaffolding/preservation only; completed training not established |
| WRN-22 sampling and WRN-16 conditional full-data SAM refits | Neither launched after qualification failed |
| KD conditional full-data student | Not launched after qualification failed |
| Weight-averaging full-data transfer | Not applied after clean validation rejection |
| Duration pilot's conditional full-data 600-epoch branch | Unrun; separate full50k ASAM600 completed |

Complete historical frozen implementations for RegMixup, ResNeXt, representation
learning and KD are not all included in the published final-model package.
Their recorded results are preserved here as experiment evidence, not as a claim
that `final_model/` can reproduce every table row.

## What the studies support

Color augmentation improved measured grayscale/appearance robustness; the frozen
AugMix recipe underperformed. At matched epoch-400 FP32 endpoints, SAM exceeded
SGD by 0.40 percentage points for Raw and 0.44 for EMA. Two-view SupCon gave no
clean improvement over its matched CE control. Sampling improved selected stress
diagnostics while losing clean accuracy. KD and weight averaging did not pass
their frozen promotion gates.

These are limited, mostly single-seed comparisons on a repeatedly used holdout.
Best-checkpoint results condition on selection. Recipe, schedule, data size,
precision and inference differences prevent treating this inventory as a uniform
causal ranking. See [RESULTS.md](RESULTS.md) for public-score interpretation.

## Teammate Shake-Shake route

The separately integrated [teammate report](../experiments/SHAOHAN_OPTIMIZATION_REPORT.md)
and [reproduction guide](../experiments/SHAKE899_REPRODUCTION.md) document the
50k scratch1000, seed2026, Shake-Shake-26 2x40d result: 4,561,354 trainable
parameters, fixed final checkpoint, 89.90% teammate-confirmed Public accuracy.
FP32 export averages original and mirrored logits; there is no independent
validation score for this full-data model. This is a separate contribution, not
one of the twenty trajectories above. Submission ID, Private score and an
independent upload-byte reconciliation are not established here. Its historical
side experiments are described by the teammate, not reconstructed by this inventory.
