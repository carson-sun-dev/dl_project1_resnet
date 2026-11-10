# ASAM training recipe

The final run is **Full50k ASAM400, seed 2026, fixed final EMA400**. The frozen
[configuration](../final_model/configs/final_refit_preact_asam400_s2026_cloud.json)
and [trainer](../final_model/src/final_refit.py) define the executable recipe.
Use the [reproduction guide](REPRODUCIBILITY.md) for commands and input identities.

## Data and preprocessing

Only the 50,000 supplied labeled CIFAR-10 images enter training. Earlier studies
used a stratified 45,000/5,000 split, seed 2026, with 500 validation images per
class. The final refit includes both subsets in training and has no validation
loader or validation-based checkpoint ranking. Its normalization is retained
from the original 45,000-image training subset:

```text
mean = [0.4915135944308404, 0.48231470971200857, 0.4468138379970015]
std  = [0.24705092701489773, 0.24347971893365983, 0.2615095063426547]
```

Class indices are `0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog,
6 frog, 7 horse, 8 ship, 9 truck`.

The training transform order is:

1. Decode RGB; random 32 x 32 crop with padding 4.
2. Horizontal flip with probability 0.5.
3. ColorJitter with brightness/contrast/saturation 0.2 and hue 0.02, applied with probability 0.8.
4. Random grayscale with probability 0.1.
5. Convert to a tensor in [0,1], then apply the fixed normalization.
6. Apply one 16-pixel Cutout square, probability 1, clipped at the boundary and filled with normalized zero.
7. Apply batch MixUp, alpha 0.2 and probability 1.

MixUp draws one Beta(0.2,0.2) coefficient per batch and a random batch permutation.
Inputs and one-hot labels use the same convex combination. The objective is mean
soft-target cross entropy with no label smoothing and no separate unmixed CE
term. MixUp training accuracy is expected correctness against mixed targets,
not clean-image validation accuracy.

## Optimization and EMA

| Setting | Final value |
| --- | --- |
| Epochs / seed / batch | 400 / 2026 / 128 |
| Base optimizer | SGD, momentum 0.9, no Nesterov |
| Weight decay | 0.0005 on all parameters |
| Learning rate | Five warmup epochs, then epochwise cosine to 0.001 |
| Warmup learning rates | 0.02, 0.04, 0.06, 0.08, 0.1 |
| ASAM | rho 0.5, eta 0.01 |
| EMA | FP32 shadows, nominal decay 0.999, update warmup |
| Deployment | Fixed EMA at epoch 400; copied online BN buffers |

SAM/ASAM compute a base gradient, perturb the parameters, compute a second
gradient at the perturbed point, restore the original parameters, and apply one
SGD update. Both passes reuse the identical augmented and mixed batch. The second
pass uses batch statistics while suppressing changes to BatchNorm running
buffers and counters. Momentum and weight decay apply once per optimizer update.

Ordinary SAM uses a global gradient-normalized perturbation; the completed
comparison uses rho 0.05. Final ASAM adapts parameters named `weight`, including
BatchNorm affine weights, by `abs(weight) + eta`. In the implementation, this
factor enters both the normalization gradient and the applied perturbation;
biases remain unadapted. Parameters are restored exactly before the base step.

At zero-based EMA update count `u`, decay is
`min(0.999, (u + 1) / (u + 10))`. Parameter shadows are averaged in FP32.
Nonparameter buffers, including BatchNorm statistics, are copied from the online
model rather than averaged. EMA does not feed back into optimization. There is
no BatchNorm recalibration, SWA transfer or validation-selected replacement of
the fixed deployment endpoint.

The saved run completed 400 epochs, **156,400 optimizer steps and EMA updates**,
with zero AMP-skipped updates. There are 391 batches per epoch. Two ASAM passes
produce 40,000,000 training model views over the complete run.

## Precision and inference

Training used native CUDA BF16 autocast with FP32 master weights, gradients and
loss calculations, no GradScaler, two data-loader workers and four CPU threads.
The saved backend policy is cuDNN deterministic enabled, benchmark disabled,
TF32 disabled, and highest float32 matmul precision. Deterministic settings do
not establish bitwise equality across different hardware/software.

The submitted prediction pipeline uses FP32 CPU inference, RGB decoding, tensor
conversion and the checkpoint's fixed normalization. The model is in evaluation
mode, with finite-logit checks and strict state loading. It uses one standard
view and one fixed model. There is no test-time augmentation, ensemble,
adaptation, test-derived normalization or BatchNorm fitting.

## Separate Shake-Shake training route

Shake-Shake uses its dedicated `src.shake_train` (45k selection) and
`src.shake_refit` (full50k) entry points, not the ASAM launcher or generic
`src.train`. Its documented full-data recipe is scratch1000, seed2026, batch128,
Nesterov SGD momentum0.9, weight decay0.0001, LR0.2 with five warmup epochs then
cosine to zero, crop/flip, CIFAR10 AutoAugment NEAREST and Cutout16. It exports
fixed final1000 with no EMA or independent validation. The
[teammate guide](../experiments/SHAKE899_REPRODUCTION.md) is authoritative for
its environment, required private source-run materials and inference policy.
