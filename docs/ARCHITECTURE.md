# ASAM model architecture

The final classifier is a custom **preactivation Wide ResNet, WRN-22-4**, trained
from scratch. The frozen model configuration is in
[final_model/configs/final_refit_preact_asam400_s2026_cloud.json](../final_model/configs/final_refit_preact_asam400_s2026_cloud.json),
and its implementation is in [final_model/src/models.py](../final_model/src/models.py).

| Component | Output channels | Spatial size | Structure |
| --- | ---: | ---: | --- |
| RGB input | 3 | 32 x 32 | Fixed train-derived normalization |
| Stem | 16 | 32 x 32 | Bias-free 3 x 3 convolution, stride 1 |
| Stage 1 | 64 | 32 x 32 | Three preactivation residual blocks |
| Stage 2 | 128 | 16 x 16 | Three blocks; first block downsamples |
| Stage 3 | 256 | 8 x 8 | Three blocks; first block downsamples |
| Final activation | 256 | 8 x 8 | BatchNorm, ReLU |
| Pooling | 256 | 1 x 1 | Adaptive global average pooling |
| Classifier | 10 | — | Biased 256-to-10 linear layer |

Each residual branch contains BN/ReLU before each of two 3 x 3 convolutions. An
equal-shape shortcut carries the original input unchanged. A dimension-changing
shortcut uses a 1 x 1 projection on the preactivated input. Stage transitions use
stride 2. There is no post-addition ReLU and dropout is zero. The model returns
ten logits.

Convolutions use Kaiming-normal initialization with `fan_out`; BatchNorm scale
and bias begin at one and zero. The linear layer keeps PyTorch's default
initialization. All learned weights start from scratch.

## Parameter budget

Both total learned parameters and trainable parameters equal **4,298,970**,
leaving 701,030 below the 5,000,000 limit. Counting includes convolution weights,
projection shortcuts, BatchNorm affine parameters, and classifier weights/bias.
BatchNorm running statistics and counters are buffers. Optimizer momentum and
EMA shadows are training state, not additional trainable classifier components.
The final model has no teacher or auxiliary head.

```bash
python final_model/run.py model
```

This command instantiates the model, counts parameters and performs a small
synthetic forward pass. It does not measure classification accuracy.

## Other studied architectures

These names identify the [historical experiment inventory](EXPERIMENTS.md).
They do not imply that every historical frozen implementation is included in the
final-model package.

| Code | Architecture | Trainable parameters |
| --- | --- | ---: |
| CW | Custom postactivation Wide; widths 64/128/256, three blocks per stage | 4,327,754 |
| P22 | Final preactivation WRN-22-4; stem 16, widths 64/128/256, three blocks per stage | 4,298,970 |
| P16 | Preactivation WRN-16-5; widths 80/160/320, two blocks per stage | 4,289,754 |
| RX | ResNeXt-29, cardinality 4, base width 24 | 4,845,002 |
| MS | Custom preactivation multiscale/dilated/SE model; stem 24, widths 56/112/224, four blocks per stage | 4,519,554 |
| MS + head | MS with the removable two-view projection head | 4,598,754 training; 4,519,554 deployment |

The MS anti-aliasing filter is fixed and nontrainable. Its no-AA control retains
the multiscale and squeeze-and-excitation mechanisms. The earlier CW model is a
custom residual network, not an exact reproduction of a published WRN protocol.

Architecture and optimization references are listed in [REFERENCES.md](REFERENCES.md).

## Shake-Shake route

The separate [Shake-Shake-26 2x40d implementation](../src/shake_shake.py) has
**4,561,354 trainable parameters**: stem16, stages40/80/160 with four two-branch
residual blocks each, and a ten-class head. Training uses independent forward
and backward mixing coefficients per image/block, and shifted two-path projection
shortcuts. In evaluation mode `forward` averages original/flip logits; `classify`
is single-view. See its [configuration](../configs/shake_full1000_s2026.json) and
[reproduction guide](../experiments/SHAKE899_REPRODUCTION.md). It is not the
frozen ASAM implementation and must not be loaded through the ASAM launcher.
