"""Independent multiscale preactivation residual CNN, trained from scratch.

Topology combines preactivation residual paths (He et al., 2016,
https://arxiv.org/abs/1603.05027), dilated convolutions (Yu and Koltun, 2016,
https://arxiv.org/abs/1511.07122), channel gating (Hu et al., 2018,
https://arxiv.org/abs/1709.01507), and fixed low-pass downsampling motivated by
Zhang (2019, https://arxiv.org/abs/1904.11486). This is a new controlled custom
architecture, not a reproduction of those papers' reported models or scores.

All code here was independently written for this project under its existing
source terms. The separable [1, 2, 1] binomial filter is implemented directly
from its mathematical coefficients; no Adobe or other third-party source code
or weights were copied. The filter is a nontrainable registered buffer.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


class BinomialDownsample(nn.Module):
    """Reflection-pad, depthwise fixed low-pass, then sample every other pixel."""

    def __init__(self):
        super().__init__()
        vector = torch.tensor([1.0, 2.0, 1.0])
        self.register_buffer("kernel", torch.outer(vector, vector)[None, None] / 16)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2] < 2 or x.shape[-1] < 2:
            raise ValueError("Downsampling requires NCHW input with both spatial dimensions >= 2")
        weight = self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 3, 3)
        # Exactly reflection-pad one pixel, expressed through slices/cat rather
        # than CUDA reflection_pad2d_backward (documented as nondeterministic).
        # This changes neither forward samples nor the fixed filter geometry.
        padded = torch.cat((x[..., 1:2], x, x[..., -2:-1]), dim=-1)
        padded = torch.cat((padded[..., 1:2, :], padded, padded[..., -2:-1, :]), dim=-2)
        return F.conv2d(padded, weight, stride=2, groups=x.shape[1])


class SqueezeExcitation(nn.Module):
    """One learned channel gate after concatenating the two residual branches."""

    def __init__(self, channels):
        super().__init__()
        _positive_int(channels, "channels")
        hidden = max(1, channels // 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Linear(channels, hidden)
        self.activation = nn.ReLU(inplace=False)
        self.expand = nn.Linear(hidden, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        gate = self.gate(self.expand(self.activation(self.reduce(self.pool(x).flatten(1)))))
        return x * gate[:, :, None, None]


class MultiScalePreActivationBlock(nn.Module):
    """Raw equal-shape identity; changing shapes project preactivated input.

    With antialiasing enabled, both residual and projection paths receive the
    same already-filtered tensor and all learned convolutions have stride one.
    Disabling the dilated branch makes its dilation one (parameter count stays
    identical). Disabling antialiasing uses ordinary strided first/projection
    convolutions. These switches exist for later explicitly controlled ablations.
    """

    def __init__(self, in_channels, out_channels, stride=1, *, antialias=True,
                 dilated_branch=True, squeeze_excitation=True):
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        if out_channels % 2:
            raise ValueError("out_channels must be even for equal branch widths")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride not in (1, 2):
            raise ValueError("stride must be one or two")
        for name, value in (("antialias", antialias), ("dilated_branch", dilated_branch),
                            ("squeeze_excitation", squeeze_excitation)):
            _boolean(value, name)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.downsample = BinomialDownsample() if antialias and stride == 2 else nn.Identity()
        convolution_stride = stride if not antialias else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=convolution_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.local_conv = nn.Conv2d(out_channels, out_channels // 2, 3, padding=1, bias=False)
        dilation = 2 if dilated_branch else 1
        self.context_conv = nn.Conv2d(out_channels, out_channels // 2, 3,
                                     padding=dilation, dilation=dilation, bias=False)
        self.se = SqueezeExcitation(out_channels) if squeeze_excitation else nn.Identity()
        self.use_identity = stride == 1 and in_channels == out_channels
        self.skip = nn.Identity() if self.use_identity else nn.Conv2d(
            in_channels, out_channels, 1, stride=convolution_stride, bias=False)

    def forward(self, x):
        preactivated = self.downsample(self.relu1(self.bn1(x)))
        residual = self.relu2(self.bn2(self.conv1(preactivated)))
        residual = self.se(torch.cat((self.local_conv(residual),
                                      self.context_conv(residual)), dim=1))
        shortcut = x if self.use_identity else self.skip(preactivated)
        return shortcut + residual


class MultiScalePreActivationResNet(nn.Module):
    """Three-stage residual CNN with frozen first-run widths56/112/224.

    Matches the project's fan-out Kaiming Conv initialization, BN weight=1,
    BN bias=0 and PyTorch default Linear initialization (including SE gates).
    """

    def __init__(self, channels=(56, 112, 224), blocks=(4, 4, 4), num_classes=10,
                 stem_channels=24, antialias=True, dilated_branch=True,
                 squeeze_excitation=True):
        super().__init__()
        if len(channels) != 3 or len(blocks) != 3:
            raise ValueError("Exactly three stage widths and block counts are required")
        for value in (*channels, *blocks, stem_channels):
            _positive_int(value, "stage/stem dimensions")
        if any(width % 2 for width in channels):
            raise ValueError("Every stage width must be even")
        if isinstance(num_classes, bool) or num_classes != 10:
            raise ValueError("This competition requires ten classes")
        for name, value in (("antialias", antialias), ("dilated_branch", dilated_branch),
                            ("squeeze_excitation", squeeze_excitation)):
            _boolean(value, name)
        self.stem = nn.Conv2d(3, stem_channels, 3, padding=1, bias=False)
        layers = []
        current = stem_channels
        for stage, (width, count) in enumerate(zip(channels, blocks)):
            for index in range(count):
                stride = 2 if stage > 0 and index == 0 else 1
                layers.append(MultiScalePreActivationBlock(
                    current, width, stride, antialias=antialias,
                    dilated_branch=dilated_branch, squeeze_excitation=squeeze_excitation))
                current = width
        self.blocks = nn.Sequential(*layers)
        self.final_bn = nn.BatchNorm2d(current)
        self.final_activation = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        features = self.final_activation(self.final_bn(self.blocks(self.stem(x))))
        return self.head(self.pool(features).flatten(1))
