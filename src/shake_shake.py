"""CIFAR Shake-Shake residual network; no external weights or data.

Independent PyTorch reimplementation of Xavier Gastaldi's BSD-licensed
CIFAR-10 architecture:
https://github.com/xgastaldi/shake-shake/tree/master/models
In particular, the projection uses two spatially offset sampling paths.
``classify`` is always one view. In eval mode ``forward`` averages original
and horizontal-flip logits, matching this repository's strong-model convention.
Validation reporting standard accuracy MUST therefore call ``classify``.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ShakeShakeFunction(torch.autograd.Function):
    """Mix with alpha in forward, independently with beta in backward.

This intentionally is not the derivative of the forward mixture. Only elementwise
operations are used. Explicit coefficient casts preserve activation/gradient
dtype under AMP, without promoting half/bfloat16 feature maps to float32.
Alpha and beta are stochastic constants, not learnable inputs.
"""

    @staticmethod
    def forward(ctx, first, second, alpha, beta):
        alpha = alpha.to(device=first.device, dtype=first.dtype)
        beta = beta.to(device=first.device, dtype=first.dtype)
        ctx.save_for_backward(beta)
        return first * alpha + second * (1 - alpha)

    @staticmethod
    def backward(ctx, grad_output):
        (beta,) = ctx.saved_tensors
        beta = beta.to(dtype=grad_output.dtype)
        return grad_output * beta, grad_output * (1 - beta), None, None


class ShiftedProjection(nn.Module):
    """Author's type-B skip: ReLU, two 1x1 sampled paths, concatenate, BN."""

    def __init__(self, inputs, width, stride):
        super().__init__()
        self.stride = stride
        self.path1 = nn.Conv2d(inputs, width // 2, 1, bias=False)
        self.path2 = nn.Conv2d(inputs, width // 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(width)

    def forward(self, x):
        x = F.relu(x)
        first = F.avg_pool2d(x, 1, self.stride)
        # Lua SpatialZeroPadding(1, -1, 1, -1): shift right/down one pixel.
        shifted = F.pad(x, (1, 0, 1, 0))[:, :, :-1, :-1]
        second = F.avg_pool2d(shifted, 1, self.stride)
        return self.bn(torch.cat((self.path1(first), self.path2(second)), dim=1))


class ShakeShakeBlock(nn.Module):
    def __init__(self, inputs, width, stride=1):
        super().__init__()

        def branch():
            return nn.Sequential(
                nn.ReLU(), nn.Conv2d(inputs, width, 3, stride, 1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
            )

        self.branch1 = branch()
        self.branch2 = branch()
        self.skip = (nn.Identity() if inputs == width and stride == 1
                     else ShiftedProjection(inputs, width, stride))

    def forward(self, x):
        first, second = self.branch1(x), self.branch2(x)
        if self.training:
            # Independent draws for every image AND block, with new backward
            # coefficients even when the forward mixture happens to be similar.
            shape = (len(x), 1, 1, 1)
            alpha = torch.rand(shape, device=x.device, dtype=torch.float32)
            beta = torch.rand(shape, device=x.device, dtype=torch.float32)
            residual = ShakeShakeFunction.apply(first, second, alpha, beta)
        else:
            residual = (first + second) * 0.5
        return self.skip(x) + residual


class ShakeShakeNet(nn.Module):
    """Default 26-layer 2x40d model has 4,561,354 trainable parameters."""

    inference_description = "single Shake-Shake residual model, mean original and horizontal-flip logits"

    def __init__(self, width=40, blocks=(4, 4, 4), num_classes=10):
        super().__init__()
        if (not isinstance(blocks, (list, tuple)) or len(blocks) != 3
                or any(type(count) is not int or count < 1 for count in blocks)
                or type(width) is not int or width < 2 or width % 2
                or type(num_classes) is not int or num_classes < 1):
            raise ValueError("Require three positive integer block counts, positive even width, positive num_classes")
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16))
        layers = []
        inputs = 16
        for stage, stage_width in enumerate((width, 2 * width, 4 * width)):
            for index in range(blocks[stage]):
                layers.append(ShakeShakeBlock(inputs, stage_width, 2 if stage and index == 0 else 1))
                inputs = stage_width
        self.stages = nn.Sequential(*layers)
        self.classifier = nn.Linear(inputs, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.classifier.bias)
        self.to(memory_format=torch.channels_last)

    def classify(self, x):
        """Return single-view logits; train mode retains stochastic shaking."""
        features = self.stages(self.stem(x.contiguous(memory_format=torch.channels_last)))
        return self.classifier(F.relu(features).mean((2, 3)))

    def forward(self, x):
        logits = self.classify(x)
        return logits if self.training else (logits + self.classify(x.flip(-1))) * 0.5
