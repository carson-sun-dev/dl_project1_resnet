"""CIFAR-sized residual networks, randomly initialized. See REFERENCES.md."""
import argparse

import torch
from torch import nn

from .utils import read_json

PARAMETER_LIMIT = 5_000_000


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.branch(x) + self.skip(x))


class CifarResNet(nn.Module):
    def __init__(self, channels=(32, 64, 128), blocks=(3, 3, 3), num_classes=10):
        super().__init__()
        if len(channels) != 3 or len(blocks) != 3 or any(x <= 0 for x in [*channels, *blocks]):
            raise ValueError("Three positive channel widths and block counts are required")
        if num_classes != 10:
            raise ValueError("This competition requires ten classes")
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers = []
        current = channels[0]
        for stage, (width, count) in enumerate(zip(channels, blocks)):
            for index in range(count):
                stride = 2 if stage > 0 and index == 0 else 1
                layers.append(BasicBlock(current, width, stride))
                current = width
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(current, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


class PreActivationBlock(nn.Module):
    """WRN preactivation unit with an untouched equal-shape identity path.

    Dimension-changing shortcuts project BN/ReLU-preactivated input, following
    the authors' topology: https://github.com/szagoruyko/wide-residual-networks/
    blob/master/models/wide-resnet.lua . No shortcut BN or post-add ReLU is used.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if any(isinstance(c, bool) or not isinstance(c, int) or c <= 0
               for c in (in_channels, out_channels)):
            raise ValueError("Block channel widths must be positive integers")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride not in (1, 2):
            raise ValueError("Block stride must be 1 or 2")
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.use_identity = stride == 1 and in_channels == out_channels
        self.skip = (nn.Identity() if self.use_identity else
                     nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False))

    def forward(self, x):
        preactivated = self.relu1(self.bn1(x))
        residual = self.conv1(preactivated)
        residual = self.conv2(self.relu2(self.bn2(residual)))
        shortcut = self.skip(x if self.use_identity else preactivated)
        return shortcut + residual


class PreActivationWideResNet(nn.Module):
    """Scratch CIFAR WRN topology; defaults implement WRN-22-4 without dropout.

    References: He et al., https://arxiv.org/abs/1603.05027 ; Zagoruyko and
    Komodakis, https://arxiv.org/abs/1605.07146 . This controlled comparison keeps
    the project's fan_out convolution/BN initialization and default Linear
    initialization, rather than the authors' different initialization choices.
    """
    def __init__(self, channels=(64, 128, 256), blocks=(3, 3, 3), num_classes=10,
                 stem_channels=16, dropout=0.0):
        super().__init__()
        if (len(channels) != 3 or len(blocks) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                       for v in [*channels, *blocks, stem_channels])):
            raise ValueError("Three positive integer widths/counts and a positive integer stem are required")
        if num_classes != 10:
            raise ValueError("This competition requires ten classes")
        if isinstance(dropout, bool) or dropout != 0:
            raise ValueError("This controlled preactivation experiment requires dropout=0")
        self.stem = nn.Conv2d(3, stem_channels, 3, padding=1, bias=False)
        layers = []
        current = stem_channels
        for stage, (width, count) in enumerate(zip(channels, blocks)):
            for index in range(count):
                stride = 2 if stage > 0 and index == 0 else 1
                layers.append(PreActivationBlock(current, width, stride))
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
        features = self.blocks(self.stem(x))
        features = self.final_activation(self.final_bn(features))
        return self.head(self.pool(features).flatten(1))


def parameter_counts(model):
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in model.parameters())}


def build_model(config):
    options = dict(config)
    architecture = options.pop("architecture", "cifar_resnet")
    if architecture == "cifar_resnet":
        model = CifarResNet(**options)
    elif architecture == "preactivation_wrn":
        model = PreActivationWideResNet(**options)
    else:
        raise ValueError(f"Unknown model architecture: {architecture!r}")
    counts = parameter_counts(model)
    if counts["trainable"] > PARAMETER_LIMIT:
        raise ValueError(f"Parameter budget exceeded: {counts['trainable']:,} > {PARAMETER_LIMIT:,}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    model = build_model(read_json(args.config)["model"])
    model.eval()
    with torch.no_grad():
        shape = list(model(torch.zeros(2, 3, 32, 32)).shape)
    print({**parameter_counts(model), "output_shape": shape})
