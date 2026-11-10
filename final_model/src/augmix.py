# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Three-view AugMix, adapted for the project's data and precision policies.

Operator parameterization and convex mixtures are adapted from Google Research's
Apache-2.0 implementation (2019):
https://github.com/google-research/augmix/blob/master/augmentations.py
https://github.com/google-research/augmix/blob/master/cifar.py
Hendrycks et al., "AugMix: A Simple Data Processing Method to Improve Robustness
and Uncertainty", ICLR 2020, https://arxiv.org/abs/1912.02781.

Project modifications: a shared crop/flip base, existing normalization, explicit
configuration validation, and log-space FP32 Jensen-Shannon calculations. No
dataset download, external benchmark, or author model code is included.
"""
import math

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision import transforms

from .utils import validate_normalization

AUGMIX_VERSION = "google_default_pool_v1_fp32_js"
OPERATOR_NAMES = ("autocontrast", "equalize", "posterize", "rotate", "solarize",
                  "shear_x", "shear_y", "translate_x", "translate_y")
_SETTING_KEYS = {"width", "depth", "severity", "mixture_concentration",
                 "js_coefficient", "all_ops"}


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def validate_augmix(training):
    """Opt-in validation; legacy training mappings are accepted unchanged."""
    settings = training.get("augmix")
    if settings is None:
        return False
    if not isinstance(settings, dict) or set(settings) != _SETTING_KEYS:
        raise ValueError("augmix must explicitly specify width, depth, severity, "
                         "mixture_concentration, js_coefficient and all_ops")
    width, depth = settings["width"], settings["depth"]
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 10:
        raise ValueError("AugMix width must be an integer in [1, 10]")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (-1, 1, 2, 3):
        raise ValueError("AugMix depth must be -1 (random 1-3) or an integer in [1, 3]")
    severity, concentration = settings["severity"], settings["mixture_concentration"]
    if not _finite_number(severity) or not 0.1 <= severity <= 10:
        raise ValueError("AugMix severity must be finite and in [0.1, 10]")
    if not _finite_number(concentration) or concentration <= 0:
        raise ValueError("AugMix mixture_concentration must be finite and positive")
    coefficient = settings["js_coefficient"]
    if not _finite_number(coefficient) or coefficient < 0:
        raise ValueError("AugMix js_coefficient must be finite and nonnegative")
    if settings["all_ops"] is not False:
        raise ValueError("Only the original default AugMix pool (all_ops=false) is supported")
    # Reject simultaneous recipes instead of silently ignoring regularization.
    if training.get("mixup_alpha", 0) != 0 or training.get("cutout_length", 0) != 0:
        raise ValueError("AugMix replaces MixUp and Cutout; disable their strengths")
    jitter = training.get("color_jitter") or {}
    if (not isinstance(jitter, dict) or any(jitter.get(key, 0) != 0
             for key in ("brightness", "contrast", "saturation", "hue"))):
        raise ValueError("AugMix replaces ColorJitter; disable its strengths")
    if training.get("random_grayscale_probability", 0) != 0:
        raise ValueError("AugMix replaces RandomGrayscale; disable its probability")
    if training.get("label_smoothing", 0) != 0:
        raise ValueError("The controlled AugMix recipe uses unsmoothed hard labels")
    return True


def _sample_level(severity):
    return np.random.uniform(0.1, severity)


def _int_parameter(level, maximum):
    return int(level * maximum / 10)


def _float_parameter(level, maximum):
    return float(level) * maximum / 10


def _signed(value):
    return -value if np.random.uniform() > 0.5 else value


def apply_operator(image, name, severity):
    """Original RGB CIFAR operators: PIL parameters and bilinear interpolation."""
    if image.mode != "RGB" or image.size != (32, 32):
        raise ValueError("AugMix requires RGB 32x32 images")
    if not _finite_number(severity) or not 0.1 <= severity <= 10:
        raise ValueError("AugMix severity must be finite and in [0.1, 10]")
    if name == "autocontrast":
        return ImageOps.autocontrast(image)
    if name == "equalize":
        return ImageOps.equalize(image)
    if name == "posterize":
        bits = 4 - _int_parameter(_sample_level(severity), 4)
        return ImageOps.posterize(image, bits)
    if name == "rotate":
        degrees = _signed(_int_parameter(_sample_level(severity), 30))
        return image.rotate(degrees, resample=Image.Resampling.BILINEAR)
    if name == "solarize":
        threshold = 256 - _int_parameter(_sample_level(severity), 256)
        return ImageOps.solarize(image, threshold)
    if name in ("shear_x", "shear_y"):
        level = _signed(_float_parameter(_sample_level(severity), 0.3))
        affine = (1, level, 0, 0, 1, 0) if name == "shear_x" else (1, 0, 0, level, 1, 0)
    elif name in ("translate_x", "translate_y"):
        level = _signed(_int_parameter(_sample_level(severity), 32 / 3))
        affine = (1, 0, level, 0, 1, 0) if name == "translate_x" else (1, 0, 0, 0, 1, level)
    else:
        raise ValueError(f"Unsupported AugMix operator: {name}")
    return image.transform((32, 32), Image.Transform.AFFINE, affine,
                           resample=Image.Resampling.BILINEAR)


class AugMixTransform:
    """One shared geometric base and two independently mixed views, same label.

    Crop/flip use the checkpointed Torch RNG; chain selection, PIL parameters and
    convex weights use checkpointed NumPy RNG. DataLoader workers seed both.
    """
    def __init__(self, mean, std, settings):
        validate_augmix({"augmix": settings})
        validate_normalization({"mean": mean, "std": std})
        self.settings = dict(settings)
        self.base_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def augment(self, base):
        """Mix only transformed copies of this base, in float32 pixel space."""
        settings = self.settings
        concentration = settings["mixture_concentration"]
        weights = np.float32(np.random.dirichlet([concentration] * settings["width"]))
        coefficient = np.float32(np.random.beta(concentration, concentration))
        original = self.to_tensor(base)
        mixture = torch.zeros_like(original)
        for weight in weights:
            chain = base.copy()
            depth = settings["depth"] if settings["depth"] > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                name = np.random.choice(OPERATOR_NAMES)
                chain = apply_operator(chain, name, settings["severity"])
            mixture.add_(self.to_tensor(chain), alpha=float(weight))
        return (1 - float(coefficient)) * original + float(coefficient) * mixture

    def __call__(self, image):
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("AugMix requires RGB 32x32 images")
        base = self.base_transform(image)
        return (self.normalize(self.to_tensor(base)),
                self.normalize(self.augment(base)),
                self.normalize(self.augment(base)))


def augmix_loss(logits_base, logits_aug1, logits_aug2, labels, coefficient=12.0):
    """Base CE + coefficient * mean KL(view || mean), entirely FP32.

    Log-space mixture avoids arbitrary probability clipping. All views and the
    mixture retain gradients; there is no teacher or detached target branch.
    """
    views = (logits_base, logits_aug1, logits_aug2)
    if (logits_base.ndim != 2 or logits_base.shape[0] == 0 or logits_base.shape[1] < 2
            or any(view.shape != logits_base.shape for view in views)):
        raise ValueError("AugMix logits must be three matching nonempty [batch, classes] tensors")
    if labels.ndim != 1 or labels.shape[0] != logits_base.shape[0] or labels.dtype != torch.long:
        raise ValueError("AugMix labels must be integer class indices matching the batch")
    if not _finite_number(coefficient) or coefficient < 0:
        raise ValueError("AugMix JS coefficient must be finite and nonnegative")
    if not torch.stack([torch.isfinite(view).all() for view in views]).all():
        raise RuntimeError("Non-finite AugMix logits")
    # Explicit FP32 inputs/dtype keep probability and log reductions FP32 even
    # when this helper is called within an autocast-enabled training context.
    logs = torch.stack([F.log_softmax(view.float(), dim=1, dtype=torch.float32) for view in views])
    log_mean = torch.logsumexp(logs, dim=0) - math.log(3)
    js = (logs.exp() * (logs - log_mean)).sum(dim=2).mean()
    ce = F.cross_entropy(logits_base.float(), labels)
    total = ce + coefficient * js
    if not torch.isfinite(torch.stack((ce, js, total))).all():
        raise RuntimeError("Non-finite AugMix loss")
    return total, ce, js
