"""Training-only regularization; randomness uses the checkpointed torch RNG."""
import math

import torch
from torch.nn import functional as F


def validate_color_augmentation(training):
    jitter = training.get("color_jitter")
    if jitter is not None:
        allowed = {"brightness", "contrast", "saturation", "hue", "probability"}
        if not isinstance(jitter, dict) or set(jitter) - allowed:
            raise ValueError("color_jitter must be a mapping of brightness/contrast/saturation/hue/probability")
        for key in ("brightness", "contrast", "saturation", "hue"):
            value = jitter.get(key, 0.0)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (key == "hue" and value > 0.5)):
                raise ValueError(f"Invalid color_jitter {key}: use finite nonnegative scalars; hue <= 0.5")
        probability = jitter.get("probability", 1.0)
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("color_jitter probability must be finite and in [0, 1]")
    probability = training.get("random_grayscale_probability", 0.0)
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("random_grayscale_probability must be finite and in [0, 1]")


def validate_augmentation(training):
    validate_color_augmentation(training)
    alpha = training.get("mixup_alpha", 0.0)
    length = training.get("cutout_length", 0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("mixup_alpha must be finite and nonnegative")
    if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= 32:
        raise ValueError("cutout_length must be an integer in [0, 32]")
    for name in ("mixup_probability", "cutout_probability"):
        value = training.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")


class Cutout:
    """Mask one clipped square after normalization; zero is the training mean."""
    def __init__(self, length=0, probability=1.0):
        validate_augmentation({"cutout_length": length, "cutout_probability": probability})
        self.length, self.probability = length, probability

    def __call__(self, image):
        if not self.length or not self.probability:
            return image
        if self.probability < 1 and torch.rand(()).item() >= self.probability:
            return image
        height, width = image.shape[-2:]
        cy, cx = torch.randint(height, ()).item(), torch.randint(width, ()).item()
        y0, x0 = cy - self.length // 2, cx - self.length // 2
        result = image.clone()
        result[:, max(0, y0):min(height, y0 + self.length),
               max(0, x0):min(width, x0 + self.length)] = 0
        return result


def mixup_batch(images, labels, alpha=0.0, probability=1.0, num_classes=10):
    """One Beta(alpha, alpha) draw per batch; disabled paths consume no RNG."""
    if alpha == 0 or probability == 0 or len(labels) < 2:
        return images, labels
    if probability < 1 and torch.rand(()).item() >= probability:
        return images, labels
    # Sample on CPU so this also works on MPS; both CPU and device RNG are saved.
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    permutation = torch.randperm(len(labels), device=labels.device)
    targets = F.one_hot(labels, num_classes).float()
    return (lam * images + (1 - lam) * images[permutation],
            lam * targets + (1 - lam) * targets[permutation])


def soft_target_cross_entropy(logits, targets, label_smoothing=0.0):
    """CE from logits, with uniform smoothing applied exactly once in FP32."""
    targets = targets.float() * (1 - label_smoothing) + label_smoothing / logits.shape[1]
    return -(targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()
