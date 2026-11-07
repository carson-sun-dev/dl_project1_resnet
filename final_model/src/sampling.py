"""Train-only sampling perturbations, independently implemented using Pillow.

This is our controlled augmentation proposal, not a reproduction of a published
algorithm. It tests a weakness in permitted validation images, without making
any claim about the hidden evaluation distribution. All draws use torch RNG.
"""
import math

from PIL import Image, ImageFilter
import torch


def validate_sampling(settings):
    if settings is None:
        return None
    allowed = {"version", "probability", "resize_probability", "resize_sizes", "blur_radius"}
    if (not isinstance(settings, dict) or set(settings) != allowed
            or type(settings["version"]) is not int or settings["version"] != 1):
        raise ValueError("Invalid sampling augmentation schema")
    for key in ("probability", "resize_probability"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Sampling probabilities must be finite in [0,1]")
    sizes = settings["resize_sizes"]
    if not isinstance(sizes, list) or not sizes or any(type(size) is not int or not 16 <= size <= 32 for size in sizes):
        raise ValueError("Sampling sizes must be integer dimensions in [16,32]")
    radii = settings["blur_radius"]
    if (not isinstance(radii, list) or len(radii) != 2 or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in radii)
            or not 0 <= radii[0] <= radii[1] <= 1):
        raise ValueError("Sampling blur radii must be finite ordered values in [0,1]")
    return settings


class SamplingAugmentation:
    def __init__(self, settings):
        validate_sampling(settings)
        self.probability = settings["probability"]
        self.resize_probability = settings["resize_probability"]
        self.sizes = tuple(settings["resize_sizes"])
        self.radii = tuple(settings["blur_radius"])

    def __call__(self, image):
        if image.mode != "RGB" or image.size != (32, 32):
            raise ValueError("Sampling augmentation expects 32x32 RGB")
        if self.probability == 0 or torch.rand(()).item() >= self.probability:
            return image
        if torch.rand(()).item() < self.resize_probability:
            size = self.sizes[torch.randint(len(self.sizes), ()).item()]
            return image.resize((size, size), Image.Resampling.BILINEAR).resize((32, 32), Image.Resampling.BILINEAR)
        radius = self.radii[0] + torch.rand(()).item() * (self.radii[1] - self.radii[0])
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
