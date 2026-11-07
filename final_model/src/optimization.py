"""Opt-in, two-pass sharpness-aware updates and independent FP32 EMA weights.

SAM: Foret et al., https://arxiv.org/abs/2010.01412 .
ASAM: Kwon et al., https://arxiv.org/abs/2102.11600 .
This implementation follows the published equations; it does not copy a research
repository implementation. ASAM uses elementwise L2 adaptation for parameters
named ``weight`` (including BN affine weights), with biases unadapted.
"""
from contextlib import contextmanager, nullcontext
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


OPTIMIZATION_VERSION = 1
EMA_VERSION = 1


def _real(value, name, *, minimum=0.0, maximum=None, upper_open=False):
    valid = (isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value) and value >= minimum)
    if maximum is not None:
        valid = valid and (value < maximum if upper_open else value <= maximum)
    if not valid:
        raise ValueError(f"{name} must be a finite scalar in its permitted range")
    return float(value)


def validate_optimization(training):
    """Return canonical opt-in settings; absent settings preserve the legacy path."""
    if "optimization" not in training:
        return None
    raw = training["optimization"]
    allowed = {"schema_version", "method", "rho", "eta", "ema"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("optimization must contain only schema_version/method/rho/eta/ema")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != OPTIMIZATION_VERSION:
        raise ValueError("Unsupported optimization schema_version")
    method = raw.get("method")
    if method not in ("sgd", "sam", "asam"):
        raise ValueError("optimization method must be sgd, sam or asam")
    if training.get("optimizer", "sgd") != "sgd":
        raise ValueError("Opt-in optimization requires the SGD base optimizer")
    rho = _real(raw.get("rho", {"sgd": 0.0, "sam": 0.05, "asam": 0.5}[method]), "rho")
    eta = _real(raw.get("eta", 0.01 if method == "asam" else 0.0), "eta")
    if method == "sgd" and (rho != 0 or eta != 0):
        raise ValueError("SGD requires rho=0 and eta=0")
    if method == "sam" and eta != 0:
        raise ValueError("SAM requires eta=0; adaptive scaling belongs to ASAM")
    ema = raw.get("ema", {"enabled": False, "decay": 0.999, "warmup": True})
    if not isinstance(ema, dict) or set(ema) - {"enabled", "decay", "warmup"}:
        raise ValueError("ema must contain only enabled/decay/warmup")
    enabled, warmup = ema.get("enabled", False), ema.get("warmup", True)
    if type(enabled) is not bool or type(warmup) is not bool:
        raise ValueError("EMA enabled and warmup must be booleans")
    decay = _real(ema.get("decay", 0.999), "EMA decay", maximum=1.0, upper_open=True)
    return {"schema_version": OPTIMIZATION_VERSION, "method": method, "rho": rho, "eta": eta,
            "ema": {"enabled": enabled, "decay": decay, "warmup": warmup}}


def _finite(tensors, description):
    values = list(tensors)
    if values and not torch.stack([torch.isfinite(value).all() for value in values]).all().item():
        raise RuntimeError(f"Non-finite {description}; optimizer update aborted")


def _norm(tensors, description):
    values = list(tensors)
    if not values:
        raise RuntimeError(f"No {description} available")
    result = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(value.detach().float()) for value in values]))
    if not torch.isfinite(result).item():
        raise RuntimeError(f"Non-finite {description} norm; optimizer update aborted")
    return result


def _gradients(named_parameters):
    active = [(name, parameter) for name, parameter in named_parameters if parameter.grad is not None]
    if not active:
        raise RuntimeError("No training gradients; optimizer update aborted")
    if any(parameter.grad.dtype != torch.float32 for _, parameter in active):
        raise RuntimeError("Optimization requires FP32 master gradients")
    _finite((parameter.grad for _, parameter in active), "training gradients")
    norm = _norm((parameter.grad for _, parameter in active), "training gradient")
    return active, norm


def _loss(logits, targets, smoothing):
    _finite([logits], "training logits")
    if targets.ndim == 2:
        if targets.shape != logits.shape:
            raise ValueError("Soft targets must match the logits shape")
        _finite([targets], "soft targets")
        if (targets < 0).any().item() or not torch.allclose(
                targets.float().sum(1), torch.ones(len(targets), device=targets.device),
                rtol=0, atol=1e-5):
            raise ValueError("Soft targets must be nonnegative probability distributions")
        softened = targets.float() * (1 - smoothing) + smoothing / logits.shape[1]
        loss = -(softened * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
    else:
        loss = F.cross_entropy(logits.float(), targets, label_smoothing=smoothing)
    _finite([loss], "training loss")
    return loss


@contextmanager
def _second_pass_batchnorm(model):
    """Use current batch statistics without changing running stats or counters."""
    modules = [(module, module.track_running_stats) for module in model.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    try:
        for module, _ in modules:
            module.track_running_stats = False
        yield
    finally:
        for module, original in modules:
            module.track_running_stats = original


def optimization_step(model, optimizer, images, targets, settings, dtype=None, label_smoothing=0):
    """Perform one atomic-boundary SGD/SAM/ASAM update on an already mixed batch.

    SAM/ASAM restore exact original parameter snapshots before the SGD update,
    including when the second forward/backward raises. A failed update raises;
    callers must not publish a partial-epoch checkpoint. No EMA or RNG sampling
    occurs here. The optimizer owns momentum/weight decay, applied exactly once.
    """
    settings = validate_optimization({"optimization": settings})
    if settings is None:
        raise ValueError("Explicit optimization settings are required")
    smoothing = _real(label_smoothing, "label_smoothing", maximum=1.0, upper_open=True)
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("optimization_step requires torch.optim.SGD")
    if dtype not in (None, torch.bfloat16):
        raise ValueError("Opt-in optimization supports BF16 CUDA or FP32, not FP16")
    if dtype is not None and images.device.type != "cuda":
        raise ValueError("BF16 optimization forward requires CUDA")
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                        if parameter.requires_grad]
    if not named_parameters or any(parameter.dtype != torch.float32 for _, parameter in named_parameters):
        raise ValueError("Optimization requires trainable FP32 master parameters")
    _finite((parameter for _, parameter in named_parameters), "model parameters")
    _finite([images], "training inputs")
    context = lambda: (torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext())
    optimizer.zero_grad(set_to_none=True)
    method = settings["method"]
    try:
        with context():
            logits = model(images)
        first_loss = _loss(logits, targets, smoothing)
        first_loss.backward()
        active, first_norm = _gradients(named_parameters)
        result = {"logits": logits.detach(), "base_loss": first_loss.item(),
                  "perturbed_loss": None, "base_gradient_norm": first_norm.item(),
                  "perturbed_gradient_norm": None, "perturbation_norm": 0.0,
                  "adaptive_perturbation_norm": 0.0, "gradient_checks": 1, "optimizer_steps": 1}
        if method != "sgd":
            scales = [(parameter.detach().abs() + settings["eta"]
                       if method == "asam" and name.rsplit(".", 1)[-1] == "weight"
                       else torch.ones_like(parameter)) for name, parameter in active]
            scaled = [parameter.grad.detach().float() * scale
                      for (_, parameter), scale in zip(active, scales)]
            denominator = _norm(scaled, "sharpness gradient")
            ratio = settings["rho"] / denominator if denominator.item() != 0 else 0.0
            perturbations = [gradient * scale * ratio for gradient, scale in zip(scaled, scales)]
            _finite(perturbations, "sharpness perturbations")
            result["perturbation_norm"] = _norm(perturbations, "perturbation").item()
            normalized = [torch.where(scale != 0, perturbation / scale, 0.0)
                          for perturbation, scale in zip(perturbations, scales)]
            _finite(normalized, "normalized perturbations")
            result["adaptive_perturbation_norm"] = _norm(normalized, "normalized perturbation").item()
            original = [parameter.detach().clone() for _, parameter in active]
            try:
                with torch.no_grad():
                    for (_, parameter), perturbation in zip(active, perturbations):
                        parameter.add_(perturbation)
                _finite((parameter for _, parameter in active), "perturbed model parameters")
                optimizer.zero_grad(set_to_none=True)
                with _second_pass_batchnorm(model), context():
                    second_logits = model(images)
                second_loss = _loss(second_logits, targets, smoothing)
                second_loss.backward()
                _, second_norm = _gradients(named_parameters)
                result.update(perturbed_loss=second_loss.item(),
                              perturbed_gradient_norm=second_norm.item(), gradient_checks=2)
            finally:
                with torch.no_grad():
                    for (_, parameter), saved in zip(active, original):
                        parameter.copy_(saved)
        optimizer.step()
        _finite((parameter for _, parameter in named_parameters), "updated model parameters")
        for state in optimizer.state.values():
            _finite((value for value in state.values() if isinstance(value, torch.Tensor)),
                    "optimizer state")
        return result
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise


class EMA:
    """Separate FP32 parameter average with copied, never averaged, buffers."""
    def __init__(self, model, decay=0.999, warmup=True):
        self.decay = _real(decay, "EMA decay", maximum=1.0, upper_open=True)
        if type(warmup) is not bool:
            raise ValueError("EMA warmup must be a boolean")
        self.warmup, self.updates = warmup, 0
        self.parameter_names = tuple(name for name, _ in model.named_parameters())
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("EMA requires FP32 model parameters")
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}
        _finite(self.shadow.values(), "initial EMA state")

    def _validate_model(self, model):
        state = model.state_dict()
        if tuple(name for name, _ in model.named_parameters()) != self.parameter_names:
            raise ValueError("EMA model parameter identity differs")
        if state.keys() != self.shadow.keys() or any(
                value.shape != self.shadow[name].shape or value.dtype != self.shadow[name].dtype
                for name, value in state.items()):
            raise ValueError("EMA model state identity differs")
        _finite(state.values(), "model state for EMA")
        return state

    @torch.no_grad()
    def update(self, model):
        state = self._validate_model(model)
        decay = min(self.decay, (self.updates + 1) / (self.updates + 10)) if self.warmup else self.decay
        for name, value in state.items():
            if name in self.parameter_names:
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                self.shadow[name].copy_(value.detach())
        self.updates += 1
        _finite(self.shadow.values(), "EMA state")

    def state_dict(self):
        return {"schema_version": EMA_VERSION, "decay": self.decay, "warmup": self.warmup,
                "updates": self.updates, "parameter_names": self.parameter_names,
                "shadow": self.model_state_dict()}

    def load_state_dict(self, state):
        required = {"schema_version", "decay", "warmup", "updates", "parameter_names", "shadow"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Invalid EMA checkpoint state")
        if (type(state["schema_version"]) is not int or state["schema_version"] != EMA_VERSION
                or state["decay"] != self.decay or state["warmup"] is not self.warmup
                or tuple(state["parameter_names"]) != self.parameter_names):
            raise ValueError("EMA checkpoint configuration/identity differs")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid EMA update count")
        incoming = state["shadow"]
        if not isinstance(incoming, dict) or incoming.keys() != self.shadow.keys() or any(
                not isinstance(value, torch.Tensor) or value.shape != self.shadow[name].shape
                or value.dtype != self.shadow[name].dtype for name, value in incoming.items()):
            raise ValueError("EMA checkpoint tensor identity differs")
        _finite(incoming.values(), "EMA checkpoint tensors")
        self.shadow = {name: value.detach().to(self.shadow[name].device).clone()
                       for name, value in incoming.items()}
        self.updates = state["updates"]

    @torch.no_grad()
    def copy_to(self, model):
        self._validate_model(model)
        model.load_state_dict(self.model_state_dict(), strict=True)

    def model_state_dict(self, model=None):
        if model is not None:
            self._validate_model(model)
        return {name: value.detach().clone() for name, value in self.shadow.items()}
