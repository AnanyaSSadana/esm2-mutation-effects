"""Minimal LoRA layers for the fair-esm ESM-2 implementation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from torch import nn

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to a frozen linear layer.

    The update is ``scaling * B(A(dropout(x)))``. ``B`` starts at zero, so
    replacing a linear layer does not change the model's initial output.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False, **factory_kwargs)

        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self):
        """Expose the frozen base weight for fair-esm compatibility."""

        return self.base.weight

    @property
    def bias(self):
        """Expose the frozen base bias for fair-esm compatibility."""

        return self.base.bias

    def forward(self, inputs):
        base_output = self.base(inputs)
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return base_output + update * self.scaling


@dataclass(frozen=True)
class LoRAInjection:
    replaced_modules: tuple[str, ...]
    trainable_parameters: int


def _module_matches(name: str, targets: tuple[str, ...]) -> bool:
    leaf_name = name.rsplit(".", 1)[-1]
    return leaf_name in targets


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> LoRAInjection:
    """Freeze ``model`` and replace selected linear modules with LoRA wrappers."""

    targets = tuple(target_modules)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("LoRA target module names must be unique and non-empty")

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name and _module_matches(name, targets)
    ]
    if not candidates:
        raise ValueError(f"No linear modules matched LoRA targets {targets}")

    for name, module in candidates:
        if isinstance(module, LoRALinear):
            raise TypeError(f"Module {name} already has LoRA")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target {name} is {type(module).__name__}, not Linear")
    matched_targets = {name.rsplit(".", 1)[-1] for name, _ in candidates}
    missing = sorted(set(targets).difference(matched_targets))
    if missing:
        raise ValueError(f"LoRA targets were not found: {missing}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced: list[str] = []
    for name, module in candidates:
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)

        # fair-esm's optimized attention path reads projection weights directly,
        # which would bypass the LoRA update. Force the ordinary module-call path.
        if child_name in {"q_proj", "k_proj", "v_proj", "out_proj"} and hasattr(
            parent, "enable_torch_version"
        ):
            parent.enable_torch_version = False

    return LoRAInjection(
        replaced_modules=tuple(replaced),
        trainable_parameters=count_trainable_parameters(model),
    )


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def lora_parameters(module: nn.Module):
    """Yield only adapter parameters, excluding any separately trainable head."""

    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield from child.lora_a.parameters()
            yield from child.lora_b.parameters()
