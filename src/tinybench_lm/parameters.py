"""Unique trainable-parameter enumeration shared by the counter, tests, and verifiers.

The competition cap counts parameters, not state-dict entries, and the input embedding
and output head are one shared ``Parameter`` (Plan Sections 3.1-3.3). Enumerating unique
``Parameter`` objects keeps the counter, the step-zero provenance record, and the release
verifier in exact agreement instead of letting each recount its own way.
"""

from __future__ import annotations

from torch import nn


def unique_trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Enumerate each trainable Parameter object once, even when it has aliases."""
    unique: dict[int, nn.Parameter] = {}
    for module in model.modules():
        for parameter in module._parameters.values():
            if parameter is not None and parameter.requires_grad:
                unique.setdefault(id(parameter), parameter)
    return tuple(unique.values())


def count_unique_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in unique_trainable_parameters(model))
