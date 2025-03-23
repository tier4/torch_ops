import random
import warnings
from typing import Any, Sequence, cast

import numpy as np
import torch

from tops.utils.dist_utils import rank

_amp_enabled: bool = False
_seed: int = 0


def set_AMP(value: bool) -> None:
    global _amp_enabled
    _amp_enabled = value


def AMP() -> bool:
    return _amp_enabled


def _to_cuda(element: torch.Tensor) -> torch.Tensor:
    return element.to(get_device(), non_blocking=True)


def to_cuda(
    elements: Sequence[torch.Tensor] | dict[Any, torch.Tensor] | torch.Tensor
) -> Sequence[torch.Tensor] | dict[Any, torch.Tensor] | torch.Tensor:
    if isinstance(elements, Sequence):
        return [cast(torch.Tensor, _to_cuda(x)) for x in elements]
    if isinstance(elements, dict):
        return {k: cast(torch.Tensor, _to_cuda(v)) for k, v in elements.items()}
    return cast(torch.Tensor, _to_cuda(elements))


def to_cpu(
    elements: Sequence[torch.Tensor] | dict[Any, torch.Tensor] | torch.Tensor
) -> Sequence[torch.Tensor] | dict[Any, torch.Tensor] | torch.Tensor:
    if isinstance(elements, Sequence):
        return [cast(torch.Tensor, to_cpu(x)) for x in elements]
    if isinstance(elements, dict):
        return {k: cast(torch.Tensor, to_cpu(v)) for k, v in elements.items()}
    return elements.cpu()


def get_device() -> torch.device:
    return (
        torch.device(f"cuda:{rank()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def set_seed(seed: int) -> None:
    global _seed
    _seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_seed() -> int:
    return _seed


def set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


try:
    symbolic_assert = torch._assert  # 1.8
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0


class suppress_tracer_warnings(warnings.catch_warnings):
    """
    Context manager to suppress known warnings in torch.jit.trace().

    This is particularly useful when working with PyTorch's JIT tracer which may raise
    TracerWarning for certain operations. The warnings are suppressed within the context
    manager's scope.

    Example:
        >>> with suppress_tracer_warnings():
        ...     traced_model = torch.jit.trace(model, example_input)
    """

    def __enter__(self) -> "suppress_tracer_warnings":
        """
        Enter the context manager and set up warning suppression.

        Returns:
            The suppress_tracer_warnings instance itself
        """
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self
