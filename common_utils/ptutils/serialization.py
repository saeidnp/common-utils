import torch
from contextlib import contextmanager


@contextmanager
def legacy_torch_load():
    """
    Context manager that restores the legacy `torch.load` default of
    `weights_only=False`.

    Starting with PyTorch 2.6, `torch.load` defaults to `weights_only=True`,
    which can break loading of checkpoints that contain pickled objects (e.g.
    configs or optimizer state). Within this context, any `torch.load` call
    that does not explicitly set `weights_only` is patched to use
    `weights_only=False`, mirroring the pre-2.6 behaviour. The original
    `torch.load` is restored on exit.

    Only load checkpoints from trusted sources, since `weights_only=False`
    allows arbitrary code execution during unpickling.

    Example:
        >>> with legacy_torch_load():
        ...     data = torch.load("checkpoint.pth")
        >>> # torch.load is restored to its original behaviour here
    """

    _real_torch_load = torch.load

    def tload(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _real_torch_load(*args, **kwargs)

    torch.load = tload
    try:
        yield
    finally:
        torch.load = _real_torch_load
