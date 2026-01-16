import itertools
import os
from functools import wraps
from filelock import FileLock
from pathlib import Path

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # You could assign a placeholder to torch here if some functions
    # might try to access torch.something without checking _TORCH_AVAILABLE first,
    # though it's better to check _TORCH_AVAILABLE.
    # For example:
    # class TorchPlaceholder:
    #     def __getattr__(self, name):
    #         raise ImportError("torch is not available, but was accessed.")
    # torch = TorchPlaceholder()


class ProtectFile(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        name = path.name if path.name.startswith(".") else f".{path.name}"
        lock_path = Path(path).parent / f"{name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)

def infinite_loader(dataloader):
    return itertools.cycle(dataloader)

def num_available_cores(cap=None):
    """
    Return the number of CPU cores available to the current job/process.

    Priority:
    1. sched_getaffinity (respects cgroups / Slurm / taskset)
    2. Slurm environment variables
    3. os.cpu_count(), capped at `cap`

    Parameters
    ----------
    cap : int
        Maximum number of cores to return when falling back to os.cpu_count()
        If None, no cap is applied.

    Returns
    -------
    int
        Number of usable CPU cores
    """

    # 1. Linux affinity (most accurate)
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, NotImplementedError):
        pass

    # 2. Slurm environment variables (ordered by reliability)
    for var in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_CPUS_ON_NODE",
    ):
        val = os.environ.get(var)
        if val:
            try:
                # SLURM_JOB_CPUS_PER_NODE can be like "8(x2)"
                return int(val.split("(")[0])
            except ValueError:
                pass

    # 3. Fallback: os.cpu_count() with cap
    cpu_count = os.cpu_count() or 1
    if cap is not None:
        cpu_count = min(cpu_count, cap)
    return cpu_count


def splitit(total_size, split_size):
    """Splits total_size into chunks of maximum size split_size and yeilds the chunk sizes.
    It is guaranteed that the sum of the returned chunks is equal to total_size.
    """
    assert total_size >= 0, f"total_size must be non-negative, got {total_size}"
    assert split_size > 0, f"split_size must be positive, got {split_size}"
    for i in range(0, total_size, split_size):
        yield min(total_size - i, split_size)


def get_register_fn(_CLASSES):
    def register_fn(cls=None, *, name=None):
        """A decorator for registering classes."""

        def _register(cls):
            if name is None:
                local_name = cls.__name__
            else:
                local_name = name
            if local_name in _CLASSES:
                raise ValueError(f"Already registered class with name: {local_name}")
            _CLASSES[local_name] = cls
            return cls

        if cls is None:
            return _register
        else:
            return _register(cls)

    return register_fn


def _batchify_helper(max_batch_size, concat_fn, batch_keys=None):
    """
    Decorator function that batches the output of another function based on the given parameters.
    The decorated function should take an integer `n` as its first argument and return either a dictionary or an array/tensor.

    Args:
        max_batch_size (int or None): The maximum batch size. If `None`, no batching will be performed.
        concat_fn (callable): A function used to concatenate the batched samples.
        dim (int, optional): The dimension along which to concatenate the batched samples. Defaults to 0.
        batch_keys (list or None, optional): A list of keys to batch. It should be `None` iff the decorated function returns an array/tensor.

    Returns:
        callable: The decorated function.

    """
    def decorator(func):
        @wraps(func)
        def wrapper(n, *args, **kwargs):
            if max_batch_size is None or n <= max_batch_size:
                return func(n, *args, **kwargs)
            if batch_keys is None:
                samples = []
                offset = 0
                for batch_size in splitit(n, max_batch_size):
                    kwargs_with_offset = kwargs.copy()
                    kwargs_with_offset['offset'] = offset
                    samples.append(func(batch_size, *args, **kwargs_with_offset))
                    offset += batch_size
                return concat_fn(samples)
            else:
                batched_dict = {k: [] for k in batch_keys}
                non_batched_dict = {}
                is_first_batch = True
                offset = 0
                for batch_size in splitit(n, max_batch_size):
                    kwargs_with_offset = kwargs.copy()
                    kwargs_with_offset['offset'] = offset
                    batch = func(batch_size, *args, **kwargs_with_offset)
                    for k in batch_keys:
                        batched_dict[k].append(batch[k])
                    if is_first_batch:
                        non_batched_dict = {k:v for k,v in batch.items() if k not in batch_keys}
                        is_first_batch = False
                    offset += batch_size
                batched_dict = {k: concat_fn(v) for k,v in batched_dict.items()}
                batched_dict.update(non_batched_dict)
                return batched_dict
        return wrapper
    return decorator

def batchify_numpy(max_batch_size, axis=0, batch_keys=None):
    concat_fn = lambda x: np.concatenate(x, axis=axis)
    return _batchify_helper(max_batch_size, concat_fn=concat_fn, batch_keys=batch_keys)

def batchify_torch(max_batch_size, dim=0, batch_keys=None):
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is not available. Please install torch to use batchify_torch.")
    concat_fn = lambda x: torch.cat(x, dim=dim)
    return _batchify_helper(max_batch_size, concat_fn=concat_fn, batch_keys=batch_keys)