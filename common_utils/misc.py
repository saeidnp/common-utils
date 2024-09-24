import itertools
import os
from functools import wraps
from filelock import FileLock
from pathlib import Path

import numpy as np
import torch

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


def num_available_cores():
    # Copied from pytorch source code https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    max_num_worker_suggest = None
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest or 1


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
        """A decorator for registering predictor classes."""

        def _register(cls):
            if name is None:
                local_name = cls.__name__
            else:
                local_name = name
            if local_name in _CLASSES:
                raise ValueError(f"Already registered model with name: {local_name}")
            _CLASSES[local_name] = cls
            return cls

        if cls is None:
            return _register
        else:
            return _register(cls)

    return register_fn


def _batchify_helper(max_batch_size, concat_fn, dim=0, batch_keys=None):
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
                while n > 0:
                    current_batch_size = min(n, max_batch_size)
                    samples.append(func(current_batch_size, *args, **kwargs))
                    n -= current_batch_size
                return concat_fn(samples)
            else:
                batched_dict = {k: [] for k in batch_keys}
                non_batched_dict = None
                while n > 0:
                    current_batch_size = min(n, max_batch_size)
                    batch = func(current_batch_size, *args, **kwargs)
                    for k in batch_keys:
                        batched_dict[k].append(batch[k])
                    non_batched_dict = {k:v for k,v in batch.items() if k not in batch_keys}
                    n -= current_batch_size
                batched_dict = {k: concat_fn(v) for k,v in batched_dict.items()}
                batched_dict.update(non_batched_dict)
                return batched_dict
        return wrapper
    return decorator

def batchify_numpy(max_batch_size, axis=0, batch_keys=None):
    concat_fn = lambda x: np.concatenate(x, axis=axis)
    return _batchify_helper(max_batch_size, concat_fn=concat_fn, batch_keys=batch_keys)

def batchify_torch(max_batch_size, dim=0, batch_keys=None):
    concat_fn = lambda x: torch.cat(x, dim=dim)
    return _batchify_helper(max_batch_size, concat_fn=concat_fn, batch_keys=batch_keys)