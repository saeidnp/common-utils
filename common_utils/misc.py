import os
import itertools
import time


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
    def register_fn(cls, *, name=None):
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

        return _register(cls)

    return register_fn
