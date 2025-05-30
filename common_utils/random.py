import random
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None # Placeholder to prevent NameError if accessed, though logic should use _TORCH_AVAILABLE

# This file is sourced from https://github.com/wsgharvey/pytorch-utils

# RNG --------------------------------------------------


def set_random_seed(seed):
    random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed + 1)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)


def get_random_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if _TORCH_AVAILABLE:
        state["torch"] = torch.get_rng_state()
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        else:
            state["cuda"] = None # Or an empty list, consistent with what get_rng_state_all might return
    else:
        state["torch"] = None
        state["cuda"] = None
    return state


def set_random_state(state):
    random.setstate(state["python"])
    if _TORCH_AVAILABLE:
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Ensure cuda state is not attempted to be set if it's None or cuda not available
            # get_rng_state_all() returns a list of tensors. set_rng_state_all() expects a list of tensors.
            # If state["cuda"] is an empty list (from a non-CUDA torch setup), this should be fine.
            torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


class RNG:
    def __init__(self, seed=None, state=None):

        self.state = get_random_state()
        with self:
            if seed is not None:
                set_random_seed(seed)
            elif state is not None:
                set_random_state(state)

    def __enter__(self):
        self.external_state = get_random_state()
        set_random_state(self.state)

    def __exit__(self, *args):
        self.state = get_random_state()
        set_random_state(self.external_state)

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class rng_decorator:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            with RNG(self.seed):
                return f(*args, **kwargs)

        return wrapped_f
