import random
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None # Placeholder to prevent NameError if accessed, though logic should use _TORCH_AVAILABLE

# This file's implementation is derived from https://github.com/wsgharvey/pytorch-utils

# RNG --------------------------------------------------


def set_random_seed(seed):
    """
    Set the random seed for random, numpy, and torch (if available).

    Args:
        seed (int): The random seed.
    """
    random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed + 1)
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)


def get_random_state():
    """
    Get the current random state of random, numpy, and torch (if available).

    Returns:
        dict: A dictionary containing the random states.
    """
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
    """
    Set the random state for random, numpy, and torch (if available) from a state dictionary.

    Args:
        state (dict): The random state dictionary.
    """
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
    """
    A context manager to run code with a temporary random seed or state.
    Restores the previous random state upon exiting.
    """
    def __init__(self, seed=None, state=None):
        """
        Initialize the RNG context manager.

        Args:
            seed (int, optional): The random seed to use.
            state (dict, optional): The random state to use.
        """

        self.state = get_random_state()
        with self:
            if seed is not None:
                set_random_seed(seed)
            elif state is not None:
                set_random_state(state)

    def __enter__(self):
        """Enter the context, saving the current state."""
        self.external_state = get_random_state()
        set_random_state(self.state)
        return self

    def __exit__(self, *args):
        """Exit the context, restoring the external state."""
        self.state = get_random_state()
        set_random_state(self.external_state)

    def get_state(self):
        """
        Get the internal state of the context manager.

        Returns:
            dict: The random state.
        """
        return self.state

    def set_state(self, state):
        """
        Set the internal state of the context manager.

        Args:
            state (dict): The random state.
        """
        self.state = state


class rng_decorator:
    """
    A decorator to run a function with a specific random seed.
    """
    def __init__(self, seed):
        """
        Initialize the rng_decorator.

        Args:
            seed (int): The random seed.
        """
        self.seed = seed

    def __call__(self, f):
        """
        Wrap the function.

        Args:
            f (callable): The function to wrap.

        Returns:
            callable: The wrapped function.
        """
        def wrapped_f(*args, **kwargs):
            with RNG(self.seed):
                return f(*args, **kwargs)

        return wrapped_f
