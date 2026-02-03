import torch
from torch import nn
from contextlib import contextmanager


def model_device(model):
    """
    Get the device of the model.

    Args:
        model (nn.Module): The model.

    Returns:
        torch.device: The device of the model.
    """
    return next(model.parameters()).device


class ModelBase(nn.Module):
    """
    A base class for models with save/load functionality.
    """
    def __init__(self):
        """Initialize the ModelBase."""
        super().__init__()

    def save(self, path, config=None, optimizer_state_dict=None, **kwargs):
        """
        Save the model state dictionary, config, and optimizer state.

        Args:
            path (str): The path to save the model to.
            config (dict, optional): The configuration dictionary. Defaults to None.
            optimizer_state_dict (dict, optional): The optimizer state dictionary. Defaults to None.
            **kwargs: Additional data to save.
        """
        to_save = {}
        to_save["state_dict"] = self.state_dict()
        if config is not None:
            to_save["config"] = config
        if optimizer_state_dict is not None:
            to_save["optimizer_state_dict"] = optimizer_state_dict
        for k, v in kwargs.items():
            to_save[k] = v
        torch.save(to_save, path)

    def load(self, path, strict=True):
        """
        Load the model state dictionary.

        Args:
            path (str): The path to load the model from.
            strict (bool, optional): Whether to strictly enforce invalid keys in the state_dict. Defaults to True.

        Returns:
            dict: The loaded data dictionary.
        """
        data = torch.load(
            path, map_location=lambda storage, loc: storage, weights_only=False
        )
        self.load_state_dict(data["state_dict"], strict=strict)
        return data

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.

        Returns:
            int: The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return model_device(self)


@contextmanager
def eval_mode(model):
    """
    Context manager for temporarily switching a PyTorch model to evaluation mode.
    This context manager saves the model's original training state, switches it to
    evaluation mode for the duration of the context, and then restores the original
    state upon exit. This is useful for running inference or validation without
    affecting the model's training state.
    Args:
        model: A PyTorch model (nn.Module) to temporarily set to evaluation mode.
    Yields:
        model: The same model object in evaluation mode.
    Example:
        >>> with eval_mode(model):
        ...     predictions = model(input_data)
        >>> # model is restored to its original state here
    """

    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)
