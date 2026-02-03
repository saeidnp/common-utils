def zero_module(module):
    """
    Zero out all parameters of a module.

    Args:
        module (nn.Module): The module to zero out.

    Returns:
        nn.Module: The module with zeroed parameters.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module