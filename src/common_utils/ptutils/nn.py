def zero_module(module):
    """
    Zero out all parameters of a module.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module