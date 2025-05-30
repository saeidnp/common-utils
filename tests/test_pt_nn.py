import pytest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define placeholders if torch or nn are used in type hints or class definitions
    # that are parsed even if skipped.
    torch = None
    nn = None

if TORCH_AVAILABLE:
    from common_utils.ptutils.nn import zero_module
else:
    # Dummy function if torch is not available, to allow parsing.
    # pytestmark will skip all tests anyway.
    def zero_module(module):
        return module
    if nn is None: # If torch imported but nn failed (unlikely but for safety)
        class Module: pass
        nn = type('nn', (object,), {'Module': Module, 'Parameter': None, 'Linear': None})()


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available or not installable in current environment")

# Helper: Simple Model Definition
# This class definition requires torch.nn, so it should only be effective if TORCH_AVAILABLE
if TORCH_AVAILABLE:
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Initialize with non-zero data to ensure zero_module has an effect
            self.param = nn.Parameter(torch.ones(5, 5) * 2.0) # Parameter directly on the module
            self.linear = nn.Linear(5, 2) # A sub-module with its own parameters
            # Initialize linear layer parameters to non-zero
            self.linear.weight.data.fill_(1.5)
            self.linear.bias.data.fill_(0.5)
else: # Define a dummy if not available so file can be parsed by linters/etc.
    class SimpleModule:
        def __init__(self):
            # Mock parameters for the purpose of iteration in tests if they were to run
            self.param = None
            self.linear = None

# Test zero_module functionality
def test_zero_module():
    model = SimpleModule()

    # Optional: Double-check that params are non-zero before zeroing, if not done in __init__
    # For example, if self.param was torch.randn:
    # model.param.data.fill_(1.0)
    # model.linear.weight.data.fill_(1.0)
    # model.linear.bias.data.fill_(1.0)

    # Ensure parameters are not all zero initially (as per SimpleModule init)
    assert not torch.all(model.param.data == 0)
    assert not torch.all(model.linear.weight.data == 0)
    assert not torch.all(model.linear.bias.data == 0)

    zero_module(model)

    # Iterate through all parameters of the model
    all_params_zero = True
    for p in model.parameters():
        if not torch.all(p.data == 0):
            all_params_zero = False
            break
    assert all_params_zero, "Not all parameters were zeroed after calling zero_module"

    # Specific checks (redundant if the loop above is comprehensive, but good for clarity)
    assert torch.all(model.param.data == 0), "Directly accessed parameter 'param' was not zeroed."
    assert torch.all(model.linear.weight.data == 0), "Sub-module 'linear.weight' was not zeroed."
    assert torch.all(model.linear.bias.data == 0), "Sub-module 'linear.bias' was not zeroed."

# Test zero_module on a module with no parameters
def test_zero_module_on_empty_module():
    if not TORCH_AVAILABLE: pytest.skip("PyTorch not available")

    class EmptyModule(nn.Module):
        def __init__(self):
            super().__init__()
            # This module intentionally has no parameters

    model = EmptyModule()

    # Call zero_module, it should not raise an error
    try:
        zero_module(model)
    except Exception as e:
        pytest.fail(f"zero_module raised an exception on an empty module: {e}")

    # Verify that the model indeed has no parameters
    assert len(list(model.parameters())) == 0, "EmptyModule should have no parameters."

# Test that zero_module returns the same module instance
def test_zero_module_returns_module():
    model = SimpleModule()
    returned_module = zero_module(model)
    assert returned_module is model, "zero_module should return the same module instance it processed."
