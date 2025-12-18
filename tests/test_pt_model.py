import pytest
import tempfile
import os

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
    from common_utils.ptutils.model import model_device, ModelBase, eval_mode
else:
    # Dummy classes/functions if torch is not available, to allow parsing
    # pytestmark will skip all tests anyway.
    def model_device(model): return "cpu" # Dummy
    class ModelBase: # Dummy
        def __init__(self): pass
        def save(self, path, **kwargs): pass
        def load(self, path, map_location=None, strict=True): pass
        def count_parameters(self): return 0
        def device(self): return "cpu"
    if nn is None: # If torch imported but nn failed (unlikely but for safety)
        class Module: pass
        nn = type('nn', (object,), {'Module': Module, 'Linear': None})()


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available or not installable in current environment")

# Helper: Simple Model Definition
if TORCH_AVAILABLE: # Only define if torch.nn is available
    class SimpleNN(ModelBase):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            return self.fc(x)
else: # Define a dummy if not available so file can be parsed
    class SimpleNN(ModelBase):
        def __init__(self, input_size, output_size): super().__init__()
        def forward(self, x): return x


# Test model_device utility function
def test_model_device():
    model = SimpleNN(10, 2)
    # model_device is expected to return a torch.device object
    assert model_device(model).type == 'cpu' # Default device is CPU

    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        model.to('cuda')
        assert model_device(model).type == 'cuda'
        # Test with specific cuda device if multiple are hypothetically available
        # For now, just checking 'cuda' type is sufficient for most single-GPU CI
        # model.to('cuda:0')
        # assert model_device(model).type == 'cuda'
        # assert model_device(model).index == 0 # If checking specific device index


# Test ModelBase.save and ModelBase.load methods
def test_model_base_save_and_load():
    model1 = SimpleNN(10, 2)
    # Initialize weights to a known value for comparison
    model1.fc.weight.data.fill_(0.5)
    model1.fc.bias.data.fill_(0.1)

    config_data = {'input_size': 10, 'output_size': 2, 'architecture': 'SimpleNN'}
    # Create a dummy optimizer for saving its state_dict
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer.step() # Make a dummy step to populate state if needed by some optimizers
    optimizer_state_to_save = optimizer.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    try:
        model1.save(
            tmp_file_path,
            config=config_data,
            optimizer_state_dict=optimizer_state_to_save,
            custom_key='custom_value',
            epoch=10
        )

        # Load into a new model instance
        model2 = SimpleNN(10, 2)
        # Ensure model2 weights are different before loading
        model2.fc.weight.data.fill_(0.0)
        model2.fc.bias.data.fill_(0.0)

        loaded_data_info = model2.load(tmp_file_path) # model2.load now returns the loaded dict

        # Verify model state
        assert torch.equal(model1.fc.weight.data, model2.fc.weight.data), "Model weights differ after loading"
        assert torch.equal(model1.fc.bias.data, model2.fc.bias.data), "Model biases differ after loading"

        # Verify other saved data directly from what model2.load returns
        assert loaded_data_info['config'] == config_data
        assert loaded_data_info['optimizer_state_dict']['param_groups'][0]['lr'] == optimizer_state_to_save['param_groups'][0]['lr']
        assert loaded_data_info['custom_key'] == 'custom_value'
        assert loaded_data_info['epoch'] == 10

        # Also check by loading the file directly with torch.load
        raw_loaded_data = torch.load(tmp_file_path)
        assert raw_loaded_data['config'] == config_data
        assert raw_loaded_data['custom_key'] == 'custom_value'

        # Test loading with strict=False
        # ModelBase.load passes strict to self.load_state_dict
        model_load_strict_false = SimpleNN(10, 2)
        model_load_strict_false.fc.weight.data.fill_(0.123) # Ensure different weights
        model_load_strict_false.load(tmp_file_path, strict=False) # Should work
        assert torch.equal(model1.fc.weight.data, model_load_strict_false.fc.weight.data)

        # Test loading with strict=True fails for mismatched keys/shapes
        # Create a model with different architecture (e.g. different layer name or shape)
        # SimpleNN's structure is fixed by __init__ args.
        # To make strict=True fail, the state_dict keys must mismatch.
        # If we saved model1 (10,2), its state_dict has 'fc.weight' and 'fc.bias'.
        # A SimpleNN(11,2) will also have 'fc.weight' and 'fc.bias' but shapes will differ.
        model_fail_strict_shape = SimpleNN(11, 2) # fc.weight shape [2,11] vs [2,10]
        with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict for SimpleNN"): # Use raw string
            model_fail_strict_shape.load(tmp_file_path, strict=True)

        # If we had a model with different layer names:
        class DifferentNN(ModelBase):
            def __init__(self): super().__init__(); self.other_layer = nn.Linear(10,2)
            def forward(self,x): return self.other_layer(x)

        model_fail_strict_keys = DifferentNN()
        with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict for DifferentNN"): # Use raw string
             model_fail_strict_keys.load(tmp_file_path, strict=True) # Fails due to key mismatch


    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# Test ModelBase.count_parameters method
def test_model_base_count_parameters():
    model = SimpleNN(input_size=10, output_size=2)
    # Expected parameters: fc.weight (10*2) + fc.bias (2) = 20 + 2 = 22
    assert model.count_parameters() == 22

    # Test with a model that has no trainable parameters
    model_no_grad = SimpleNN(input_size=5, output_size=1) # 5*1 + 1 = 6 params
    for param in model_no_grad.parameters():
        param.requires_grad = False
    assert model_no_grad.count_parameters() == 0 # Only counts params with requires_grad=True

    # Test with mixed requires_grad
    model_mixed_grad = SimpleNN(input_size=3, output_size=3) # 3*3 + 3 = 12 params
    model_mixed_grad.fc.weight.requires_grad = False # 3*3 = 9 params non-trainable
    # Bias (3 params) remains trainable
    assert model_mixed_grad.count_parameters() == 3


# Test ModelBase.device method (which uses model_device)
def test_model_base_device_method():
    model = SimpleNN(10, 2)
    assert model.device().type == 'cpu'

    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        model.to('cuda')
        assert model.device().type == 'cuda'
        # model.to('cpu')
        # assert model.device().type == 'cpu'


# Test eval_mode context manager
def test_eval_mode():
    model = SimpleNN(10, 2)

    # Scenario 1: Model initially in training mode
    model.train()
    assert model.training is True
    with eval_mode(model):
        assert model.training is False
    assert model.training is True

    # Scenario 2: Model initially in eval mode
    model.eval()
    assert model.training is False
    with eval_mode(model):
        assert model.training is False
    assert model.training is False

    # Scenario 3: Exception handling
    model.train()
    assert model.training is True
    try:
        with eval_mode(model):
            assert model.training is False
            raise RuntimeError("Test Exception")
    except RuntimeError:
        pass
    assert model.training is True
