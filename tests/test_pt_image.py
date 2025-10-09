import pytest
import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None # Make sure torch is defined for type hints if needed, or for hasattr checks

if TORCH_AVAILABLE:
    from common_utils.ptutils.image import tensor2pil
else:
    # If torch is not available, tensor2pil cannot be imported.
    # We define a dummy here so the file can be parsed, but tests will be skipped.
    def tensor2pil(tensor, drange=[0,1]):
        pass

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available or not installable in current environment")

# Test tensor2pil for a single image (3D tensor)
def test_tensor2pil_single_image():
    # Create a sample 3D tensor (C, H, W)
    tensor_data = torch.rand(3, 32, 32)
    img = tensor2pil(tensor_data)

    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)  # PIL uses (W, H)
    assert img.mode == 'RGB'

# Test tensor2pil for a batch of images (4D tensor)
def test_tensor2pil_batch_of_images():
    # Create a sample 4D tensor (B, C, H, W)
    tensor_data = torch.rand(2, 3, 32, 32)
    imgs = tensor2pil(tensor_data)

    assert isinstance(imgs, list)
    assert len(imgs) == 2
    for img in imgs:
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)
        assert img.mode == 'RGB'

# Test tensor2pil with drange scaling
def test_tensor2pil_drange():
    # Create a tensor with values outside [0,1]
    # Shape (1, 3, 2, 2) for B, C, H, W
    raw_channel_data = torch.tensor([[-1., 0.], [1., 2.]], dtype=torch.float32)
    # Stack to make 3 channels, then add batch dimension
    tensor_data = torch.stack([raw_channel_data, raw_channel_data, raw_channel_data], dim=0).unsqueeze(0)

    imgs = tensor2pil(tensor_data, drange=[-1, 2])
    assert isinstance(imgs, list)
    assert len(imgs) == 1
    img = imgs[0]

    assert isinstance(img, Image.Image)
    assert img.mode == 'RGB'

    # Convert PIL image to numpy array to check pixel values
    img_np = np.array(img) # Shape (H, W, C) e.g. (2, 2, 3)

    # Expected pixel values after scaling from drange=[-1, 2] to [0, 255]
    # val_out = ((val_in - drange_min) / (drange_max - drange_min)) * 255
    # -1 should map to 0: ((-1 - (-1)) / (2 - (-1))) * 255 = 0
    #  0 should map to 85: ((0 - (-1)) / (2 - (-1))) * 255 = (1/3)*255 = 85
    #  1 should map to 170: ((1 - (-1)) / (2 - (-1))) * 255 = (2/3)*255 = 170
    #  2 should map to 255: ((2 - (-1)) / (2 - (-1))) * 255 = 255

    # Check pixel values (H, W, C)
    # img_np[row, col, channel_rgb_idx]
    assert np.allclose(img_np[0, 0, :], [0, 0, 0])   # Original value -1.0
    assert np.allclose(img_np[0, 1, :], [85, 85, 85]) # Original value 0.0
    assert np.allclose(img_np[1, 0, :], [170, 170, 170]) # Original value 1.0
    assert np.allclose(img_np[1, 1, :], [255, 255, 255]) # Original value 2.0

# Test tensor2pil with invalid tensor dimensions
def test_tensor2pil_invalid_ndim():
    # Test with a 2D tensor
    tensor_2d = torch.rand(32, 32)
    with pytest.raises(AssertionError): # Removed match, as default AssertionError has no specific message
        tensor2pil(tensor_2d)

    # Test with a 5D tensor
    tensor_5d = torch.rand(1, 2, 3, 32, 32)
    with pytest.raises(AssertionError): # Removed match
        tensor2pil(tensor_5d)

# Test tensor2pil with grayscale and RGB inputs
def test_tensor2pil_channel_modes():
    # Test with single channel input (should result in 'L' mode image)
    tensor_gray = torch.rand(1, 1, 32, 32) # B, C, H, W where C=1
    imgs_gray = tensor2pil(tensor_gray)

    assert isinstance(imgs_gray, list)
    assert len(imgs_gray) == 1
    assert isinstance(imgs_gray[0], Image.Image)
    assert imgs_gray[0].size == (32, 32)
    assert imgs_gray[0].mode == 'L' # Single channel image

    # Test with 3 channel input (should result in 'RGB' mode image)
    tensor_rgb = torch.rand(1, 3, 32, 32) # B, C, H, W where C=3
    imgs_rgb = tensor2pil(tensor_rgb)

    assert isinstance(imgs_rgb, list)
    assert len(imgs_rgb) == 1
    assert isinstance(imgs_rgb[0], Image.Image)
    assert imgs_rgb[0].size == (32, 32)
    assert imgs_rgb[0].mode == 'RGB' # 3 channel image

# Test with a single grayscale image (3D tensor, C=1)
def test_tensor2pil_single_grayscale_image():
    tensor_data = torch.rand(1, 32, 32) # (C, H, W) where C=1
    img = tensor2pil(tensor_data)

    assert isinstance(img, Image.Image)
    assert img.size == (32, 32)
    assert img.mode == 'L'

# Test with drange where min=max
def test_tensor2pil_drange_min_max_equal():
    tensor_data = torch.ones(1, 3, 2, 2) * 5.0 # All values are 5.0
    imgs = tensor2pil(tensor_data, drange=[5, 5])
    img = imgs[0]
    img_np = np.array(img)
    # Expect all pixels to be 0, as (val - min) / (max - min) is undefined, code handles this as 0
    assert np.all(img_np == 0)

# Test with default drange [0,1]
def test_tensor2pil_default_drange():
    tensor_data = torch.tensor([[[0.0, 0.5], [0.75, 1.0]]], dtype=torch.float32) # (1,H,W)
    tensor_data = tensor_data.unsqueeze(0) # (B,1,H,W) -> (1,1,2,2)

    imgs = tensor2pil(tensor_data) # Default drange [0,1]
    img = imgs[0]
    assert img.mode == 'L'
    img_np = np.array(img)

    assert img_np[0,0] == 0    # 0.0 * 255
    assert img_np[0,1] == 127  # 0.5 * 255 (approx)
    assert img_np[1,0] == 191  # 0.75 * 255 (approx)
    assert img_np[1,1] == 255  # 1.0 * 255

    # Check closeness for float results
    assert abs(img_np[0,1] - 0.5*255) < 1
    assert abs(img_np[1,0] - 0.75*255) < 1
