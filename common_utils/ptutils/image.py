import torch
import PIL
import numpy as np

def tensor2pil(tensor, drange=[0,1]):
    """Given a tensor of shape (Bx)Cxwxh with pixel values in drange, returns a PIL image
       of the tensor. Returns a list of images if the input tensor is a batch.
       C can be 1 or 3. If C=1, the image is grayscale. If C=3, the image is RGB.

    Args:
        tensor: A tensor of shape (Bx)Cxwxh
        drange (list, optional): Range of pixel values in the input tensor. Defaults to [0,1].
    """
    assert tensor.ndim == 3 or tensor.ndim == 4, "Input tensor must be 3D or 4D"
    if tensor.ndim == 3:
        # If 3D, assume (C, H, W) and add a batch dimension
        return tensor2pil(tensor.unsqueeze(0), drange=drange)[0]

    img_batch = tensor.cpu().numpy().transpose([0, 2, 3, 1]) # B, H, W, C

    # Handle drange scaling
    # Avoid division by zero if drange[0] == drange[1]
    if drange[1] - drange[0] != 0:
        img_batch = (img_batch - drange[0]) / (drange[1] - drange[0])
    else: # If drange is something like [5,5], then the image is constant.
          # We can either set it to 0 or based on its value relative to a common range like [0,1]
          # For simplicity, if drange is flat, we assume values are constant and outside the typical dynamic range,
          # mapping them to 0.
        img_batch = np.zeros_like(img_batch)

    img_batch = (img_batch * 255).clip(0, 255).astype(np.uint8)

    # Check if the image is grayscale or RGB and handle accordingly
    if img_batch.shape[-1] == 1:
        # Grayscale image, squeeze the channel dimension
        return [PIL.Image.fromarray(img.squeeze(-1)) for img in img_batch]
    else:
        # RGB image
        return [PIL.Image.fromarray(img) for img in img_batch]