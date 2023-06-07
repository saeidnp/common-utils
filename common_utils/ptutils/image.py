import torch
import PIL
import numpy as np

def tensor2pil(tensor, drange=[0,1]):
    """Given a tensor of shape (Bx)3xwxh with pixel values in drange, returns a PIL image
       of the tensor. Returns a list of images if the input tensor is a batch.

    Args:
        tensor: A tensor of shape (Bx)3xwxh
        drange (list, optional): Range of pixel values in the input tensor. Defaults to [0,1].
    """
    assert tensor.ndim == 3 or tensor.ndim == 4
    if tensor.ndim == 3:
        return tensor2pil(tensor.unsqueeze(0), drange=drange)[0]
    img_batch = tensor.cpu().numpy().transpose([0, 2, 3, 1])
    img_batch = (img_batch - drange[0]) / (drange[1] - drange[0])  * 255# img_batch with pixel values in [0, 255]
    img_batch = img_batch.astype(np.uint8)
    return [PIL.Image.fromarray(img) for img in img_batch]