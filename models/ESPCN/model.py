

import torch.nn as nn

"""ESPCN Model

Efficient Sub-Pixel Convolutional Neural Networks (ESPCN)

Site:
  "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (2016)" (https://arxiv.org/abs/1609.05158v2)
"""

class ESPCN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f1: int = 5, f2: int = 3, f3: int = 3, n1: int = 64, n2: int = 32) -> None:
    """FSRCNN's Constructor

    Args:
      scale (int): number of scaling factor
      c (int): number of channel the input/output image.
      f1 (int): spatial size of a Patch extraction and representation filter.
      f2 (int): spatial size of a Non-linear Mapping filter.
      f3 (int): spatial size of a Reconstruction filter.
      n1 (int): number of Patch extraction and representation feature map.
      n2 (int): number of Non-linear feature map. typically n1 // 2.

    Examples:
      >>> ESPCN(scale) # typical ESPCN parameters
    """
    super().__init__()
    # Patch extraction and representation
    self.patch_extraction = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n1, kernel_size=f1, padding=f1//2, bias=True),
      nn.Tanh(),
    )
    # Non-linear Mapping
    self.nonlinear_mapping = nn.Sequential(
      nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=f2, padding=f2//2, bias=True),
      nn.Tanh(),
    )
    # SubPixel Convolution
    self.subpixel_convolution = nn.Sequential(
      nn.Conv2d(in_channels=n2, out_channels=c * (scale ** 2), kernel_size=f3, padding=f3//2, bias=True),
      nn.PixelShuffle(scale)
    )

  def forward(self, x):
    x = self.patch_extraction(x)
    x = self.nonlinear_mapping(x)
    x = self.subpixel_convolution(x)
    return x
