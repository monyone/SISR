

import torch.nn as nn

"""SRCNN Model

Super-Resolution Convolutional Neural Network (SRCNN)

Site:
  "Image Super-Resolution Using Deep Convolutional Networks (2014)" (https://arxiv.org/abs/1501.00092, https://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
"""

class SRCNN(nn.Module):
  def __init__(self, c: int = 3, f1: int = 9, f2: int = 1, f3: int = 5, n1: int = 64, n2: int = 32) -> None:
    """SRCNN's Constructor

    Args:
      c (int): number of channel the input/output image.
      f1 (int): spatial size of a Patch extraction and representation filter.
      f2 (int): spatial size of a Non-linear Mapping filter. typically 1.
      f3 (int): spatial size of a Reconstruction filter.
      n1 (int): number of Patch extraction and representation feature map.
      n2 (int): number of Non-linear feature map. typically n1 // 2.

    Examples:
      >>> SRCNN() # typical SRCNN parameters
      >>> SRCNN(n1 = 128, n2 = 64) # Large feature map variant in 4.3.1 Filter number
      >>> SRCNN(n1 = 32, n2 = 16) # Small feature map variant in 4.3.1 Filter number
      >>> SRCNN(f2 = 5) # Large spatial size variant in 4.3.2 Filter size
      >>> SRCNN(f2 = 3) # Large spatial size variant in 4.3.2 Filter size
    """
    super().__init__()
    # Patch extraction and representation
    self.patch_extraction = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n1, kernel_size=f1, padding=f1//2, bias=True),
      nn.ReLU(inplace=True),
    )
    # Non-linear Mapping
    self.nonlinear_mapping = nn.Sequential(
      nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=f2, padding=f2//2, bias=True),
      nn.ReLU(inplace=True),
    )
    # Reconstruction
    self.reconstruction = nn.Conv2d(in_channels=n2, out_channels=c, kernel_size=f3, padding=f3//2, bias=True)

  def forward(self, x):
    x = self.patch_extraction(x)
    x = self.nonlinear_mapping(x)
    x = self.reconstruction(x)
    return x
