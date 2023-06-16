
import torch
import torch.nn as nn

from math import sqrt

"""VDSR Model

Very Deep Super Resolution (VDSR)

Site:
  "Accurate Image Super-Resolution Using Very Deep Convolutional Networks (2015)" (https://arxiv.org/abs/1511.04587)
"""

class VDSR(nn.Module):
  def __init__(self, c: int = 3, f: int = 3, n: int = 64, d: int = 20) -> None:
    """VDSR's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      d (int): number of layers. d - 1 residual layers and one output layer.

    Examples:
      >>> VSDR() # typical VDSR parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True)
    )
    # Residual Layer
    self.residual = nn.Sequential(
      *sum([[
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
        nn.ReLU(inplace=True)
      ] for _ in range(max(0, d - 1))], []),
    )
    # Output Layer
    self.output = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))

  def forward(self, x):
    input = x
    x = self.input(x)
    x = self.residual(x)
    x = self.output(x)
    x = torch.add(x, input)
    return x
