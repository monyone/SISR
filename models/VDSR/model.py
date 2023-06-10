
import torch
import torch.nn as nn

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
    self.layers = nn.Sequential(
      # Input
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=True),
      nn.ReLU(inplace=True),
      # Residual Layer
      *sum([[nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=True), nn.ReLU(inplace=True)] for _ in range(d - 1)], []),
      # Output
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=True)
    )

  def forward(self, x):
    input = x
    x = self.layers(x)
    x = torch.add(x, input)
    return x
