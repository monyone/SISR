
import torch
import torch.nn as nn

from math import log, sqrt

"""EDSR Model

Site:
  "Enhanced Deep Residual Networks for Single Image Super-Resolution (2017)" (https://arxiv.org/abs/1707.02921)
"""

class ResidualBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, w: float = 0.1) -> None:
    super().__init__()
    self.w = w
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )

  def forward(self, x):
    input = x
    x = self.block(x)
    x *= self.w
    x = torch.add(x, input)
    return x

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(n, (scale ** 2) * n, kernel_size=f, padding=1, bias=True),
      nn.PixelShuffle(scale),
      nn.PReLU(n)
    )

  def forward(self, x):
    return self.block(x)

class EDSR(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 256, l: int = 32) -> None:
    """EDSR's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      l (int): number of Residual Blocks.

    Examples:
      >>> EDSR() # typical EDSR parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )
    # Residubal Blocks
    self.residual = nn.Sequential(
      *[ResidualBlock(f=f, n=n, w=0.1) for _ in range(l)]
    )
    self.skip = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.input(x)
    skip = x
    x = self.residual(x)
    x = self.skip(x) + skip
    x = self.upscale(x)
    x = self.output(x)
    return x
