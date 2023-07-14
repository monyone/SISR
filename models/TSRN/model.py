
import torch
import torch.nn as nn

from math import log, sqrt

"""TSRN Model

Site:
  "The Unreasonable Effectiveness of Texture Transfer for Single Image Super-resolution (2018)" (https://arxiv.org/abs/1808.00043)
"""

class ResidualBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )

  def forward(self, x):
    input = x
    x = self.block(x)
    x = torch.add(x, input)
    return x

class TSRN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, l: int = 10) -> None:
    """EnhanceNet's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      l (int): number of Residual Blocks.

    Examples:
      >>> EnhanceNet() # typical EnhanceNet parameters
    """
    super().__init__()
    self.scale = scale
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU()
    )
    # Residubal Blocks
    self.residual = nn.Sequential(
      *[ResidualBlock(f=f, n=n) for _ in range(l)]
    )
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[nn.Sequential(
        nn.Conv2d(n, (2 ** 2) * n, kernel_size=f, padding=f//2, bias=False),
        nn.PixelShuffle(2),
        nn.ReLU()
      ) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(),
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False),
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()
    for m in self.residual.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)) / l)

  def forward(self, x):
    input = x
    x = self.input(x)
    x = self.residual(x)
    x = self.upscale(x)
    x = self.output(x)
    x = torch.add(x, nn.functional.interpolate(input, scale_factor=self.scale, mode='bicubic', align_corners=False))
    return x
