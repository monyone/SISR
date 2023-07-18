
import torch
import torch.nn as nn

from math import log, sqrt

"""RCAN model

Site:
  "Image Super-Resolution Using Very Deep Residual Channel Attention Networks (2018)" (https://arxiv.org/abs/1807.02758)
"""

class ResidualChannelAttensionBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, r: int = 16) -> None:
    super().__init__()
    self.convs = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )
    self.channel_attension = nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=1),
      nn.Conv2d(in_channels=n, out_channels=n//r, kernel_size=1, padding=1//2, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=n//r, out_channels=n, kernel_size=1, padding=1//2, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    input = x
    x = self.convs(x)
    x = torch.mul(x, self.channel_attension(x))
    return torch.add(input, x)

class ResidualGroup(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, r: int = 16, b = 20) -> None:
    super().__init__()
    self.blocks = nn.Sequential(
      *[ResidualChannelAttensionBlock(f=f, n=n, r=r) for _ in range(b)],
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )

  def forward(self, x):
    input = x
    x = self.blocks(x)
    return torch.add(input, x)

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(n, (scale ** 2) * n, kernel_size=f, padding=f//2, bias=True),
      nn.PixelShuffle(scale),
    )

  def forward(self, x):
    return self.block(x)

class RCAN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f: int = 3, n: int = 64,  g: int = 10, b: int = 20, r: int = 16) -> None:
    """RCAN's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.
      f1 (int): spatial size of region.
      n (int): number of feature map.
      g (int): number of Residual Groups
      b (int): number of Residual Channel Attension Block in Residual Groups

    Examples:
      >>> RCAN() # typical SRResNet parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    # Residubal Groups
    self.residual_groups = nn.Sequential(
      *[ResidualGroup(f=f, n=n, r=r, b=b) for _ in range(g)]
    )
    self.skip = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
    )
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.1 * sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.input(x)
    skip = x
    x = self.residual_groups(x)
    x = self.skip(x) + skip
    x = self.upscale(x)
    x = self.output(x)
    return x
