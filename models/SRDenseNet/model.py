
import torch
import torch.nn as nn

from math import log, sqrt

"""SRDenseNet Model

Site:
  "Image Super-Resolution Using Dense Skip Connections] (2017)" (https://ieeexplore.ieee.org/document/8237776)
"""

class DenseBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 128, g: int = 16, b: int = 8) -> None:
    super().__init__()
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=g, kernel_size=f, padding=f//2, bias=True),
      nn.ReLU(inplace=True)
    )
    self.layers = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(in_channels=(index + 1) * g, out_channels=g, kernel_size=f, padding=f//2, bias=True),
        nn.ReLU(inplace=True)
      ) for index in range(b - 1)
    ])

  def forward(self, x):
    x = self.input(x)
    for block in self.layers:
      x = torch.cat([x, block(x)], dim=1)
    return x

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.upscale = nn.Sequential(
      nn.ConvTranspose2d(in_channels=n, out_channels=n, kernel_size=f, stride=scale, padding=f//2, output_padding=(scale-1), bias=True),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.upscale(x)

class SRDenseNet(nn.Module):
  def __init__(self, scale: int, c: int = 3, f: int = 3, n: int = 256,  d: int = 8, g: int = 16, b: int = 8) -> None:
    """SRDenseNet's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.
      f (int): spatial size of region
      n (int): number of feature map.
      d (int): number of DenseBlock.
      g (int): number of Growth feature map in DenseLayer.
      b (int): number of DenseLayer in DenseBlock.

    Examples:
      >>> SRDenseNet() # typical SRDenseNet parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Conv2d(in_channels=c, out_channels=(g * b), kernel_size=f, padding=f//2, bias=True)
    # Residubal Blocks
    self.dense_blocks = nn.ModuleList([
      DenseBlock(f=f, n=((index + 1) * g * b), g=g, b=b) for index in range(d)
    ])
    # bottleneck
    self.bottleneck = nn.Sequential(
      nn.Conv2d(in_channels=((d + 1) * (g * b)), out_channels=n, kernel_size=1, padding=1//2, bias=True),
      nn.ReLU(inplace=True)
    )
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.reconstruction = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=True),
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()
    for m in self.dense_blocks.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)) / (d * b))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.input(x)
    for dense_block in self.dense_blocks:
      x = torch.cat([x, dense_block(x)], dim=1)
    x = self.bottleneck(x)
    x = self.upscale(x)
    x = self.reconstruction(x)
    return x
