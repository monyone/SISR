
import torch
import torch.nn as nn

from math import log, sqrt

"""Reak-ESRNet/Real-ESRGAN Model

Site:
  "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data (2021)" (https://arxiv.org/abs/2107.10833)
"""

class ResidualDenseBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, g: int = 32, b: int = 5, w: float = 0.2) -> None:
    super().__init__()
    self.w = w
    self.dense = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(in_channels=n + index * g, out_channels=(g if index != b - 1 else n), kernel_size=f, padding=f//2, bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=True) if index != b - 1 else nn.Identity()
      ) for index in range(b)
    ])

  def forward(self, x):
    input = x
    for block in self.dense[:-1]:
      x = torch.cat([x, block(x)], dim=1)
    x = self.dense[-1](x)
    x = torch.mul(x, self.w)
    x = torch.add(x, input)
    return x

class ResidualInResidualDenseBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, g: int = 32, l: int = 3, b: int = 5, w: float = 0.2) -> None:
    super().__init__()
    self.w = w
    self.rdbs = nn.Sequential(*[
      ResidualDenseBlock(f=f, n=n, g=g, b=b, w=w) for _ in range(l)
    ])

  def forward(self, x):
    input = x
    x = self.rdbs(x)
    x = torch.mul(x, self.w)
    x = torch.add(x, input)
    return x

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Upsample(scale_factor=scale, mode='nearest'),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )

  def forward(self, x):
    return self.block(x)

class RealESRNet(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, g: int = 32, d: int = 23, l: int = 3, b: int = 5) -> None:
    """RealESRNet's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.
      f (int): spatial size of region
      n (int): number of feature map.
      d (int): number of ResidualInResidualDenceBlocks.
      l (int): number of ResidualDenseBlock in ResidualInResidualDenceBlocks.
      b (int): number of block in ResidualDenseBlock.

    Examples:
      >>> RealESRNet() # typical RealESRNet parameters
    """
    super().__init__()
    downscale = 1 if scale >= 4 else 4 - scale
    scale *= downscale
    # Input Layer
    self.input = nn.Sequential(
      nn.PixelUnshuffle(downscale),
      nn.Conv2d(in_channels=c * (downscale ** 2), out_channels=n, kernel_size=f, padding=f//2, bias=False)
    )
    # Residubal Blocks
    self.residual = nn.Sequential(
      *[ResidualInResidualDenseBlock(f=f, n=n, g=g, l=l, b=b, w=0.2) for _ in range(d)]
    )
    # skip
    self.skip = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.1 * sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.input(x)
    skip = x
    x = self.residual(x)
    x = torch.add(self.skip(x), skip)
    x = self.upscale(x)
    x = self.output(x)
    return x

class RealESRGAN(nn.Module):
  def __init__(self, c: int = 3, f: int = 3, n: int = 64, d: int = 3) -> None:
    """RealESRGAN's Constructor

    Args:
      c (int): number of channel the input image.
      f (int): number of spatial region
      n (int): number of feature map

    Examples:
      >>> RealESRGAN() # typical ESRGAN parameters
    """
    super().__init__()

    self.input = nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    self.down_blocks = nn.ModuleList([
      nn.Sequential(
        nn.utils.spectral_norm(nn.Conv2d(in_channels=(n * (2 ** index)), out_channels=(n * (2 ** (index + 1))), kernel_size=4, padding=4//2, stride=2)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      ) for index in range(d)
    ])
    self.up_blocks = nn.ModuleList([
      nn.Sequential(
        nn.utils.spectral_norm(nn.Conv2d(in_channels=(n * (2 ** (d - index))), out_channels=(n * (2 ** (d - (index + 1)))), kernel_size=f, padding=f//2)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      ) for index in range(d)
    ])
    self.output = nn.Sequential(
      nn.utils.spectral_norm(nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)),
      nn.LeakyReLU(0.2, True),
      nn.utils.spectral_norm(nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)),
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(in_channels=n, out_channels=1, kernel_size=f, padding=f//2)
    )

  def forward(self, x):
    x = self.input(x)
    skip = x
    down = [x := block(x) for block in self.down_blocks]
    for index, block in enumerate(self.up_blocks):
      x = block(nn.functional.interpolate(torch.add(x, down[(len(down) - 1) - index]) if index != 0 else x, scale_factor=2, mode='bilinear', align_corners=False))
    x = torch.add(skip, x)
    x = self.output(x)
    return x
