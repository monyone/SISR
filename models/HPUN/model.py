import torch
import torch.nn as nn

from math import sqrt

"""HPUN Model

Hybrid Pixel-Unshuffled Network (HPUN)

Site:
  "Hybrid Pixel-Unshuffled Network for Lightweight Image Super-Resolution (2022)" (https://arxiv.org/abs/2203.08921)

Caution:
  This algorithm is covered by patent US20230153946A1. (https://patents.google.com/patent/US20230153946A1/)
  If you run in production, you might be patent issue.
"""

class SelfResidualDSCResBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool, depth: int = 2) -> None:
    super().__init__()
    self.depthwise = nn.ModuleList([
      nn.Conv2d(in_channels=in_channels, groups=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias) for _ in range(depth)
    ])
    self.pointwise = nn.ModuleList([
      nn.Sequential(*[
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=1//2, bias=bias),
        nn.ReLU(inplace=True) if idx != depth - 1 else nn.Identity()
      ]) for idx in range(depth)
    ])

  def forward(self, x: torch.Tensor):
    input = x
    for d, p in zip(self.depthwise, self.pointwise):
      x = p(torch.add(x, d(x)))
    return torch.add(input, x)

class PixelUnshuffledDownsampling(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int, stride: int, padding: int, bias: bool) -> None:
    super().__init__()
    self.depthwise = nn.Sequential(
      nn.PixelUnshuffle(downscale_factor=scale_factor),
      nn.MaxPool2d(kernel_size=scale_factor),
      nn.Conv2d(in_channels=(in_channels * (scale_factor ** 2)), groups=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  bias=bias),
      nn.Upsample(scale_factor=(scale_factor * 2), mode='bilinear', align_corners=False)
    )
    self.pointwise = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=1//2, bias=bias)

  def forward(self, x: torch.Tensor):
    return self.pointwise(torch.add(self.depthwise(x), x))

class PixelUnshuffledBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int, stride: int, padding: int, bias: bool) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      PixelUnshuffledDownsampling(in_channels=in_channels, out_channels=out_channels, scale_factor=scale_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(inplace=True),
      SelfResidualDSCResBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )

  def forward(self, x: torch.Tensor):
    return torch.add(self.layers(x), x)

class HybridPixelUnshuffledBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int, stride: int, padding: int, bias: bool) -> None:
    super().__init__()
    self.pud = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(inplace=True),
      PixelUnshuffledDownsampling(in_channels=in_channels, out_channels=out_channels, scale_factor=scale_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )
    self.pub = PixelUnshuffledBlock(in_channels=in_channels, out_channels=out_channels, scale_factor=scale_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

  def forward(self, x: torch.Tensor):
    x = torch.add(self.pud(x), x)
    x = torch.add(self.pub(x), x)
    return x

class UpscaleBlock(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, scale_factor: int, kernel_size: int, stride: int, padding: int, bias: bool) -> None:
    super().__init__()
    self.upscale = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=(out_channels * (scale_factor ** 2)), kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.PixelShuffle(upscale_factor=scale_factor),
    )

  def forward(self, x: torch.Tensor):
    return self.upscale(x)

class HPUN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, l: int = 8) -> None:
    """HPUN's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      l (int): number of HPUB.

    Examples:
      >>> HPUN() # typical HPUN-M parameters
      >>> HPUN(l=12) # HPUN-L parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=True),
    )
    # Hybrid PixelUnshuffled Block
    self.hpub = nn.Sequential(*[
      HybridPixelUnshuffledBlock(in_channels=n, out_channels=n, scale_factor=2, kernel_size=f, stride=1, padding=f//2, bias=False) for _ in range(l)
    ])
    # skip connection block
    self.skip = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    # Upscale and Output Blocks
    self.upscale = UpscaleBlock(in_channels=n, out_channels=c, scale_factor=scale, kernel_size=f, stride=1, padding=f//2, bias=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.1 * sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.input(x)
    skip = x
    x = self.hpub(x)
    x = self.skip(x) + skip
    x = self.upscale(x)
    return x
