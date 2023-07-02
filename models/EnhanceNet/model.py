
import torch
import torch.nn as nn

from math import log, sqrt

"""EnhanceNet Model

Site:
  "EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis (2017)" (https://arxiv.org/abs/1612.07919)
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

class EnhanceNet(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, l: int = 10) -> None:
    """EnhanceNet's Constructor

    Args:
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
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
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

class EnhanceNetDiscriminator(nn.Module):
  def __init__(self, c: int = 3, size: tuple[int, int] = (128, 128)) -> None:
    """EnhanceNet's Discriminator Constructor

    Args:
      c (int): number of channel the input/output image.
      size (tuple(int, int)): image size (default: 128x128)

    Examples:
      >>> EnhanceNetDiscriminator() # typical EnhanceNetDiscriminator parameters
    """
    super().__init__()
    self.size = size
    # layers
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, padding=3//2),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=3//2),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      *[nn.Sequential(
        nn.Conv2d(in_channels=32 * (2 ** idx), out_channels=64 * (2 ** idx), kernel_size=3, padding=3//2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64 * (2 ** idx), out_channels=64 * (2 ** idx), kernel_size=3, stride=2, padding=3//2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
      ) for idx in range(4)],

      nn.Flatten(start_dim=1),

      nn.Linear(in_features=512 * ((self.size[0] // (2 ** 5)) * (self.size[1] // (2 ** 5))), out_features=1024),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      nn.Linear(in_features=1024, out_features=1),
      nn.Sigmoid()
    )

  def forward(self, x):
    assert x.size()[-2:] == self.size, f'Input image size must be is {self.size}, got {x.size()[-2:]}'
    return self.layers(x)
