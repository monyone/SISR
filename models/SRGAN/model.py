
import torch
import torch.nn as nn

from math import log, sqrt

"""SRResNet/SRGAN Model

Site:
  "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2016)" (https://arxiv.org/abs/1609.04802)
"""

class ResidualBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.BatchNorm2d(n),
      nn.PReLU(n),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.BatchNorm2d(n)
    )

  def forward(self, x):
    input = x
    x = self.block(x)
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

class SRResNet(nn.Module):
  def __init__(self, scale: int, c: int = 3, f1: int = 9, f2 = 3, n: int = 64, l: int = 16) -> None:
    """SRResNet's Constructor

    Args:
      c (int): number of channel the input/output image.
      f1 (int): spatial size of input/output region.
      f2 (int): spatial size of residual region.
      n (int): number of feature map.
      l (int): number of Residual Blocks.

    Examples:
      >>> SRResNet() # typical SRResNet parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f1, padding=f1//2, bias=False),
      nn.PReLU(n)
    )
    # Residubal Blocks
    self.residual = nn.Sequential(
      *[ResidualBlock(f=f2, n=n) for _ in range(l)]
    )
    self.skip = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f2, padding=f2//2, bias=False),
      nn.BatchNorm2d(n),
    )
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f2, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f1, padding=f1//2, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0, sqrt(2. / (n * n)))
        if m.bias is not None: m.bias.data.zero_()


  def forward(self, x):
    x = self.input(x)
    skip = x
    x = self.residual(x)
    x = self.skip(x) + skip
    x = self.upscale(x)
    x = self.output(x)
    return x

class SRGAN(nn.Module):
  def __init__(self, c: int = 3, size: tuple[int, int] = (96, 96)) -> None:
    """SRGAN's Constructor

    Args:
      c (int): number of channel the input/output image.
      size (tuple(int, int)): image szie

    Examples:
      >>> SRGAN() # typical SRGAN parameters
    """
    super().__init__()
    self.size = size
    # layers
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=64, kernel_size=3, padding=3//2),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      *[nn.Sequential(
        nn.Conv2d(in_channels=64 * (2 ** idx), out_channels=64 * (2 ** idx), kernel_size=3, stride=2, padding=3//2),
        nn.BatchNorm2d(num_features=64 * (2 ** idx)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64 * (2 ** idx), out_channels=128 * (2 ** idx), kernel_size=3, padding=3//2),
        nn.BatchNorm2d(num_features=128 * (2 ** idx)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
      ) for idx in range(3)],

      nn.Flatten(start_dim=1),

      nn.Linear(in_features=512 * ((self.size[0] // (2 ** 3)) * (self.size[1] // (2 ** 3))), out_features=1024),
      nn.LeakyReLU(negative_slope=0.2, inplace=True),
      nn.Linear(in_features=1024, out_features=1),
      nn.Sigmoid()
    )

  def forward(self, x):
    assert x.size()[-2:] == self.size, f'Input image size must be is {self.size}, got {x.size()[-2:]}'
    return self.layers(x)
