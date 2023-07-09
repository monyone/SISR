
import torch
import torch.nn as nn

from math import log, sqrt

"""ESRNet/ESRGAN Model

Site:
  "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (2018)" (https://arxiv.org/abs/1809.00219)
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

class ESRNet(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, g: int = 32, d: int = 23, l: int = 3, b: int = 5) -> None:
    """ESRNet's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region
      n (int): number of feature map.
      d (int): number of ResidualInResidualDenceBlocks.
      l (int): number of ResidualDenseBlock in ResidualInResidualDenceBlocks.
      b (int): number of block in ResidualDenseBlock.

    Examples:
      >>> ESRNet() # typical ESRNet parameters
    """
    super().__init__()
    # Input Layer
    self.input = nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False)
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

class ESRGAN(nn.Module):
  def __init__(self, c: int = 3, size: tuple[int, int] = (96, 96)) -> None:
    """ESRGAN's Constructor

    Args:
      c (int): number of channel the input/output image.
      size (tuple(int, int)): image szie

    Examples:
      >>> ESRGAN() # typical ESRGAN parameters
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
    )

  def forward(self, x):
    assert x.size()[-2:] == self.size, f'Input image size must be is {self.size}, got {x.size()[-2:]}'
    return self.layers(x)
