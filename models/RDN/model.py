
import torch
import torch.nn as nn

from math import log, sqrt

"""RDN Model

Residual Dense Network (RDN)

Site:
  "Residual Dense Network for Image Super-Resolution (2018)" (https://arxiv.org/abs/1802.08797)
"""

class ResidualDenseBlock(nn.Module):
  def __init__(self, f: int = 3, n: int = 64, g: int = 32, b: int = 5) -> None:
    super().__init__()
    self.dense = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(in_channels=n + index * g, out_channels=g, kernel_size=f, padding=f//2, bias=False),
        nn.ReLU(inplace=True)
      ) for index in range(b)
    ])
    self.bottleneck = nn.Conv2d(in_channels=n + b * g, out_channels=n, kernel_size=1, padding=1//2, bias=False)

  def forward(self, x):
    input = x
    for block in self.dense:
      x = torch.cat([x, block(x)], dim=1)
    x = self.bottleneck(x)
    x = torch.add(x, input)
    return x

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(n, (scale ** 2) * n, kernel_size=f, padding=f//2, bias=True),
      nn.PixelShuffle(scale),
    )

  def forward(self, x):
    return self.block(x)

class RDN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, g: int = 32, d: int = 20, b: int = 6) -> None:
    """RDN's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region
      n (int): number of feature map.
      d (int): number of ResidualDenceBlocks.
      b (int): number of Convolutions in ResidualDenseBlock.
    Examples:
      >>> RDN() or RDN(n=64, g=32, d=20, b=6) # typical RDN parameters (5.3)
      >>> RDN(n=64, g=64, d=16, b=8) # 5.4 fair comparison for using 64 feature map
    """
    super().__init__()
    # Input Layer
    self.shallow_feature_extraction1 = nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    self.shallow_feature_extraction2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    # Residubal Blocks
    self.residual_dense_blocks = nn.ModuleList([
      ResidualDenseBlock(f=f, n=n, g=g, b=b) for _ in range(d)
    ])
    # bottleneck
    self.global_future_fusion = nn.Sequential(
      nn.Conv2d(in_channels=n * d, out_channels=n, kernel_size=1, padding=1//2, bias=False),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
    )
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.1 * sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    x = self.shallow_feature_extraction1(x)
    global_residual_learning = x
    x = self.shallow_feature_extraction2(x)
    x = self.global_future_fusion(torch.cat([x := rdb(x) for rdb in self.residual_dense_blocks], dim=1))
    x = torch.add(x, global_residual_learning)
    x = self.upscale(x)
    x = self.output(x)
    return x
