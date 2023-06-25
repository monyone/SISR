
import torch
import torch.nn as nn

from math import sqrt

"""DRRN Model

Deep Recursive Residual Network (DRRN)

Site:
  "Image Super-Resolution via Deep Recursive Residual Network (2017)" (https://ieeexplore.ieee.org/document/8099781)
"""

class DRRN(nn.Module):
  def __init__(self, c: int = 3, f: int = 3, n: int = 128, B: int = 1, U: int = 25) -> None:
    """DRRN's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      B (int): number of Residual Blocks.
      U (int): number of Residual Units in Redisual Blocks.

    Examples:
      >>> DRRN() # typical DRRN parameters
    """
    super().__init__()
    self.units = U
    # Input Layer
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True)
    )
    # Residual Unit
    self.redisual_units = nn.Sequential(*[
      nn.Sequential(
        nn.BatchNorm2d(num_features=n),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
        nn.BatchNorm2d(num_features=n),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False)
      ) for _ in range(B)]
    )
    # Residual Block
    self.residual_blocks = nn.Sequential(*[
      nn.Sequential(
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
        nn.ReLU(inplace=True),
      ) for _ in range(B)]
    )
    # Output Layer
    self.output = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0, sqrt(2. / (n * n)))
        if m.bias is not None: m.bias.data.zero_()
    for m in self.residual_blocks.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)) / self.units)
    for m in self.redisual_units.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)) / (self.units * 2))

  def forward(self, x):
    input = x
    x = self.input(x)
    for residual_block, redisual_unit in zip(self.residual_blocks, self.redisual_units):
      x = residual_block(x)
      block = x
      for _ in range(self.units):
        x = torch.add(redisual_unit(x), block)
    x = self.output(x)
    x = torch.add(x, input)
    return x
