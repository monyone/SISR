
import torch
import torch.nn as nn

from math import sqrt

"""DRCN Model

Deeply-Recursive Convolutional Network (DRCN)

Site:
  "Deeply-Recursive Convolutional Network for Image Super-Resolution (2015)" (https://arxiv.org/abs/1511.04491)
"""

class DRCN(nn.Module):
  def __init__(self, c: int = 3, f: int = 3, n: int = 256, d: int = 16) -> None:
    """DRCN's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map.
      d (int): number of recursion.

    Examples:
      >>> DRCN() # typical DRCN parameters
    """
    super().__init__()
    self.recursion = d
    # embedding net
    self.embedding = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True)
    )
    # Inference Net
    self.inference = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.ReLU(inplace=True)
    )
    # Reconstruction Net
    self.reconstruction = nn.Sequential(
      nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False),
    )
    self.weight = nn.Parameter(torch.mul(torch.ones(d) , 1 / d))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    input = x
    x = self.embedding(x)
    x = [self.reconstruction(x := self.inference(x)) for _ in range(self.recursion)]
    return torch.add(torch.mul(sum(x), 1 / self.recursion), input), torch.add(torch.mul(sum(x * w for x, w in zip(x, self.weight)), 1 / torch.sum(self.weight)), input)
