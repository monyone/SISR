
import torch
import torch.nn as nn

from math import log, sqrt

"""SRGAN Model

Site:
  "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (2016)" (https://arxiv.org/abs/1609.04802)
"""

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
