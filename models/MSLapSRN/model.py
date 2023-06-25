import torch
import torch.nn as nn

from math import log, sqrt

"""MSLapSRN Model

Multi Scale Laplacian Pyramid Super-Resolution Network (MSLapSRN)

Site:
  "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks (2017)" (https://arxiv.org/abs/1710.01992)
"""

class MSLapSRN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f: int = 3, n: int = 64, r: int = 8, d: int = 5) -> None:
    """MSLapSRN's Constructor

    Args:
      scale (int): number of scaling factor
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of texture map
      r (int): number of recursion of residential blocks.
      d (int): number of feature extraction layers in residential block.

    Examples:
      >>> MSLapSRN(scale) # typical LapSRN parameters
    """
    super().__init__()
    self.recursion = r
    self.scale = scale
    # Input
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )
    # Feature Extraction
    self.residual_block = nn.Sequential(
      *[nn.Sequential(
        nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      ) for _ in range(d)]
    )
    # upsaling image
    self.upscale_feature = nn.ConvTranspose2d(in_channels=n, out_channels=n, kernel_size=f, stride=2, padding=f//2, output_padding=1, bias=False)
    # feature to image
    self.feature_to_image = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False)
    # Upscaling image
    self.upscale_image = nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=f, stride=2, padding=f//2, output_padding=1, bias=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    output = [x]
    x = self.input(x)
    for _ in range(int(log(self.scale, 2))):
      residual = x
      for _ in range(self.recursion):
        x = torch.add(residual, self.residual_block(x))
      x = self.upscale_feature(x)
      output.append(torch.add(self.feature_to_image(x), self.upscale_image(output[-1])))
    return output[1:]

