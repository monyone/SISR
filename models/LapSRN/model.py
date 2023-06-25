import torch
import torch.nn as nn

from math import log, sqrt

"""LapSRN Model

Laplacian Pyramid Super-Resolution Network (LapSRN)

Site:
  "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution (2017)" (https://arxiv.org/abs/1704.03915)
"""

class LapSRN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f: int = 3, n: int = 64, d: int = 10) -> None:
    """LapSRN's Constructor

    Args:
      scale (int): number of scaling factor
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of texture map
      d (int): number of feature extraction layers.

    Examples:
      >>> LapSRN(scale) # typical LapSRN parameters on scale = 2 or scale = 4
      >>> LapSRN(scale=8, d=5) # typical LapSRN parameters on scale = 8
    """
    super().__init__()
    iterations = int(log(scale, 2))
    # Input
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=False),
      nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )
    # Feature Extraction
    self.feature_extraction = nn.Sequential(
      *[nn.Sequential(
        *[nn.Sequential(
          nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=False),
          nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ) for _ in range(d)],
        nn.ConvTranspose2d(in_channels=n, out_channels=n, kernel_size=f, stride=2, padding=f//2, output_padding=1, bias=False),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
      ) for _ in range(iterations)]
    )
    # Output
    self.feature_to_image = nn.Sequential(
      *[nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=False) for _ in range(iterations)]
    )
    # Upscaling Image
    self.upscale_image = nn.Sequential(
      *[nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=f, stride=2, padding=f//2, output_padding=1, bias=True) for _ in range(iterations)]
    )

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x):
    output = [x]
    x = self.input(x)
    for feature_extraction, feature_to_image, upscale_image in zip(self.feature_extraction, self.feature_to_image, self.upscale_image):
      x = feature_extraction(x)
      output.append(torch.add(feature_to_image(x), upscale_image(output[-1])))
    return output[1:]

