

import torch.nn as nn

"""FSRCNN Model

Fast Super-Resolution Convolutional Neural Networks (FSRCNN)

Site:
  "Accelerating the Super-Resolution Convolutional Neural Network (2016)" (https://arxiv.org/abs/1608.00367, https://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
"""

class FSRCNN(nn.Module):
  def __init__(self, scale: int, c: int = 3, d: int = 64, s: int = 12, m: int = 4) -> None:
    """FSRCNN's Constructor

    Args:
      scale (int): number of scaling
      c (int): number of channel the input/output image.
      d (int): number of feature map.
      m (int): number of mapping layers.

    Examples:
      >>> FSRCNN(scale) # typical FSRCNN parameters
      >>> FSRCNN(scale, d=32, s=5, m=1) # typical FSRCNN-s paramters
      >>> FSRCNN(scale, d=8, s=8, m=4) # FSRCNNX (https://github.com/igv/FSRCNN-TensorFlow) 8-0-4-1 parameters
      >>> FSRCNN(scale, d=16, s=16, m=4) # FSRCNNX (https://github.com/igv/FSRCNN-TensorFlow) 16-0-4-1 parameters
    """
    super().__init__()
    self.layers = nn.Sequential(
      # Feature Extraction
      nn.Conv2d(in_channels=c, out_channels=d, kernel_size=5, padding=5//2, bias=True),
      nn.PReLU(d),
      # Shrinking (if s != d)
      *([
        nn.Conv2d(in_channels=d, out_channels=s, kernel_size=1, padding=1//2, bias=True),
        nn.PReLU(s),
      ] if s != d else []),
      # Mapping
      *sum([[nn.Conv2d(in_channels=s, out_channels=s, kernel_size=3, padding=3//2, bias=True), nn.PReLU(s)] for _ in range(m)], []),
      # Expanding (if s != d)
      *([
        nn.Conv2d(in_channels=s, out_channels=d, kernel_size=1, padding=1//2, bias=True),
        nn.PReLU(d),
      ] if s != d else []),
      # Deconvolution
      nn.ConvTranspose2d(in_channels=d, out_channels=c, kernel_size=9, stride=scale, padding=9//2, output_padding=scale-1)
    )

  def forward(self, x):
    return self.layers(x)
