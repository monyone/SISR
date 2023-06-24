import torch
import torch.nn as nn

"""REDNET Model

very deep Residual Encoder-Decoder Networks (RED-NET)

Site:
  "Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections" ()
"""

class REDNET(nn.Module):
  def __init__(self, c: int = 3, f: int = 3, n: int = 64, l: int = 15) -> None:
    """REDNET's Constructor

    Args:
      c (int): number of channel the input/output image.
      f (int): spatial size of region.
      n (int): number of feature map. typically.
      l (int): number of layers

    Examples:
      >>> REDNET() # typical RED-NET30 parameters
      >>> REDNET(l=10) # typical RED-NET20 parameters
    """
    super().__init__()
    self.input = nn.Sequential(
      nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=True),
      nn.ReLU(inplace=True),
    )
    self.relu = nn.ReLU(inplace=True)
    self.convolution = nn.Sequential(
      *[nn.Conv2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=True) for _ in range(l)]
    )
    self.deconvolution = nn.Sequential(
      *[nn.ConvTranspose2d(in_channels=n, out_channels=n, kernel_size=f, padding=f//2, bias=True) for _ in range(l)]
    )
    self.output = nn.ConvTranspose2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=True)

  def forward(self, x):
    input = x
    x = self.input(x)
    skip = [x := self.relu(conv(x)) for conv in self.convolution]
    for index, deconv in enumerate(self.deconvolution):
      x = self.relu(deconv(x))
      if index % 2 == 1:
        x = self.relu(torch.add(x, skip[(len(self.convolution) - 1) - index]))

    x = self.output(x)
    x = torch.add(x, input)
    return x
