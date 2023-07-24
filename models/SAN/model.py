import torch
import torch.nn as nn

from math import log, sqrt

"""SAN Model

Second-order Attention Network (SAN)

Site:
  " Second-Order Attention Network for Single Image Super-Resolution (2019)" (https://openaccess.thecvf.com/content_CVPR_2019/html/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.html)

Caution:
  Unverified (TODO: need long run)
"""

def covarianceMatrix(x: torch.Tensor):
  B, C, H, W = x.shape
  M = H * W
  x = x.reshape(B, C, M)
  I_hat = (-1 / (M ** 2)) * torch.ones((M, M), device=x.device, dtype=x.dtype) + (1 / M) * torch.eye(M, device=x.device, dtype=x.dtype)
  I_hat = I_hat.reshape(1, M, M).repeat_interleave(B, dim=0)
  return x.bmm(I_hat).bmm(x.transpose(1, 2))

def eigenDecomposition(x: torch.Tensor, iteration: int):
  B, C, _ = x.shape
  I3 = 3.0 * torch.eye(C, device=x.device, dtype=x.dtype).reshape(1, C, C).repeat_interleave(B, dim=0)
  normA = ((1/3) * x.mul(I3)).sum(dim=1).sum(dim=1)
  A = x.div(normA.reshape(B, 1, 1).expand_as(x))
  if iteration < 2:
    ZY = 0.5 * (I3 - A)
    Y = A.bmm(ZY)
  else:
    ZY = 0.5 * (I3 - A)
    Y = A.bmm(ZY)
    Z = ZY
    for i in range(1, iteration - 1):
      ZY = 0.5 * (I3 - Z).bmm(Y)
      Y = Y.bmm(ZY)
      Z = ZY.bmm(Z)
    ZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
  return ZY * (torch.sqrt(normA).reshape(B, 1, 1).expand_as(x))

class SecondOrderChannelAttension(nn.Module):
  def __init__(self, in_channels: int, reduction: int = 8) -> None:
    super().__init__()
    self.attension = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1, padding=1//2, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=in_channels//reduction, out_channels=in_channels, kernel_size=1, padding=1//2, bias=True),
      nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor):
    B, C, _, _ = x.shape
    covariance = covarianceMatrix(x)
    covariance_sqrt = eigenDecomposition(covariance, 5)
    covariance_sum = torch.mean(covariance_sqrt, dim=1).reshape(B, C, 1, 1)
    return torch.mul(x, self.attension(covariance_sum))

class NonLocalEmbeddedGaussianBlock2D(nn.Module):
  def __init__(self, in_channels: int, inter_channels: int, use_bn: bool = True) -> None:
    super().__init__()
    self.inter_channels = inter_channels
    self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, padding=0, bias=True)
    self.theta = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, padding=0, bias=True)
    self.phi = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, padding=0, bias=True)
    self.W = nn.Sequential(
      nn.Conv2d(in_channels=inter_channels, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
      nn.BatchNorm2d(num_features=in_channels) if use_bn else nn.Identity()
    )

  def forward(self, x: torch.Tensor):
    B, _, H, W = x.shape
    g_x = self.g(x).reshape(B, self.inter_channels, -1).permute(0, 2, 1)
    theta_x = self.theta(x).reshape(B, self.inter_channels, -1).permute(0, 2, 1)
    phi_x = self.phi(x).reshape(B, self.inter_channels, -1)
    f = torch.matmul(theta_x, phi_x)
    f_div_C = nn.functional.softmax(f, dim=-1)
    y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous().reshape(B, self.inter_channels, H, W)
    W_y = self.W(y)
    return torch.add(x, W_y)

class NonLocalChannelAttension(nn.Module):
  def __init__(self, in_channels: int = 64, inter_channels: int = 32, use_bn: bool = True) -> None:
    super().__init__()
    self.non_local = NonLocalEmbeddedGaussianBlock2D(in_channels=in_channels, inter_channels=inter_channels, use_bn=use_bn)

  def forward(self, x: torch.Tensor):
    _, _, H, W = x.shape
    H_, W_ = int(H / 2), int(W / 2)
    # divide four parts
    non_local_feat = torch.zeros_like(x)
    non_local_feat[:,:,:H_,:W_] = self.non_local(x[:,:,:H_,:W_])
    non_local_feat[:,:,H_:,:W_] = self.non_local(x[:,:,H_:,:W_])
    non_local_feat[:,:,:H_,W_:] = self.non_local(x[:,:,:H_,W_:])
    non_local_feat[:,:,H_:,W_:] = self.non_local(x[:,:,H_:,W_:])
    return non_local_feat

class RedisualBlock(nn.Module):
  def __init__(self, features: int, kernel_size: int) -> None:
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2, bias=True),
    )

  def forward(self, x: torch.Tensor):
    return torch.add(x, self.layers(x))

class UpscaleBlock(nn.Module):
  def __init__(self, scale: int, f: int = 3, n: int = 64) -> None:
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(n, (scale ** 2) * n, kernel_size=f, padding=f//2, bias=True),
      nn.PixelShuffle(scale),
    )

  def forward(self, x):
    return self.block(x)

class LocalSourceResidualAttentionGroup(nn.Module):
  def __init__(self, features: int, reduction: int, kernel_size: int, blocks: int) -> None:
    super().__init__()
    self.layers = nn.Sequential(*([
      RedisualBlock(features=features, kernel_size=kernel_size) for _ in range(blocks)
    ] + [
      SecondOrderChannelAttension(in_channels=features, reduction=reduction),
      # for output
      nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
    ]))

  def forward(self, x: torch.Tensor):
    return torch.add(x, self.layers(x))

class SAN(nn.Module):
  def __init__(self, scale: int, c: int = 3, f = 3, n: int = 64, g: int = 20, l: int = 10) -> None:
    super().__init__()
    # input convolution
    self.input = nn.Conv2d(in_channels=c, out_channels=n, kernel_size=f, padding=f//2, bias=True)
    # Non Local Attension
    self.non_local = NonLocalChannelAttension(in_channels=n, inter_channels=n//8, use_bn=False)
    # Residual Group
    self.gamma = nn.Parameter(torch.zeros(1))
    self.redisual_groups = nn.ModuleList([
      LocalSourceResidualAttentionGroup(features=n, reduction=8, kernel_size=f, blocks=l) for _ in range(g)
    ])
    # Upscale Blocks
    self.upscale = nn.Sequential(
      *[UpscaleBlock(scale=2, f=f, n=n) for _ in range(int(log(scale, 2)))]
    )
    # Output Layer
    self.output = nn.Conv2d(in_channels=n, out_channels=c, kernel_size=f, padding=f//2, bias=True)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.1 * sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels)))
        if m.bias is not None: m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.zero_()
        if m.bias is not None: m.bias.data.zero_()

  def forward(self, x: torch.Tensor):
    x = self.input(x)
    input = x
    x = self.non_local(x)
    residual = x
    for group in self.redisual_groups:
      x = group(x) + self.gamma * residual
    x = self.non_local(x) + input
    x = self.upscale(x)
    x = self.output(x)
    return x

if __name__ == '__main__':
  print(eigenDecomposition(covarianceMatrix(torch.randn((5, 3, 10, 10))), 5))
