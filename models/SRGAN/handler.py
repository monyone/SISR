import torch
import torch.nn as nn

from torchvision.models.vgg import vgg19, VGG19_Weights

from typing import cast
from math import log10

from ..handler import Handler

class VGGLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg_net = nn.Sequential(*list(vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features)[:36]).eval()
    for param in self.vgg_net.parameters():
      param.requires_grad = False
    self.criterion = nn.MSELoss()
    self.register_buffer(name='vgg_mean', tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False))
    self.register_buffer(name='vgg_std', tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False))

  def forward(self, sr, target):
    sr = sr.sub(self.vgg_mean).div(self.vgg_std)
    target = target.sub(self.vgg_mean).div(self.vgg_std)
    return self.criterion(self.vgg_net(sr), self.vgg_net(target))

class SRGANGeneratorHandler(Handler):
  def __init__(self, model: nn.Module, handler: Handler | None = None):
    super().__init__()
    self.model = model
    self.handler = handler
    self.mse_loss = nn.MSELoss()
    self.content_loss = VGGLoss()

  def to(self, device: str) -> Handler:
    self.content_loss.to(device=device)
    return self

  def train(self, input, target):
    if self.handler:
      sr, _ = self.handler.train(input, target)
    else:
      sr = self.model(input)
    return sr, self.content_loss(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      if self.handler:
        return self.handler.statistics(input, target)
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.mse_loss(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def test(self, input):
    return self.model(input).clamp_(0, 1)

class SRGANDiscriminatorHandler(Handler):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
    self.criterion = nn.BCELoss()

  def train(self, fake, real):
    d_fake = self.model(fake.detach())
    d_real = self.model(real.detach())
    g_fake = self.model(fake)
    d_loss = self.criterion(d_fake, torch.zeros_like(d_fake)) + self.criterion(d_real, torch.ones_like(d_real))
    g_loss = self.criterion(g_fake, torch.ones_like(g_fake))
    return d_loss.item(), d_loss, g_loss * 0.001

  def statistics(self, fake, real):
    with torch.no_grad():
      fake = self.model(fake)
      real = self.model(real)
      loss = self.criterion(fake, torch.zeros_like(fake)) + self.criterion(real, torch.ones_like(real)).item()
      return loss

  def test(self, input):
    return self.model(input).clamp_(0, 1)