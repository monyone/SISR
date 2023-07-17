import torch
import torch.nn as nn

from torchvision.transforms import Normalize
from torchvision.models.vgg import vgg19, VGG19_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from typing import cast
from math import log10

from ..handler import Handler

def if_y_then_gray(tensor: torch.Tensor):
  _, c, _, _ = tensor.size()
  if c != 1: return tensor
  return tensor.repeat_interleave(repeats=3, dim=1)

class VGGLoss(nn.Module):
  def __init__(self, weights: dict[int, float]):
    super().__init__()
    self.weights = weights
    self.vgg_net = create_feature_extractor(vgg19(weights=VGG19_Weights.IMAGENET1K_V1), return_nodes={f'features.{layer}': f'{layer}' for layer in weights}).eval()
    for param in self.vgg_net.parameters(): param.requires_grad = False
    self.criterion = nn.MSELoss()
    self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  def forward(self, sr, target):
    sr = self.vgg_net(self.normalize(if_y_then_gray(sr)))
    target = self.vgg_net(self.normalize(if_y_then_gray(target)))
    return sum(self.criterion(sr[f'{layer}'], target[f'{layer}']) * weight for layer, weight in self.weights.items())

class RealESRGANGeneratorHandler(Handler):
  def __init__(self, model: nn.Module, handler: Handler | None = None):
    super().__init__()
    self.model = model
    self.handler = handler
    self.pixel_loss = nn.L1Loss()
    self.mse_loss = nn.MSELoss()
    self.content_loss = VGGLoss(weights={2: 0.1, 7: 0.1, 16: 1, 25: 1, 34: 1})

  def to(self, device: str) -> Handler:
    self.content_loss.to(device)
    return self

  def train(self, input, target):
    if self.handler:
      sr, _ = self.handler.train(input, target)
    else:
      sr = self.model(input)

    return sr, self.pixel_loss(sr, target) + self.content_loss(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      if self.handler:
        return self.handler.statistics(input, target)
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.mse_loss(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def test(self, input):
    return self.model(input).clamp_(0, 1)

class RealESRGANDiscriminatorHandler(Handler):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
    self.criterion = nn.BCEWithLogitsLoss()

  def train(self, fake, real):
    d_fake = self.model(fake.detach())
    d_real = self.model(real.detach())
    g_fake = self.model(fake)
    g_real = d_real.detach()
    d_loss = (self.criterion(d_fake - torch.mean(d_real, dim=0), torch.zeros_like(d_fake)) + self.criterion(d_real - torch.mean(d_fake, dim=0), torch.ones_like(d_real))) / 2
    g_loss = (self.criterion(g_fake - torch.mean(g_real, dim=0), torch.ones_like(g_fake)) + self.criterion(g_real - torch.mean(g_fake, dim=0), torch.zeros_like(g_real))) / 2
    return d_loss.item(), d_loss, g_loss * 0.1

  def statistics(self, fake, real):
    with torch.no_grad():
      fake = self.model(fake)
      real = self.model(real)
      loss = self.criterion(fake, torch.zeros_like(fake)) + self.criterion(real, torch.ones_like(real)).item()
      return loss

  def test(self, input):
    return self.model(input).clamp_(0, 1)
