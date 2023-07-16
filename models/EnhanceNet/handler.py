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

class TextureLoss(nn.Module):
  def __init__(self, weights: dict[int, float]):
    super().__init__()
    self.weights = weights
    self.vgg_net = create_feature_extractor(vgg19(weights=VGG19_Weights.IMAGENET1K_V1), return_nodes={f'features.{layer}': f'{layer}' for layer in weights}).eval()
    for param in self.vgg_net.parameters(): param.requires_grad = False
    self.criterion = nn.MSELoss()
    self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  @staticmethod
  def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    return (torch.bmm(features, features.transpose(1, 2))).div(c * h * w)

  @staticmethod
  def into_patch(tensor: torch.Tensor, patch: int) -> torch.Tensor:
    _, c, _, _ = tensor.size()
    return tensor.unfold(2, size=patch, step=patch).unfold(3, size=patch, step=patch).permute([0, 2, 3, 1, 4, 5]).reshape(-1, c, patch, patch)

  def forward(self, sr, target):
    sr = self.vgg_net(self.normalize(if_y_then_gray(sr)))
    target = self.vgg_net(self.normalize(if_y_then_gray(target)))
    return sum(self.criterion(TextureLoss.gram_matrix(TextureLoss.into_patch(sr[f'{layer}'], 16)), TextureLoss.gram_matrix(TextureLoss.into_patch(target[f'{layer}'], 16))) * weight for layer, weight in self.weights.items())

class EnhanceNetGeneratorHandler(Handler):
  def __init__(self, model: nn.Module, handler: Handler | None = None):
    super().__init__()
    self.model = model
    self.handler = handler
    self.mse_loss = nn.MSELoss()
    self.feature_loss = VGGLoss(weights={9: 0.2, 36: 0.02})
    self.texture_loss = TextureLoss(weights={0: 10, 5: 100/3, 10: 100/3})

  def to(self, device: str) -> Handler:
    self.feature_loss.to(device=device)
    self.texture_loss.to(device=device)
    return self

  def train(self, input, target):
    import time
    begin = time.monotonic_ns()
    if self.handler:
      sr, _ = self.handler.train(input, target)
    else:
      sr = self.model(input)
    end = time.monotonic_ns()
    print('g inf: gen', (end - begin) / 1000000)

    # texture_loss = (3e-7 * self.texture_1_1_loss(sr, target) + 1e-6 * self.texture_2_1_loss(sr, target) + 1e-6 * self.texture_3_1_loss(sr, target)) # Squred Sum Total Loss (in paper, assume 128x128 train, vgg19 mean activation required)
    return sr, self.feature_loss(sr, target) + self.texture_loss(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      if self.handler:
        return self.handler.statistics(input, target)
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.mse_loss(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def test(self, input):
    return self.model(input).clamp_(0, 1)

class EnhanceNetDiscriminatorHandler(Handler):
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
    return d_loss.item(), d_loss, 2 * g_loss

  def statistics(self, fake, real):
    with torch.no_grad():
      fake = self.model(fake)
      real = self.model(real)
      loss = self.criterion(fake, torch.zeros_like(fake)) + self.criterion(real, torch.ones_like(real)).item()
      return loss

  def test(self, input):
    return self.model(input).clamp_(0, 1)
