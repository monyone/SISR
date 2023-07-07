import torch
import torch.nn as nn

from torchvision.transforms import Normalize
from torchvision.models.vgg import vgg19, VGG19_Weights

from typing import cast
from math import log10

from ..handler import Handler

def if_y_then_gray(tensor: torch.Tensor):
  _, c, _, _ = tensor.size()
  if c != 1: return tensor
  return tensor.repeat_interleave(repeats=3, dim=1)

class TextureLoss(nn.Module):
  def __init__(self, layer: int):
    super().__init__()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.vgg_net = nn.Sequential(*list(vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features)[:layer + 1]).to(device=device).eval()
    for param in self.vgg_net.parameters(): param.requires_grad = False
    self.criterion = nn.MSELoss()
    self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  @staticmethod
  def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    return (torch.bmm(features, features.transpose(1, 2))).div(c * h * w)

  def forward(self, sr, target):
    sr = TextureLoss.gram_matrix(self.vgg_net(self.normalize(if_y_then_gray(sr))))
    target = TextureLoss.gram_matrix(self.vgg_net(self.normalize(if_y_then_gray(target))))
    return self.criterion(sr, target)

class TSRNHandler(Handler):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
    self.mse_loss = nn.MSELoss()
    self.alpha = 1
    self.texture_2_2_loss = TextureLoss(layer=8)
    self.texture_3_4_loss = TextureLoss(layer=17)
    self.texture_4_4_loss = TextureLoss(layer=26)
    self.texture_5_4_loss = TextureLoss(layer=35)

  def to(self, device: str) -> Handler:
    self.texture_2_2_loss.to(device=device)
    self.texture_3_4_loss.to(device=device)
    self.texture_4_4_loss.to(device=device)
    self.texture_5_4_loss.to(device=device)
    return self

  def train(self, input, target):
    sr = self.model(input)

    mse_loss = self.mse_loss(sr, target)
    texture_loss = self.texture_2_2_loss(sr, target) + self.texture_3_4_loss(sr, target) + self.texture_4_4_loss(sr, target) + self.texture_5_4_loss(sr, target)
    return sr, self.alpha * mse_loss + (1 - self.alpha) * texture_loss

  def statistics(self, input, target):
    with torch.no_grad():
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.mse_loss(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def step(self, epoch: int, epochs: int) -> None:
    self.alpha = 1 if epoch <= 10 else 0

  def test(self, input):
    return self.model(input).clamp_(0, 1)
