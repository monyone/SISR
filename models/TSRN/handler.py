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

  def forward(self, sr, target):
    sr = self.vgg_net(self.normalize(if_y_then_gray(sr)))
    target = self.vgg_net(self.normalize(if_y_then_gray(target)))
    return sum(self.criterion(TextureLoss.gram_matrix(sr[f'{layer}']), TextureLoss.gram_matrix(target[f'{layer}'])) * weight for layer, weight in self.weights.items())

class TSRNHandler(Handler):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
    self.mse_loss = nn.MSELoss()
    self.alpha = 1
    self.texture_loss = TextureLoss(weights={8: 1, 17: 1, 26: 1, 35: 1})

  def to(self, device: str) -> Handler:
    self.texture_loss.to(device=device)
    return self

  def train(self, input, target):
    sr = self.model(input)
    return sr, self.alpha * self.mse_loss(sr, target) + (1 - self.alpha) * self.texture_loss(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.mse_loss(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def step(self, epoch: int, epochs: int) -> None:
    self.alpha = 1 if epoch < 10 else 0

  def test(self, input):
    return self.model(input).clamp_(0, 1)
