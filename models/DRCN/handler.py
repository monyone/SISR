import torch
import torch.nn as nn

from typing import cast
from math import log10

from ..handler import Handler
from .model import DRCN

class DRCNHandler(Handler):
  def __init__(self, model: DRCN):
    super().__init__()
    self.model = model
    self.alpha = 1
    self.criterion = nn.MSELoss()
    self.mse = nn.MSELoss()

  def loss(self, input, target):
    average, sr = self.model(input)
    loss1 = self.criterion(average, target)
    loss2 = self.criterion(sr, target)
    loss = self.alpha * loss1 + (1 - self.alpha) * loss2
    return loss

  def statistics(self, input, target):
    with torch.no_grad():
      _, sr = self.model(input)
      sr = sr.clamp_(0, 1)
      loss = cast(float, self.mse(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def step(self, epoch: int, epochs: int) -> None:
    self.alpha = max(0, min(1, 1 - (epoch * 4 / epochs)))

  def test(self, input):
    _, sr = self.model(input).clamp_(0, 1)
    return sr
