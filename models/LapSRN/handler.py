import torch
import torch.nn as nn

from typing import cast
from math import log10

from ..handler import Handler
from .model import LapSRN

class LapSRNHandler(Handler):
  def __init__(self, model: LapSRN):
    super().__init__()
    self.model = model
    self.alpha = 1
    self.criterion = nn.MSELoss()
    self.mse = nn.MSELoss()

  def to(self, device) -> Handler:
    return self

  def train(self, input, target):
    sr = self.model(input)
    loss = sum(self.criterion(sr, tg) for sr, tg in zip(sr, target))
    return sr[-1], loss

  def statistics(self, input, targets):
    with torch.no_grad():
      outputs = self.model(input)
      sr = outputs[-1].clamp_(0, 1)
      loss = cast(float, self.mse(sr, targets[-1]).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def test(self, input):
    outputs = self.model(input).clamp_(0, 1)
    return outputs[-1]
