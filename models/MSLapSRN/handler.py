import torch
import torch.nn as nn

from typing import cast
from math import log10

from ..handler import Handler
from .model import MSLapSRN

class MSLapSRNHandler(Handler):
  def __init__(self, model: MSLapSRN):
    super().__init__()
    self.model = model
    self.alpha = 1
    self.criterion = nn.MSELoss()
    self.mse = nn.MSELoss()

  def loss(self, input, targets):
    loss = sum(self.criterion(sr, tg) for sr, tg in zip(self.model(input), targets))
    return loss

  def statistics(self, input, targets):
    with torch.no_grad():
      outputs = self.model(input)
      sr = outputs[-1].clamp_(0, 1)
      loss = cast(float, self.mse(sr, targets[-1]).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def step(self, epoch: int, epochs: int) -> None:
    pass

  def test(self, input):
    outputs = self.model(input).clamp_(0, 1)
    return outputs[-1]
