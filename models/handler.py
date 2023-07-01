import torch
import torch.nn as nn

from typing import cast
from math import log10

from abc import ABC, abstractmethod

class Handler(ABC):
  def __init__(self):
    pass

  def to(self, device: str):
    return self

  @abstractmethod
  def train(self, input, target):
    pass

  def step(self, epoch: int, epochs: int):
    pass

  @abstractmethod
  def statistics(self, input, target):
    pass

  @abstractmethod
  def test(self, input):
    pass

class DefaultHandler(Handler):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model
    self.criterion = nn.MSELoss()

  def to(self, device: str) -> Handler:
    return self

  def train(self, input, target):
    sr = self.model(input)
    return sr, self.criterion(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      sr = self.model(input).clamp_(0, 1)
      loss = cast(float, self.criterion(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def test(self, input):
    return self.model(input).clamp_(0, 1)
