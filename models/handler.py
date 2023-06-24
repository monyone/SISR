import torch
import torch.nn as nn

from typing import cast
from math import log10

from abc import ABC, abstractmethod

class Handler(ABC):
  def __init__(self):
    self.mseloss = nn.MSELoss()

  @abstractmethod
  def loss(self, input, target):
    pass

  @abstractmethod
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

  def loss(self, input, target):
    sr = self.model(input)
    return self.criterion(sr, target)

  def statistics(self, input, target):
    with torch.no_grad():
      sr = self.model(input)
      loss = cast(float, self.criterion(sr, target).item())
      return loss, 10 * log10(1 / loss) if loss != 0 else 100

  def step(self, epoch: int, epochs: int) -> None:
    pass

  def test(self, input):
    return self.model(input)
