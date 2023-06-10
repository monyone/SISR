import os
from pathlib import Path
from math import log10

from typing import cast

import torch
from torch import cuda
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Trainer:
  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.modules.loss._Loss, seed: int | None, train_loader: DataLoader, test_loader: DataLoader) -> None:
    super().__init__()
    self.device: str = 'cuda' if cuda.is_available() else 'cpu'
    self.model: nn.Module = model.to(self.device)
    self.optimizer: optim.Optimizer = optimizer
    self.criterion: nn.modules.loss._Loss = criterion
    self.train_loader: DataLoader = train_loader
    self.test_loader: DataLoader = test_loader
    if seed is None: return
    torch.manual_seed(seed)
    if self.device != 'cuda': return
    torch.cuda.manual_seed(seed)

  def train(self, epoch) -> None:
    self.model.train()
    epoch_loss, epoch_psnr = 0, 0
    for batch in self.train_loader:
      batch = list(map(lambda n: n.to(self.device), batch))
      highres, lowres = Variable(batch[0]), Variable(batch[1])

      self.optimizer.zero_grad()
      prediction = self.model(lowres)
      loss = self.criterion(prediction, highres)
      epoch_loss += cast(float, loss.data)
      epoch_psnr += 10 * log10(1 / cast(float, loss.data))

      loss.backward()
      self.optimizer.step()
    print('[epoch:{}, train]: Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch, epoch_loss / len(self.train_loader), epoch_psnr / len(self.train_loader)))

  def test(self, epoch: int) -> None:
    self.model.eval()
    test_loss, test_psnr = 0, 0
    with torch.no_grad():
      for batch in self.test_loader:
        batch = list(map(lambda n: n.to(self.device), batch))
        highres, lowres = Variable(batch[0]), Variable(batch[1])

        prediction = self.model(lowres)
        loss = cast(float, self.criterion(prediction, highres).data)
        test_loss += loss
        test_psnr += 10 * log10(1 / loss)
    print("[epoch:{}, validate] Loss: {:.4f}, PSNR: {:.4f} dB".format(epoch, test_loss / len(self.test_loader), test_psnr / len(self.test_loader)))

  def run(self, epochs: int, save_dir: Path = Path('./'), save_prefix: str = 'result') -> None:
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
      self.train(epoch=epoch)
      self.test(epoch=epoch)
      torch.save(self.model.state_dict(), save_dir / f'{save_prefix}_{epoch}.pth')
