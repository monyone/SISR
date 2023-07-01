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

from models.handler import Handler

class GeneratorTrainer:
  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, handler: Handler, state: str | None, seed: int | None, train_loader: DataLoader, test_loader: DataLoader) -> None:
    super().__init__()
    self.device: str = 'cuda' if cuda.is_available() else 'cpu'
    self.model: nn.Module = model.to(self.device)
    if state is not None: self.model.load_state_dict(torch.load(str(state), map_location=self.device))
    self.optimizer: optim.Optimizer = optimizer
    self.handler: Handler = handler.to(self.device)
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
      highres, lowres = batch
      highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
      lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

      self.optimizer.zero_grad()
      _, loss = self.handler.train(lowres, highres)
      loss.backward()
      self.optimizer.step()

      epoch_loss += cast(float, loss.item())
      epoch_psnr += 10 * log10(1 / cast(float, loss.item())) if loss.item() != 0 else 100

    print('[epoch:{}, train]: Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch, epoch_loss / len(self.train_loader), epoch_psnr / len(self.train_loader)))

  def test(self, epoch: int) -> None:
    self.model.eval()
    test_loss, test_psnr = 0, 0
    with torch.no_grad():
      for batch in self.test_loader:
        highres, lowres = batch
        highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
        lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

        loss, psnr = self.handler.statistics(lowres, highres)
        test_loss += loss
        test_psnr += psnr
    print("[epoch:{}, validate] Loss: {:.4f}, PSNR: {:.4f} dB".format(epoch, test_loss / len(self.test_loader), test_psnr / len(self.test_loader)))

  def run(self, epochs: int, save_dir: Path = Path('./'), save_prefix: str = 'result') -> None:
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
      self.train(epoch=epoch)
      self.handler.step(epoch, epochs)

      self.test(epoch=epoch)
      torch.save(self.model.state_dict(), save_dir / f'{save_prefix}_{epoch}.pth')
