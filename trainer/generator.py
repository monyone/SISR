import os
from pathlib import Path
from math import log10
import time
from datetime import timedelta

from typing import cast

import torch
from torch import cuda
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.handler import Handler

class GeneratorTrainer:
  def __init__(self, model: nn.Module, optimizer: optim.Optimizer, handler: Handler, state: str | None, train_loader: DataLoader, test_loader: DataLoader, seed: int | None = None, use_amp: bool = True) -> None:
    super().__init__()
    self.device: str = 'cuda' if cuda.is_available() else 'cpu'
    self.model: nn.Module = model.to(self.device)
    if state is not None: self.model.load_state_dict(torch.load(str(state), map_location=self.device))
    self.optimizer: optim.Optimizer = optimizer
    self.handler: Handler = handler.to(self.device)
    self.train_loader: DataLoader = train_loader
    self.test_loader: DataLoader = test_loader
    self.use_amp = use_amp
    self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if seed is None: return
    torch.manual_seed(seed)
    if self.device != 'cuda': return
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = seed is None
    torch.backends.cudnn.deterministic = seed is not None

  def train(self, epoch) -> None:
    elapsed = 0

    self.model.train()
    epoch_loss, epoch_psnr = 0, 0
    for batch in self.train_loader:
      begin = time.monotonic_ns()

      highres, lowres = batch
      highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
      lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

      self.optimizer.zero_grad()
      with torch.autocast(device_type=self.device, enabled=self.use_amp):
        _, loss = self.handler.train(lowres, highres)
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()

      epoch_loss += cast(float, loss.item())
      epoch_psnr += 10 * log10(1 / cast(float, loss.item())) if loss.item() != 0 else 100

      end = time.monotonic_ns()
      print("\r", f'epoch:{epoch}', f'{elapsed}/{len(self.train_loader)}', f'ETA:{timedelta(seconds=((len(self.train_loader) - elapsed) * ((end - begin) / 1000000000)))}', end="")
      elapsed += 1

    print()
    print('[epoch:{}, train]: Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch, epoch_loss / len(self.train_loader), epoch_psnr / len(self.train_loader)))

  def test(self, epoch: int) -> None:
    self.model.eval()
    test_loss, test_psnr, test_inference_time = 0, 0, 0
    with torch.no_grad():
      for batch in self.test_loader:
        highres, lowres = batch
        highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
        lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

        begin = time.monotonic_ns()
        loss, psnr = self.handler.statistics(lowres, highres)
        end = time.monotonic_ns()

        test_loss += loss
        test_psnr += psnr
        test_inference_time += (end - begin) / 1000000
    print("[epoch:{}, validate] Loss: {:.4f}, PSNR: {:.4f} dB, Time: {:.4f} ms".format(epoch, test_loss / len(self.test_loader), test_psnr / len(self.test_loader), test_inference_time / len(self.test_loader)))

  def run(self, epochs: int, save_dir: Path = Path('./'), save_prefix: str = 'result') -> None:
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
      self.train(epoch=epoch)
      self.handler.step(epoch, epochs)

      self.test(epoch=epoch)
      torch.save(self.model.state_dict(), save_dir / f'{save_prefix}_{epoch}.pth')
