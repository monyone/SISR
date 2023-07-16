import os
from pathlib import Path
from math import log10

from typing import cast

import torch
from torch import cuda
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models.handler import Handler

class GANTrainer:
  def __init__(self, g_model: nn.Module, d_model: nn.Module, g_optimizer: optim.Optimizer, d_optimizer: optim.Optimizer, g_handler: Handler, d_handler: Handler, g_state: str | None, train_loader: DataLoader, test_loader: DataLoader, seed: int | None = None, use_amp: bool = True) -> None:
    super().__init__()
    self.device: str = 'cuda' if cuda.is_available() else 'cpu'
    self.g_model: nn.Module = g_model.to(self.device)
    self.d_model: nn.Module = d_model.to(self.device)
    if g_state is not None: self.g_model.load_state_dict(torch.load(str(g_state), map_location=self.device))
    self.g_optimizer: optim.Optimizer = g_optimizer
    self.d_optimizer: optim.Optimizer = d_optimizer
    self.g_handler: Handler = g_handler.to(self.device)
    self.d_handler: Handler = d_handler.to(self.device)
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

    self.d_model.train()
    self.g_model.train()

    epoch_loss, epoch_psnr = 0, 0
    for batch in self.train_loader:
      highres, lowres = batch
      highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
      lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

      with torch.autocast(device_type=self.device, enabled=self.use_amp):
        sr, content_loss = self.g_handler.train(lowres, highres)

      self.d_optimizer.zero_grad()
      with torch.autocast(device_type=self.device, enabled=self.use_amp):
        _, d_loss, adversarial_loss = self.d_handler.train(sr, highres)

      self.scaler.scale(d_loss).backward()
      self.scaler.step(self.d_optimizer)

      self.g_optimizer.zero_grad()
      g_loss = (content_loss + adversarial_loss)
      self.scaler.scale(g_loss).backward()
      self.scaler.step(self.g_optimizer)

      self.scaler.update()

      epoch_loss += cast(float, g_loss.item())
      epoch_psnr += 10 * log10(1 / cast(float, g_loss.item())) if g_loss.item() != 0 else 100

      print("\r", elapsed, ':', len(self.train_loader), end="")
      elapsed += 1

    print('[epoch:{}, train]: Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch, epoch_loss / len(self.train_loader), epoch_psnr / len(self.train_loader)))

  def test(self, epoch: int) -> None:
    self.g_model.eval()
    self.d_model.eval()
    test_loss, test_psnr = 0, 0
    with torch.no_grad():
      for batch in self.test_loader:
        highres, lowres = batch
        highres = tuple(map(lambda n: n.to(self.device), highres)) if type(highres) is list else highres.to(self.device)
        lowres = tuple(map(lambda n: n.to(self.device), lowres)) if type(lowres) is list else lowres.to(self.device)

        loss, psnr = self.g_handler.statistics(lowres, highres)
        test_loss += loss
        test_psnr += psnr
    print("[epoch:{}, validate] Loss: {:.4f}, PSNR: {:.4f} dB".format(epoch, test_loss / len(self.test_loader), test_psnr / len(self.test_loader)))

  def run(self, epochs: int, save_dir: Path = Path('./'), save_prefix: str = 'result') -> None:
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(epochs):
      self.train(epoch=epoch)
      self.g_handler.step(epoch=epoch, epochs=epochs)
      self.d_handler.step(epoch=epoch, epochs=epochs)

      self.test(epoch=epoch)
      torch.save(self.g_model.state_dict(), save_dir / f'{save_prefix}_{epoch}.pth')
