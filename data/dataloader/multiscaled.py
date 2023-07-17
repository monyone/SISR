import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms.functional as F
import torchvision.transforms as T

from math import log
from glob import glob

from typing import Any

from ..degradation.real_esrgan import degradation_real_esrgan
from ..degradation.bsrgan import degradation_bsrgan
from ..degradation.jpeg import degradation_jpeg

def _degrade(image: torch.Tensor, scale_factor: int, interpolation: T.InterpolationMode, distort: str | None = None):
  if distort == 'Real-ESRGAN':
    return degradation_real_esrgan(image=image, scale_factor=scale_factor)
  elif distort == 'BSRGAN':
    return degradation_bsrgan(image=image, scale_factor=scale_factor)
  elif distort == 'JPEG':
    return degradation_jpeg(image=image, scale_factor=scale_factor, interpolation=interpolation)
  else:
    return F.resize(img=image, antialias=True, size=(tuple(map(lambda n: n // scale_factor, image.size()[1:3]))), interpolation=interpolation)

class MultiScaledImageDataset(Dataset):
  def __init__(self, path: str or list[str], crop: int | None = None, scale: int = 2, interpolation: T.InterpolationMode = T.InterpolationMode.BICUBIC, distort: str | None = None, y_only: bool = False, dividable = False) -> None:
    super().__init__()
    self.paths: list[str] = sum(map(list, map(glob, path if type(path) is list else [path])), [])
    self.crop = crop
    self.scale = scale
    self.interpolation = interpolation
    self.distort = distort
    self.y_only = y_only
    self.dividable = dividable

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index) -> Any:
    path = self.paths[index % len(self)]
    hires = io.read_image(path, mode=io.ImageReadMode.RGB) / 255
    if tuple(map(lambda n: n // self.scale * self.scale, hires.size()[1:3])) != tuple(hires.size()[1:3]) and self.dividable:
      hires = F.resize(img=hires, antialias=True, size=(tuple(map(lambda n: n // self.scale * self.scale, hires.size()[1:3]))), interpolation=self.interpolation)
    if self.crop is not None: hires = T.RandomCrop(self.crop)(hires)

    lowres = [_degrade(hires, scale_factor=(2 ** (idx + 1)), interpolation=self.interpolation, distort=self.distort) for idx in range(int(log(self.scale, 2)))]

    if self.y_only:
      hires = torch.unsqueeze(input=((16 + (64.738 * hires[0, :, :] + 129.057 * hires[1, :, :] + 25.064 * hires[2, :, :])) / 255), dim=0)
      lowres = [torch.unsqueeze(input=((16 + (64.738 * low[0, :, :] + 129.057 * low[1, :, :] + 25.064 * low[2, :, :])) / 255), dim=0) for low in lowres]

    return list(reversed([hires, *lowres[:-1]])), lowres[-1]
