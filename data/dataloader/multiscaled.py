import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms.functional as F
import torchvision.transforms as T

import random
from math import log
from glob import glob

from typing import Any

class MultiScaledImageDataset(Dataset):
  def __init__(self, path: str or list[str], crop: int | None = None, scale: int = 2, interpolation: T.InterpolationMode = T.InterpolationMode.BICUBIC, distort: bool = False, y_only: bool = False, dividable = False) -> None:
    super().__init__()
    self.paths = sum(map(list, map(glob, path if type(path) is list else [path])), [])
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
    lowres = [F.resize(img=hires, antialias=True, size=(tuple(map(lambda n: n // (2 ** (idx + 1)), hires.size()[1:3]))), interpolation=self.interpolation) for idx in range(int(log(self.scale, 2)))]

    if self.distort and random.random() < 0.25:
      lowres = io.decode_jpeg(io.encode_jpeg((lowres * 255).clamp_(0, 255).to(torch.uint8), quality=random.randrange(70, 95)), io.ImageReadMode.RGB) / 255

    if self.y_only:
      hires = torch.unsqueeze(input=((16 + (64.738 * hires[0, :, :] + 129.057 * hires[1, :, :] + 25.064 * hires[2, :, :])) / 255), dim=0)
      lowres = [torch.unsqueeze(input=((16 + (64.738 * low[0, :, :] + 129.057 * low[1, :, :] + 25.064 * low[2, :, :])) / 255), dim=0) for low in lowres]

    return list(reversed([hires, *lowres[:-1]])), lowres[-1]