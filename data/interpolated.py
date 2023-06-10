from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms.functional as F
import torchvision.transforms as T

from pathlib import Path
from glob import glob

from typing import Any

class InterpolatedImageDataset(Dataset):
  def __init__(self, path: str, crop: int | None = None, scale: int = 2, interpolation: T.InterpolationMode = T.InterpolationMode.BICUBIC) -> None:
    super().__init__()
    self.paths = list(glob(path))
    self.crop = crop
    self.scale = scale
    self.interpolation = interpolation

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index) -> Any:
    path = self.paths[index % len(self)]
    hires = io.read_image(path) / 255
    if self.crop is not None: hires = T.RandomCrop(self.crop)(hires)

    lowres = F.resize(
      img=F.resize(
        img=hires,
        antialias=True,
        size=(tuple(map(lambda n: n // self.scale, hires.size()[1:3]))),
        interpolation=self.interpolation
      ),
      antialias=True,
      size=hires.size()[1:3],
      interpolation=self.interpolation
    )

    return hires, lowres
