import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms.functional as F
import torchvision.transforms as T

from pathlib import Path
from glob import glob

from typing import Any


class PairwiseDataSet(Dataset):
  def __init__(self, path: str or list[str], y_only: bool = False) -> None:
    super().__init__()
    self.paths: list[str] = sum(map(list, map(glob, path if type(path) is list else [path])), [])
    self.y_only = y_only

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index) -> Any:
    hires_path = self.paths[index % len(self)]
    hires = io.read_image(hires_path, mode=io.ImageReadMode.RGB) / 255
    lowres_path_obj = Path(hires_path)
    lowres_path = lowres_path_obj.parent / (lowres_path_obj.stem.removesuffix('HR') + 'LR' + lowres_path_obj.suffix)
    lowres = io.read_image(str(lowres_path), mode=io.ImageReadMode.RGB) / 255

    if self.y_only:
      hires = torch.unsqueeze(input=((16 + (64.738 * hires[0, :, :] + 129.057 * hires[1, :, :] + 25.064 * hires[2, :, :])) / 255), dim=0)
      lowres = torch.unsqueeze(input=((16 + (64.738 * lowres[0, :, :] + 129.057 * lowres[1, :, :] + 25.064 * lowres[2, :, :])) / 255), dim=0)

    return hires, lowres
