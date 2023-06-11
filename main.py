#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from torch import cuda
from torch.utils.data import DataLoader
import torchvision.utils as utils

from data.interpolated import InterpolatedImageDataset
from data.noninterpolated import NonInterpolatedImageDataset

from models.SRCNN.model import SRCNN
from models.VDSR.model import VDSR
from models.FSRCNN.model import FSRCNN

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--image', type=Path, required=True, help="Input Image Path")
  parser.add_argument('--model', type=str, required=True, help="SISR model")
  parser.add_argument('--crop', type=int, help="Crop size")
  parser.add_argument('--scale', type=int, default=1, help="Downscale factor")
  parser.add_argument('--state', type=Path, required=True, help="Trained state Path")
  args = parser.parse_args()

  models = {
    'SRCNN': tuple([SRCNN(), InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale)]),
    'VDSR': tuple([VDSR(), InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale)]),
    'FSRCNN': tuple([FSRCNN(scale=args.scale), NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale)])
  }

  device: str = 'cuda' if cuda.is_available() else 'cpu'
  model, dataset = models[args.model]
  model.load_state_dict(torch.load(str(args.state), map_location=device))
  dataloader = DataLoader(dataset=dataset, batch_size=1)

  model = model.to(device)

  model.eval()
  with torch.no_grad():
    for _, lowres in dataloader:
      lowres = lowres.to(device)
      upscaled = model(lowres)
      #utils.save_image(lowres, str(f'./{args.image.stem}_lr{args.image.suffix}'), nrow=1)
      #utils.save_image(_, str(f'./{args.image.stem}_hr{args.image.suffix}'), nrow=1)
      #utils.save_image(upscaled - lowres, str(f'./{args.image.stem}_sr_d{args.image.suffix}'), nrow=1)
      utils.save_image(upscaled, str(f'./{args.image.stem}_sr{args.image.suffix}'), nrow=1)
