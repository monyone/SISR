#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from torch import cuda
from torch.utils.data import DataLoader
import torchvision.utils as utils

from data.interpolated import InterpolatedImageDataset
from data.noninterpolated import NonInterpolatedImageDataset
from data.multiscaled import MultiScaledImageDataset

from models.handler import DefaultHandler
from models.SRCNN.model import SRCNN
from models.VDSR.model import VDSR
from models.FSRCNN.model import FSRCNN
from models.DRCN.model import DRCN
from models.DRCN.handler import DRCNHandler
from models.ESPCN.model import ESPCN
from models.REDNet.model import REDNet
from models.DRRN.model import DRRN
from models.LapSRN.model import LapSRN
from models.LapSRN.handler import LapSRNHandler
from models.MSLapSRN.model import MSLapSRN
from models.MSLapSRN.handler import MSLapSRNHandler
from models.EnhanceNet.model import EnhanceNet
from models.SRResNet.model import SRResNet

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--image', type=Path, required=True, help="Input Image Path")
  parser.add_argument('--model', type=str, required=True, help="SISR model")
  parser.add_argument('--crop', type=int, help="Crop size")
  parser.add_argument('--scale', type=int, default=1, help="Downscale factor")
  parser.add_argument('--state', type=Path, required=True, help="Trained state Path")
  parser.add_argument('--y_only', action='store_true', help="Train y color only")
  parser.add_argument('--dividable', action='store_true', help="Input Image adjust to downscale factor")

  args = parser.parse_args()

  models = {
    'SRCNN': tuple([SRCNN(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'VDSR': tuple([VDSR(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'FSRCNN': tuple([FSRCNN(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'DRCN': tuple([DRCN(c=(1 if args.y_only else 3)), DRCNHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'ESPCN': tuple([ESPCN(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'RED-Net': tuple([REDNet(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'DRRN': tuple([DRRN(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'LapSRN': tuple([LapSRN(c=(1 if args.y_only else 3), scale=args.scale), LapSRNHandler, MultiScaledImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'MSLapSRN': tuple([MSLapSRN(c=(1 if args.y_only else 3), scale=args.scale), MSLapSRNHandler, MultiScaledImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'EnhanceNet': tuple([EnhanceNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
    'SRResNet': tuple([SRResNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, y_only=args.y_only, dividable=args.dividable)]),
  }

  device: str = 'cuda' if cuda.is_available() else 'cpu'
  model, handler_class, dataset = models[args.model]
  model.load_state_dict(torch.load(str(args.state), map_location=device))
  dataloader = DataLoader(dataset=dataset, batch_size=1)

  model = model.to(device)
  handler = handler_class(model)

  model.eval()
  with torch.no_grad():
    for _, lowres in dataloader:
      lowres = lowres.to(device)
      upscaled = handler.test(lowres)
      from math import log10
      print(10 * log10(1 / torch.nn.MSELoss()(upscaled, _.to(device)).data))
      #utils.save_image(lowres, str(f'./{args.image.stem}_lr{args.image.suffix}'), nrow=1)
      utils.save_image(_, str(f'./{args.image.stem}_hr{args.image.suffix}'), nrow=1)
      utils.save_image(torch.abs(upscaled - _.to(device)), str(f'./{args.image.stem}_sr_ud{args.image.suffix}'), nrow=1)
      #utils.save_image(torch.abs(upscaled - lowres), str(f'./{args.image.stem}_sr_ld{args.image.suffix}'), nrow=1)
      utils.save_image(upscaled, str(f'./{args.image.stem}_sr{args.image.suffix}'), nrow=1)
