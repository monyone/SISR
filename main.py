#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from torch import cuda
from torch.utils.data import DataLoader
import torchvision.utils as utils

from data.dataloader.interpolated import InterpolatedImageDataset
from data.dataloader.noninterpolated import NonInterpolatedImageDataset
from data.dataloader.multiscaled import MultiScaledImageDataset

from models.handler import DefaultMSEHandler, DefaultMAEHandler
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
from models.SRGAN.model import SRResNet
from models.EDSR.model import EDSR
from models.TSRN.model import TSRN
from models.TSRN.handler import TSRNHandler
from models.ESRGAN.model import RRDBNet
from models.SRDenseNet.model import SRDenseNet
from models.RDN.model import RDN
from models.RealESRGAN.model import RealESRNet

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--image', type=Path, required=True, help="Input Image Path")
  parser.add_argument('--model', type=str, required=True, help="SISR model")
  parser.add_argument('--crop', type=int, help="Crop size")
  parser.add_argument('--scale', type=int, default=1, help="upscale factor")
  parser.add_argument('--state', type=Path, required=True, help="Trained state Path")
  parser.add_argument('--y_only', action='store_true', help="inference Y color only")
  parser.add_argument('--test', action='store_true', help="inference for testing")
  parser.add_argument('--distort', type=str, help="image distortion method using testing")

  args = parser.parse_args()

  models = {
    'SRCNN': tuple([SRCNN(c=(1 if args.y_only else 3)), DefaultMSEHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'FSRCNN': tuple([FSRCNN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'ESPCN': tuple([ESPCN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'VDSR': tuple([VDSR(c=(1 if args.y_only else 3)), DefaultMSEHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'DRCN': tuple([DRCN(c=(1 if args.y_only else 3)), DRCNHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'DRRN': tuple([DRRN(c=(1 if args.y_only else 3)), DefaultMSEHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'RED-Net': tuple([REDNet(c=(1 if args.y_only else 3)), DefaultMSEHandler, InterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'LapSRN': tuple([LapSRN(c=(1 if args.y_only else 3), scale=args.scale), LapSRNHandler, MultiScaledImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'MSLapSRN': tuple([MSLapSRN(c=(1 if args.y_only else 3), scale=args.scale), MSLapSRNHandler, MultiScaledImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'EnhanceNet': tuple([EnhanceNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'SRDenseNet': tuple([SRDenseNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'TSRN': tuple([TSRN(c=(1 if args.y_only else 3), scale=args.scale), TSRNHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'SRResNet': tuple([SRResNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'EDSR': tuple([EDSR(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'RDN': tuple([RDN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'RRDBNet': tuple([RRDBNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
    'Real-ESRNet': tuple([RealESRNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, NonInterpolatedImageDataset(path=str(args.image), crop=args.crop, scale=args.scale, distort=args.distort, y_only=args.y_only, upscale=(not args.test), dividable=args.test)]),
  }

  device: str = 'cuda' if cuda.is_available() else 'cpu'
  model, handler_class, dataset = models[args.model]
  model.load_state_dict(torch.load(str(args.state), map_location=device))
  dataloader = DataLoader(dataset=dataset, batch_size=1)

  model = model.to(device)
  handler = handler_class(model)

  model.eval()
  with torch.no_grad():
    for highres, lowres in dataloader:
      lowres = lowres.to(device)
      highres = highres.to(device)
      upscaled = handler.test(highres if not args.test else lowres)
      utils.save_image(upscaled, str(f'./{args.image.stem}_sr{args.image.suffix}'), nrow=1)
      if args.test:
        from math import log10
        print(10 * log10(1 / torch.nn.MSELoss()(upscaled, highres.to(device)).data))
        utils.save_image(lowres, str(f'./{args.image.stem}_lr{args.image.suffix}'), nrow=1)
        utils.save_image(torch.abs(upscaled - highres.to(device)), str(f'./{args.image.stem}_sr_ud{args.image.suffix}'), nrow=1)
