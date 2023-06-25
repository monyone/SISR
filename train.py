#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch.optim as optim
from torch.utils.data import DataLoader

from trainer.train import Train
from data.interpolated import InterpolatedImageDataset
from data.noninterpolated import NonInterpolatedImageDataset

from models.handler import DefaultHandler
from models.SRCNN.model import SRCNN
from models.VDSR.model import VDSR
from models.FSRCNN.model import FSRCNN
from models.DRCN.model import DRCN
from models.DRCN.handler import DRCNHandler
from models.ESPCN.model import ESPCN
from models.REDNet.model import REDNet
from models.DRRN.model import DRRN
from models.SRResNet.model import SRResNet

# PREFERENCE
crop = None
#train_path = './data/DIV2K/DIV2K_train_HR_Patches/*'
train_path = ['./data/T91/Patches/*.png']
validate_path = './data/SET5/*'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--epochs', type=int, default=10, help="Number of Epochs")
  parser.add_argument('--model', type=str, required=True, help="SISR model")
  parser.add_argument('--scale', type=int, default=2, help="Upscaling scale factor")
  parser.add_argument('--batch', type=int, default=1, help="Batch size")
  parser.add_argument('--y_only', action='store_true', help="Train y color only")

  args = parser.parse_args()

  models = {
    'SRCNN': tuple([SRCNN(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'VDSR': tuple([VDSR(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'FSRCNN': tuple([FSRCNN(scale=args.scale, c=(1 if args.y_only else 3)), DefaultHandler, NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'DRCN': tuple([DRCN(c=(1 if args.y_only else 3)), DRCNHandler, InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'ESPCN': tuple([ESPCN(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'RED-Net': tuple([REDNet(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'DRRN': tuple([DRRN(c=(1 if args.y_only else 3)), DefaultHandler, InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'SRResNet': tuple([SRResNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultHandler, NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
  }

  model, handler_class, train_set, validation_set, lr = models[args.model]
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ConstantLR(optimizer=optimizer, last_epoch=-1)
  handler = handler_class(model)

  train_loader = DataLoader(dataset=train_set, batch_size=args.batch, num_workers=os.cpu_count(), pin_memory=True)
  validation_loader = DataLoader(dataset=validation_set, num_workers=os.cpu_count(), pin_memory=True)

  train = Train(model=model, optimizer=optimizer, scheduler=scheduler, handler=handler, seed=None, train_loader=train_loader, test_loader=validation_loader)
  train.run(epochs=args.epochs, save_dir=Path(f'./result/{args.model}_x{args.scale}'), save_prefix='state')
