#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train.trainer import Trainer
from data.interpolated import InterpolatedImageDataset
from data.noninterpolated import NonInterpolatedImageDataset

from models.SRCNN.model import SRCNN
from models.VDSR.model import VDSR
from models.FSRCNN.model import FSRCNN
from models.DRCN.model import DRCN
from models.ESPCN.model import ESPCN

# PREFERENCE
crop = 128
train_path = './data/DIV2K/DIV2K_train_HR/*'
validate_path = './data/DIV2K/DIV2K_valid_HR/*'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--epochs', type=int, default=10, help="Number of Epochs")
  parser.add_argument('--model', type=str, required=True, help="SISR model")
  parser.add_argument('--scale', type=int, default=2, help="Upscaling scale factor")
  parser.add_argument('--batch', type=int, default=1, help="Batch size")

  args = parser.parse_args()

  models = {
    'SRCNN': tuple([SRCNN(), InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale), InterpolatedImageDataset(path=validate_path, crop=crop, scale=args.scale), nn.MSELoss(), 0.0001]),
    'VDSR': tuple([VDSR(), InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale), InterpolatedImageDataset(path=validate_path, crop=crop, scale=args.scale), nn.MSELoss(), 0.0001]),
    'FSRCNN': tuple([FSRCNN(scale=args.scale), NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale), NonInterpolatedImageDataset(path=validate_path, crop=crop, scale=args.scale), nn.MSELoss(), 0.0001]),
    'DRCN': tuple([DRCN(), InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale), InterpolatedImageDataset(path=validate_path, crop=crop, scale=args.scale), nn.MSELoss(), 0.0001]),
    'ESPCN': tuple([ESPCN(scale=args.scale), NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale), NonInterpolatedImageDataset(path=validate_path, crop=crop, scale=args.scale), nn.MSELoss(), 0.0001]),
  }

  model, train_set, validation_set, loss, lr = models[args.model]
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.ConstantLR(optimizer, last_epoch=-1)
  if args.model == 'VDSR':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 4, gamma=0.1)

  train_loader = DataLoader(dataset=train_set, batch_size=args.batch, shuffle=True)
  validation_loader = DataLoader(dataset=validation_set, batch_size=args.batch, shuffle=True)

  trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=loss, seed=None, train_loader=train_loader, test_loader=validation_loader)
  trainer.run(epochs=args.epochs, save_dir=Path(f'./result/{args.model}_x{args.scale}'), save_prefix='state')
