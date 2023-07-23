#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from trainer.generator import GeneratorTrainer
from trainer.gan import GANTrainer

from data.dataloader.interpolated import InterpolatedImageDataset
from data.dataloader.noninterpolated import NonInterpolatedImageDataset
from data.dataloader.multiscaled import MultiScaledImageDataset
from data.dataloader.pairwise import PairwiseDataSet

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
from models.EnhanceNet.model import EnhanceNet, EnhanceNetDiscriminator
from models.EnhanceNet.handler import EnhanceNetGeneratorHandler, EnhanceNetDiscriminatorHandler
from models.SRGAN.model import SRResNet, SRGAN
from models.SRGAN.handler import SRGANGeneratorHandler, SRGANDiscriminatorHandler
from models.EDSR.model import EDSR
from models.TSRN.model import TSRN
from models.TSRN.handler import TSRNHandler
from models.ESRGAN.model import RRDBNet, ESRGAN
from models.ESRGAN.handler import ESRGANGeneratorHandler, ESRGANDiscriminatorHandler
from models.SRDenseNet.model import SRDenseNet
from models.RDN.model import RDN
from models.RealESRGAN.model import RealESRNet, RealESRGAN
from models.RealESRGAN.handler import RealESRGANGeneratorHandler, RealESRGANDiscriminatorHandler
from models.RCAN.model import RCAN
from models.SwiftSRGAN.model import SwiftSRResNet, SwiftSRGAN
from models.HPUN.model import HPUN

# PREFERENCE
crop = None
pairwise = False
train_path = './data/dataset/DIV2K/DIV2K_train_HR_Patches_128_128/*'
#train_path = ['./data/dataset/T91/Patches/*.png']
#pairwise=True
#train_path='./data/dataset/DIV2K/DIV2K_train_HR_Patches_256_128_RealESRGANx4/*_HR.png'
validate_path = './data/dataset/SET5/*'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--epochs', type=int, default=10, help="Number of Epochs")
  parser.add_argument('--generator', type=str, required=True, help="SISR generator model")
  parser.add_argument('--generator_state', type=str, help="SISR generator state")
  parser.add_argument('--discriminator', type=str, help="SISR discriminator model")
  parser.add_argument('--discriminator_patch', type=int, default=128, help="SISR discriminator patch size")
  parser.add_argument('--scale', type=int, default=2, help="Upscaling scale factor")
  parser.add_argument('--batch', type=int, default=1, help="Batch size")
  parser.add_argument('--amp', action='store_true', help="Train with amp")
  parser.add_argument('--distort', type=str, help="image distortion method")
  parser.add_argument('--y_only', action='store_true', help="Train y color only")

  args = parser.parse_args()

  generator_models = {
    'SRCNN': tuple([SRCNN(c=(1 if args.y_only else 3)), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'FSRCNN': tuple([FSRCNN(scale=args.scale, c=(1 if args.y_only else 3)), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'ESPCN': tuple([ESPCN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'VDSR': tuple([VDSR(c=(1 if args.y_only else 3)), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'DRCN': tuple([DRCN(c=(1 if args.y_only else 3)), DRCNHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'DRRN': tuple([DRRN(c=(1 if args.y_only else 3)), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'RED-Net': tuple([REDNet(c=(1 if args.y_only else 3)), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else InterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), InterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'LapSRN': tuple([LapSRN(c=(1 if args.y_only else 3), scale=args.scale), LapSRNHandler, MultiScaledImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), MultiScaledImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'MSLapSRN': tuple([MSLapSRN(c=(1 if args.y_only else 3), scale=args.scale), MSLapSRNHandler, MultiScaledImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), MultiScaledImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'EnhanceNet': tuple([EnhanceNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'SRDenseNet': tuple([SRDenseNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'TSRN': tuple([TSRN(c=(1 if args.y_only else 3), scale=args.scale), TSRNHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'SRResNet': tuple([SRResNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'EDSR': tuple([EDSR(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'RDN': tuple([RDN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'RRDBNet': tuple([RRDBNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'RCAN': tuple([RCAN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'Swift-SRResNet': tuple([SwiftSRResNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMSEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'Real-ESRNet': tuple([RealESRNet(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
    'HPUN': tuple([HPUN(c=(1 if args.y_only else 3), scale=args.scale), DefaultMAEHandler, PairwiseDataSet(path=train_path, y_only=args.y_only) if pairwise else NonInterpolatedImageDataset(path=train_path, crop=crop, scale=args.scale, distort=args.distort, y_only=args.y_only), NonInterpolatedImageDataset(path=validate_path, scale=args.scale, y_only=args.y_only), 0.0001]),
  }

  discriminator_models = {
    'EnhanceNet': tuple([EnhanceNetDiscriminator((1 if args.y_only else 3), size=tuple([args.discriminator_patch] * 2)), EnhanceNetGeneratorHandler, EnhanceNetDiscriminatorHandler, 0.0001]),
    'SRGAN': tuple([SRGAN((1 if args.y_only else 3), size=(tuple([args.discriminator_patch] * 2))), SRGANGeneratorHandler, SRGANDiscriminatorHandler, 0.0001]),
    'ESRGAN': tuple([ESRGAN((1 if args.y_only else 3), size=(tuple([args.discriminator_patch] * 2))), ESRGANGeneratorHandler, ESRGANDiscriminatorHandler, 0.0001]),
    'Swift-SRGAN': tuple([SwiftSRGAN((1 if args.y_only else 3)), SRGANGeneratorHandler, SRGANDiscriminatorHandler, 0.0001]),
    'Real-ESRGAN': tuple([RealESRGAN((1 if args.y_only else 3)), RealESRGANGeneratorHandler, RealESRGANDiscriminatorHandler, 0.0001]),
  }

  generator_model, geneartor_handler_class, train_set, validation_set, generator_lr = generator_models[args.generator]
  train_loader = DataLoader(dataset=train_set, batch_size=args.batch, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
  validation_loader = DataLoader(dataset=validation_set, num_workers=os.cpu_count(), pin_memory=True, persistent_workers=True)
  if args.discriminator in discriminator_models:
    discriminator_model, updated_generator_handler_class, discriminator_handler_class, discriminator_lr = discriminator_models[args.discriminator]

    g_optimizer = optim.Adam(generator_model.parameters(), lr=generator_lr)
    g_handler = updated_generator_handler_class(generator_model, geneartor_handler_class(generator_model)) if geneartor_handler_class is not None else updated_generator_handler_class(generator_model)
    d_optimizer = optim.Adam(discriminator_model.parameters(), lr=discriminator_lr)
    d_handler = discriminator_handler_class(discriminator_model)
    g_state = args.generator_state

    train = GANTrainer(g_model=generator_model, d_model=discriminator_model, g_optimizer=g_optimizer, g_handler=g_handler, g_state=g_state, d_optimizer=d_optimizer, d_handler=d_handler, train_loader=train_loader, test_loader=validation_loader, seed=None, use_amp=args.amp)
    train.run(epochs=args.epochs, save_dir=Path(f'./result/{args.discriminator}_x{args.scale}'), save_prefix='state')
  else:
    optimizer = optim.Adam(generator_model.parameters(), lr=generator_lr)
    handler = geneartor_handler_class(generator_model)
    state = args.generator_state

    train = GeneratorTrainer(model=generator_model, optimizer=optimizer, handler=handler, train_loader=train_loader, state=state, test_loader=validation_loader, seed=None, use_amp=args.amp)
    train.run(epochs=args.epochs, save_dir=Path(f'./result/{args.generator}_x{args.scale}'), save_prefix='state')
