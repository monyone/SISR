#!/usr/bin/env python3

import torchvision.io as io
import torchvision.utils as utils
import torchvision.transforms.functional as F

import argparse
import os
from pathlib import Path
from glob import glob

from degradation.real_esrgan import degradation_real_esrgan
from degradation.jpeg import degradation_jpeg

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--crop', type=int, default=32, help='crop size')
  parser.add_argument('--stride', type=int, default=14, help='crop size')
  parser.add_argument('--input_path', type=Path, required=True, help="input path")
  parser.add_argument('--output_dir', type=Path, required=True, help="output path")
  parser.add_argument('--pairwise', action='store_true', help="generate pairwise dataset")
  parser.add_argument('--scale', type=int, default=1, help="LR downscale factor if pairwise specified")
  parser.add_argument('--distort', type=str, default='bicubic', help="LR distotion algorithm if pairwise specified")

  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  for path_str in glob(str(args.input_path)):
    path = Path(path_str)
    img = io.read_image(path_str, mode=io.ImageReadMode.RGB) / 255
    _, w, h = img.size()
    for top in range(0, h - args.crop + 1, args.stride):
      for left in range(0, w - args.crop + 1, args.stride):
        bottom = min(h, top + args.crop)
        right = min(w, left + args.crop)
        cropped = F.crop(img, top=top, left=left, width=args.crop, height=args.crop)
        if args.pairwise:
          if tuple(map(lambda n: n // args.scale * args.scale, cropped.size()[1:3])) != tuple(cropped.size()[1:3]):
            cropped = F.resize(img=cropped, antialias=True, size=(tuple(map(lambda n: n // args.scale * args.scale, cropped.size()[1:3]))), interpolation=F.InterpolationMode.BICUBIC)
          # Gen HR
          utils.save_image(cropped, str(args.output_dir / f'{path.stem}_{top}_{left}_HR{path.suffix}'), nrow=1)
          # Gen LR
          if args.distort == 'Real-ESRGAN+':
            utils.save_image(degradation_real_esrgan(image=cropped, scale_factor=args.scale, use_sharpness=True), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
          elif args.distort == 'Real-ESRGAN':
            utils.save_image(degradation_real_esrgan(image=cropped, scale_factor=args.scale, use_sharpness=False), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
          elif args.distort == 'JPEG':
            utils.save_image(degradation_jpeg(image=cropped, scale_factor=args.scale, use_sharpness=False, interpolation=F.InterpolationMode.BICUBIC), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
          elif args.distort == 'BILINER':
            utils.save_image(F.resize(img=cropped, antialias=True, size=(tuple(map(lambda n: n // args.scale, cropped.size()[1:3]))), interpolation=F.InterpolationMode.BILINEAR), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
          elif args.distort == 'BICUBIC':
            utils.save_image(F.resize(img=cropped, antialias=True, size=(tuple(map(lambda n: n // args.scale, cropped.size()[1:3]))), interpolation=F.InterpolationMode.BICUBIC), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
          else:
            utils.save_image(F.resize(img=cropped, antialias=True, size=(tuple(map(lambda n: n // args.scale, cropped.size()[1:3]))), interpolation=F.InterpolationMode.BICUBIC), str(args.output_dir / f'{path.stem}_{top}_{left}_LR{path.suffix}'), nrow=1)
        else:
          # Gen HR
          utils.save_image(cropped, str(args.output_dir / f'{path.stem}_{top}_{left}{path.suffix}'), nrow=1)
