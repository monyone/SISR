#!/usr/bin/env python3

import torchvision.io as io
import torchvision.utils as utils
import torchvision.transforms.functional as F

import argparse
import os
from pathlib import Path
from glob import glob

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--crop', type=int, default=128, help='crop size')
  parser.add_argument('--input_path', type=Path, required=True, help="input path")
  parser.add_argument('--output_dir', type=Path, required=True, help="output path")

  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  for path_str in glob(str(args.input_path)):
    path = Path(path_str)
    img = io.read_image(path_str, mode=io.ImageReadMode.RGB) / 255
    _, w, h = img.size()
    for top in range(0, h - (h % args.crop), args.crop):
      for left in range(0, w - (w % args.crop), args.crop):
        bottom = min(h, top + args.crop)
        right = min(w, left + args.crop)
        utils.save_image(F.crop(img, top=top, left=left, width=args.crop, height=args.crop), str(args.output_dir / f'{path.stem}_{top}_{left}{path.suffix}'), nrow=1)
