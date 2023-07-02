#!/usr/bin/env python3

import torch
import torch.nn as nn

import torchvision.io as io
from torchvision.transforms import Normalize
from torchvision.transforms.functional import crop
from torchvision.models.vgg import vgg19, VGG19_Weights

import argparse
import os
from pathlib import Path
from glob import glob

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch SISR (Single Image Super Resolution)')
  parser.add_argument('--input', type=Path, required=True, help="input path")
  parser.add_argument('--output', type=Path, required=True, help="output path")
  parser.add_argument('--crop', type=int, default=128, help='crop size')
  parser.add_argument('--stride', type=int, default=64, help='crop size')
  parser.add_argument('--average_pooling', action='store_true', help="replace vgg maxpool to avgpool")

  args = parser.parse_args()
  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
  normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
  if args.average_pooling:
    for idx in range(len(vgg.features)):
      if isinstance(vgg.features[idx], nn.MaxPool2d): vgg.features[idx] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
  for param in vgg.parameters(): param.requires_grad = False
  vgg = vgg.to(device=device)
  vgg.eval()

  for idx, path_str in enumerate(glob(str(args.input))):
    path = Path(path_str)
    img = normalize(torch.unsqueeze(io.read_image(path_str, mode=io.ImageReadMode.RGB) / 255, dim=0))
    _, _, w, h = img.size()
    for top in range(0, h - args.crop + 1, args.stride):
      for left in range(0, w - args.crop + 1, args.stride):
        cropped = crop(img=img, top=top, left=left, height=args.crop, width=args.crop).to(device=device)
        with torch.no_grad():
          x: torch.Tensor = vgg.features[0](cropped)
          prev_conv_layer_mean: torch.Tensor | None = None
          for layer in vgg.features[1:]:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
              layer.weight /= x.mean()
              layer.bias /= x.mean()
              x /= x.mean()
    print(f'[{idx}] {path} done!')
    torch.save(vgg.state_dict(), args.output)


