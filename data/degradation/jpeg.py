import torch
import torchvision.io as io
import torchvision.transforms.functional as F

import random

def degradation_jpeg(image: torch.Tensor, scale_factor: int, interpolation: F.InterpolationMode):
  jpeg = io.decode_jpeg(io.encode_jpeg((image * 255).clamp_(0, 255).to(torch.uint8), quality=random.randrange(30, 95)), io.ImageReadMode.RGB) / 255
  return F.resize(img=jpeg, antialias=True, size=(tuple(map(lambda n: n // scale_factor, jpeg.size()[1:3]))), interpolation=interpolation)
