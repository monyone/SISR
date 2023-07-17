import torch
import torchvision.io as io
import torchvision.transforms.functional as F

import random

def degration_dncnn(image: torch.Tensor, scale_factor: int = 4, interpolation: F.InterpolationMode = F.InterpolationMode.BICUBIC):
  _, H, W = image.shape

  # Gausian Noise
  sigma = random.randint(0, 55) / 255
  image = image + torch.normal(0, sigma, image.shape)

  # Resize
  downscale_factor = random.uniform(1, 4)
  image = F.resize(img=F.resize(img=image, antialias=True, size=tuple(map(int, (H / downscale_factor, W / downscale_factor))), interpolation=interpolation), antialias=True, size=(H, W), interpolation=interpolation)

  # Jpeg
  jpeg_quality = random.randrange(5, 99)
  image = io.decode_jpeg(io.encode_jpeg((image * 255).clamp_(0, 255).to(torch.uint8), quality=jpeg_quality), io.ImageReadMode.RGB) / 255

  # DownScale
  return F.resize(img=image, antialias=True, size=tuple(map(int, (H / scale_factor, W / scale_factor))), interpolation=interpolation)

