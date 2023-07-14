import torch
import torchvision.io as io

import random

def degradation_jpeg(image: torch.Tensor):
  return io.decode_jpeg(io.encode_jpeg((image * 255).clamp_(0, 255).to(torch.uint8), quality=random.randrange(70, 95)), io.ImageReadMode.RGB) / 255
