import torch
import torchvision.io as io
import torchvision.transforms.functional as F

import random
import math

# FIXME: It is Experimental

def gm_blur_kernel(mean: torch.Tensor, cov: torch.Tensor, size: int = 15):
  dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
  center = size / 2.0 + 0.5
  k = torch.zeros((size, size))
  for y in range(size):
    for x in range(size):
      cy = y - center + 1
      cx = x - center + 1
      k[y, x] = torch.exp(dist.log_prob(torch.Tensor([cx, cy])))
  k = k / torch.sum(k)
  return k

def anisotropic_gaussian(kernel_size: int = 15, theta: float = math.pi, l1: int = 6, l2: int = 6):
  rot = torch.Tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  v = rot @ torch.Tensor([1, 0])
  V = torch.Tensor([[v[0], v[1]], [v[1], -v[0]]])
  D = torch.Tensor([[l1, 0], [0, l2]])
  sigma = V @ D @ torch.linalg.inv(V)
  return gm_blur_kernel(mean=torch.zeros(2), cov=sigma, size=kernel_size)

def fspecial_gaussian(hsize: float, sigma: float):
  size = [(hsize - 1.0) / 2.0, (hsize - 1.0) / 2.0]
  [x, y] = torch.meshgrid([torch.arange(-size[1], size[1] + 1), torch.arange(-size[0], size[0] + 1)], indexing='xy')
  h = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
  h[h < torch.finfo(torch.float).eps * h.max()] = 0
  return h / h.sum() if h.sum() != 0 else h

def add_blur(image: torch.Tensor, scale_factor: int = 4):
  wd2 = 4.0 + scale_factor
  wd = 2.0 + 0.2 * scale_factor
  if random.random() < 0.5:
    l1 = wd2 * random.random()
    l2 = wd2 * random.random()
    k = anisotropic_gaussian(kernel_size=(2 * random.randint(2,11) + 3), theta=(random.random() * math.pi), l1=l1, l2=l2)
  else:
    k = fspecial_gaussian(hsize=(2 * random.randint(2,11) + 3), sigma=(wd * random.random()))
  k = k.unsqueeze(dim=0).unsqueeze(dim=0)
  return torch.nn.functional.conv2d(input=image.reshape(-1, 1, *image.shape[1:]), weight=k, padding=tuple(map(lambda s: s // 2, k.shape[2:]))).reshape(-1, *image.shape[1:])

def add_Gaussian_noise(image: torch.Tensor, noise_level1=2, noise_level2=25):
  noise_level = random.randint(noise_level1, noise_level2)
  rnum = random.random()
  if  rnum > 0.6: # add color Gaussian noise
    image += torch.randn(image.shape) * (noise_level / 255)
  elif rnum < 0.4: # add grayscale Gaussian noise
    image += torch.randn(image.shape[1:]).unsqueeze(dim=0).repeat_interleave(3, dim=0) * (noise_level / 255)
  else: # add  noise
    L = noise_level2 / 255
    D = torch.diag(torch.rand(3))
    svd = torch.linalg.svd(torch.rand([3, 3]))
    U = svd[0] @ svd[2]
    conv = torch.transpose(U, dim0=0, dim1=1) @ D @ U
    image += torch.distributions.MultivariateNormal(loc=torch.zeros([3]), covariance_matrix=torch.abs(conv * (L ** 2))).sample(image.shape[1:]).reshape(-1, *image.shape[1:])
    image = image.clamp(0, 1)
  return image

def add_JPEG_noise(image: torch.Tensor):
  return io.decode_jpeg(io.encode_jpeg((image * 255).clamp_(0, 255).to(torch.uint8), quality=random.randint(30, 95)), io.ImageReadMode.RGB) / 255

def degradation_bsrgan(image: torch.Tensor, scale_factor: int = 4):
  jpeg_prob, scale2_prob = 0.9, 0.25
  shuffle_order = random.sample(range(7), 7)
  idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
  if idx1 > idx2:  # keep downsample3 last
    shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

  # TODO: downsample1

  _, H, W = image.shape

  for i in shuffle_order:
    if i == 0:
      image = add_blur(image=image, scale_factor=scale_factor)
    elif i == 1:
      image = add_blur(image=image, scale_factor=scale_factor)
    elif i == 2:
      if random.random() < 0.75:
        sf1 = random.uniform(1, 2 * scale_factor)
        image = F.resize(img=image, size=[int(1 / sf1 * H), int(1 / sf1 * W)], antialias=False, interpolation=random.choice([F.InterpolationMode.NEAREST, F.InterpolationMode.BILINEAR, F.InterpolationMode.BICUBIC]))
      else:
        k = fspecial_gaussian(hsize=(25 + (scale_factor - 1) / 2), sigma=random.uniform(0.1, 0.6 * scale_factor))
        k_shifted = F.crop(img=k.unsqueeze(dim=0), top=0, left=int((scale_factor - 1) / 2), height=25, width=25).reshape(25, 25)
        k_shifted = (k_shifted/k_shifted.sum()).unsqueeze(dim=0).unsqueeze(dim=0)  # blur with shifted kernel
        image = torch.nn.functional.conv2d(input=image.reshape(-1, 1, *image.shape[1:]), weight=k_shifted, padding=tuple(map(lambda s: s // 2, k_shifted.shape[2:]))).reshape(-1, *image.shape[1:])
      image = image.clamp(0, 1)
    elif i == 3:
      image = F.resize(img=image, size=[int(1 / scale_factor * H), int(1 / scale_factor * W)], antialias=False, interpolation=random.choice([F.InterpolationMode.NEAREST, F.InterpolationMode.BILINEAR, F.InterpolationMode.BICUBIC]))
      image = image.clamp(0, 1)
    elif i == 4:
      pass
      # add Gaussian noise
      image = add_Gaussian_noise(image, noise_level1=2, noise_level2=25)
    elif i == 5:
      if random.random() < jpeg_prob:
        image = add_JPEG_noise(image)
    elif i == 6:
      # Camera ISP model
      pass

  image = add_JPEG_noise(image)
  return image

