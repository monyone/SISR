
import torch
import torch.nn as nn

from torchvision.models.swin_transformer import SwinTransformerBlock

from math import log, sqrt

from typing import Callable

"""SwinIR Model

Site:
  "SwinIR: Image Restoration Using Swin Transformer (2021)" (https://arxiv.org/abs/2108.10257)

Caution:
  It is Experimental (FIXME)
"""

class BasicLayer(nn.Module):
  """ A basic Swin Transformer layer for one stage.

  """

  def __init__(self, depth: int, dim: int, num_heads, window_size: tuple[int, int], mlp_ratio: float = 4., dropout: float = 0., attention_dropout: float = 0., stochastic_depth_prob: float =0., norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm):
    super().__init__()
    self.blocks = nn.Sequential(*[
      SwinTransformerBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        shift_size= [0 for _ in window_size] if i % 2 == 0 else [w // 2 for w in window_size],
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=(norm_layer or nn.LayerNorm)
      )
      for i in range(depth)
    ])

  def forward(self, x: torch.Tensor):
    return self.blocks(x)

class PatchEmbed(nn.Module):
  """ Image to Patch Embedding

  Args:
    embed_dim (int): Number of linear projection output channels. Default: 96.
    norm_layer (Callable[..., nn.Module], optional): Normalization layer. Default: None
  """

  def __init__(self, embed_dim: int = 96, norm_layer: Callable[..., nn.Module] | None = None):
    super().__init__()
    self.embed_dim = embed_dim
    self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

  def forward(self, x: torch.Tensor):
    x = x.permute(0, 2, 3, 1)
    x = self.norm(x)
    return x

class PatchUnEmbed(nn.Module):
  """ Image to Patch Unembedding

  Args:
    embed_dim (int): Number of linear projection output channels. Default: 96.
  """

  def __init__(self, embed_dim: int = 96):
    super().__init__()
    self.embed_dim = embed_dim

  def forward(self, x: torch.Tensor):
    return x.permute(0, 3, 1, 2)

class RSTB(nn.Module):
  """Residual Swin Transformer Block (RSTB).

  """

  def __init__(self, depth: int, dim: int, num_heads, window_size: tuple[int, int], mlp_ratio: float = 4., dropout: float = 0., attention_dropout: float = 0., stochastic_depth_prob: float =0., norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm):
    super().__init__()

    self.residual_group = BasicLayer(
      depth=depth,
      dim=dim,
      num_heads=num_heads,
      window_size=window_size,
      mlp_ratio=mlp_ratio,
      dropout=dropout,
      attention_dropout=attention_dropout,
      stochastic_depth_prob=stochastic_depth_prob,
      norm_layer=norm_layer
    )
    self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=3//2, bias=False)
    self.patch_embed = PatchEmbed(embed_dim=dim)
    self.patch_unembed = PatchUnEmbed(embed_dim=dim)

  def forward(self, x: torch.Tensor):
    input = x
    x = self.residual_group(x)
    x = self.patch_unembed(x)
    x = self.conv(x)
    x = self.patch_embed(x)
    return x + input

class SwinIR(nn.Module):
  def __init__(self, scale: int, c: int = 3, embed_dim: int = 96, depth_and_num_heads: list[tuple[int, int]] = [(6, 6), (6, 6), (6, 6), (6, 6)], window_size: tuple[int, int] = (7, 7), mlp_ratio: float = 4., dropout: float = 0., attention_dropout: float = 0., stochastic_depth_prob: float = 0., norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm) -> None:
    """SwinIR's Constructor

    Args:
      scale (int): number of scaling factor.
      c (int): number of channel the input/output image.

    Examples:
      >>> SwinIR() # typical SRResNet parameters
    """
    super().__init__()

    self.scale = scale
    self.num_features = 64
    self.window_size = window_size
    self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=norm_layer)
    self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
    self.pos_drop = nn.Dropout(p=dropout)
    self.norm = norm_layer(embed_dim)
    self.conv_fst = nn.Conv2d(in_channels=c, out_channels=embed_dim, kernel_size=3, padding=3//2, bias=False)
    self.conv_snd = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=3//2, bias=False)

    # nearest + conv (less artifact)
    self.upscale = nn.Sequential(
      # pre upscale
      nn.Conv2d(in_channels=embed_dim, out_channels=self.num_features, kernel_size=3, padding=3//2, bias=False),
      nn.LeakyReLU(inplace=True),
      # Upscale
      *[
        nn.Sequential(
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=3, padding=3//2, bias=False),
          nn.LeakyReLU(inplace=True)
        )
        for _ in range(int(log(scale, 2)))
      ],
      # post Upscale
      nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=3, padding=3//2, bias=False),
      nn.LeakyReLU(inplace=True),
    )
    self.conv_lst = nn.Conv2d(in_channels=self.num_features, out_channels=c, kernel_size=3, padding=3//2, bias=False)

    self.rstbs = nn.ModuleList([
      RSTB(
        dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        norm_layer=norm_layer
      )
      for depth, num_heads in depth_and_num_heads
    ])

    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

  def forward(self, x):
    H, W = x.shape[2:]
    mod_pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
    mod_pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
    x = nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    #
    x = self.conv_fst(x)
    #
    residual = x
    x = self.patch_embed(x)
    x = self.pos_drop(x)
    for rstb in self.rstbs:
      x = rstb(x)
    x = self.norm(x)
    x = self.patch_unembed(x)
    x = self.conv_snd(x) + residual
    #
    x = self.upscale(x)
    x = self.conv_lst(x)
    return x[:, :, :H*self.scale, :W*self.scale]
